"""
methylbert.data.nanopore.finetune_extract

ONT spanning-read extractor for MethylBERT fine-tuning data generation.

This is the long-read replacement for the bisulfite `read_extract` in
methylbert.data.finetune_data_generate. It is designed to be passed as the
`read_extract_sequences_func` hook to `finetune_data_generate`, so that all
the surrounding orchestration (DMR selection, dmr_label re-numbering,
per-BAM ctype assignment, read-name train/eval split) is reused unchanged.

Design (locked decisions):
  - Window: 512 bases centred on the DMR midpoint  ->  510 stride-1 3-mers.
    mid = (start + end) // 2 ;  window = [mid - 256, mid + 256).
  - DNA channel: REFERENCE sequence of the window, tokenised to 3-mer ids.
    Identical for every read over a DMR -> forces the classifier onto
    methylation, not COLO829 somatic SNVs.
  - Methylation channel: the READ's calls, transported onto reference
    coordinates. We REUSE featurize_read (untouched) for the read-space
    state logic (thresholds, 5mC+5hmC sum, reverse-strand handling), then
    remap via get_aligned_pairs. The three CpG cases fall out of one
    initialisation (see _methyl_states_for_window).
  - Read filter: FULL-WINDOW SPAN only. A read must cover the entire 512-base
    window (reference_start <= w_start and reference_end >= w_end). Partial
    reads are dropped (counted, not handled - parked).
  - Token-count convention: L - 2 = 510 (matches featurize_read / pretraining;
    NOT the original finetune kmers() L-3 off-by-one).
  - Output dna_seq is SPACE-SEPARATED INTEGER TOKEN IDS (Option B), so the
    fine-tune loader parses ints directly instead of letters->to_seq.

Returned DataFrame columns (consumed by finetune_data_generate):
    name        read query name (used for the train/eval split key)
    RF          dna_seq: space-separated int token ids (510 of them)
    ME          methyl_seq: 510-char digit string in {0,1,2,3}
    dmr_ctype   the DMR's characteristic cell type (from the DMR row)
    dmr_label   contiguous 0..N-1 DMR id (assigned by finetune_data_generate)
finetune_data_generate renames RF->dna_seq, ME->methyl_seq and adds `ctype`.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pysam

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.featurize import (
    featurize_read,
    STATE_NON_CPG,
    STATE_UNKNOWN,
)

logger = logging.getLogger(__name__)

WINDOW_BASES = 512          # reference window width
HALF = WINDOW_BASES // 2    # 256
N_TOKENS = WINDOW_BASES - 2  # 510 stride-1 3-mers (L-2 convention)


def _window_for_dmr(start: int, end: int):
    """Single source of truth for the window coordinates of a DMR.

    Returns (w_start, w_end) as a half-open interval [w_start, w_end) of
    exactly WINDOW_BASES bases, centred on the DMR midpoint.
    """
    mid = (int(start) + int(end)) // 2
    return mid - HALF, mid + HALF


def _reference_tokens(ref_window: str, vocab: MethylVocab):
    """Tokenise a WINDOW_BASES-long reference string into N_TOKENS 3-mer ids.

    N-containing or otherwise unknown 3-mers map to vocab.unk_index, exactly
    as featurize_read does for read tokens.
    """
    stoi = vocab.stoi
    unk = vocab.unk_index
    toks = np.empty(N_TOKENS, dtype=np.int32)
    for j in range(N_TOKENS):
        toks[j] = stoi.get(ref_window[j:j + 3], unk)
    return toks


def _methyl_states_for_window(read, ref_window, w_start, vocab, mm_flag):
    """Build the reference-anchored methylation state array for one read.

    Returns an int array of length N_TOKENS with values in {0,1,2,3}, or None
    if featurize_read fails on the read.

    The three cases fall out of the initialisation order, no special-casing:
      init every position NON_CPG (2)                        -> case 3 default
      mark reference CpG positions UNKNOWN (3)               -> case 2 default
      overwrite a reference CpG with the read's {0,1,3} call -> case 1
    A read CpG that maps to a non-CpG reference position has nowhere to land
    and is ignored (case 3). A reference CpG the read didn't call / didn't
    reach / disagrees on stays UNKNOWN (case 2).
    """
    # 1. featurize_read gives read-space (token, state) arrays with all the
    #    reverse-strand handling already correct. We only need the states.
    try:
        _read_tokens, read_states = featurize_read(read, vocab, mm_flag=mm_flag)
    except (ValueError, RuntimeError) as e:
        logger.debug("featurize_read failed on %r: %s", read.query_name, e)
        return None

    # Recover {query position of CpG-C -> state} for every read CpG.
    # featurize sets state at token index i for the CpG whose C is at query
    # position i+1; non-CpG tokens are STATE_NON_CPG.
    read_cpg = {
        i + 1: int(read_states[i])
        for i in range(len(read_states))
        if read_states[i] != STATE_NON_CPG
    }

    # 2. Initialise the reference window's states.
    states = np.full(N_TOKENS, STATE_NON_CPG, dtype=np.int8)
    for j in range(N_TOKENS):
        # 3-mer j has its middle base at ref_window[j+1]; CpG iff "CG" there.
        if ref_window[j + 1] == "C" and ref_window[j + 2] == "G":
            states[j] = STATE_UNKNOWN

    # 3. Transport read calls onto reference coordinates via the alignment.
    #    get_aligned_pairs query indices share featurize's query coordinate
    #    system (both index query_sequence / BAM SEQ), so the join is direct.
    q2r = {}
    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
        q2r[qpos] = rpos

    for qpos, state in read_cpg.items():
        rpos = q2r.get(qpos)
        if rpos is None:
            continue  # read CpG-C in an insertion / soft-clip -> no ref pos
        j = rpos - w_start - 1  # token whose middle base sits at rpos
        if 0 <= j < N_TOKENS and states[j] == STATE_UNKNOWN:
            # states[j]==UNKNOWN means reference confirms a CpG here (case 1).
            # states[j]==NON_CPG means reference says non-CpG -> leave it (case 3).
            states[j] = state

    return states


def ont_read_extract(
    bam_file_path: str,
    dict_ref: dict,
    k: int = 3,
    dmrs: Optional[pd.DataFrame] = None,
    ncores: int = 1,
    methyl_caller: str = "dorado",
    *,
    mm_flag: str = "?",
):
    """ONT replacement for finetune_data_generate.read_extract.

    Matches the hook signature (bam_file_path, dict_ref, k, dmrs, ncores,
    methyl_caller); mm_flag is bound separately (e.g. via functools.partial)
    because the Dorado COLO829 BAMs use the '?' MM flag and that is not part
    of the hook contract.

    Processed serially over DMRs - at ~100 DMRs and modest coverage this is
    fast enough; parallelising over DMRs is trivial later if needed.
    """
    if k != 3:
        raise ValueError(f"This extractor is 3-mer only; got k={k}")
    if dmrs is None or dmrs.shape[0] == 0:
        return pd.DataFrame([])

    vocab = MethylVocab(k=3)  # deterministic; identical to the loader's vocab

    rows = []
    n_span = 0      # reads fully spanning a window (kept)
    n_clip = 0      # reads overlapping but not fully spanning (dropped)
    n_offedge = 0   # DMRs whose window spills off the contig (skipped)

    aln = pysam.AlignmentFile(bam_file_path, "rb")
    try:
        for dmr in dmrs.to_dict("records"):
            chromo = dmr["chr"]
            w_start, w_end = _window_for_dmr(dmr["start"], dmr["end"])

            contig_seq = dict_ref.get(chromo)
            if contig_seq is None:
                continue
            if w_start < 0 or w_end > len(contig_seq):
                n_offedge += 1
                continue

            ref_window = contig_seq[w_start:w_end].upper()
            if len(ref_window) != WINDOW_BASES:
                n_offedge += 1
                continue

            ref_tokens = _reference_tokens(ref_window, vocab)
            ref_token_str = " ".join(str(int(t)) for t in ref_tokens)

            for read in aln.fetch(chromo, w_start, w_end, until_eof=True):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                # Full-window span: read must cover the entire window.
                if read.reference_start > w_start or read.reference_end < w_end:
                    n_clip += 1
                    continue

                states = _methyl_states_for_window(
                    read, ref_window, w_start, vocab, mm_flag
                )
                if states is None:
                    continue

                n_span += 1
                rows.append({
                    "name": read.query_name,
                    "RF": ref_token_str,                       # same per DMR
                    "ME": "".join(str(int(s)) for s in states),
                    "dmr_ctype": dmr["ctype"],
                    "dmr_label": int(dmr["dmr_id"]),
                })
    finally:
        aln.close()

    logger.info(
        "%s: %d windows kept (full-span), %d reads dropped (partial), "
        "%d DMRs skipped (off-edge)",
        bam_file_path, n_span, n_clip, n_offedge,
    )
    if not rows:
        return pd.DataFrame([])
    return pd.DataFrame(rows)