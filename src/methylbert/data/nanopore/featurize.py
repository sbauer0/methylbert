"""
methylbert.data.nanopore.featurize

Convert a single nanopore-sequenced read (with MM/ML tags) into a pair of
arrays suitable for MethylBERT pretraining:

    (token_ids, methyl_states)

where token_ids are 3-mer indices into a MethylVocab and methyl_states are in
{0, 1, 2, 3} (0 = unmethylated CpG, 1 = methylated CpG, 2 = non-CpG, 3 = unknown CpG).
Both arrays are in forward-strand canonical orientation.

Design contract:
    - 4-state methylation embedding
    - threshold: state 1 if P(5mC) + P(5hmC) >= 0.8;
                 state 0 if P(5mC) + P(5hmC) <= 0.2;
                 state 3 otherwise
    - unlisted Cs in CpG context: state 0 if BAM flag '.', state 3 if BAM flag '?'
    - non-CpG Cs in forward orientation: state 2
    - 3-mer ownership: the 3-mer whose MIDDLE base is the C of the CpG carries
      the label (matches original MethylBERT's `kmers()` in finetune_data_generate.py,
      which sets `mid = int(k/2)` and rejects even k. For k=3, the C of the CpG
      is at seq[c]; the labeled 3-mer is seq[c-1 : c+2] = "XCG", at index c-1.)
    - forward-strand canonicalization: handled via BAM SEQ (always reference forward)
      and explicit position translation for reverse-mapped MM/ML calls

Strand handling — important and easy to get wrong:

    BAM SEQ (i.e. pysam's query_sequence) is always in reference forward orientation
    by SAM spec. For a forward-mapped read, this matches the original sequenced read.
    For a reverse-mapped read, BAM SEQ is the reverse complement of what was sequenced.

    pysam.modified_bases returns positions translated to BAM SEQ orientation
    (verified empirically against pysam 0.22+). For a forward-mapped read, the base
    at the reported position is the C that was modified. For a reverse-mapped read,
    the base at the reported position is a G (the complement of the C in the
    original read), and the C of the symmetric forward-strand CpG is at position-1.
"""

import numpy as np


# ---- Constants -----------------------------------------------------------

P_HIGH = 0.8
P_LOW = 0.2

STATE_UNMETH = 0
STATE_METH = 1
STATE_NON_CPG = 2
STATE_UNKNOWN = 3

FLAG_IMPLICIT = "."   # unlisted CpG Cs are implicitly canonical -> STATE_UNMETH
FLAG_EXPLICIT = "?"   # unlisted CpG Cs are explicitly unknown   -> STATE_UNKNOWN


# ---- Main entry point ----------------------------------------------------

def featurize_read(read, vocab, mm_flag):
    """
    Convert a pysam AlignedSegment into (token_ids, methyl_states) arrays.

    Parameters
    ----------
    read : pysam.AlignedSegment
        A primary, mapped nanopore read with MM and ML tags. The caller is
        responsible for filtering (e.g. discarding unmapped/secondary/supplementary
        reads, MAPQ filtering).
    vocab : MethylVocab
        4-base 3-mer vocabulary. Used to convert 3-mer strings to token IDs.
    mm_flag : str
        The MM tag flag character for this BAM, either '.' or '?'.
        Pulled from the manifest entry. Determines how unlisted CpG Cs are
        treated.

    Returns
    -------
    token_ids : np.ndarray, shape (n_tokens,), dtype int32
        3-mer indices into vocab. Length is len(query_sequence) - 2.
        N-containing 3-mers (or any 3-mer not in vocab) map to vocab.unk_index.
    methyl_states : np.ndarray, shape (n_tokens,), dtype int8
        Per-token methylation state in {0, 1, 2, 3}.

    Raises
    ------
    ValueError
        If mm_flag is not '.' or '?', or the read is unmapped, or the read has
        no query sequence.
    """
    if mm_flag not in (FLAG_IMPLICIT, FLAG_EXPLICIT):
        raise ValueError(
            f"mm_flag must be {FLAG_IMPLICIT!r} or {FLAG_EXPLICIT!r}, got {mm_flag!r}"
        )
    if read.is_unmapped:
        raise ValueError(f"read {read.query_name!r} is unmapped")

    seq = read.query_sequence
    if seq is None:
        raise ValueError(f"read {read.query_name!r} has no query sequence")
    L = len(seq)
    is_reverse = read.is_reverse

    # ---- 1. Aggregate per-CpG methylation probabilities -----------------
    # Keys: position of the C of the forward-strand CpG (in BAM SEQ coords).
    p_m_per_cpg = {}
    p_h_per_cpg = {}

    mod_bases = read.modified_bases or {}
    for key, positions in mod_bases.items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        base, _strand, code = key
        if base != "C":
            continue
        if code not in ("m", "h"):
            continue
        for pos, qual in positions:
            # Translate to "C of forward-strand CpG" position.
            # pysam returns positions in BAM SEQ orientation (verified empirically).
            #   - forward-mapped: seq[pos] is the modified C; CpG's C is at pos.
            #   - reverse-mapped: seq[pos] is a G (complement of read C);
            #                     symmetric forward CpG's C is at pos - 1.
            forward_c_pos = (pos - 1) if is_reverse else pos
            # Defensive: skip if not a valid forward CpG context. This naturally
            # discards all-context calls on non-CpG Cs (HG008 normals scenario)
            # and any out-of-bounds edges.
            if forward_c_pos < 0 or forward_c_pos + 1 >= L:
                continue
            if seq[forward_c_pos] != "C" or seq[forward_c_pos + 1] != "G":
                continue
            # ML byte N represents probability bin [N/256, (N+1)/256). Use N/256
            # as the lower-bound interpretation (the common convention).
            p = qual / 256.0
            if code == "m":
                p_m_per_cpg[forward_c_pos] = p
            else:  # 'h'
                p_h_per_cpg[forward_c_pos] = p

    # ---- 2. Per-CpG state from accumulated probabilities ----------------
    cpg_state = {}
    for i in range(L - 1):
        if seq[i] != "C" or seq[i + 1] != "G":
            continue
        p_m = p_m_per_cpg.get(i)
        p_h = p_h_per_cpg.get(i)
        if p_m is None and p_h is None:
            # Unlisted CpG C -> use BAM-level flag semantic
            cpg_state[i] = (
                STATE_UNMETH if mm_flag == FLAG_IMPLICIT else STATE_UNKNOWN
            )
        else:
            p_total = (p_m or 0.0) + (p_h or 0.0)
            if p_total >= P_HIGH:
                cpg_state[i] = STATE_METH
            elif p_total <= P_LOW:
                cpg_state[i] = STATE_UNMETH
            else:
                cpg_state[i] = STATE_UNKNOWN

    # ---- 3. Tokenize 3-mers and assign methylation states ---------------
    n_tokens = L - 2
    if n_tokens <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)

    token_ids = np.empty(n_tokens, dtype=np.int32)
    methyl_states = np.full(n_tokens, STATE_NON_CPG, dtype=np.int8)

    unk = vocab.unk_index
    stoi = vocab.stoi
    for i in range(n_tokens):
        kmer = seq[i:i + 3]
        token_ids[i] = stoi.get(kmer, unk)
        # Middle-base anchoring: the 3-mer at index i has its middle base at
        # seq[i+1]. If that base is the C of a forward-strand CpG, the 3-mer
        # (shape "XCG") carries the methylation label.
        c_pos = i + 1
        if c_pos in cpg_state:
            methyl_states[i] = cpg_state[c_pos]

    return token_ids, methyl_states