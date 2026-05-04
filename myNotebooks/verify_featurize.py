#!/usr/bin/env python3
"""
verify_featurize.py — independent verification of the nanopore featurizer.

Strategy:
    Re-implement (token_ids, methyl_states) computation from scratch using
    only raw MM/ML tag strings and explicit string operations, completely
    bypassing pysam's mod-base helpers. Run both the reference (this file)
    and the production featurize_read on the same real reads. Report any
    disagreement.

    Also produce a visual spot-check: for the first few CpGs in each read,
    print the surrounding context, the 3-mer that carries the methylation
    label, the ML probability, and the assigned state — so a human can
    eyeball that the labels look right.

Usage:
    python verify_featurize.py PATH/TO/BAM --mm-flag '?' [--n-reads 5]

The BAM path and mm-flag should match a row of your preprocessing manifest.
If you have a manifest, paste the value from the appropriate row.

Exit code: 0 iff every checked read had perfect token AND state agreement.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pysam

# Import the production code under test:
from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.featurize import (
    featurize_read,
    STATE_UNMETH, STATE_METH, STATE_NON_CPG, STATE_UNKNOWN,
    FLAG_IMPLICIT, FLAG_EXPLICIT,
    P_HIGH, P_LOW,
)


STATE_NAMES = {
    STATE_UNMETH: "unmeth(0)",
    STATE_METH: "meth(1)",
    STATE_NON_CPG: "non-CpG(2)",
    STATE_UNKNOWN: "unknown(3)",
}

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(s: str) -> str:
    return s.translate(COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# MM tag parsing (pure string operations — no pysam helpers)
# ---------------------------------------------------------------------------

def parse_mm_tag(mm_str: str):
    """
    Parse an MM tag value into a list of blocks.

    Each block is a dict: {base, strand, code, flag, skips}.
    'skips' is a list of integer skip-counts as listed in the tag.
    Pure string operations; does not call into pysam mod-base helpers.
    """
    blocks = []
    if not mm_str:
        return blocks
    for block in mm_str.rstrip(";").split(";"):
        if not block:
            continue
        parts = block.split(",")
        header = parts[0]
        skips = [int(p) for p in parts[1:]]
        # Header: <base><strand><code>[?|.]
        base = header[0]
        strand = header[1]
        if header[-1] in "?.":
            code = header[2:-1]
            flag = header[-1]
        else:
            code = header[2:]
            flag = ""
        blocks.append({
            "base": base, "strand": strand, "code": code,
            "flag": flag, "skips": skips,
        })
    return blocks


# ---------------------------------------------------------------------------
# Reference featurizer (independent reimplementation)
# ---------------------------------------------------------------------------

def reference_featurize_read(read, vocab, mm_flag):
    """
    Compute (token_ids, methyl_states) from scratch.

    Uses only:
        - read.query_sequence  (BAM SEQ, in reference forward orientation)
        - read.is_reverse
        - read.get_tag("MM"), read.get_tag("ML")  (raw values)
        - hand-rolled MM parsing
        - hand-rolled position translation for reverse-mapped reads

    Does NOT use pysam.modified_bases / .modified_bases_forward; this is the
    whole point of the verification.
    """
    seq = read.query_sequence
    L = len(seq)
    is_reverse = read.is_reverse

    mm_str = read.get_tag("MM") if read.has_tag("MM") else read.get_tag("Mm")
    ml_raw = read.get_tag("ML") if read.has_tag("ML") else read.get_tag("Ml")
    ml_bytes = list(ml_raw)

    blocks = parse_mm_tag(mm_str)

    # MM positions are listed in *original-read* (i.e. as-sequenced) orientation.
    # For forward-mapped reads, original_read == seq. For reverse-mapped reads,
    # original_read is the reverse complement of seq.
    original_read = reverse_complement(seq) if is_reverse else seq

    # Pre-compute Cs in original-read orientation; MM skip-counts walk this list.
    c_positions_in_original = [i for i, b in enumerate(original_read) if b == "C"]

    # Walk blocks to fill per-CpG probabilities. ML bytes are concatenated
    # across blocks in MM block order; each block consumes len(block.skips)
    # bytes. Track ml_offset accordingly.
    p_m_per_cpg = {}
    p_h_per_cpg = {}
    ml_offset = 0

    for block in blocks:
        n_block_bytes = len(block["skips"])

        # We only care about cytosine modifications m and h.
        if block["base"] != "C" or block["code"] not in ("m", "h"):
            ml_offset += n_block_bytes
            continue

        c_index = -1   # which entry of c_positions_in_original we're on
        for skip_idx, n_skip in enumerate(block["skips"]):
            # MM semantic: skip n_skip Cs, then the next C is modified.
            c_index += n_skip + 1
            if c_index >= len(c_positions_in_original):
                # Tag refers past end of read; should not happen in valid data.
                break
            original_pos = c_positions_in_original[c_index]

            # Translate to "C of forward-strand CpG" position in seq.
            # Forward-mapped: original_read == seq, so original_pos is already
            # the position of a C in seq.
            # Reverse-mapped: the C at original_pos in original_read corresponds
            # to a G in seq at position L-1-original_pos. The C of the symmetric
            # forward-strand CpG is one base earlier, at L-2-original_pos.
            if is_reverse:
                forward_c_pos = L - 2 - original_pos
            else:
                forward_c_pos = original_pos

            # Anchor only if this position is the C of a forward CpG. Non-CpG C
            # calls (all-context model output) are silently dropped — same as
            # the production featurizer.
            if 0 <= forward_c_pos and forward_c_pos + 1 < L:
                if seq[forward_c_pos] == "C" and seq[forward_c_pos + 1] == "G":
                    qual = ml_bytes[ml_offset + skip_idx]
                    p = qual / 256.0
                    if block["code"] == "m":
                        p_m_per_cpg[forward_c_pos] = p
                    else:
                        p_h_per_cpg[forward_c_pos] = p

        ml_offset += n_block_bytes

    # Build per-CpG state.
    cpg_state = {}
    for i in range(L - 1):
        if seq[i] == "C" and seq[i + 1] == "G":
            p_m = p_m_per_cpg.get(i)
            p_h = p_h_per_cpg.get(i)
            if p_m is None and p_h is None:
                cpg_state[i] = STATE_UNMETH if mm_flag == FLAG_IMPLICIT else STATE_UNKNOWN
            else:
                p_total = (p_m or 0.0) + (p_h or 0.0)
                if p_total >= P_HIGH:
                    cpg_state[i] = STATE_METH
                elif p_total <= P_LOW:
                    cpg_state[i] = STATE_UNMETH
                else:
                    cpg_state[i] = STATE_UNKNOWN

    # Tokenize with middle-base anchoring.
    n_tokens = L - 2
    if n_tokens <= 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8),
                cpg_state, p_m_per_cpg, p_h_per_cpg)

    token_ids = np.empty(n_tokens, dtype=np.int32)
    methyl_states = np.full(n_tokens, STATE_NON_CPG, dtype=np.int8)

    for i in range(n_tokens):
        kmer = seq[i:i + 3]
        token_ids[i] = vocab.stoi.get(kmer, vocab.unk_index)
        c_pos = i + 1   # middle of 3-mer at index i
        if c_pos in cpg_state:
            methyl_states[i] = cpg_state[c_pos]

    # Return the auxiliary dicts too so the spot-check section can use them.
    return token_ids, methyl_states, cpg_state, p_m_per_cpg, p_h_per_cpg


# ---------------------------------------------------------------------------
# Comparison + visual spot-check
# ---------------------------------------------------------------------------

def compare_reads(ref_tokens, ref_states, prod_tokens, prod_states):
    """Return (n_total, n_token_disagree, n_state_disagree, sample_disagreements)."""
    assert len(ref_tokens) == len(prod_tokens), \
        f"length mismatch: ref={len(ref_tokens)}, prod={len(prod_tokens)}"
    n = len(ref_tokens)
    token_disagree = (ref_tokens != prod_tokens)
    state_disagree = (ref_states != prod_states)
    sample = []
    for i in np.where(token_disagree | state_disagree)[0][:10]:
        sample.append({
            "index": int(i),
            "ref_token": int(ref_tokens[i]),
            "prod_token": int(prod_tokens[i]),
            "ref_state": int(ref_states[i]),
            "prod_state": int(prod_states[i]),
        })
    return n, int(token_disagree.sum()), int(state_disagree.sum()), sample


def print_visual_spot_check(read, vocab, mm_flag, ref_tokens, ref_states,
                             cpg_state, p_m, p_h, n_examples=6):
    """
    Print first n_examples CpGs in the read with rich context: position,
    surrounding bases, 3-mer that carries the label, the m/h probabilities,
    and the assigned state. A human reading this should see "yes the C of the
    CpG is in the middle of that 3-mer; yes the probability translates to that
    state under our threshold rule."
    """
    seq = read.query_sequence
    L = len(seq)

    # Find all forward-CpG C positions in order.
    cpg_c_positions = [i for i in range(L - 1) if seq[i] == "C" and seq[i + 1] == "G"]
    if not cpg_c_positions:
        print("    (no forward-strand CpGs in this read)")
        return

    print(f"    Visual spot-check on first {min(n_examples, len(cpg_c_positions))} "
          f"of {len(cpg_c_positions)} forward CpGs in the read:")
    for c_pos in cpg_c_positions[:n_examples]:
        # Surrounding context: 4 bases before and 4 after the C.
        win_lo, win_hi = max(0, c_pos - 4), min(L, c_pos + 5)
        context = seq[win_lo:win_hi]
        # The 3-mer carrying the label has its middle at c_pos, so its index is c_pos - 1
        kmer_idx = c_pos - 1
        kmer_str = seq[kmer_idx:kmer_idx + 3] if 0 <= kmer_idx <= L - 3 else None
        # Probabilities — produced by the reference path
        p_m_val = p_m.get(c_pos)
        p_h_val = p_h.get(c_pos)
        # State recorded at the labelled 3-mer in production output
        state_at_kmer = (int(ref_states[kmer_idx])
                        if kmer_idx is not None and 0 <= kmer_idx < len(ref_states)
                        else None)
        # Compute expected state from the probabilities for the explanation
        if p_m_val is None and p_h_val is None:
            p_total = None
            expected = STATE_UNMETH if mm_flag == FLAG_IMPLICIT else STATE_UNKNOWN
            why = (f"no MM/ML entry; flag={mm_flag!r} -> "
                   f"{STATE_NAMES[expected]}")
        else:
            p_total = (p_m_val or 0.0) + (p_h_val or 0.0)
            if p_total >= P_HIGH:
                expected = STATE_METH
            elif p_total <= P_LOW:
                expected = STATE_UNMETH
            else:
                expected = STATE_UNKNOWN
            why = (f"P_m={p_m_val if p_m_val is not None else '-'}, "
                   f"P_h={p_h_val if p_h_val is not None else '-'}, "
                   f"sum={p_total:.3f} -> {STATE_NAMES[expected]}")

        print(f"      CpG @ forward C-pos {c_pos}:")
        print(f"        seq[{win_lo}:{win_hi}] = {context!r}  (C is at index {c_pos - win_lo} of context)")
        if kmer_str is not None:
            mid_check = "✓" if kmer_str[1] == "C" else "✗ ANCHORING WRONG"
            print(f"        labelled 3-mer at index {kmer_idx}: {kmer_str!r} "
                  f"(middle base = '{kmer_str[1]}' {mid_check})")
        else:
            print(f"        labelled 3-mer: not addressable (CpG too close to read edge)")
        print(f"        threshold rule: {why}")
        if state_at_kmer is not None:
            match = "✓" if state_at_kmer == expected else "✗ DISAGREEMENT"
            print(f"        production state at index {kmer_idx}: "
                  f"{STATE_NAMES[state_at_kmer]} {match}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def find_usable_reads(bam_path, n, max_scan):
    """Iterate the BAM and return the first n primary-mapped reads with
    MM and ML tags. Stops scanning after max_scan reads to bound runtime."""
    out = []
    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if len(out) >= n or i >= max_scan:
                break
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if not (read.has_tag("MM") or read.has_tag("Mm")):
                continue
            if not (read.has_tag("ML") or read.has_tag("Ml")):
                continue
            out.append(read)
    finally:
        bam.close()
    return out


def verify_one_read(read, vocab, mm_flag, n_spotcheck):
    print(f"\n--- READ: name={read.query_name!r} chr={read.reference_name} "
          f"start={read.reference_start} strand={'-' if read.is_reverse else '+'} "
          f"len={len(read.query_sequence)} mapq={read.mapping_quality}")

    ref_tokens, ref_states, cpg_state, p_m, p_h = reference_featurize_read(
        read, vocab, mm_flag)
    prod_tokens, prod_states = featurize_read(read, vocab, mm_flag)

    n, n_tok_dis, n_state_dis, sample = compare_reads(
        ref_tokens, ref_states, prod_tokens, prod_states)

    state_dist = np.bincount(prod_states, minlength=4)
    print(f"    n_tokens          : {n:,}")
    print(f"    token agreement   : {n - n_tok_dis:,} / {n:,} "
          f"({100 * (n - n_tok_dis) / max(n, 1):.4f}%)")
    print(f"    state agreement   : {n - n_state_dis:,} / {n:,} "
          f"({100 * (n - n_state_dis) / max(n, 1):.4f}%)")
    print(f"    state distribution: "
          f"unmeth={state_dist[0]:,} meth={state_dist[1]:,} "
          f"non-CpG={state_dist[2]:,} unknown={state_dist[3]:,}")
    print(f"    forward CpGs found: {len(cpg_state):,}")

    if sample:
        print(f"    First {len(sample)} disagreements:")
        for d in sample:
            print(f"      idx={d['index']:>6}  "
                  f"token: ref={d['ref_token']} prod={d['prod_token']}  "
                  f"state: ref={STATE_NAMES.get(d['ref_state'], '?')} "
                  f"prod={STATE_NAMES.get(d['prod_state'], '?')}")

    print_visual_spot_check(
        read, vocab, mm_flag, ref_tokens, ref_states,
        cpg_state, p_m, p_h, n_examples=n_spotcheck)

    return n_tok_dis == 0 and n_state_dis == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bam", help="Path to BAM to verify")
    parser.add_argument("--mm-flag", required=True, choices=[".", "?"],
                        help="MM tag flag character for this BAM (per your manifest).")
    parser.add_argument("--n-reads", type=int, default=5,
                        help="Number of reads to verify (default 5).")
    parser.add_argument("--max-scan", type=int, default=10_000,
                        help="Cap on reads scanned to find usable ones (default 10000).")
    parser.add_argument("--n-spotcheck", type=int, default=6,
                        help="CpGs per read to print in the visual spot-check (default 6).")
    args = parser.parse_args(argv)

    bam_path = Path(args.bam)
    if not bam_path.is_file():
        print(f"BAM not found: {bam_path}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(" VERIFY featurize_read AGAINST INDEPENDENT REFERENCE")
    print("=" * 78)
    print(f"BAM     : {bam_path}")
    print(f"mm_flag : {args.mm_flag!r}")
    print(f"reads   : up to {args.n_reads} (scanning up to {args.max_scan})")

    print("Building Vocab", flush=True)
    vocab = MethylVocab(k=3)

    print("Finding usable reads...", flush=True)
    reads = find_usable_reads(bam_path, args.n_reads, args.max_scan)
    if not reads:
        print("No usable reads found.", file=sys.stderr)
        return 2
    print(f"  Found {len(reads)} reads to verify "
          f"(strands: {sum(r.is_reverse for r in reads)} reverse, "
          f"{sum(not r.is_reverse for r in reads)} forward).")

    all_clean = True
    for read in reads:
        clean = verify_one_read(read, vocab, args.mm_flag, args.n_spotcheck)
        all_clean = all_clean and clean

    print()
    print("=" * 78)
    if all_clean:
        print(f"✅ PASS: all {len(reads)} reads agree on every token AND state.")
        print("   The production featurizer matches an independent reimplementation.")
        return 0
    else:
        print(f"❌ FAIL: at least one read has disagreements between the two paths.")
        print("   Inspect the per-read disagreement listings above to localise the bug.")
        return 1


if __name__ == "__main__":
    sys.exit(main())