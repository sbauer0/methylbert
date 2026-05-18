"""
Tests for methylbert.data.nanopore.featurize.

Most tests construct synthetic pysam reads in-memory so we can verify exact
expected outputs. A separate smoke test runs against the real test BAM at
myNotebooks/test.bam and just confirms the featurizer doesn't blow up.
"""

import array
import os

import numpy as np
import pysam
import pytest

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.featurize import (
    featurize_read,
    STATE_UNMETH,
    STATE_METH,
    STATE_NON_CPG,
    STATE_UNKNOWN,
    FLAG_IMPLICIT,
    FLAG_EXPLICIT,
    P_HIGH,
    P_LOW,
)


REAL_BAM = "/home/bauerste/Methylbert_methylation_encoding/methylbert/myNotebooks/test.bam"


# ---- Fixtures ------------------------------------------------------------

@pytest.fixture
def vocab():
    return MethylVocab(k=3)


@pytest.fixture
def header():
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
    })


def _make_read(header, query_sequence, mm_tag, ml_bytes, is_reverse=False,
               query_name="r"):
    """Construct an in-memory pysam.AlignedSegment for testing."""
    read = pysam.AlignedSegment(header)
    read.query_name = query_name
    read.query_sequence = query_sequence
    read.flag = 16 if is_reverse else 0          # is_reverse via BAM_FREVERSE
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.cigartuples = [(0, len(query_sequence))]   # all match
    read.set_tag("MM", mm_tag)
    read.set_tag("ML", array.array("B", ml_bytes))
    return read


# ---- Core state assignment ----------------------------------------------

def test_forward_mapped_methylated(vocab, header):
    """Forward-mapped read, one CpG, high P(5mC) -> state 1."""
    # Forward seq "ACGAT": CpG at positions 1-2 (C at 1, G at 2).
    # MM "C+m?,0;" -> first C in original read (pos 1) modified.
    # ML 230 -> P = 230/256 = 0.898 >= 0.8 -> STATE_METH.
    read = _make_read(header, "ACGAT", "C+m?,0;", [230])
    token_ids, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)

    # 3-mers: ACG (i=0), CGA (i=1), GAT (i=2)
    # Middle-base anchoring: the C of the CpG (at seq pos 1) is the MIDDLE
    # base of the 3-mer at i=0 ("ACG"), so that 3-mer carries the label.
    assert len(token_ids) == 3
    assert states.tolist() == [STATE_METH, STATE_NON_CPG, STATE_NON_CPG]
    assert token_ids[0] == vocab.stoi["ACG"]
    assert token_ids[1] == vocab.stoi["CGA"]
    assert token_ids[2] == vocab.stoi["GAT"]


def test_forward_mapped_unmethylated(vocab, header):
    """Low P(5mC) -> state 0."""
    # ML 25 -> P ≈ 0.098 <= 0.2.
    read = _make_read(header, "ACGAT", "C+m?,0;", [25])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_UNMETH


def test_forward_mapped_ambiguous(vocab, header):
    """Middle P(5mC) -> state 3."""
    # ML 128 -> P = 0.5, which is between 0.2 and 0.8.
    read = _make_read(header, "ACGAT", "C+m?,0;", [128])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_UNKNOWN


def test_threshold_boundary_low(vocab, header):
    """P exactly at 0.2 -> state 0 (boundary inclusive on low side)."""
    # 51/256 = 0.1992 -> just below 0.2, should be STATE_UNMETH
    read = _make_read(header, "ACGAT", "C+m?,0;", [51])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_UNMETH

    # 52/256 = 0.2031 -> just above 0.2, should be STATE_UNKNOWN
    read = _make_read(header, "ACGAT", "C+m?,0;", [52])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_UNKNOWN


def test_threshold_boundary_high(vocab, header):
    """P at/above 0.8 -> state 1."""
    # 204/256 = 0.7969 -> just below 0.8, should be STATE_UNKNOWN
    read = _make_read(header, "ACGAT", "C+m?,0;", [204])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_UNKNOWN

    # 205/256 = 0.8008 -> just above 0.8, should be STATE_METH
    read = _make_read(header, "ACGAT", "C+m?,0;", [205])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_METH


# ---- Alpha (5mC + 5hmC summation) ---------------------------------------

def test_alpha_5hmc_summation_promotes_to_methylated(vocab, header):
    """Neither m nor h alone exceeds 0.8, but the sum does -> STATE_METH."""
    # ML order: m calls then h calls (matching MM block order).
    # m=102/256 ≈ 0.398, h=128/256 = 0.5; sum ≈ 0.898 >= 0.8.
    read = _make_read(header, "ACGAT", "C+m?,0;C+h?,0;", [102, 128])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_METH


def test_alpha_only_5hmc(vocab, header):
    """High P(5hmC) alone -> STATE_METH (alpha treats any modification)."""
    # Only an h block; ML=230 -> P_h ≈ 0.898 >= 0.8.
    read = _make_read(header, "ACGAT", "C+h?,0;", [230])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert states[0] == STATE_METH


# ---- Unlisted CpG handling (flag-driven) --------------------------------

def test_unlisted_cpg_with_explicit_flag(vocab, header):
    """An in-sequence CpG with no MM/ML entry -> STATE_UNKNOWN if flag is '?'."""
    # Two CpGs in sequence: ACGCGT (CpGs at C-positions 1 and 3).
    # MM lists only the first C (skip 0 -> first C at pos 1). The second CpG's C
    # at pos 3 has no entry.
    read = _make_read(header, "ACGCGT", "C+m?,0;", [230])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)

    # 3-mers: ACG(i=0), CGC(i=1), GCG(i=2), CGT(i=3)
    # Middle-base anchoring -- 3-mer at i carries the label of the C at seq[i+1]:
    #   i=0 ("ACG"): middle base seq[1]='C' (CpG with call) -> STATE_METH
    #   i=2 ("GCG"): middle base seq[3]='C' (CpG, unlisted, flag '?') -> STATE_UNKNOWN
    #   i=1, i=3: middle bases are 'G', not C of CpG -> STATE_NON_CPG
    assert states[0] == STATE_METH
    assert states[2] == STATE_UNKNOWN
    assert states[1] == STATE_NON_CPG
    assert states[3] == STATE_NON_CPG


def test_unlisted_cpg_with_implicit_flag(vocab, header):
    """Same as above, but with flag '.' -> STATE_UNMETH for the unlisted CpG."""
    read = _make_read(header, "ACGCGT", "C+m?,0;", [230])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_IMPLICIT)
    assert states[2] == STATE_UNMETH


# ---- Non-CpG behavior ---------------------------------------------------

def test_non_cpg_c_with_call_is_ignored(vocab, header):
    """An all-context C call on a non-CpG C should not produce a methylation state.

    The featurizer should skip such calls (since the C isn't in CpG context),
    and the resulting 3-mers should all be STATE_NON_CPG.
    """
    # "ACAT" -- C at pos 1 is NOT in CpG context (next base is A).
    read = _make_read(header, "ACAT", "C+m.,0;", [230])
    _, states = featurize_read(read, vocab, mm_flag=FLAG_IMPLICIT)
    assert all(s == STATE_NON_CPG for s in states)


def test_no_cpg_in_sequence(vocab, header):
    """Sequence with no CpGs at all -> all states are STATE_NON_CPG."""
    read = _make_read(header, "AAATTT", "C+m?,;", [])  # empty position list
    _, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert all(s == STATE_NON_CPG for s in states)


# ---- Strand canonicalization (the most error-prone part) ----------------

def test_reverse_mapped_same_biology_as_forward(vocab, header):
    """A reverse-mapped read of the same biological CpG must give the same output.

    Forward strand reference at this region: ACGAT, with CpG C at forward pos 1.
    A reverse-mapped read aligned here would have its original sequenced read
    being RC(ACGAT) = ATCGT, with the C of the symmetric CpG at *read* pos 2.
    BAM SEQ is stored in reference forward orientation, so query_sequence is ACGAT.
    The MM tag is in original-read orientation: 'C+m?,0;' -> first C in ATCGT
    (pos 2 in original read).
    """
    fwd = _make_read(header, "ACGAT", "C+m?,0;", [230], is_reverse=False)
    rev = _make_read(header, "ACGAT", "C+m?,0;", [230], is_reverse=True)

    fwd_tokens, fwd_states = featurize_read(fwd, vocab, mm_flag=FLAG_EXPLICIT)
    rev_tokens, rev_states = featurize_read(rev, vocab, mm_flag=FLAG_EXPLICIT)

    np.testing.assert_array_equal(fwd_tokens, rev_tokens)
    np.testing.assert_array_equal(fwd_states, rev_states)
    # Both should mark the ACG token (i=0, middle base is the C of the CpG)
    # as methylated.
    assert rev_states[0] == STATE_METH


def test_reverse_mapped_with_two_cpgs(vocab, header):
    """A more involved reverse-mapped case with two CpGs and different states."""
    # Forward strand (= BAM SEQ since stored in forward orientation): "ACGCGT".
    # Forward CpGs at i=1 (CGC) and i=3 (CGT).
    # Original sequenced read (reverse-mapped) = RC("ACGCGT") = "ACGCGT".
    # ...wait, that's a palindrome. Let's pick a non-palindromic example.
    #
    # Forward: "ACGTACGAT", CpGs at positions 1-2 and 5-6.
    # RC: "ATCGTACGT", first C at original-read pos 2, second C at pos 6.
    # MM "C+m?,0,0;" -> two calls, first C (skip 0) and next C (skip 0 more).
    # ML [230, 25] -> first methylated, second unmethylated.
    #
    # In original-read coords: pos 2 (high) and pos 6 (low).
    # In BAM SEQ coords (after pysam translation): L-1-2=6 and L-1-6=2.
    # The forward CpG C corresponding to a call at BAM SEQ pos p (reverse-mapped)
    # is at p-1: positions 5 and 1.
    # CpG at i=1: low (state 0)
    # CpG at i=5: high (state 1)
    fwd_seq = "ACGTACGAT"
    rev = _make_read(header, fwd_seq, "C+m?,0,0;", [230, 25], is_reverse=True)
    _, states = featurize_read(rev, vocab, mm_flag=FLAG_EXPLICIT)

    # 3-mers (i): ACG(0), CGT(1), GTA(2), TAC(3), ACG(4), CGA(5), GAT(6)
    # Middle-base anchoring -- 3-mer at i carries the label of the C at seq[i+1]:
    #   i=0 ("ACG"): middle is seq[1]='C' (CpG at forward pos 1) -> low call -> STATE_UNMETH
    #   i=4 ("ACG"): middle is seq[5]='C' (CpG at forward pos 5) -> high call -> STATE_METH
    assert states[0] == STATE_UNMETH
    assert states[4] == STATE_METH
    # Other positions: middle base is not the C of a CpG -> STATE_NON_CPG
    assert states[1] == STATE_NON_CPG
    assert states[2] == STATE_NON_CPG
    assert states[3] == STATE_NON_CPG
    assert states[5] == STATE_NON_CPG
    assert states[6] == STATE_NON_CPG


# ---- Vocab edge cases ---------------------------------------------------

def test_n_in_sequence_maps_to_unk(vocab, header):
    """3-mers containing N should map to vocab.unk_index."""
    read = _make_read(header, "ANCGAT", "C+m?,0;", [230])
    token_ids, _ = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    # 3-mers: ANC, NCG, CGA, GAT
    assert token_ids[0] == vocab.unk_index
    assert token_ids[1] == vocab.unk_index
    assert token_ids[2] == vocab.stoi["CGA"]
    assert token_ids[3] == vocab.stoi["GAT"]


def test_short_read_returns_empty(vocab, header):
    """A 2-base read produces no 3-mers."""
    # "AC" (length 2) — note: pysam refuses zero-length sequences, hence 2 not 0
    read = _make_read(header, "AC", "C+m?,;", [])
    tokens, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
    assert len(tokens) == 0
    assert len(states) == 0


# ---- Validation ---------------------------------------------------------

def test_invalid_mm_flag_raises(vocab, header):
    read = _make_read(header, "ACGAT", "C+m?,0;", [230])
    with pytest.raises(ValueError, match="mm_flag"):
        featurize_read(read, vocab, mm_flag="!")


# ---- Smoke test against real BAM ----------------------------------------

@pytest.mark.skipif(not os.path.exists(REAL_BAM), reason="test BAM not present")
def test_smoke_real_bam(vocab):
    """First N usable reads of the real test BAM run without exception
    and produce sensible-looking outputs."""
    bam = pysam.AlignmentFile(REAL_BAM, "rb")
    n_processed = 0
    n_with_meth_call = 0
    try:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if not read.has_tag("MM") or not read.has_tag("ML"):
                continue
            # We don't know the BAM's flag a priori; pick '?' (the modern default).
            # If the smoke test fails because the BAM uses '.', you'll need to
            # parameterise this — but the featurizer will run either way; the
            # only difference is how unlisted CpG Cs are labeled.
            tokens, states = featurize_read(read, vocab, mm_flag=FLAG_EXPLICIT)
            assert len(tokens) == len(states)
            assert len(tokens) > 0, "expected non-empty arrays for a real read"
            assert ((states >= 0) & (states <= 3)).all()
            assert (tokens >= 0).all()
            assert (tokens < len(vocab)).all()
            if (states == STATE_METH).any() or (states == STATE_UNMETH).any():
                n_with_meth_call += 1
            n_processed += 1
            if n_processed >= 20:
                break
    finally:
        bam.close()
    assert n_processed > 0, "no usable reads in the test BAM"
    # At least some real reads should have actual methylation calls.
    assert n_with_meth_call > 0, "no reads had any 0/1 methylation states; suspicious"