"""
Tests for MethylBertPretrainDatasetBinary (the nanopore two-int format reader).

We construct synthetic shards via ShardWriter (the actual production writer),
then load them with the dataset class and verify the __getitem__ contract.
"""

import json
import os
import random as random_module

import numpy as np
import pytest
import torch

from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertPretrainDatasetBinary
from methylbert.data.nanopore.featurize import (
    STATE_UNMETH, STATE_METH, STATE_NON_CPG, STATE_UNKNOWN,
)
from methylbert.data.nanopore.manifest import ManifestEntry
from methylbert.data.nanopore.shards import ShardWriter


WINDOW_LEN = 510


@pytest.fixture
def vocab():
    return MethylVocab(k=3)


@pytest.fixture
def manifest_entry():
    return ManifestEntry(
        bam_path="/data/example.bam",
        mm_flag="?",
        reference_build="hg38",
        basecaller_version="dorado_test",
        mod_codes="C+m,C+h",
        cohort="t",
        sample_label="t",
    )


def _row(rng, vocab_size=69):
    """Make a synthetic (tokens, states) row.

    Tokens are real (non-special, >= 5) so they're eligible for MLM masking.
    States are a mix of {0, 1, 2, 3} so we can verify per-state behavior.
    """
    tokens = rng.integers(5, vocab_size, size=WINDOW_LEN, dtype=np.int16)
    # Mix all four states so masking-related changes are observable.
    states = rng.integers(0, 4, size=WINDOW_LEN, dtype=np.int8)
    return tokens, states


def _write_shard(out_dir, basename, manifest_entry, vocab, n_rows, seed=0):
    rng = np.random.default_rng(seed)
    with ShardWriter(out_dir, basename, WINDOW_LEN,
                     manifest_entry, vocab_size=len(vocab)) as w:
        for i in range(n_rows):
            tokens, states = _row(rng, vocab_size=len(vocab))
            w.append(tokens, states)


# ---- Construction & len -------------------------------------------------

def test_load_single_shard(tmp_path, vocab, manifest_entry):
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=5)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    assert len(ds) == 5


def test_load_multiple_shards(tmp_path, vocab, manifest_entry):
    _write_shard(tmp_path, "a", manifest_entry, vocab, n_rows=3, seed=1)
    _write_shard(tmp_path, "b", manifest_entry, vocab, n_rows=7, seed=2)
    _write_shard(tmp_path, "c", manifest_entry, vocab, n_rows=11, seed=3)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    assert len(ds) == 21


def test_empty_directory_raises(tmp_path, vocab):
    with pytest.raises(FileNotFoundError, match="No shard JSON"):
        MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                         seq_len=WINDOW_LEN)


def test_window_len_mismatch_raises(tmp_path, vocab, manifest_entry):
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=2)
    with pytest.raises(ValueError, match="window_len"):
        MethylBertPretrainDatasetBinary(str(tmp_path), vocab, seq_len=99)


def test_zero_row_shard_skipped(tmp_path, vocab, manifest_entry):
    """A shard with n_rows=0 must not blow up memmap construction."""
    # Empty shard.
    with ShardWriter(tmp_path, "empty", WINDOW_LEN,
                     manifest_entry, vocab_size=len(vocab)) as w:
        pass
    # And one real shard so the dataset isn't fully empty.
    _write_shard(tmp_path, "real", manifest_entry, vocab, n_rows=4)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    assert len(ds) == 4


def test_old_format_raises(tmp_path, vocab):
    """Pointing at an old-format shard dir (with `seq_len`/`n_reads` keys but
    no `n_rows`/`window_len`) should raise rather than silently doing the
    wrong thing."""
    # Mimic the old sidecar format.
    (tmp_path / "old.json").write_text(json.dumps({
        "n_reads": 10, "seq_len": WINDOW_LEN, "vocab_size": len(vocab),
    }))
    with pytest.raises(ValueError, match="window_len"):
        MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                         seq_len=WINDOW_LEN)


# ---- __getitem__ output shape & keys ------------------------------------

def test_item_returns_four_keys(tmp_path, vocab, manifest_entry):
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=2)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    item = ds[0]
    assert set(item.keys()) == {"bert_input", "bert_label", "bert_mask",
                                "methyl_seq"}


def test_item_shapes_after_sos_prepend(tmp_path, vocab, manifest_entry):
    """All four arrays should have length seq_len + 1 after the SOS prepend."""
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=2)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    item = ds[0]
    expected_len = WINDOW_LEN + 1
    assert item["bert_input"].shape[0] == expected_len
    assert item["bert_label"].shape[0] == expected_len
    assert item["bert_mask"].shape[0] == expected_len
    assert item["methyl_seq"].shape[0] == expected_len


def test_methyl_seq_state_values_in_range(tmp_path, vocab, manifest_entry):
    """Every methyl state must be in {0, 1, 2, 3}."""
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=10)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    for i in range(len(ds)):
        states = ds[i]["methyl_seq"]
        assert ((states >= 0) & (states <= 3)).all()


def test_sos_position_is_special(tmp_path, vocab, manifest_entry):
    """Index 0 of bert_input is SOS; methyl_seq[0] is STATE_NON_CPG."""
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=4)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["bert_input"][0]) == vocab.sos_index
        assert int(item["methyl_seq"][0]) == STATE_NON_CPG


def test_eos_position_is_special(tmp_path, vocab, manifest_entry):
    """The position with EOS in bert_input has STATE_NON_CPG in methyl_seq."""
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=4)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    for i in range(len(ds)):
        item = ds[i]
        # Find EOS in bert_input. Note: random replacement *can* produce an
        # EOS token at non-EOS positions, so we look for the *last* position
        # where bert_input == EOS — that must be the placed EOS.
        eos_positions = (item["bert_input"] == vocab.eos_index).nonzero(as_tuple=True)[0]
        assert len(eos_positions) >= 1, "expected at least one EOS in bert_input"
        last_eos = int(eos_positions[-1])
        assert int(item["methyl_seq"][last_eos]) == STATE_NON_CPG


# ---- Option B: hidden methylation at MLM-selected positions -------------

def test_masked_positions_have_state_unknown(tmp_path, vocab, manifest_entry):
    """At every NON-SPECIAL position where bert_mask is True, methyl_seq is
    STATE_UNKNOWN. Special-token positions (EOS) are STATE_NON_CPG even when
    they happen to be in bert_mask, since they aren't real CpG positions.

    This is the core Option-B contract: when DNA at a CpG position is selected
    for the MLM objective, the methylation track must not leak the CpG-context.
    """
    # Use enough rows to amortize the bernoulli randomness.
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=20)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    found_at_least_one_masked = False
    for i in range(len(ds)):
        item = ds[i]
        masked_positions = item["bert_mask"].bool()
        # Positions that are special tokens in bert_input — exclude these from
        # the all-STATE_UNKNOWN check, because they get STATE_NON_CPG
        # regardless of mask status (they aren't biological CpGs).
        is_special = item["bert_input"] < 5   # MethylVocab special-token range
        check_positions = masked_positions & ~is_special
        if check_positions.any():
            found_at_least_one_masked = True
            assert (item["methyl_seq"][check_positions] == STATE_UNKNOWN).all()
    assert found_at_least_one_masked, \
        "no masked positions across 20 items; raise n_rows or check threshold"


def test_unmasked_methyl_states_preserved_outside_specials(
    tmp_path, vocab, manifest_entry
):
    """Outside the SOS slot, EOS slot, and MLM-selected positions, the
    methylation state should be exactly what was written to disk."""
    rng = np.random.default_rng(7)
    # Write a known row so we can recover the original states for comparison.
    written_tokens, written_states = _row(rng, vocab_size=len(vocab))
    with ShardWriter(tmp_path, "ex", WINDOW_LEN,
                     manifest_entry, vocab_size=len(vocab)) as w:
        w.append(written_tokens, written_states)

    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    item = ds[0]
    # Where bert_mask is False AND it's not the SOS slot AND it's not the
    # EOS slot, methyl_seq should match the original states (which were
    # prepended-by-one to align with bert_input).
    eos_pos_in_padded = int(
        (item["bert_input"] == vocab.eos_index).nonzero(as_tuple=True)[0][-1]
    )
    n = item["methyl_seq"].shape[0]
    for pos in range(n):
        if pos == 0:                       # SOS
            continue
        if pos == eos_pos_in_padded:       # EOS
            continue
        if bool(item["bert_mask"][pos]):   # MLM-selected
            continue
        # Original row index = pos - 1 (because of SOS prepend)
        original_state = int(written_states[pos - 1])
        assert int(item["methyl_seq"][pos]) == original_state, (
            f"mismatch at pos {pos}: got {int(item['methyl_seq'][pos])}, "
            f"expected original {original_state}"
        )


# ---- Multi-shard locate -------------------------------------------------

def test_multi_shard_indexing_is_consistent(tmp_path, vocab, manifest_entry):
    """Indexing across shard boundaries should hit the right shard's row."""
    _write_shard(tmp_path, "a", manifest_entry, vocab, n_rows=3, seed=10)
    _write_shard(tmp_path, "b", manifest_entry, vocab, n_rows=4, seed=20)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN)
    assert len(ds) == 7
    # Items 0..2 come from shard a; items 3..6 from shard b.
    # Just check we can fetch every item without error.
    for i in range(len(ds)):
        item = ds[i]
        assert item["bert_input"].shape[0] == WINDOW_LEN + 1


# ---- Random length truncation -------------------------------------------

def test_random_len_keeps_output_shape(tmp_path, vocab, manifest_entry):
    """random_len truncates internally but the output is padded back to
    seq_len before SOS prepend, so the final shape is unchanged."""
    _write_shard(tmp_path, "ex", manifest_entry, vocab, n_rows=10)
    ds = MethylBertPretrainDatasetBinary(str(tmp_path), vocab,
                                          seq_len=WINDOW_LEN, random_len=True)
    for i in range(len(ds)):
        item = ds[i]
        assert item["bert_input"].shape[0] == WINDOW_LEN + 1
        assert item["methyl_seq"].shape[0] == WINDOW_LEN + 1