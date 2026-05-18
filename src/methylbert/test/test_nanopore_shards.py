"""
Tests for methylbert.data.nanopore.shards.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from methylbert.data.nanopore.shards import (
    ShardWriter,
    shard_exists,
    SHARD_VERSION,
)
from methylbert.data.nanopore.manifest import ManifestEntry


WINDOW_LEN = 510


@pytest.fixture
def manifest_entry():
    return ManifestEntry(
        bam_path="/data/example.bam",
        mm_flag="?",
        reference_build="hg38",
        basecaller_version="dorado_0.5.3",
        mod_codes="C+m,C+h",
        cohort="example",
        sample_label="ex",
    )


def _row(seed):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 69, size=WINDOW_LEN, dtype=np.int16)
    states = rng.integers(0, 4, size=WINDOW_LEN, dtype=np.int8)
    return tokens, states


# ---- Happy path ---------------------------------------------------------

def test_writes_three_files_atomically(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
        w.append(*_row(2))
        w.append(*_row(3))

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["ex.json", "ex.states.bin", "ex.tokens.bin"]

    # No leftover .tmp files
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_metadata_records_correct_counts(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
        w.append(*_row(2))

    with open(tmp_path / "ex.json") as f:
        meta = json.load(f)
    assert meta["shard_version"] == SHARD_VERSION
    assert meta["n_rows"] == 2
    assert meta["window_len"] == WINDOW_LEN
    assert meta["vocab_size"] == 69
    assert meta["tokens_dtype"] == "int16"
    assert meta["states_dtype"] == "int8"
    assert meta["tokens_file"] == "ex.tokens.bin"
    assert meta["states_file"] == "ex.states.bin"
    me = meta["manifest_entry"]
    assert me["bam_path"] == manifest_entry.bam_path
    assert me["mm_flag"] == "?"
    assert me["cohort"] == "example"


def test_file_sizes_match_metadata(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        for i in range(5):
            w.append(*_row(i))

    # int16 = 2 bytes, int8 = 1 byte
    assert (tmp_path / "ex.tokens.bin").stat().st_size == 5 * WINDOW_LEN * 2
    assert (tmp_path / "ex.states.bin").stat().st_size == 5 * WINDOW_LEN * 1


def test_roundtrip_via_memmap(tmp_path, manifest_entry):
    """Write a few rows, then memmap and verify content matches."""
    rows = [_row(i) for i in range(4)]
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        for tokens, states in rows:
            w.append(tokens, states)

    tokens_mm = np.memmap(
        tmp_path / "ex.tokens.bin",
        dtype=np.int16, mode="r", shape=(4, WINDOW_LEN),
    )
    states_mm = np.memmap(
        tmp_path / "ex.states.bin",
        dtype=np.int8, mode="r", shape=(4, WINDOW_LEN),
    )
    for i, (tokens, states) in enumerate(rows):
        np.testing.assert_array_equal(tokens_mm[i], tokens)
        np.testing.assert_array_equal(states_mm[i], states)


def test_zero_rows_is_valid(tmp_path, manifest_entry):
    """A BAM with no usable reads should still produce valid (empty) shards."""
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        pass
    assert (tmp_path / "ex.json").exists()
    assert (tmp_path / "ex.tokens.bin").exists()
    assert (tmp_path / "ex.states.bin").exists()
    with open(tmp_path / "ex.json") as f:
        meta = json.load(f)
    assert meta["n_rows"] == 0


# ---- Dtype handling -----------------------------------------------------

def test_dtype_conversion_on_append(tmp_path, manifest_entry):
    """Inputs are converted to int16/int8 even if passed as int32/int64."""
    tokens = np.zeros(WINDOW_LEN, dtype=np.int32) + 5
    states = np.zeros(WINDOW_LEN, dtype=np.int64) + 1
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(tokens, states)

    tokens_mm = np.memmap(
        tmp_path / "ex.tokens.bin",
        dtype=np.int16, mode="r", shape=(1, WINDOW_LEN),
    )
    assert (tokens_mm[0] == 5).all()


def test_window_length_mismatch_raises(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        bad = np.zeros(WINDOW_LEN - 1, dtype=np.int16)
        good = np.zeros(WINDOW_LEN, dtype=np.int8)
        with pytest.raises(ValueError, match="Window length mismatch"):
            w.append(bad, good)


# ---- Atomicity ----------------------------------------------------------

def test_exception_inside_with_leaves_no_files(tmp_path, manifest_entry):
    """If we raise inside the with-block, no shard files should appear."""
    with pytest.raises(RuntimeError):
        with ShardWriter(tmp_path, "ex", WINDOW_LEN,
                         manifest_entry, vocab_size=69) as w:
            w.append(*_row(1))
            raise RuntimeError("boom")

    # Neither final files nor leftover .tmp files
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"unexpected leftover files: {leftover}"


def test_can_write_after_failed_run(tmp_path, manifest_entry):
    """A failed run must not block a subsequent retry on the same basename."""
    with pytest.raises(RuntimeError):
        with ShardWriter(tmp_path, "ex", WINDOW_LEN,
                         manifest_entry, vocab_size=69) as w:
            w.append(*_row(1))
            raise RuntimeError("first run failed")

    # Now retry; should succeed cleanly.
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(99))
    assert (tmp_path / "ex.json").exists()


def test_stale_tmp_files_are_cleared_on_enter(tmp_path, manifest_entry):
    """Manually drop a .tmp file from a previous crashed run; the writer should
    clear it on enter and produce a clean shard."""
    (tmp_path / "ex.tokens.bin.tmp").write_bytes(b"garbage")
    (tmp_path / "ex.states.bin.tmp").write_bytes(b"more garbage")

    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))

    # The shard files should be sized for exactly 1 row, not contaminated.
    assert (tmp_path / "ex.tokens.bin").stat().st_size == WINDOW_LEN * 2
    assert (tmp_path / "ex.states.bin").stat().st_size == WINDOW_LEN * 1


# ---- shard_exists -------------------------------------------------------

def test_shard_exists_true_for_complete_shard(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
    assert shard_exists(tmp_path, "ex") is True


def test_shard_exists_false_when_missing(tmp_path):
    assert shard_exists(tmp_path, "missing") is False


def test_shard_exists_false_when_only_json(tmp_path):
    """If JSON is there but the .bin files aren't, treat as not-done."""
    (tmp_path / "ex.json").write_text(json.dumps({
        "shard_version": SHARD_VERSION, "n_rows": 0,
        "window_len": WINDOW_LEN, "vocab_size": 69,
    }))
    assert shard_exists(tmp_path, "ex") is False


def test_shard_exists_false_when_size_mismatch(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
    # Tamper: make the tokens file the wrong size.
    (tmp_path / "ex.tokens.bin").write_bytes(b"\x00" * 10)
    assert shard_exists(tmp_path, "ex") is False


def test_shard_exists_false_for_old_version(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
    # Tamper: change version in JSON to a value our code doesn't accept.
    with open(tmp_path / "ex.json") as f:
        meta = json.load(f)
    meta["shard_version"] = SHARD_VERSION + 99
    with open(tmp_path / "ex.json", "w") as f:
        json.dump(meta, f)
    assert shard_exists(tmp_path, "ex") is False


def test_shard_exists_false_for_corrupt_json(tmp_path, manifest_entry):
    with ShardWriter(tmp_path, "ex", WINDOW_LEN, manifest_entry, vocab_size=69) as w:
        w.append(*_row(1))
    (tmp_path / "ex.json").write_text("{not valid json")
    assert shard_exists(tmp_path, "ex") is False