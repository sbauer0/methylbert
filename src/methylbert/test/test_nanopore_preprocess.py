"""
Tests for methylbert.data.nanopore.preprocess.

We construct synthetic BAM files on disk in tmp_path, run the preprocessor,
and verify the resulting shards. The featurizer/chunker/shard-writer logic
itself is covered by their own test modules; the tests here focus on the
orchestration: filtering, stats accounting, resumability, and the manifest
loop.
"""

import array
import json
from pathlib import Path

import numpy as np
import pysam
import pytest

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.manifest import ManifestEntry
from methylbert.data.nanopore.shards import shard_exists
from methylbert.data.nanopore.preprocess import (
    bam_basename,
    preprocess_bam,
    preprocess_manifest,
)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic BAMs
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab():
    return MethylVocab(k=3)


def _make_header(ref_name="chr1", ref_len=10_000):
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": ref_name, "LN": ref_len}],
    })


def _build_methylated_seq(n_cpgs=20, between=20):
    """
    Build a deterministic test sequence with several CpGs, well-spaced, plus
    flanking padding so it's at least a couple of windows long after tokenisation.

    Returns (sequence_string, mm_tag, ml_bytes_list).
    The MM tag is in original-read orientation; we'll only use this for
    forward-mapped reads in tests.
    """
    # Block: "ATGAT" (5bp, no CpG) + "ACGAT" (CpG at offset 1)
    # Repeat to make a long sequence.
    block = "ATGATACGAT"   # 10 bp; CpG at offset 6 within the block
    seq = block * 80        # 800 bp
    # We won't bother computing the precise MM positions per Cs; for the
    # preprocessor tests we want a read with *some* methylation calls and
    # a usable length. Use a simple "every C in the read is modified" pattern.
    # Count Cs in seq.
    n_cs = seq.count("C")
    mm = "C+m?," + ",".join(["0"] * n_cs) + ";"
    ml = [200] * n_cs   # P ≈ 0.78, in the unknown band; doesn't matter for these tests
    return seq, mm, ml


def _make_aligned_read(header, query_sequence, mm_tag, ml_bytes,
                      query_name="r", flag=0, mapq=60):
    read = pysam.AlignedSegment(header)
    read.query_name = query_name
    read.query_sequence = query_sequence
    read.flag = flag
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = mapq
    read.cigartuples = [(0, len(query_sequence))]
    read.set_tag("MM", mm_tag)
    read.set_tag("ML", array.array("B", ml_bytes))
    return read


def _make_unmapped_read(header, query_name="unmapped"):
    read = pysam.AlignedSegment(header)
    read.query_name = query_name
    read.query_sequence = "ACGT" * 200
    read.flag = 4   # unmapped
    return read


def _make_no_mm_read(header, query_name="no_mm"):
    """A mapped read with no MM/ML tags at all."""
    read = pysam.AlignedSegment(header)
    read.query_name = query_name
    read.query_sequence = "ACGT" * 200
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.cigartuples = [(0, 800)]
    return read


def _write_bam(path: Path, header, reads):
    """Write a list of reads to a BAM file at `path`. Sorts + indexes."""
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for r in reads:
            out.write(r)


def _make_entry(bam_path, mm_flag="?", cohort="t", sample_label="t"):
    return ManifestEntry(
        bam_path=str(bam_path),
        mm_flag=mm_flag,
        reference_build="chm13v2",
        basecaller_version="dorado_test",
        mod_codes="C+m",
        cohort=cohort,
        sample_label=sample_label,
    )


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def test_bam_basename_strips_extension():
    assert bam_basename("/x/y/z.bam") == "z"
    assert bam_basename("/x/y/Z.BAM") == "Z"
    assert bam_basename("foo.bar.bam") == "foo.bar"
    assert bam_basename("/x/y/no_ext") == "no_ext"


# ---------------------------------------------------------------------------
# preprocess_bam: end-to-end on synthetic BAM
# ---------------------------------------------------------------------------

def test_single_usable_read_produces_shard(tmp_path, vocab):
    bam_path = tmp_path / "single.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    read = _make_aligned_read(header, seq, mm, ml, query_name="ok")
    _write_bam(bam_path, header, [read])

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab)

    assert stats.reads_total == 1
    assert stats.reads_kept == 1
    assert stats.windows_written >= 1
    assert shard_exists(tmp_path, bam_basename(bam_path))


def test_filtering_unmapped_secondary_no_mm(tmp_path, vocab):
    """One usable read + several skipworthy reads should give correct stats."""
    bam_path = tmp_path / "mixed.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()

    ok = _make_aligned_read(header, seq, mm, ml, query_name="ok")
    unmapped = _make_unmapped_read(header)
    no_mm = _make_no_mm_read(header)
    secondary = _make_aligned_read(header, seq, mm, ml,
                                   query_name="sec", flag=0x100)  # secondary
    supplementary = _make_aligned_read(header, seq, mm, ml,
                                       query_name="supp", flag=0x800)  # supplementary
    low_mapq = _make_aligned_read(header, seq, mm, ml,
                                  query_name="lowmapq", mapq=5)

    _write_bam(bam_path, header,
               [ok, unmapped, no_mm, secondary, supplementary, low_mapq])

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab, min_mapq=20)

    assert stats.reads_total == 6
    assert stats.reads_kept == 1
    assert stats.reads_unmapped == 1
    assert stats.reads_no_mm_tag == 1
    assert stats.reads_secondary_supplementary == 2
    assert stats.reads_low_mapq == 1


def test_max_reads_caps_iteration(tmp_path, vocab):
    bam_path = tmp_path / "many.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    reads = [_make_aligned_read(header, seq, mm, ml, query_name=f"r{i}")
             for i in range(10)]
    _write_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab, max_reads=3)
    assert stats.reads_total == 3


def test_shard_metadata_records_manifest_entry(tmp_path, vocab):
    bam_path = tmp_path / "meta.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    read = _make_aligned_read(header, seq, mm, ml, query_name="ok")
    _write_bam(bam_path, header, [read])

    entry = _make_entry(bam_path, mm_flag="?", cohort="some_cohort",
                        sample_label="some_label")
    preprocess_bam(entry, tmp_path, vocab)

    json_path = tmp_path / f"{bam_basename(bam_path)}.json"
    with open(json_path) as f:
        meta = json.load(f)
    me = meta["manifest_entry"]
    assert me["mm_flag"] == "?"
    assert me["cohort"] == "some_cohort"
    assert me["sample_label"] == "some_label"
    assert me["bam_path"] == str(bam_path)


def test_shards_have_consistent_n_rows_with_stats(tmp_path, vocab):
    """The .json's n_rows should equal stats.windows_written."""
    bam_path = tmp_path / "consistent.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    reads = [_make_aligned_read(header, seq, mm, ml, query_name=f"r{i}")
             for i in range(3)]
    _write_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab)

    with open(tmp_path / f"{bam_basename(bam_path)}.json") as f:
        meta = json.load(f)
    assert meta["n_rows"] == stats.windows_written


# ---------------------------------------------------------------------------
# preprocess_manifest: multi-BAM loop
# ---------------------------------------------------------------------------

def _write_manifest_tsv(path: Path, entries):
    cols = ["bam_path", "mm_flag", "reference_build", "basecaller_version",
            "mod_codes", "cohort", "sample_label"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for e in entries:
            f.write("\t".join(getattr(e, c) for c in cols) + "\n")


def test_manifest_loop_processes_all_bams(tmp_path, vocab):
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()

    # Two synthetic BAMs.
    bam_a = tmp_path / "A.bam"
    bam_b = tmp_path / "B.bam"
    _write_bam(bam_a, header,
               [_make_aligned_read(header, seq, mm, ml, query_name="a1")])
    _write_bam(bam_b, header,
               [_make_aligned_read(header, seq, mm, ml, query_name="b1"),
                _make_aligned_read(header, seq, mm, ml, query_name="b2")])

    entry_a = _make_entry(bam_a, cohort="A", sample_label="a")
    entry_b = _make_entry(bam_b, cohort="B", sample_label="b")
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry_a, entry_b])

    out_dir = tmp_path / "out"
    all_stats = preprocess_manifest(manifest_path, out_dir, vocab=vocab)

    assert len(all_stats) == 2
    assert shard_exists(out_dir, bam_basename(bam_a))
    assert shard_exists(out_dir, bam_basename(bam_b))
    # Manifest should be copied for traceability.
    assert (out_dir / "manifest.tsv").exists()


def test_manifest_loop_skips_existing_shards(tmp_path, vocab):
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    bam_a = tmp_path / "A.bam"
    _write_bam(bam_a, header,
               [_make_aligned_read(header, seq, mm, ml, query_name="a1")])
    entry_a = _make_entry(bam_a)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry_a])

    out_dir = tmp_path / "out"

    # First run: actually processes.
    stats1 = preprocess_manifest(manifest_path, out_dir, vocab=vocab)
    assert len(stats1) == 1

    # Second run: should skip (no stats returned for already-done BAMs).
    stats2 = preprocess_manifest(manifest_path, out_dir, vocab=vocab)
    assert len(stats2) == 0


def test_manifest_loop_force_reprocess(tmp_path, vocab):
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    bam_a = tmp_path / "A.bam"
    _write_bam(bam_a, header,
               [_make_aligned_read(header, seq, mm, ml, query_name="a1")])
    entry_a = _make_entry(bam_a)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry_a])

    out_dir = tmp_path / "out"
    preprocess_manifest(manifest_path, out_dir, vocab=vocab)
    stats2 = preprocess_manifest(manifest_path, out_dir, vocab=vocab,
                                 skip_existing=False)
    assert len(stats2) == 1   # reprocessed


def test_manifest_loop_warns_for_missing_bam(tmp_path, vocab, caplog):
    """If a manifest entry points to a non-existent BAM, we log and skip
    rather than crashing."""
    import logging
    caplog.set_level(logging.WARNING)
    entry = _make_entry(tmp_path / "does_not_exist.bam")
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry])

    out_dir = tmp_path / "out"
    all_stats = preprocess_manifest(manifest_path, out_dir, vocab=vocab)
    assert all_stats == []
    assert any("not found" in rec.message for rec in caplog.records)


def test_round_trip_to_memmap(tmp_path, vocab):
    """End-to-end: BAM -> shards -> memmap; verify we can read back tokens
    and methylation states with the right shapes and reasonable values."""
    bam_path = tmp_path / "rt.bam"
    header = _make_header()
    seq, mm, ml = _build_methylated_seq()
    reads = [_make_aligned_read(header, seq, mm, ml, query_name=f"r{i}")
             for i in range(2)]
    _write_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab)
    assert stats.windows_written > 0

    bn = bam_basename(bam_path)
    with open(tmp_path / f"{bn}.json") as f:
        meta = json.load(f)
    n_rows = meta["n_rows"]
    window_len = meta["window_len"]

    tokens = np.memmap(tmp_path / f"{bn}.tokens.bin",
                       dtype=np.int16, mode="r", shape=(n_rows, window_len))
    states = np.memmap(tmp_path / f"{bn}.states.bin",
                       dtype=np.int8, mode="r", shape=(n_rows, window_len))

    # Tokens: all in [0, vocab_size).
    assert (tokens >= 0).all()
    assert (tokens < len(vocab)).all()
    # States: all in {0, 1, 2, 3}.
    assert ((states >= 0) & (states <= 3)).all()