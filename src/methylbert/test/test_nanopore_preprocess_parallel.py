"""
Tests for the per-region and parallel paths of preprocess.py.

The serial / single-BAM behaviour is covered by test_nanopore_preprocess.py.
This file adds tests for:
    - shard_basename naming convention
    - region= parameter to preprocess_bam
    - per_region work-item construction and shard naming
    - n_workers > 1 dispatch via multiprocessing.Pool
"""

import array
import os
from pathlib import Path

import numpy as np
import pysam
import pytest

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.manifest import ManifestEntry
from methylbert.data.nanopore.shards import shard_exists
from methylbert.data.nanopore.preprocess import (
    bam_basename,
    shard_basename,
    preprocess_bam,
    preprocess_manifest,
    _build_work_items,
    _list_bam_regions,
    DEFAULT_PRIMARY_REGEX,
)


# ---------------------------------------------------------------------------
# Test fixtures (reuse the helpers from the main preprocess tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab():
    return MethylVocab(k=3)


def _make_header(refs):
    """refs: list of (name, length) tuples."""
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": n, "LN": L} for n, L in refs],
    })


def _build_seq_with_calls(n_blocks=80):
    block = "ATGATACGAT"
    seq = block * n_blocks
    n_cs = seq.count("C")
    mm = "C+m?," + ",".join(["0"] * n_cs) + ";"
    ml = [200] * n_cs
    return seq, mm, ml


def _make_aligned_read(header, query_sequence, mm_tag, ml_bytes,
                      ref_id, ref_start, query_name="r"):
    read = pysam.AlignedSegment(header)
    read.query_name = query_name
    read.query_sequence = query_sequence
    read.flag = 0
    read.reference_id = ref_id
    read.reference_start = ref_start
    read.mapping_quality = 60
    read.cigartuples = [(0, len(query_sequence))]
    read.set_tag("MM", mm_tag)
    read.set_tag("ML", array.array("B", ml_bytes))
    return read


def _write_indexed_bam(path: Path, header, reads):
    """Write a sorted, indexed BAM. Region fetches require an index."""
    raw_path = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(raw_path), "wb", header=header) as out:
        for r in reads:
            out.write(r)
    pysam.sort("-o", str(path), str(raw_path))
    pysam.index(str(path))
    raw_path.unlink()


def _make_entry(bam_path):
    return ManifestEntry(
        bam_path=str(bam_path),
        mm_flag="?",
        reference_build="t",
        basecaller_version="t",
        mod_codes="C+m",
        cohort="t",
        sample_label="t",
    )


def _write_manifest_tsv(path: Path, entries):
    cols = ["bam_path", "mm_flag", "reference_build", "basecaller_version",
            "mod_codes", "cohort", "sample_label"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for e in entries:
            f.write("\t".join(getattr(e, c) for c in cols) + "\n")


# ---------------------------------------------------------------------------
# shard_basename
# ---------------------------------------------------------------------------

def test_shard_basename_no_region():
    assert shard_basename("/x/y/HG008.bam", None) == "HG008"


def test_shard_basename_with_region():
    assert shard_basename("/x/y/HG008.bam", "chr1") == "HG008__chr1"


def test_shard_basename_sanitizes_unsafe_chars():
    # Filesystem-unfriendly characters (like ':' or '/') get rewritten.
    assert shard_basename("/x/y/HG008.bam", "chr1:1-1000") == "HG008__chr1_1-1000"


# ---------------------------------------------------------------------------
# Region listing + filter regex
# ---------------------------------------------------------------------------

def test_list_bam_regions_filters_by_regex(tmp_path):
    """Default regex keeps chr1, chrX, chrM and rejects alts."""
    refs = [
        ("chr1", 10_000),
        ("chr2", 10_000),
        ("chrX", 10_000),
        ("chrM", 1_000),
        ("chr1_KI270706v1_random", 5_000),       # alt -> rejected
        ("chrUn_KI270302v1", 5_000),             # unplaced -> rejected
    ]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [
        _make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100,
                          query_name="r0"),
    ]
    bam_path = tmp_path / "t.bam"
    _write_indexed_bam(bam_path, header, reads)

    kept = _list_bam_regions(str(bam_path), DEFAULT_PRIMARY_REGEX)
    assert kept == ["chr1", "chr2", "chrX", "chrM"]


# ---------------------------------------------------------------------------
# preprocess_bam with region=
# ---------------------------------------------------------------------------

def test_region_filter_only_processes_one_contig(tmp_path, vocab):
    """A read on chr2 should not appear in the chr1 shard, and vice versa."""
    refs = [("chr1", 10_000), ("chr2", 10_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [
        _make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100,
                          query_name="r_chr1"),
        _make_aligned_read(header, seq, mm, ml, ref_id=1, ref_start=200,
                          query_name="r_chr2"),
    ]
    bam_path = tmp_path / "two_contigs.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)

    # Only chr1
    stats_chr1 = preprocess_bam(entry, tmp_path, vocab, region="chr1")
    assert stats_chr1.reads_total == 1
    assert stats_chr1.reads_kept == 1
    assert shard_exists(tmp_path, f"{bam_basename(bam_path)}__chr1")

    # Only chr2 (separate output dir to avoid name confusion)
    sub = tmp_path / "chr2_only"
    sub.mkdir()
    stats_chr2 = preprocess_bam(entry, sub, vocab, region="chr2")
    assert stats_chr2.reads_total == 1
    assert shard_exists(sub, f"{bam_basename(bam_path)}__chr2")


def test_region_stats_records_region(tmp_path, vocab):
    refs = [("chr1", 10_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [_make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100)]
    bam_path = tmp_path / "one.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    stats = preprocess_bam(entry, tmp_path, vocab, region="chr1")
    assert stats.region == "chr1"


# ---------------------------------------------------------------------------
# Work-item construction
# ---------------------------------------------------------------------------

def test_build_work_items_per_region(tmp_path, vocab):
    refs = [("chr1", 10_000), ("chr2", 10_000), ("chr_alt_1", 5_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [_make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100)]
    bam_path = tmp_path / "t.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    items = _build_work_items(
        [entry], tmp_path / "out",
        per_region=True,
        region_regex=DEFAULT_PRIMARY_REGEX,
        skip_existing=True,
    )
    # chr1 + chr2 match; chr_alt_1 is filtered out.
    regions = [r for (_, r) in items]
    assert sorted(regions) == ["chr1", "chr2"]


def test_build_work_items_skip_existing(tmp_path, vocab):
    refs = [("chr1", 10_000), ("chr2", 10_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [_make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100)]
    bam_path = tmp_path / "t.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)

    # Process chr1 first.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    preprocess_bam(entry, out_dir, vocab, region="chr1")

    # Now build work items: only chr2 should remain.
    items = _build_work_items(
        [entry], out_dir,
        per_region=True,
        region_regex=DEFAULT_PRIMARY_REGEX,
        skip_existing=True,
    )
    assert [(e.bam_path, r) for (e, r) in items] == [(str(bam_path), "chr2")]


# ---------------------------------------------------------------------------
# Manifest-level per-region path (serial)
# ---------------------------------------------------------------------------

def test_manifest_per_region_serial(tmp_path, vocab):
    refs = [("chr1", 10_000), ("chr2", 10_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [
        _make_aligned_read(header, seq, mm, ml, ref_id=0, ref_start=100,
                          query_name="r_chr1"),
        _make_aligned_read(header, seq, mm, ml, ref_id=1, ref_start=200,
                          query_name="r_chr2"),
    ]
    bam_path = tmp_path / "t.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry])

    out_dir = tmp_path / "out"
    all_stats = preprocess_manifest(
        manifest_path, out_dir, vocab=vocab,
        per_region=True, n_workers=1,
    )
    # Two regions match the default regex; both should be processed.
    assert len(all_stats) == 2
    base = bam_basename(bam_path)
    assert shard_exists(out_dir, f"{base}__chr1")
    assert shard_exists(out_dir, f"{base}__chr2")


# ---------------------------------------------------------------------------
# Parallel path (multiprocessing)
# ---------------------------------------------------------------------------

def test_manifest_per_region_parallel(tmp_path, vocab):
    """End-to-end: multiprocessing.Pool with n_workers=2 dispatches work items
    and produces the same shard set as serial mode."""
    refs = [("chr1", 10_000), ("chr2", 10_000), ("chr3", 10_000)]
    header = _make_header(refs)
    seq, mm, ml = _build_seq_with_calls()
    reads = [
        _make_aligned_read(header, seq, mm, ml, ref_id=i, ref_start=100,
                          query_name=f"r_{i}")
        for i in range(3)
    ]
    bam_path = tmp_path / "t.bam"
    _write_indexed_bam(bam_path, header, reads)

    entry = _make_entry(bam_path)
    manifest_path = tmp_path / "manifest.tsv"
    _write_manifest_tsv(manifest_path, [entry])

    out_dir = tmp_path / "out"
    all_stats = preprocess_manifest(
        manifest_path, out_dir, vocab=vocab,
        per_region=True, n_workers=2,
    )
    assert len(all_stats) == 3
    base = bam_basename(bam_path)
    for r in ["chr1", "chr2", "chr3"]:
        assert shard_exists(out_dir, f"{base}__{r}")