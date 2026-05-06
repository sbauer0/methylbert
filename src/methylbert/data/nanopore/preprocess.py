"""
methylbert.data.nanopore.preprocess

Driver that takes a manifest of BAMs and produces binary training shards.

Usage as a library:

    from methylbert.data.nanopore.preprocess import preprocess_manifest
    stats = preprocess_manifest(
        "manifest.tsv",
        "/path/to/output_dir",
        n_workers=64,
        per_region=True,
    )

Usage as a CLI:

    python -m methylbert.data.nanopore.preprocess manifest.tsv \\
        --output-dir /path/to/output_dir \\
        --per-region \\
        --n-workers 64

The pipeline for each (BAM, region) work item:

    pysam reads -> filter (mapped, primary, MM/ML present, MAPQ)
                -> featurize_read    -> (token_ids, methyl_states)
                -> chunk_read        -> fixed-length windows
                -> ShardWriter       -> <basename>.{tokens.bin, states.bin, json}

When per_region=False, "region" is None and one shard is produced per BAM.
When per_region=True, one shard is produced per (BAM, contig) pair, with
basename `<bam>__<contig>`.

Resumability:
    Work items whose shard already exists (per shards.shard_exists) are
    skipped at queue-build time. Pass skip_existing=False to force.
"""

import argparse
import logging
import multiprocessing as mp
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pysam

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.featurize import featurize_read
from methylbert.data.nanopore.chunker import chunk_read, WINDOW_LEN, MIN_TAIL
from methylbert.data.nanopore.manifest import ManifestEntry, load_manifest
from methylbert.data.nanopore.shards import ShardWriter, shard_exists


logger = logging.getLogger(__name__)


# Default region filter for per_region mode: human primary chromosomes only.
# Matches both CHM13v2 and hg38 UCSC-style names. Excludes alt contigs,
# decoys, unplaced scaffolds. Override via --region-regex if you need them.
DEFAULT_PRIMARY_REGEX = r"^chr([0-9]+|X|Y|M|MT)$"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class BamStats:
    """Per-(BAM, region) processing stats."""
    bam_path: str
    region: Optional[str] = None
    reads_total: int = 0
    reads_unmapped: int = 0
    reads_secondary_supplementary: int = 0
    reads_low_mapq: int = 0
    reads_no_mm_tag: int = 0
    reads_featurize_error: int = 0
    reads_too_short: int = 0
    reads_kept: int = 0
    windows_written: int = 0
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bam_basename(bam_path) -> str:
    """Strip the .bam extension to get the BAM basename."""
    name = Path(bam_path).name
    if name.lower().endswith(".bam"):
        name = name[:-4]
    return name


def shard_basename(bam_path, region: Optional[str]) -> str:
    """Compute the shard basename for a (BAM, region) pair.

        region=None    -> "<bam>"
        region="chr1"  -> "<bam>__chr1"
    """
    base = bam_basename(bam_path)
    if region is None:
        return base
    safe_region = re.sub(r"[^A-Za-z0-9._-]", "_", region)
    return f"{base}__{safe_region}"


# ---------------------------------------------------------------------------
# Per-(BAM, region) preprocessor
# ---------------------------------------------------------------------------

def preprocess_bam(
    manifest_entry: ManifestEntry,
    output_dir,
    vocab: MethylVocab,
    *,
    region: Optional[str] = None,
    window_len: int = WINDOW_LEN,
    min_tail: int = MIN_TAIL,
    min_mapq: int = 0,
    max_reads: Optional[int] = None,
    log_every: int = 100_000,
) -> BamStats:
    """
    Process one BAM (or one region of one BAM) into a single shard set.
    Returns BamStats.

    region : str or None
        If None, iterate the whole BAM (`fetch(until_eof=True)`).
        If a contig name, only that region (`fetch(contig=region)`); BAM must
        be indexed. Output basename includes the contig: `<bam>__<region>`.
    """
    output_dir = Path(output_dir)
    basename = shard_basename(manifest_entry.bam_path, region)
    stats = BamStats(bam_path=manifest_entry.bam_path, region=region)
    start = time.monotonic()

    bam = pysam.AlignmentFile(manifest_entry.bam_path, "rb")
    try:
        if region is None:
            iterator = bam.fetch(until_eof=True)
        else:
            iterator = bam.fetch(contig=region)

        with ShardWriter(output_dir, basename, window_len,
                         manifest_entry, vocab_size=len(vocab)) as writer:
            for read in iterator:
                if max_reads is not None and stats.reads_total >= max_reads:
                    break
                stats.reads_total += 1

                if log_every and stats.reads_total % log_every == 0:
                    logger.info(
                        f"  [{basename}] {stats.reads_total:,} examined, "
                        f"{stats.reads_kept:,} kept, "
                        f"{stats.windows_written:,} windows..."
                    )

                if read.is_unmapped:
                    stats.reads_unmapped += 1
                    continue
                if read.is_secondary or read.is_supplementary:
                    stats.reads_secondary_supplementary += 1
                    continue
                if read.mapping_quality < min_mapq:
                    stats.reads_low_mapq += 1
                    continue
                has_mm = read.has_tag("MM") or read.has_tag("Mm")
                has_ml = read.has_tag("ML") or read.has_tag("Ml")
                if not (has_mm and has_ml):
                    stats.reads_no_mm_tag += 1
                    continue

                try:
                    token_ids, methyl_states = featurize_read(
                        read, vocab, mm_flag=manifest_entry.mm_flag,
                    )
                except (ValueError, RuntimeError) as e:
                    stats.reads_featurize_error += 1
                    logger.debug(
                        f"featurize error on {read.query_name!r}: {e}"
                    )
                    continue

                n_windows_this_read = 0
                for window_tokens, window_states in chunk_read(
                    token_ids, methyl_states,
                    window_len=window_len, min_tail=min_tail,
                ):
                    writer.append(window_tokens, window_states)
                    n_windows_this_read += 1

                if n_windows_this_read == 0:
                    stats.reads_too_short += 1
                else:
                    stats.reads_kept += 1
                    stats.windows_written += n_windows_this_read
    finally:
        bam.close()

    stats.elapsed_sec = time.monotonic() - start
    return stats


# ---------------------------------------------------------------------------
# Work-item construction
# ---------------------------------------------------------------------------

def _list_bam_regions(bam_path, regex: str) -> List[str]:
    """Return contig names in the BAM matching `regex`, in BAM-header order."""
    pat = re.compile(regex)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return [ref for ref in bam.references if pat.match(ref)]


def _build_work_items(
    entries: List[ManifestEntry],
    output_dir: Path,
    per_region: bool,
    region_regex: str,
    skip_existing: bool,
) -> List[Tuple[ManifestEntry, Optional[str]]]:
    """Build (entry, region) work items, with resumability filtering applied."""
    work_items: List[Tuple[ManifestEntry, Optional[str]]] = []
    n_skipped_resume = 0

    for entry in entries:
        if not Path(entry.bam_path).exists():
            logger.warning(f"BAM not found, skipping: {entry.bam_path}")
            continue

        if per_region:
            try:
                regions = _list_bam_regions(entry.bam_path, region_regex)
            except Exception as e:
                logger.warning(
                    f"Could not list regions for {entry.bam_path}: {e}"
                )
                continue
            if not regions:
                logger.warning(
                    f"No contigs match {region_regex!r} in "
                    f"{entry.bam_path}; this BAM will produce no shards."
                )
            for region in regions:
                basename = shard_basename(entry.bam_path, region)
                if skip_existing and shard_exists(output_dir, basename):
                    n_skipped_resume += 1
                    continue
                work_items.append((entry, region))
        else:
            basename = shard_basename(entry.bam_path, None)
            if skip_existing and shard_exists(output_dir, basename):
                n_skipped_resume += 1
                continue
            work_items.append((entry, None))

    if n_skipped_resume:
        logger.info(f"Skipping {n_skipped_resume} work items "
                    f"already complete on disk.")
    return work_items


# ---------------------------------------------------------------------------
# Worker entry point (must be at module level for multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _process_one_workitem(args):
    """Pool worker. Args is a tuple to be picklable across the fork boundary."""
    (manifest_entry, region, output_dir, vocab, kwargs) = args
    try:
        return preprocess_bam(
            manifest_entry, output_dir, vocab, region=region, **kwargs
        )
    except Exception as e:
        # Don't let one bad work item poison the pool. Log and return a
        # zero-stats record so the main process can keep going.
        logger.error(
            f"Worker failed on bam={manifest_entry.bam_path!r} "
            f"region={region!r}: {type(e).__name__}: {e}"
        )
        return BamStats(bam_path=manifest_entry.bam_path, region=region)


# ---------------------------------------------------------------------------
# Manifest-level driver
# ---------------------------------------------------------------------------

def preprocess_manifest(
    manifest_path,
    output_dir,
    vocab: Optional[MethylVocab] = None,
    *,
    window_len: int = WINDOW_LEN,
    min_tail: int = MIN_TAIL,
    min_mapq: int = 0,
    max_reads_per_bam: Optional[int] = None,
    skip_existing: bool = True,
    log_every: int = 100_000,
    copy_manifest: bool = True,
    n_workers: int = 1,
    per_region: bool = False,
    region_regex: str = DEFAULT_PRIMARY_REGEX,
) -> List[BamStats]:
    """
    Process all BAMs in a manifest. With per_region=True, each BAM is split by
    contig (filtered by `region_regex`); each (BAM, region) pair becomes one
    work item. With n_workers > 1, work items run in a multiprocessing.Pool.
    Returns one BamStats per work item that actually ran.
    """
    if vocab is None:
        vocab = MethylVocab(k=3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    logger.info(f"Loaded {len(entries)} entries from {manifest_path}")

    if copy_manifest:
        shutil.copy2(manifest_path, output_dir / "manifest.tsv")

    work_items = _build_work_items(
        entries, output_dir, per_region, region_regex, skip_existing
    )
    logger.info(
        f"Built {len(work_items)} work items "
        f"(per_region={per_region}, n_workers={n_workers})."
    )
    if not work_items:
        return []

    kwargs = dict(
        window_len=window_len,
        min_tail=min_tail,
        min_mapq=min_mapq,
        max_reads=max_reads_per_bam,
        log_every=log_every,
    )
    work_args = [
        (entry, region, output_dir, vocab, kwargs)
        for (entry, region) in work_items
    ]

    all_stats: List[BamStats] = []

    if n_workers <= 1:
        # Serial path (also useful for debugging).
        for i, args in enumerate(work_args, start=1):
            entry, region, _, _, _ = args
            tag = shard_basename(entry.bam_path, region)
            logger.info(f"[{i}/{len(work_args)}] {tag}")
            stats = _process_one_workitem(args)
            all_stats.append(stats)
            logger.info(
                f"  -> {stats.windows_written:,} windows from "
                f"{stats.reads_kept:,} kept reads "
                f"({stats.reads_total:,} examined) in {stats.elapsed_sec:.1f}s"
            )
    else:
        logger.info(f"Dispatching {len(work_args)} work items "
                    f"to {n_workers} workers...")
        n_done = 0
        with mp.Pool(n_workers) as pool:
            for stats in pool.imap_unordered(_process_one_workitem, work_args):
                all_stats.append(stats)
                n_done += 1
                tag = shard_basename(stats.bam_path, stats.region)
                logger.info(
                    f"[{n_done}/{len(work_args)}] {tag} -> "
                    f"{stats.windows_written:,} windows from "
                    f"{stats.reads_kept:,} reads in "
                    f"{stats.elapsed_sec:.1f}s"
                )

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_final_summary(all_stats: List[BamStats]) -> None:
    if not all_stats:
        print("\nNo work items were processed.")
        return
    print()
    print("=" * 78)
    print(" FINAL SUMMARY")
    print("=" * 78)
    by_bam = {}
    for s in all_stats:
        by_bam.setdefault(s.bam_path, []).append(s)
    total_windows = 0
    total_reads = 0
    total_kept = 0
    max_elapsed = 0.0
    for bam_path, items in by_bam.items():
        bn = bam_basename(bam_path)
        n_regions = len(items)
        wins = sum(s.windows_written for s in items)
        reads = sum(s.reads_total for s in items)
        kept = sum(s.reads_kept for s in items)
        bam_max = max((s.elapsed_sec for s in items), default=0.0)
        print(f"\n  {bn}  ({n_regions} region{'s' if n_regions != 1 else ''})")
        print(f"    reads examined:           {reads:>14,}")
        print(f"    reads kept:               {kept:>14,}")
        print(f"    windows written:          {wins:>14,}")
        print(f"    elapsed (max region):     {bam_max:>13.1f}s")
        total_windows += wins
        total_reads += reads
        total_kept += kept
        max_elapsed = max(max_elapsed, bam_max)
    print()
    print(f"  TOTAL: {total_windows:,} windows from {total_kept:,} kept "
          f"({total_reads:,} examined) across {len(by_bam)} BAM(s).")
    print(f"  Wall time (max single work item): {max_elapsed:.1f}s")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess nanopore BAMs into MethylBERT training shards."
    )
    parser.add_argument("manifest", help="Path to TSV manifest")
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write shard files into",
    )
    parser.add_argument(
        "--min-mapq", type=int, default=0,
        help="Skip reads with MAPQ below this value (default 0)",
    )
    parser.add_argument(
        "--max-reads-per-bam", type=int, default=None,
        help="Cap reads examined per BAM (default: process all). "
             "Per-region mode caps per (BAM, region).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Reprocess work items even if a complete shard exists.",
    )
    parser.add_argument(
        "--log-every", type=int, default=100_000,
        help="Emit progress line every this-many reads per worker (default 100k).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Multiprocessing workers (default 1 = serial).",
    )
    parser.add_argument(
        "--per-region", action="store_true",
        help="Split each BAM by reference contig into independent shards. "
             "Required for fine-grained parallelism on big BAMs.",
    )
    parser.add_argument(
        "--region-regex", type=str, default=DEFAULT_PRIMARY_REGEX,
        help=f"Regex for contigs to keep in per-region mode "
             f"(default: {DEFAULT_PRIMARY_REGEX!r}).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    all_stats = preprocess_manifest(
        args.manifest, args.output_dir,
        min_mapq=args.min_mapq,
        max_reads_per_bam=args.max_reads_per_bam,
        skip_existing=not args.no_skip_existing,
        log_every=args.log_every,
        n_workers=args.n_workers,
        per_region=args.per_region,
        region_regex=args.region_regex,
    )

    _print_final_summary(all_stats)


if __name__ == "__main__":
    main()