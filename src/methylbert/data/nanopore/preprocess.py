"""
methylbert.data.nanopore.preprocess

Driver that takes a manifest of BAMs and produces binary training shards.

Usage as a library:

    from methylbert.data.nanopore.preprocess import preprocess_manifest
    stats = preprocess_manifest("manifest.tsv", "/path/to/output_dir")

Usage as a CLI:

    python -m methylbert.data.nanopore.preprocess manifest.tsv \\
        --output-dir /path/to/output_dir [--min-mapq 0] [--max-reads-per-bam N]

The pipeline for each BAM:

    pysam reads -> filter (mapped, primary, MM/ML present, MAPQ)
                -> featurize_read    -> (token_ids, methyl_states)
                -> chunk_read        -> fixed-length windows
                -> ShardWriter       -> <basename>.{tokens.bin, states.bin, json}

Resumability:
    BAMs whose shard already exists (per shards.shard_exists) are skipped.
    Pass skip_existing=False to force reprocessing.

Single-process for now. Each BAM's writer is independent, so per-BAM
multiprocessing is the natural parallelisation point later.
"""

import argparse
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import pysam

from methylbert.data.vocab import MethylVocab
from methylbert.data.nanopore.featurize import featurize_read
from methylbert.data.nanopore.chunker import chunk_read, WINDOW_LEN, MIN_TAIL
from methylbert.data.nanopore.manifest import ManifestEntry, load_manifest
from methylbert.data.nanopore.shards import ShardWriter, shard_exists


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class BamStats:
    """Per-BAM processing stats. Reported at end of run."""
    bam_path: str
    reads_total: int = 0
    reads_unmapped: int = 0
    reads_secondary_supplementary: int = 0
    reads_low_mapq: int = 0
    reads_no_mm_tag: int = 0
    reads_featurize_error: int = 0
    reads_too_short: int = 0           # featurized but produced no windows
    reads_kept: int = 0                # featurized and produced >=1 window
    windows_written: int = 0
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bam_basename(bam_path) -> str:
    """Strip the .bam extension to get the shard basename."""
    name = Path(bam_path).name
    if name.lower().endswith(".bam"):
        name = name[:-4]
    return name


# ---------------------------------------------------------------------------
# Per-BAM preprocessor
# ---------------------------------------------------------------------------

def preprocess_bam(
    manifest_entry: ManifestEntry,
    output_dir,
    vocab: MethylVocab,
    *,
    window_len: int = WINDOW_LEN,
    min_tail: int = MIN_TAIL,
    min_mapq: int = 0,
    max_reads: Optional[int] = None,
    log_every: int = 100_000,
) -> BamStats:
    """
    Process one BAM into binary shards. Returns processing stats.

    Parameters
    ----------
    manifest_entry : ManifestEntry
        Manifest row for this BAM. Provides the path and mm_flag.
    output_dir : str or Path
        Directory the shards are written into.
    vocab : MethylVocab
        4-base 3-mer vocab passed to featurize_read.
    window_len, min_tail : int
        Chunker parameters.
    min_mapq : int
        Skip reads with MAPQ < this value. Default 0 (no extra filter on top
        of whatever upstream filtering produced this BAM).
    max_reads : int or None
        Cap on reads iterated per BAM. None = no cap. Useful for smoke tests.
    log_every : int
        Emit a progress line every this-many reads.

    Returns
    -------
    BamStats
        Counts of read disposition + windows written + elapsed time.

    Raises
    ------
    Whatever pysam raises for a malformed BAM. Per-read errors are caught and
    counted in stats.reads_featurize_error rather than propagated.
    """
    output_dir = Path(output_dir)
    basename = bam_basename(manifest_entry.bam_path)
    stats = BamStats(bam_path=manifest_entry.bam_path)
    start = time.monotonic()

    bam = pysam.AlignmentFile(manifest_entry.bam_path, "rb")
    try:
        with ShardWriter(output_dir, basename, window_len,
                         manifest_entry, vocab_size=len(vocab)) as writer:
            for read in bam.fetch(until_eof=True):
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
                # MM and ML must both be present (we require ML to read probabilities).
                # Accept legacy lowercase tag names too.
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
# Manifest-level preprocessor
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
) -> List[BamStats]:
    """
    Process all BAMs in a manifest. Returns one BamStats per BAM actually
    processed (skipped BAMs are not in the returned list).

    The manifest TSV is copied into output_dir as `manifest.tsv` for
    reproducibility (overwritten on each call).
    """
    if vocab is None:
        vocab = MethylVocab(k=3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    logger.info(f"Loaded {len(entries)} entries from {manifest_path}")

    if copy_manifest:
        shutil.copy2(manifest_path, output_dir / "manifest.tsv")

    all_stats: List[BamStats] = []
    for i, entry in enumerate(entries, start=1):
        basename = bam_basename(entry.bam_path)
        logger.info(f"[{i}/{len(entries)}] {basename}")

        if skip_existing and shard_exists(output_dir, basename):
            logger.info("  -> shard already exists, skipping")
            continue

        if not Path(entry.bam_path).exists():
            logger.warning(f"  -> BAM not found, skipping: {entry.bam_path}")
            continue

        stats = preprocess_bam(
            entry, output_dir, vocab,
            window_len=window_len, min_tail=min_tail,
            min_mapq=min_mapq, max_reads=max_reads_per_bam,
            log_every=log_every,
        )
        all_stats.append(stats)

        logger.info(
            f"  -> {stats.windows_written:,} windows from "
            f"{stats.reads_kept:,} kept reads "
            f"({stats.reads_total:,} examined) in {stats.elapsed_sec:.1f}s"
        )

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_final_summary(all_stats: List[BamStats]) -> None:
    if not all_stats:
        print("\nNo BAMs were processed.")
        return
    print()
    print("=" * 78)
    print(" FINAL SUMMARY")
    print("=" * 78)
    total_windows = 0
    total_reads = 0
    total_kept = 0
    total_elapsed = 0.0
    for s in all_stats:
        bn = bam_basename(s.bam_path)
        print(f"\n  {bn}")
        print(f"    reads examined:           {s.reads_total:>14,}")
        print(f"    reads kept:               {s.reads_kept:>14,}")
        if s.reads_unmapped:
            print(f"    skipped: unmapped         {s.reads_unmapped:>14,}")
        if s.reads_secondary_supplementary:
            print(f"    skipped: secondary/supp.  {s.reads_secondary_supplementary:>14,}")
        if s.reads_low_mapq:
            print(f"    skipped: low MAPQ         {s.reads_low_mapq:>14,}")
        if s.reads_no_mm_tag:
            print(f"    skipped: no MM/ML tag     {s.reads_no_mm_tag:>14,}")
        if s.reads_featurize_error:
            print(f"    skipped: featurize error  {s.reads_featurize_error:>14,}")
        if s.reads_too_short:
            print(f"    skipped: too short        {s.reads_too_short:>14,}")
        print(f"    windows written:          {s.windows_written:>14,}")
        print(f"    elapsed:                  {s.elapsed_sec:>13.1f}s")
        total_windows += s.windows_written
        total_reads += s.reads_total
        total_kept += s.reads_kept
        total_elapsed += s.elapsed_sec
    print()
    print(f"  TOTAL: {total_windows:,} windows from {total_kept:,} kept reads "
          f"({total_reads:,} examined) in {total_elapsed:.1f}s "
          f"across {len(all_stats)} BAM(s).")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess nanopore BAMs into MethylBERT training shards."
    )
    parser.add_argument("manifest", help="Path to TSV manifest")
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write shard files (.tokens.bin / .states.bin / .json) into",
    )
    parser.add_argument(
        "--min-mapq", type=int, default=0,
        help="Skip reads with MAPQ below this value (default 0, no extra filter)",
    )
    parser.add_argument(
        "--max-reads-per-bam", type=int, default=None,
        help="Cap reads examined per BAM (default: process all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Reprocess BAMs even if a complete shard already exists.",
    )
    parser.add_argument(
        "--log-every", type=int, default=100_000,
        help="Emit a progress line every this-many reads (default 100,000).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging (per-read featurize errors, etc.).",
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
    )

    _print_final_summary(all_stats)


if __name__ == "__main__":
    main()