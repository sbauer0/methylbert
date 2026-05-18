#!/usr/bin/env python3
"""
build_balanced_corpus.py

One-off script: subsample the full preprocessed nanopore corpus into a
balanced training set + chromosome-22 held-out test set.

Spec:
    - 5 biological conditions, weighted equally:
          HG008 normal duodenal, HG008 normal pancreatic, HG008 tumor PDAC,
          COLO829 cancer (2 flow-cell BAMs), COLO829BL normal (2 flow-cell BAMs)
    - Each condition contributes target_rows / 5 rows to training.
    - Within a condition, target rows are split equally across member BAMs
      (so each COLO829 BAM gets half what each HG008 BAM gets).
    - Within a BAM, rows are sampled uniformly across all non-chr22 shards.
      Chromosome representation falls out of preprocessing coverage.
    - chr22 shards from every BAM are held out as test set, copied unchanged.

Output layout:
    <output_dir>/
    ├── train/                 168 subsampled balanced train shards
    ├── test/                    7 chr22 shards (full, one per BAM)
    └── subsampling_summary.txt  per-BAM/condition row counts

Each shard preserves its original basename. Output is read-compatible with
MethylBertPretrainDatasetBinary by pointing data_dir at the train/ or test/
subdir.

Usage:
    python build_balanced_corpus.py \\
        --source-dir /data/gidb/shared/datasets/MethylBERT/pretrain_shards_4state_v1 \\
        --output-dir /home/bauerste/pretrain_data/pretrain_shards_4state_v1_balanced \\
        --target-train-rows 123000000 \\
        --workers 32

Resumability: completed shards detected via shards.shard_exists are skipped.
Safe to re-run after interruption.
"""

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from methylbert.data.nanopore.shards import ShardWriter, shard_exists
from methylbert.data.nanopore.manifest import ManifestEntry


HELD_OUT_CHROM_SUFFIX = "__chr22"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_shards(source_dir: Path):
    """Discover all shards from source_dir. Returns list of dicts."""
    shards = []
    for json_path in sorted(source_dir.glob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        if "n_rows" not in meta or "window_len" not in meta:
            logger.warning(f"Skipping non-shard JSON: {json_path.name}")
            continue
        shard_basename = json_path.stem
        tokens_path = source_dir / meta.get(
            "tokens_file", f"{shard_basename}.tokens.bin"
        )
        states_path = source_dir / meta.get(
            "states_file", f"{shard_basename}.states.bin"
        )
        if not tokens_path.exists() or not states_path.exists():
            logger.warning(f"Skipping shard with missing bins: {shard_basename}")
            continue
        manifest_entry = ManifestEntry(**meta["manifest_entry"])
        # bam_basename is the part before the __chrN suffix
        if "__" in shard_basename:
            bam_basename = shard_basename.rsplit("__", 1)[0]
        else:
            bam_basename = shard_basename
        shards.append({
            "shard_basename": shard_basename,
            "json_path": json_path,
            "tokens_path": tokens_path,
            "states_path": states_path,
            "n_rows": int(meta["n_rows"]),
            "window_len": int(meta["window_len"]),
            "vocab_size": int(meta["vocab_size"]),
            "manifest_entry": manifest_entry,
            "is_chr22": shard_basename.endswith(HELD_OUT_CHROM_SUFFIX),
            "bam_basename": bam_basename,
        })
    return shards


# ---------------------------------------------------------------------------
# Sample-size assignment
# ---------------------------------------------------------------------------

def assign_sample_sizes(train_shards, target_train_rows: int):
    """Compute per-shard sample size, balanced across 5 conditions.

    Conditions are identified by (cohort, sample_label) tuples from each
    shard's manifest_entry. Each condition is allocated target_train_rows / 5
    rows; that allocation is split equally across BAMs in the condition;
    within each BAM, the per-BAM target is distributed across the BAM's
    shards proportional to each shard's row count (i.e. uniform sampling
    across all non-chr22 rows of that BAM).

    Modifies each shard dict in place, adding 'sample_size'.
    Returns a summary dict for logging.
    """
    conditions = defaultdict(list)
    for shard in train_shards:
        me = shard["manifest_entry"]
        conditions[(me.cohort, me.sample_label)].append(shard)

    n_conditions = len(conditions)
    if n_conditions != 5:
        raise ValueError(
            f"Expected exactly 5 conditions, found {n_conditions}: "
            f"{sorted(conditions.keys())}"
        )

    target_per_condition = target_train_rows // n_conditions
    summary = {
        "target_train_rows": target_train_rows,
        "n_conditions": n_conditions,
        "target_per_condition": target_per_condition,
        "by_condition": {},
    }

    for cond_key, cond_shards in conditions.items():
        bams = sorted(set(s["bam_basename"] for s in cond_shards))
        n_bams = len(bams)
        target_per_bam = target_per_condition // n_bams

        cond_info = {
            "bams": bams,
            "n_bams": n_bams,
            "target_per_bam": target_per_bam,
            "per_bam": {},
        }

        for bam in bams:
            bam_shards = [s for s in cond_shards if s["bam_basename"] == bam]
            total_rows = sum(s["n_rows"] for s in bam_shards)
            if total_rows == 0:
                logger.warning(f"BAM {bam}: 0 total rows, skipping")
                fraction = 0.0
            else:
                fraction = target_per_bam / total_rows
                if fraction > 1.0:
                    logger.warning(
                        f"BAM {bam}: needs {target_per_bam:,} rows but only "
                        f"{total_rows:,} available; capping at all rows."
                    )
                    fraction = 1.0
            actual_total = 0
            for s in bam_shards:
                size = int(round(s["n_rows"] * fraction))
                s["sample_size"] = size
                actual_total += size
            cond_info["per_bam"][bam] = {
                "total_rows_available": total_rows,
                "fraction": fraction,
                "actual_rows_sampled": actual_total,
                "n_shards": len(bam_shards),
            }
        summary["by_condition"]["+".join(cond_key)] = cond_info

    return summary


# ---------------------------------------------------------------------------
# Workers (must be at module level for multiprocessing.Pool)
# ---------------------------------------------------------------------------

def subsample_shard(args):
    """Worker: subsample one train shard."""
    shard, train_dir, seed = args
    train_dir = Path(train_dir)

    if shard["sample_size"] == 0:
        return shard["shard_basename"], 0, 0.0
    if shard_exists(train_dir, shard["shard_basename"]):
        return shard["shard_basename"], -1, 0.0  # -1 => skipped (already done)

    t0 = time.monotonic()

    tokens_mm = np.memmap(
        shard["tokens_path"], dtype=np.int16, mode="r",
        shape=(shard["n_rows"], shard["window_len"]),
    )
    states_mm = np.memmap(
        shard["states_path"], dtype=np.int8, mode="r",
        shape=(shard["n_rows"], shard["window_len"]),
    )

    n_select = min(shard["sample_size"], shard["n_rows"])
    rng = np.random.default_rng(seed)
    indices = rng.choice(shard["n_rows"], size=n_select, replace=False)
    indices.sort()  # sequential reads are kinder to memmap than random

    with ShardWriter(
        train_dir, shard["shard_basename"], shard["window_len"],
        shard["manifest_entry"], vocab_size=shard["vocab_size"],
    ) as writer:
        for i in indices:
            writer.append(np.asarray(tokens_mm[i]),
                          np.asarray(states_mm[i]))

    return shard["shard_basename"], n_select, time.monotonic() - t0


def copy_chr22_shard(args):
    """Worker: copy one chr22 shard verbatim to test_dir (atomic per file)."""
    shard, test_dir = args
    test_dir = Path(test_dir)

    if shard_exists(test_dir, shard["shard_basename"]):
        return shard["shard_basename"], -1, 0.0

    t0 = time.monotonic()
    test_dir.mkdir(parents=True, exist_ok=True)
    for src in [shard["tokens_path"], shard["states_path"], shard["json_path"]]:
        dst = test_dir / src.name
        dst_tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, dst_tmp)
        dst_tmp.replace(dst)
    return shard["shard_basename"], shard["n_rows"], time.monotonic() - t0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(output_dir: Path, summary: dict,
                  train_results, test_results) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "subsampling_summary.txt"

    train_actual = sum(r[1] for r in train_results if r[1] >= 0)
    test_actual = sum(r[1] for r in test_results if r[1] >= 0)
    train_skipped = sum(1 for r in train_results if r[1] == -1)
    test_skipped = sum(1 for r in test_results if r[1] == -1)

    with open(out, "w") as f:
        f.write("=" * 78 + "\n")
        f.write(" SUBSAMPLING SUMMARY\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Train target            : {summary['target_train_rows']:>14,} rows\n")
        f.write(f"Conditions              : {summary['n_conditions']:>14}\n")
        f.write(f"Per-condition target    : {summary['target_per_condition']:>14,} rows\n\n")
        f.write(f"Train rows this run     : {train_actual:>14,}\n")
        f.write(f"Test rows this run      : {test_actual:>14,}\n")
        f.write(f"Train shards skipped    : {train_skipped:>14}  (resume mode)\n")
        f.write(f"Test shards skipped     : {test_skipped:>14}  (resume mode)\n\n")
        f.write("-" * 78 + "\n")
        f.write(" Per-condition breakdown (computed targets, not run-specific)\n")
        f.write("-" * 78 + "\n")
        for cond, info in summary["by_condition"].items():
            f.write(f"\n  Condition: {cond}\n")
            f.write(f"    BAMs in condition  : {info['n_bams']}\n")
            f.write(f"    Target per BAM     : {info['target_per_bam']:,}\n")
            for bam, b in info["per_bam"].items():
                f.write(f"    BAM: {bam[:65]}\n")
                f.write(f"      rows available  : {b['total_rows_available']:>14,}\n")
                f.write(f"      sampling rate   : {b['fraction']:>14.4%}\n")
                f.write(f"      rows assigned   : {b['actual_rows_sampled']:>14,}\n")
                f.write(f"      shards          : {b['n_shards']:>14}\n")
    logger.info(f"Wrote summary: {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _log_train_result(r, n_done, n_total):
    name, n, dt = r
    if n == -1:
        logger.info(f"  [train {n_done}/{n_total}] {name}  (skipped, already done)")
    else:
        logger.info(f"  [train {n_done}/{n_total}] {name}: "
                    f"{n:,} rows in {dt:.1f}s")


def _log_test_result(r, n_done, n_total):
    name, n, dt = r
    if n == -1:
        logger.info(f"  [test {n_done}/{n_total}] {name}  (skipped, already done)")
    else:
        logger.info(f"  [test {n_done}/{n_total}] {name}: "
                    f"{n:,} rows copied in {dt:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Full preprocessed corpus directory.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write balanced train/ + test/ subdirs.")
    parser.add_argument("--target-train-rows", type=int, default=123_000_000,
                        help="Total training rows in the output (default 123M).")
    parser.add_argument("--workers", type=int, default=16,
                        help="Multiprocessing workers (default 16).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for per-shard sampling RNGs.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source dir : {source_dir}")
    logger.info(f"Output dir : {output_dir}")

    # 1. Discover all shards
    logger.info("Discovering shards...")
    shards = discover_shards(source_dir)
    train_shards = [s for s in shards if not s["is_chr22"]]
    test_shards  = [s for s in shards if s["is_chr22"]]
    logger.info(f"  Total : {len(shards)}  "
                f"(train={len(train_shards)}, test={len(test_shards)})")
    if len(train_shards) == 0 or len(test_shards) == 0:
        raise RuntimeError("Empty train or test shard set; check source dir.")

    # 2. Compute per-shard sampling targets
    summary = assign_sample_sizes(train_shards, args.target_train_rows)
    expected_train = sum(s["sample_size"] for s in train_shards)
    expected_test = sum(s["n_rows"] for s in test_shards)
    logger.info(f"  Expected train rows : {expected_train:,}")
    logger.info(f"  Expected test rows  : {expected_test:,}")

    # 3. Build work queues
    train_args = [(s, str(train_dir), args.seed + i)
                  for i, s in enumerate(train_shards)]
    test_args = [(s, str(test_dir)) for s in test_shards]

    # 4. Dispatch
    logger.info(f"Subsampling {len(train_args)} train shards "
                f"and copying {len(test_args)} test shards "
                f"with {args.workers} worker(s)...")
    t0 = time.monotonic()

    train_results = []
    if args.workers <= 1:
        for a in train_args:
            r = subsample_shard(a)
            train_results.append(r)
            _log_train_result(r, len(train_results), len(train_args))
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(subsample_shard, train_args):
                train_results.append(r)
                _log_train_result(r, len(train_results), len(train_args))

    test_results = []
    # Test copy is IO-bound; modest worker count is plenty.
    test_workers = min(args.workers, max(1, len(test_args)))
    if test_workers <= 1:
        for a in test_args:
            r = copy_chr22_shard(a)
            test_results.append(r)
            _log_test_result(r, len(test_results), len(test_args))
    else:
        with mp.Pool(test_workers) as pool:
            for r in pool.imap_unordered(copy_chr22_shard, test_args):
                test_results.append(r)
                _log_test_result(r, len(test_results), len(test_args))

    elapsed = time.monotonic() - t0

    # 5. Summary
    write_summary(output_dir, summary, train_results, test_results)
    train_total = sum(r[1] for r in train_results if r[1] >= 0)
    test_total = sum(r[1] for r in test_results if r[1] >= 0)
    logger.info("")
    logger.info(f"Done in {elapsed:.1f}s.")
    logger.info(f"  Train: {train_total:,} rows in {len(train_results)} shards")
    logger.info(f"  Test : {test_total:,} rows in {len(test_results)} shards")


if __name__ == "__main__":
    main()