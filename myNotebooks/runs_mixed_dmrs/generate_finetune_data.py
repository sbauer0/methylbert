#!/usr/bin/env python3
"""
build_mixed_dmr_data.py

One shot for the mixed-areaStat 1000-DMR experiment:
  1. Stratified-random selection of 1000 DMRs across four |areaStat| bands,
     balanced 50/50 T vs N within each band, fixed seed -> a dmr1000 TSV.
  2. MethylBERT fine-tune data generation:
       run-1 (PAU5*) -> train/eval  (split_ratio = 0.85)
       run-2 (PAU6*) -> test        (split_ratio = None)

The selection is done HERE, not by MethylBERT. finetune_data_generate is called
with n_dmrs=-1 so it uses every row of the pre-selected file (with n_dmrs>0 it
would sort by |areaStat| and keep only the top-n per ctype, which would discard
exactly the low bands this experiment is about).

Run detached:
  nohup python build_mixed_dmr_data.py > build_mixed.log 2>&1 &
  tail -f build_mixed.log
"""
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config -- paths on the working cluster
# ----------------------------------------------------------------------------
HOME     = "/home/bauerste"
DMRS_IN  = f"{HOME}/methylbertDMRs/dmrs_filtered_min200.tsv"
RUNDIR   = f"{HOME}/methylbert_finetune/methylbert/myNotebooks/runs_mixed_dmrs"
DMR_OUT  = f"{RUNDIR}/dmr1000_mixed.tsv"
REF      = f"{HOME}/GRCh38_no_alt_analysis_set/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
TRAINDIR = f"{HOME}/finetuneDatasetsMixedDMR"
TESTDIR  = f"{HOME}/finetuneTestdataMixedDMR"

BAM  = "/tmp/bauerste/COLO829_BL"
RUN1 = [(f"{BAM}/colo829/PAU59949.d052sup4305mCG_5hmCGvHg38_pass.bam",   "T"),   # train/eval
        (f"{BAM}/colo829bl/PAU59807.d052sup4305mCG_5hmCGvHg38_pass.bam", "N")]
RUN2 = [(f"{BAM}/colo829/PAU61426.d052sup4305mCG_5hmCGvHg38_pass.bam",   "T"),   # test
        (f"{BAM}/colo829bl/PAU61427.d052sup4305mCG_5hmCGvHg38_pass.bam", "N")]

SEED = 42
# (low_exclusive, high_inclusive, n_per_ctype) -> band totals 400/300/200/100
BANDS = [(200, 300, 200),
         (300, 400, 150),
         (400, 600, 100),
         (600, float("inf"), 50)]


# ----------------------------------------------------------------------------
# 1. DMR selection  (no MethylBERT import -> unit-testable on its own)
# ----------------------------------------------------------------------------
def select_dmrs(in_path, out_path, seed=SEED, bands=BANDS):
    df = pd.read_csv(in_path, sep="\t", index_col=None)
    for col in ("chr", "start", "end", "ctype", "areaStat"):
        if col not in df.columns:
            raise SystemExit(f"input DMR file is missing required column: {col!r}")
    orig_cols = list(df.columns)
    df = df.copy()
    df["_abs"] = df["areaStat"].abs()

    rng = np.random.RandomState(seed)
    picks, summary = [], []
    for lo, hi, k in bands:
        for c in ("T", "N"):
            pool = df[(df["_abs"] > lo) & (df["_abs"] <= hi) & (df["ctype"] == c)]
            if len(pool) < k:
                raise SystemExit(
                    f"band ({lo}, {hi}] ctype {c}: need {k}, only {len(pool)} "
                    f"available. Adjust BANDS.")
            picks.append(pool.sample(n=k, random_state=rng))
            summary.append((f"{lo}<s<={hi}", c, k, len(pool)))

    sel = pd.concat(picks).sort_values("_abs", ascending=False)[orig_cols]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sel.to_csv(out_path, sep="\t", index=False)

    # validate + report
    assert len(sel) == sum(k for _, _, k in bands) * 2, "unexpected total"
    n_t = int((sel["ctype"] == "T").sum())
    n_n = int((sel["ctype"] == "N").sum())
    print(f"selected {len(sel)} DMRs  ({n_t} T / {n_n} N)  seed={seed}")
    print(f"{'band':14s} {'ctype':5s} {'picked':>6s} {'pool':>7s}")
    for band, c, k, pool_n in summary:
        print(f"{band:14s} {c:5s} {k:>6d} {pool_n:>7d}")
    print(f"written -> {out_path}")
    return sel


# ----------------------------------------------------------------------------
# 2. MethylBERT data generation  (import deferred to keep step 1 importable)
# ----------------------------------------------------------------------------
def generate_data(dmr_file):
    from functools import partial
    from methylbert.data.finetune_data_generate import finetune_data_generate
    from methylbert.data.nanopore.finetune_extract import ont_read_extract

    os.makedirs(TRAINDIR, exist_ok=True)
    os.makedirs(TESTDIR, exist_ok=True)

    def write_sc(path, bams):
        with open(path, "w") as f:
            for bam, label in bams:
                f.write(f"{bam}\t{label}\n")
        return path

    sc_run1 = write_sc(os.path.join(TRAINDIR, "run1_bams.tsv"), RUN1)
    sc_run2 = write_sc(os.path.join(TESTDIR,  "run2_bams.tsv"), RUN2)

    passes = [("run1 (train/eval)", os.path.join(TRAINDIR, "dmr1000"),      sc_run1, 0.85),
              ("test",              os.path.join(TESTDIR,  "dmr1000_test"),  sc_run2, None)]

    for tag, out, sc, split in passes:
        os.makedirs(out, exist_ok=True)
        print(f"\n=== {tag}  split={split} -> {out} ===", flush=True)
        finetune_data_generate(
            f_dmr=dmr_file,
            output_dir=out,
            f_ref=REF,
            sc_dataset=sc,
            n_mers=3,
            n_dmrs=-1,                 # use ALL pre-selected DMRs; do NOT re-select
            split_ratio=split,
            ignore_sex_chromo=False,
            methyl_caller="dorado",
            read_extract_sequences_func=partial(ont_read_extract, mm_flag="?"),
        )
        print(f"=== done {out} ===", flush=True)


def main():
    sel = select_dmrs(DMRS_IN, DMR_OUT)
    generate_data(DMR_OUT)
    print("\nALL DONE")


if __name__ == "__main__":
    main()