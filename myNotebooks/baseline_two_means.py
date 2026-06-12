#!/usr/bin/env python3
"""
Two-means mean-methylation baseline for MethylBERT ONT fine-tuning.

The "can a transformer beat one number per window" floor. For each DMR it
estimates the mean methylation of T-origin and N-origin reads ON THE TRAIN
SPLIT ONLY, then classifies each eval/test window by which profile its own
mean methylation is closer to. Window-level, no read aggregation - matches
the model's primary metric.

Comparison label is identical to the model's: ctype_label = int(ctype ==
dmr_ctype) ("does this read's origin match the DMR's characteristic type").
AUROC is computed over that label so the number is directly comparable to the
methylation-aware and ablation models.

Methylation of a window = confident-methylated / confident-total, i.e.
count('1') / (count('0') + count('1')) over the methyl_seq string. States 2
(non-CpG) and 3 (unknown) are excluded from the denominator, not counted as
unmethylated.

DMR-count-agnostic: point --data-dir at any of dmr100/ dmr250/ dmr500/ dmr1000/.
Profiles are fit on train_seq.csv; AUROC reported on the eval file (default
test_seq.csv - which is the 15% eval split here, NOT the run-2 test set; point
--eval-file at run-2's data.csv for the final test number).

Usage:
    python baseline_two_means.py --data-dir ~/finetuneDatasets/dmr100
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

COLS = ["methyl_seq", "ctype", "dmr_ctype", "dmr_label"]


def window_methylation(s: str) -> float:
    """Confident-methylated fraction of one window's methyl_seq string."""
    if not isinstance(s, str):
        return np.nan
    n1 = s.count("1")
    n0 = s.count("0")
    tot = n0 + n1
    return (n1 / tot) if tot > 0 else np.nan


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", usecols=COLS,
        dtype={"methyl_seq": str, "ctype": str, "dmr_ctype": str, "dmr_label": int},
    )
    df["m"] = df["methyl_seq"].map(window_methylation)
    return df.drop(columns="methyl_seq")


def main():
    ap = argparse.ArgumentParser(description="Two-means methylation baseline.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-file", default="train_seq.csv")
    ap.add_argument("--eval-file", default="test_seq.csv",
                    help="15%% eval split (generator names it test_seq.csv), "
                         "or run-2 data.csv for the real test set.")
    args = ap.parse_args()

    train_path = os.path.join(os.path.expanduser(args.data_dir), args.train_file)
    eval_path  = os.path.join(os.path.expanduser(args.data_dir), args.eval_file)
    print(f"train profiles from: {train_path}")
    print(f"evaluate on:         {eval_path}\n")

    train = load(train_path)
    ev    = load(eval_path)

    # Drop windows with no confident calls (undefined mean).
    n_tr_nan = int(train["m"].isna().sum())
    n_ev_nan = int(ev["m"].isna().sum())
    train = train.dropna(subset=["m"])
    ev    = ev.dropna(subset=["m"])

    # ---- Fit per-DMR T/N profiles on TRAIN ONLY (mean of per-read fractions) ----
    prof_mean = train.pivot_table(index="dmr_label", columns="ctype", values="m", aggfunc="mean")
    prof_n    = train.pivot_table(index="dmr_label", columns="ctype", values="m", aggfunc="size")
    for c in ("T", "N"):
        if c not in prof_mean.columns:
            prof_mean[c] = np.nan
            prof_n[c] = 0

    # ---- Score eval windows (vectorised) ----
    ev = ev.join(prof_mean.rename(columns={"T": "profT", "N": "profN"}), on="dmr_label")
    is_T = ev["dmr_ctype"].values == "T"
    H  = np.where(is_T, ev["profT"].values, ev["profN"].values)   # dmr_ctype (hyper) profile
    Lo = np.where(is_T, ev["profN"].values, ev["profT"].values)   # other-ctype profile
    m  = ev["m"].values

    valid = ~np.isnan(H) & ~np.isnan(Lo)
    n_skipped = int((~valid).sum())

    # score higher => window looks like the DMR's characteristic type => label 1
    score = np.abs(m - Lo) - np.abs(m - H)
    label = (ev["ctype"].values == ev["dmr_ctype"].values).astype(int)

    score_v = score[valid]
    label_v = label[valid]

    # ---- Diagnostics ----
    n_dmrs = prof_mean.shape[0]
    both = (prof_n.get("T", 0) > 0) & (prof_n.get("N", 0) > 0)
    print(f"DMRs with train reads: {n_dmrs}  | with BOTH T and N profiles: {int(both.sum())}")
    reads_per_profile = prof_n[["T", "N"]].replace(0, np.nan).stack()
    print(f"train reads per (DMR,ctype) profile: "
          f"min={int(reads_per_profile.min())}, median={int(reads_per_profile.median())}, "
          f"max={int(reads_per_profile.max())}")
    print(f"windows dropped (no confident calls): train={n_tr_nan}, eval={n_ev_nan}")
    print(f"eval windows scored: {int(valid.sum())}  | skipped (missing DMR profile): {n_skipped}")
    pos = int(label_v.sum()); tot = len(label_v)
    print(f"eval ctype_label balance: {pos}/{tot} positive ({100*pos/tot:.1f}%)\n")

    # ---- Metrics (window-level, over ctype_label) ----
    auroc = roc_auc_score(label_v, score_v)
    pred = (score_v > 0).astype(int)          # nearest-mean hard decision
    bacc = balanced_accuracy_score(label_v, pred)
    cm = confusion_matrix(label_v, pred)
    print(f"window-level AUROC:        {auroc:.4f}")
    print(f"balanced accuracy (m>0):   {bacc:.4f}")
    print(f"confusion matrix [rows=true 0/1, cols=pred 0/1]:\n{cm}")


if __name__ == "__main__":
    main()