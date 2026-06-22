#!/usr/bin/env python3
"""Two-means mean-methylation baseline (cross-directory, importable).
Fits per-DMR T/N profiles on a train window CSV and scores an eval/test window
CSV with the signed nearest-mean margin s(w,d)=|m-Lo|-|m-H|, evaluated against
the relative match label ctype==dmr_ctype. Returns AUROC, accuracy, balanced acc."""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix)

def window_methylation(s):
    if not isinstance(s, str): return np.nan
    n1, n0 = s.count("1"), s.count("0"); tot = n0 + n1
    return (n1 / tot) if tot > 0 else np.nan

def load(path):
    head = pd.read_csv(path, sep="\t", nrows=0)
    ctype_col = "ctype" if "ctype" in head.columns else ("sample_ctype" if "sample_ctype" in head.columns else None)
    if ctype_col is None:
        raise KeyError(f"{path}: no 'ctype'/'sample_ctype'; cols={list(head.columns)}")
    use = ["methyl_seq", ctype_col, "dmr_ctype", "dmr_label"]
    df = pd.read_csv(path, sep="\t", usecols=use,
                     dtype={"methyl_seq": str, ctype_col: str, "dmr_ctype": str, "dmr_label": int})
    df = df.rename(columns={ctype_col: "ctype"})
    df["m"] = df["methyl_seq"].map(window_methylation)
    return df.drop(columns="methyl_seq")

def compute_baseline(train_path, eval_path, verbose=True):
    train, ev = load(train_path), load(eval_path)
    n_tr_nan, n_ev_nan = int(train["m"].isna().sum()), int(ev["m"].isna().sum())
    train, ev = train.dropna(subset=["m"]), ev.dropna(subset=["m"])
    prof = train.pivot_table(index="dmr_label", columns="ctype", values="m", aggfunc="mean")
    prof_n = train.pivot_table(index="dmr_label", columns="ctype", values="m", aggfunc="size")
    for c in ("T", "N"):
        if c not in prof.columns: prof[c] = np.nan; prof_n[c] = 0
    ev = ev.join(prof.rename(columns={"T": "profT", "N": "profN"}), on="dmr_label")
    is_T = ev["dmr_ctype"].values == "T"
    H  = np.where(is_T, ev["profT"].values, ev["profN"].values)
    Lo = np.where(is_T, ev["profN"].values, ev["profT"].values)
    m  = ev["m"].values
    valid = ~np.isnan(H) & ~np.isnan(Lo)
    score = (np.abs(m - Lo) - np.abs(m - H))[valid]
    label = ((ev["ctype"].values == ev["dmr_ctype"].values).astype(int))[valid]
    pred = (score > 0).astype(int)
    res = {"auroc": roc_auc_score(label, score),
           "accuracy": accuracy_score(label, pred),
           "balanced_accuracy": balanced_accuracy_score(label, pred),
           "n_eval": int(valid.sum()), "n_skipped": int((~valid).sum()),
           "n_dmrs": int(prof.shape[0]), "pos_rate": float(label.mean())}
    if verbose:
        both = (prof_n.get("T", 0) > 0) & (prof_n.get("N", 0) > 0)
        print(f"DMRs with train reads: {res['n_dmrs']} | both T&N: {int(both.sum())}")
        print(f"windows dropped (no confident calls): train={n_tr_nan}, eval={n_ev_nan}")
        print(f"eval scored: {res['n_eval']} | skipped (missing profile): {res['n_skipped']}")
        print(f"match-label balance: {100*res['pos_rate']:.1f}% positive")
        print(f"AUROC {res['auroc']:.4f} | acc {res['accuracy']:.4f} | bal.acc {res['balanced_accuracy']:.4f}")
        print("confusion:\n", confusion_matrix(label, pred))
    return res

def main():
    ap = argparse.ArgumentParser(description="Two-means methylation baseline.")
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--eval-path", required=True)
    a = ap.parse_args()
    compute_baseline(os.path.expanduser(a.train_path), os.path.expanduser(a.eval_path), verbose=True)

if __name__ == "__main__":
    main()