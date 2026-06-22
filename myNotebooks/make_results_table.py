#!/usr/bin/env python3
import os, sys
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

BASELINE_DIR = os.path.expanduser("~/methylbert_finetune/methylbert/myNotebooks")
sys.path.insert(0, BASELINE_DIR)
from baseline_two_means import compute_baseline

HOME       = os.path.expanduser("~")
PRED_DIR   = os.path.join(HOME, "methylbert_finetune/predictions/predictions100")
BASE_TRAIN = os.path.join(HOME, "finetuneDatasets/dmr100/train_seq.csv")
BASE_TEST  = os.path.join(HOME, "finetuneTestdata/dmr100_test/data.csv")

# (model, seed, best-checkpoint step, best eval loss)  -- read off the fine-tuning figure
# prediction tag is built as preds_s{seed}_{w|wo}_methylation_test.csv
MODELS = [
    ("Methylation-aware", 42,  974, 0.403, "w"),
    ("Methylation-aware", 67,  849, 0.409, "w"),
    ("Methylation-aware", 123, 574, 0.424, "w"),
    ("Ablation",          42,  199, 0.493, "wo"),
    ("Ablation",          67,  749, 0.432, "wo"),
    ("Ablation",          123, 274, 0.492, "wo"),
]

def test_metrics(seed, wtag):
    path = os.path.join(PRED_DIR, f"preds_s{seed}_{wtag}_methylation_test.csv")
    df = pd.read_csv(path)                       # prediction CSVs are comma-separated
    y = df["y_oriented"].astype(int).values
    p = df["p_oriented"].astype(float).values
    pred = (p > 0.5).astype(int)
    return roc_auc_score(y, p), accuracy_score(y, pred), balanced_accuracy_score(y, pred)

rows = []
for model, seed, ckpt, eloss, wtag in MODELS:
    auroc, acc, bacc = test_metrics(seed, wtag)
    rows.append([model, str(seed), str(ckpt), f"{eloss:.3f}",
                 f"{auroc:.4f}", f"{acc:.4f}", f"{bacc:.4f}"])

b = compute_baseline(BASE_TRAIN, BASE_TEST, verbose=False)
rows.append(["Baseline (two-means)", "--", "--", "--",
             f"{b['auroc']:.4f}", f"{b['accuracy']:.4f}", f"{b['balanced_accuracy']:.4f}"])

headers = ["Model", "Seed", "Best ckpt", "Eval loss",
           "Test AUROC", "Test acc", "Test bal.acc"]

# console
table = [headers] + rows
w = [max(len(r[i]) for r in table) for i in range(len(headers))]
for ri, r in enumerate(table):
    print("  ".join(r[i].ljust(w[i]) for i in range(len(headers))))
    if ri == 0:
        print("  ".join("-" * w[i] for i in range(len(headers))))

# LaTeX
print("\n% --- LaTeX (needs \\usepackage{booktabs}) ---")
print(r"\begin{tabular}{llrrrrr}")
print(r"\toprule")
print(" & ".join(h.replace("%", r"\%") for h in headers) + r" \\")
print(r"\midrule")
for r in rows:
    print(" & ".join(r) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")