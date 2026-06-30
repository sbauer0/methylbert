#!/usr/bin/env python3
"""
Dump per-read predictions for all 6 dmr100 models x 3 splits (train/eval/test).
Uses dump_predictions.dump_predictions (path-in -> DataFrame).

Output: /workspace/predictions/preds_s{seed}_{variant}_{split}.csv   (18 files)
Skips any output that already exists, so an interrupted run on the rented
(hourly-billed) box resumes cheaply.

Run:  python dump_all_dmr100.py
"""
import os
import traceback
from dump_predictions import dump_predictions

MODEL_BASE = "/home/bauerste/methylbert_finetune"
OUT_DIR    = "/home/bauerste/methylbert_finetune/predictions"

# split -> data csv
SPLITS = {
    "train": "/tmp/bauerste/finetuneDatasets/dmr100/train_seq.csv",  # run-1 train
    "eval":  "/tmp/bauerste/finetuneDatasets/dmr100/test_seq.csv",   # run-1 15% eval split
    "test":  "/tmp/bauerste/finetuneDatasets/dmr100_test/data.csv",  # run-2 held-out test
}

MODELS = [
    (seed, variant)
    for seed in (42, 67, 123)
    for variant in ("w_methylation", "wo_methylation")
]

os.makedirs(OUT_DIR, exist_ok=True)

done, skipped, failed = 0, 0, 0
for seed, variant in MODELS:
    model_path = os.path.join(MODEL_BASE, f"finetune_out_s{seed}_{variant}")
    for split, data_path in SPLITS.items():
        out_csv = os.path.join(OUT_DIR, f"preds_s{seed}_{variant}_{split}.csv")
        tag = f"s{seed}_{variant}/{split}"

        if os.path.exists(out_csv):
            print(f"SKIP (exists): {tag}", flush=True)
            skipped += 1
            continue

        print(f"\n=== {tag} ===\n  model: {model_path}\n  data:  {data_path}", flush=True)
        try:
            df = dump_predictions(model_path, data_path)
            df.to_csv(out_csv, index=False)
            # quick eyeball stat so you catch problems live, no GPU needed later
            n = len(df)
            pos = int(df["y_oriented"].sum())
            print(f"  wrote {n} rows -> {out_csv}  (y_oriented +{pos}/{n} = {100*pos/n:.1f}%)", flush=True)
            print(df.head(3).to_string(index=False), flush=True)
            done += 1
        except Exception:
            print(f"  FAILED: {tag}", flush=True)
            traceback.print_exc()
            failed += 1

print(f"\nDONE. wrote={done} skipped={skipped} failed={failed} -> {OUT_DIR}", flush=True)
