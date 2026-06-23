#!/usr/bin/env python3
"""
Dump per-read predictions for the dmr1000 models (seed 42, w/wo methylation)
x 3 splits (train/eval/test). Uses dump_predictions.dump_predictions.

Output: /workspace/predictions1000/preds_dmr1000_s42_{variant}_{split}.csv  (6 files)
Skips existing outputs (resumable on the hourly box).

Run:  python dump_all_dmr1000.py
"""
import os
import traceback
from dump_predictions import dump_predictions

DMR_TOTAL  = 1000
MODEL_BASE = "/workspace/methylbert_finetune"
OUT_DIR    = f"/workspace/predictions{DMR_TOTAL}"

SPLITS = {
    "train": f"/workspace/finetuneDatasets/dmr{DMR_TOTAL}/train_seq.csv",      # run-1 train
    "eval":  f"/workspace/finetuneDatasets/dmr{DMR_TOTAL}/test_seq.csv",       # run-1 15% eval
    "test":  f"/workspace/finetuneTestdata/dmr{DMR_TOTAL}_test/data.csv",      # run-2 held-out
}

VARIANTS = ("w_methylation", "wo_methylation")
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

done, skipped, failed = 0, 0, 0
for variant in VARIANTS:
    model_path = os.path.join(MODEL_BASE, f"finetune_out_dmr{DMR_TOTAL}_s{SEED}_{variant}")
    for split, data_path in SPLITS.items():
        out_csv = os.path.join(OUT_DIR, f"preds_dmr{DMR_TOTAL}_s{SEED}_{variant}_{split}.csv")
        tag = f"dmr{DMR_TOTAL}_s{SEED}_{variant}/{split}"

        if os.path.exists(out_csv):
            print(f"SKIP (exists): {tag}", flush=True)
            skipped += 1
            continue

        print(f"\n=== {tag} ===\n  model: {model_path}\n  data:  {data_path}", flush=True)
        try:
            df = dump_predictions(model_path, data_path)
            df.to_csv(out_csv, index=False)
            n = len(df); pos = int(df["y_oriented"].sum())
            print(f"  wrote {n} rows -> {out_csv}  (y_oriented +{pos}/{n} = {100*pos/n:.1f}%)", flush=True)
            print(df.head(3).to_string(index=False), flush=True)
            done += 1
        except Exception:
            print(f"  FAILED: {tag}", flush=True)
            traceback.print_exc()
            failed += 1

print(f"\nDONE. wrote={done} skipped={skipped} failed={failed} -> {OUT_DIR}", flush=True)