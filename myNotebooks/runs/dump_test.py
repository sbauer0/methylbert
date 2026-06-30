#!/usr/bin/env python3
"""
Dump TEST-split per-read predictions for all fine-tuned models.

  dmr100  : 3 seeds (42,67,123) x 2 variants  -> 6 test files
  dmr250  : seed 42 x 2 variants              -> 2 test files
  dmr500  : seed 42 x 2 variants              -> 2 test files
  dmr1000 : seed 42 x 2 variants              -> 2 test files

Test data is the run-2 held-out set in finetuneTestdata/dmr{N}_test/data.csv.
Predictions are written ALONGSIDE the existing train/eval files in
predictions/predictions{N}/, using the same filename convention so downstream
readers find a set's three splits together.

Skips outputs that already exist (resumable). Each model is scored in its own
subprocess so the GPU is fully freed between models (no OOM accumulation).

Run:  python dump_test_all.py
"""
import os, sys, argparse, subprocess, traceback

MODEL_BASE = "/home/bauerste/methylbert_finetune/finetuned_models"
PRED_BASE  = "/home/bauerste/methylbert_finetune/predictions"
TESTDATA   = "/tmp/bauerste/finetuneTestdata"   # run-2 held-out test data lives here

# Each job: (dmr_total, seed, variant)
#   dmr100 model dir  = finetune_out_s{seed}_{variant}            (no 'dmr100')
#   dmr{N} model dir  = finetune_out_dmr{N}_s{seed}_{variant}     (N in 250/500/1000)
JOBS = []
for variant in ("w_methylation", "wo_methylation"):
    for seed in (42, 67, 123):
        JOBS.append((100, seed, variant))
    for n in (250, 500, 1000):
        JOBS.append((n, 42, variant))


def model_dir(n, seed, variant):
    if n == 100:
        return os.path.join(MODEL_BASE, f"finetune_out_s{seed}_{variant}")
    return os.path.join(MODEL_BASE, f"finetune_out_dmr{n}_s{seed}_{variant}")


def out_dir(n):
    return os.path.join(PRED_BASE, f"predictions{n}")


def out_csv(n, seed, variant):
    # match the existing train/eval filename convention per set
    if n == 100:
        fname = f"preds_s{seed}_{variant}_test.csv"
    else:
        fname = f"preds_dmr{n}_s{seed}_{variant}_test.csv"
    return os.path.join(out_dir(n), fname)


def test_data(n):
    return os.path.join(TESTDATA, f"dmr{n}_test", "data.csv")


def do_one(n, seed, variant):
    """Score one model on its test split (own process -> GPU freed on exit)."""
    from dump_predictions import dump_predictions
    mdir = model_dir(n, seed, variant)
    data = test_data(n)
    dest = out_csv(n, seed, variant)
    tag  = f"dmr{n}_s{seed}_{variant}/test"

    os.makedirs(out_dir(n), exist_ok=True)
    if os.path.exists(dest):
        print(f"SKIP (exists): {tag}", flush=True)
        return 0
    for p, what in ((mdir, "model dir"), (data, "test data")):
        if not os.path.exists(p):
            print(f"  MISSING {what}: {p}\n  FAILED: {tag}", flush=True)
            return 1

    print(f"\n=== {tag} ===\n  model: {mdir}\n  data:  {data}", flush=True)
    try:
        df = dump_predictions(mdir, data)
        df.to_csv(dest, index=False)
        n_rows = len(df); pos = int(df["y_oriented"].sum())
        print(f"  wrote {n_rows} rows -> {dest}  (y_oriented +{pos}/{n_rows} = {100*pos/n_rows:.1f}%)", flush=True)
        print(df.head(3).to_string(index=False), flush=True)
        return 0
    except Exception:
        print(f"  FAILED: {tag}", flush=True)
        traceback.print_exc()
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-index", type=int, default=None,
                    help="internal: run a single job in this process")
    args = ap.parse_args()

    if args.job_index is not None:
        sys.exit(do_one(*JOBS[args.job_index]))

    script = os.path.abspath(__file__)
    done = skipped = failed = 0
    for i, (n, seed, variant) in enumerate(JOBS):
        dest = out_csv(n, seed, variant)
        if os.path.exists(dest):
            print(f"SKIP (exists): dmr{n}_s{seed}_{variant}/test", flush=True)
            skipped += 1
            continue
        # subprocess per model -> clean GPU between runs
        r = subprocess.run([sys.executable, script, "--job-index", str(i)])
        if r.returncode == 0:
            done += 1
        else:
            failed += 1
    print(f"\nDONE. wrote={done} skipped={skipped} failed={failed} -> {PRED_BASE}/predictions*/", flush=True)


if __name__ == "__main__":
    main()