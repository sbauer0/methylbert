#!/usr/bin/env python3
"""
Fine-tune the mixed-areaStat dmr1000 models (seed 42) for BOTH pretraining
variants and dump their run-2 test predictions:

    with_methylation   (methylation-aware pretraining)
    wo_methylation     (ablation)

Same fine-tune data feeds both models -- the ablation lives in the PRETRAINED
checkpoint, not in the fine-tune data. Hyperparameters are unchanged from the
earlier 1000-DMR runs (300-step schedule).

For each variant, training and prediction run in separate subprocesses so the
GPU allocation from training is fully released before inference.

Run:  nohup python train_mixed_dmr.py > train_mixed_dmr.log 2>&1 & ; tail -f train_mixed_dmr.log
Dry:  python train_mixed_dmr.py --dry-run       # resolve + check paths, do nothing
"""
import os, sys, argparse, subprocess

DMR_TOTAL = 1000
SEED      = 42

# ---- shared fine-tune data (identical for both variants) ----
DATA_DIR   = f"/home/bauerste/finetuneDatasetsMixedDMR/dmr{DMR_TOTAL}"
TRAIN_CSV  = os.path.join(DATA_DIR, "train_seq.csv")
EVAL_CSV   = os.path.join(DATA_DIR, "test_seq.csv")   # run-1 15% eval split (NOT run-2)
TEST_CSV   = f"/home/bauerste/finetuneTestdataMixedDMR/dmr{DMR_TOTAL}_test/data.csv"  # run-2 held-out

# ---- outputs (mixed-DMR experiment dirs) ----
MODEL_BASE = "/home/bauerste/methylbert_finetune/finetuned_models_mixed_dmr"
PRED_DIR   = "/home/bauerste/methylbert_finetune/predictions_mixed_dmr"

# ---- pretrained checkpoints, one per variant ----
VARIANTS = {
    "with_methylation": "/home/bauerste/methylbert_pretrain_with_methylation/"
                        "pretrained_model_120k_512_w_methylation/step_120000",
    "wo_methylation":   "/home/bauerste/methylbert_pretrain_without_methylation/"
                        "pretrained_model_120k_512_wo_methylation/step_120000",
}

def out_dir(variant):  return os.path.join(MODEL_BASE, f"finetune_out_dmr{DMR_TOTAL}_s{SEED}_{variant}")
def pred_csv(variant): return os.path.join(PRED_DIR,   f"preds_dmr{DMR_TOTAL}_s{SEED}_{variant}_test.csv")

# ---- training hyperparams (300-step schedule; unchanged) ----
SEQ_LEN, N_MERS                = 511, 3
BATCH, GRAD_ACCUM, NUM_WORKERS = 32, 2, 8
LR, WARMUP, DECREASE_STEPS     = 1e-4, 30, 100
EVAL_FREQ, STEPS, LOSS         = 15, 300, "bce"


def train(variant):
    from torch.utils.data import DataLoader
    from methylbert.data.vocab import MethylVocab
    from methylbert.data.dataset import MethylBertFinetuneDataset
    from methylbert.utils import set_seed
    from methylbert.trainer import MethylBertFinetuneTrainer

    pretrained, output = VARIANTS[variant], out_dir(variant)
    print(f"[train] dmr{DMR_TOTAL} seed={SEED} variant={variant}")
    print(f"  pretrained: {pretrained}")
    print(f"  output:     {output}", flush=True)

    for p, what in ((pretrained, "pretrained dir"), (TRAIN_CSV, "train csv"), (EVAL_CSV, "eval csv")):
        if not os.path.exists(p):
            sys.exit(f"  MISSING {what}: {p}")

    if os.path.exists(os.path.join(output, "config.json")):
        print("  SKIP: model already exists (config.json present). Delete dir to retrain.", flush=True)
        return

    set_seed(SEED)
    tokenizer = MethylVocab(N_MERS)
    train_dataset = MethylBertFinetuneDataset(TRAIN_CSV, tokenizer, seq_len=SEQ_LEN)
    eval_dataset  = MethylBertFinetuneDataset(EVAL_CSV,  tokenizer, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=False)
    print(f"  train rows: {len(train_dataset)} | num_dmrs: {train_dataset.num_dmrs()}", flush=True)

    os.makedirs(output, exist_ok=True)
    trainer = MethylBertFinetuneTrainer(
        len(tokenizer), save_path=output,
        train_dataloader=train_loader, test_dataloader=eval_loader,
        lr=LR, with_cuda=True, log_freq=10, eval_freq=EVAL_FREQ,
        warmup_step=WARMUP, decrease_steps=DECREASE_STEPS,
        gradient_accumulation_steps=GRAD_ACCUM, loss=LOSS,
    )
    trainer.load(pretrained)
    trainer.train(steps=STEPS)
    print(f"  DONE training: {output}", flush=True)


def dump(variant):
    # dump_predictions.py lives beside this script / in the scripts dir
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dump_predictions import dump_predictions

    output, pcsv = out_dir(variant), pred_csv(variant)
    print(f"\n[dump] {variant} test predictions -> {pcsv}", flush=True)

    if not os.path.exists(os.path.join(output, "config.json")):
        sys.exit(f"  cannot dump: no trained model at {output}")
    if not os.path.exists(TEST_CSV):
        sys.exit(f"  MISSING test data: {TEST_CSV}")
    if os.path.exists(pcsv):
        print(f"  SKIP: predictions already exist: {pcsv}", flush=True)
        return

    os.makedirs(PRED_DIR, exist_ok=True)
    df = dump_predictions(output, TEST_CSV)
    df.to_csv(pcsv, index=False)
    n = len(df); pos = int(df["y_oriented"].sum())
    print(f"  wrote {n} rows -> {pcsv}  (y_oriented +{pos}/{n} = {100*pos/n:.1f}%)", flush=True)
    print(df.head(3).to_string(index=False), flush=True)


def dry_run():
    print("=== dry run: resolved paths (mixed-DMR dmr1000, seed 42) ===")
    def status(p): return "OK " if os.path.exists(p) else "MISSING"
    print("shared data:")
    for label, p in (("train", TRAIN_CSV), ("eval", EVAL_CSV), ("test", TEST_CSV)):
        print(f"  [{status(p)}] {label:5s} {p}")
    for v in VARIANTS:
        print(f"variant {v}:")
        print(f"  [{status(VARIANTS[v])}] pretrained {VARIANTS[v]}")
        print(f"  [{'exists' if os.path.exists(os.path.join(out_dir(v),'config.json')) else 'to-train'}] model      {out_dir(v)}")
        print(f"  [{'exists' if os.path.exists(pred_csv(v)) else 'to-dump ' }] preds      {pred_csv(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "dump"], default=None,
                    help="internal: run a single phase for one variant in its own process")
    ap.add_argument("--variant", choices=list(VARIANTS), default=None)
    ap.add_argument("--dry-run", action="store_true", help="resolve + check paths, do nothing")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(); return
    if args.phase == "train":
        train(args.variant); return
    if args.phase == "dump":
        dump(args.variant);  return

    # orchestrator: per variant, train then dump, each in its own subprocess
    script = os.path.abspath(__file__)
    failures = []
    for v in VARIANTS:
        print(f"\n########## variant: {v} ##########", flush=True)
        print("=== phase 1: train ===", flush=True)
        if subprocess.run([sys.executable, script, "--phase", "train", "--variant", v]).returncode != 0:
            print(f"  training failed for {v}; skipping its dump.", flush=True)
            failures.append(f"{v}:train"); continue
        print("\n=== phase 2: dump test predictions ===", flush=True)
        if subprocess.run([sys.executable, script, "--phase", "dump", "--variant", v]).returncode != 0:
            failures.append(f"{v}:dump")

    print("\n=== SUMMARY ===", flush=True)
    if failures:
        print("FAILED:", ", ".join(failures), flush=True)
        sys.exit(1)
    print("both variants trained and predictions dumped.", flush=True)


if __name__ == "__main__":
    main()