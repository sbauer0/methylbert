#!/usr/bin/env python3
"""
Repair run: train the one missing model (dmr1000, seed 42, wo_methylation)
and dump its run-2 test predictions. Everything else is already done.

The wo_methylation run previously crashed on a bad PRETRAINED_WO path
(missing underscore), leaving an empty output dir. This fixes that and the
/tmp-vs-/home data paths, trains, then scores the held-out test split.

Run:  nohup python repair_dmr1000_wo.py > repair_dmr1000_wo.log 2>&1 & ; tail -f repair_dmr1000_wo.log
"""
import os, sys, gc, argparse, subprocess

# ---- target ----
DMR_TOTAL = 1000
SEED      = 42
VARIANT   = "wo_methylation"

# ---- paths (corrected for this cluster session; data kept in /home) ----
DATA_DIR   = f"/home/bauerste/finetuneDatasets/dmr{DMR_TOTAL}"
TRAIN_CSV  = os.path.join(DATA_DIR, "train_seq.csv")
EVAL_CSV   = os.path.join(DATA_DIR, "test_seq.csv")            # run-1 15% eval split (NOT run-2)
TEST_CSV   = f"/home/bauerste/finetuneTestdata/dmr{DMR_TOTAL}_test/data.csv"  # run-2 held-out

MODEL_BASE = "/home/bauerste/methylbert_finetune/finetuned_models"
PRED_DIR   = f"/home/bauerste/methylbert_finetune/predictions/predictions{DMR_TOTAL}"
OUTPUT     = os.path.join(MODEL_BASE, f"finetune_out_dmr{DMR_TOTAL}_s{SEED}_{VARIANT}")
PRED_CSV   = os.path.join(PRED_DIR, f"preds_dmr{DMR_TOTAL}_s{SEED}_{VARIANT}_test.csv")

# fixed: was "methylbert_pretrainwithout_methylation" (missing underscore) -> the crash
PRETRAINED = "/home/bauerste/methylbert_pretrain_without_methylation/pretrained_model_120k_512_wo_methylation/step_120000"

# ---- training hyperparams (300-step schedule, scaled from the 1000-step config) ----
SEQ_LEN, N_MERS                = 511, 3
BATCH, GRAD_ACCUM, NUM_WORKERS = 32, 2, 8
LR, WARMUP, DECREASE_STEPS     = 1e-4, 30, 100
EVAL_FREQ, STEPS, LOSS         = 15, 300, "bce"


def train():
    from torch.utils.data import DataLoader
    from methylbert.data.vocab import MethylVocab
    from methylbert.data.dataset import MethylBertFinetuneDataset
    from methylbert.utils import set_seed
    from methylbert.trainer import MethylBertFinetuneTrainer

    print(f"[train] dmr{DMR_TOTAL} seed={SEED} variant={VARIANT}")
    print(f"  pretrained: {PRETRAINED}")
    print(f"  output:     {OUTPUT}", flush=True)

    for p, what in ((PRETRAINED, "pretrained dir"), (TRAIN_CSV, "train csv"), (EVAL_CSV, "eval csv")):
        if not os.path.exists(p):
            sys.exit(f"  MISSING {what}: {p}")

    if os.path.exists(os.path.join(OUTPUT, "config.json")):
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

    os.makedirs(OUTPUT, exist_ok=True)
    trainer = MethylBertFinetuneTrainer(
        len(tokenizer), save_path=OUTPUT,
        train_dataloader=train_loader, test_dataloader=eval_loader,
        lr=LR, with_cuda=True, log_freq=10, eval_freq=EVAL_FREQ,
        warmup_step=WARMUP, decrease_steps=DECREASE_STEPS,
        gradient_accumulation_steps=GRAD_ACCUM, loss=LOSS,
    )
    trainer.load(PRETRAINED)
    trainer.train(steps=STEPS)
    print(f"  DONE training: {OUTPUT}", flush=True)


def dump():
    from dump_predictions import dump_predictions
    print(f"\n[dump] test predictions -> {PRED_CSV}", flush=True)

    if not os.path.exists(os.path.join(OUTPUT, "config.json")):
        sys.exit(f"  cannot dump: no trained model at {OUTPUT}")
    if not os.path.exists(TEST_CSV):
        sys.exit(f"  MISSING test data: {TEST_CSV}")
    if os.path.exists(PRED_CSV):
        print(f"  SKIP: predictions already exist: {PRED_CSV}", flush=True)
        return

    os.makedirs(PRED_DIR, exist_ok=True)
    df = dump_predictions(OUTPUT, TEST_CSV)
    df.to_csv(PRED_CSV, index=False)
    n = len(df); pos = int(df["y_oriented"].sum())
    print(f"  wrote {n} rows -> {PRED_CSV}  (y_oriented +{pos}/{n} = {100*pos/n:.1f}%)", flush=True)
    print(df.head(3).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "dump"], default=None,
                    help="internal: run a single phase in its own process")
    args = ap.parse_args()

    if args.phase == "train":
        train(); return
    if args.phase == "dump":
        dump();  return

    # Run each phase in its own subprocess so the GPU is fully released between
    # training and inference (avoids holding the training allocation during dump).
    script = os.path.abspath(__file__)
    print("=== phase 1: train ===", flush=True)
    r = subprocess.run([sys.executable, script, "--phase", "train"])
    if r.returncode != 0:
        sys.exit(f"training subprocess failed (exit {r.returncode}); not dumping.")
    print("\n=== phase 2: dump test predictions ===", flush=True)
    r = subprocess.run([sys.executable, script, "--phase", "dump"])
    if r.returncode != 0:
        sys.exit(f"dump subprocess failed (exit {r.returncode}).")
    print("\n=== REPAIR COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()