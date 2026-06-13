#!/usr/bin/env python3
"""
Fine-tune dmr500, seed 42, both variants (w / wo methylation pretrain).
Incremental sweep step 1: train these, eyeball the eval curves, THEN decide
whether 1000 steps holds for dmr500/dmr1000.

Each run in its own subprocess (clean GPU, isolated failures, resumable).
Run:  nohup python finetune_dmr500.py > ft_dmr500.log 2>&1 & ; tail -f ft_dmr500.log
"""
import os, sys, argparse, subprocess

DMR_TOTAL = 500
DATA_DIR  = f"/workspace/finetuneDatasets/dmr{DMR_TOTAL}/"
TRAIN_CSV = os.path.join(DATA_DIR, "train_seq.csv")
EVAL_CSV  = os.path.join(DATA_DIR, "test_seq.csv")   # 15% eval split (NOT run-2)
OUT_BASE  = "/workspace/methylbert_finetune"

PRETRAINED_W  = "/workspace/methylbert_pretrain_with_methylation/pretrained_model_120k_512_w_methylation/step_120000"
PRETRAINED_WO = "/workspace/methylbert_pretrain_without_methylation/pretrained_model_120k_512_wo_methylation/step_120000"

SEED = 42
SEQ_LEN, N_MERS              = 511, 3
BATCH, GRAD_ACCUM, NUM_WORKERS = 32, 2, 8
LR, WARMUP, DECREASE_STEPS   = 1e-4, 100, 300
EVAL_FREQ, STEPS, LOSS       = 25, 1000, "bce"

RUNS = [
    (PRETRAINED_W,  "w_methylation",  SEED),
    (PRETRAINED_WO, "wo_methylation", SEED),
]


def output_dir(seed, tag):
    return os.path.join(OUT_BASE, f"finetune_out_dmr{DMR_TOTAL}_s{seed}_{tag}/")


def do_run(idx):
    from torch.utils.data import DataLoader
    from methylbert.data.vocab import MethylVocab
    from methylbert.data.dataset import MethylBertFinetuneDataset
    from methylbert.utils import set_seed
    from methylbert.trainer import MethylBertFinetuneTrainer

    pretrained, tag, seed = RUNS[idx]
    output = output_dir(seed, tag)
    print(f"[run {idx+1}/{len(RUNS)}] dmr{DMR_TOTAL} seed={seed} variant={tag}")
    print(f"  pretrained: {pretrained}")
    print(f"  output:     {output}", flush=True)

    if os.path.exists(os.path.join(output, "config.json")):
        print("  SKIP: model already exists. Delete dir to re-run.", flush=True)
        return

    set_seed(seed)
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
    print(f"  DONE: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-index", type=int, default=None)
    args = ap.parse_args()
    if args.run_index is not None:
        do_run(args.run_index)
        return
    script = os.path.abspath(__file__)
    for i in range(len(RUNS)):
        print(f"\n{'='*72}\n### launching dmr{DMR_TOTAL} run {i+1}/{len(RUNS)} ###\n{'='*72}", flush=True)
        r = subprocess.run([sys.executable, script, "--run-index", str(i)])
        if r.returncode != 0:
            print(f"### run {i+1} FAILED (exit {r.returncode}) — continuing ###", flush=True)
    print("\nDMR500 RUNS DISPATCHED")


if __name__ == "__main__":
    main()