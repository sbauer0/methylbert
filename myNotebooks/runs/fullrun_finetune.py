#!/usr/bin/env python3
"""
Fine-tune all 6 models on the regenerated (NMS'd) dmr100 data.
3 seeds x 2 variants (methylation-aware / ablation). Each run in its own
subprocess so the GPU is fully freed between runs and failures don't cascade.

Run detached:
  nohup python finetune_dmr100_all6.py > ft_dmr100.log 2>&1 & ; tail -f ft_dmr100.log
"""
import os, sys, argparse, subprocess

DATA_DIR  = "/tmp/bauerste/finetuneDatasets/dmr100/"
TRAIN_CSV = os.path.join(DATA_DIR, "train_seq.csv")
EVAL_CSV  = os.path.join(DATA_DIR, "test_seq.csv")   # 15% eval split (NOT run-2 test)
OUT_BASE  = "/home/bauerste/methylbert_finetune"

PRETRAINED_W  = "/home/bauerste/methylbert_pretrain_with_methylation/pretrained_model_120k_512_w_methylation/step_120000"
PRETRAINED_WO = "/home/bauerste/methylbert_pretrain_without_methylation/pretrained_model_120k_512_wo_methylation/step_120000"

SEQ_LEN, N_MERS              = 511, 3
BATCH, GRAD_ACCUM, NUM_WORKERS = 32, 2, 8
LR, WARMUP, DECREASE_STEPS   = 1e-4, 30, 100
EVAL_FREQ, STEPS, LOSS       = 15, 300, "bce"

# (pretrained_path, variant_tag, seed)
RUNS = []
for seed in (42, 67, 123):
    RUNS.append((PRETRAINED_W,  "w_methylation",  seed))
    RUNS.append((PRETRAINED_WO, "wo_methylation", seed))


def output_dir(seed, tag):
    return os.path.join(OUT_BASE, f"finetune_out_s{seed}_{tag}/")


def do_run(idx):
    from torch.utils.data import DataLoader
    from methylbert.data.vocab import MethylVocab
    from methylbert.data.dataset import MethylBertFinetuneDataset
    from methylbert.utils import set_seed
    from methylbert.trainer import MethylBertFinetuneTrainer

    pretrained, tag, seed = RUNS[idx]
    output = output_dir(seed, tag)
    print(f"[run {idx+1}/{len(RUNS)}] seed={seed} variant={tag}")
    print(f"  pretrained: {pretrained}")
    print(f"  output:     {output}", flush=True)

    if os.path.exists(os.path.join(output, "config.json")):
        print("  SKIP: a model already exists here. Delete the dir to re-run.", flush=True)
        return

    set_seed(seed)
    tokenizer = MethylVocab(N_MERS)
    train_dataset = MethylBertFinetuneDataset(TRAIN_CSV, tokenizer, seq_len=SEQ_LEN)
    eval_dataset  = MethylBertFinetuneDataset(EVAL_CSV,  tokenizer, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH, num_workers=NUM_WORKERS,
                              pin_memory=True, shuffle=False)

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
        print(f"\n{'='*72}\n### launching run {i+1}/{len(RUNS)} ###\n{'='*72}", flush=True)
        r = subprocess.run([sys.executable, script, "--run-index", str(i)])
        if r.returncode != 0:
            print(f"### run {i+1} FAILED (exit {r.returncode}) — continuing ###", flush=True)
    print("\nALL RUNS DISPATCHED")


if __name__ == "__main__":
    main()