#!/usr/bin/env python3
"""
Verbose per-read prediction dump for a fine-tuned MethylBERT model.

Interface:
    dump_predictions(model_path: str, data_path: str) -> pd.DataFrame

Loads the best-eval checkpoint at `model_path`, scores every (read, DMR)
window in `data_path`, and returns one row per window with both the
trained (oriented) label/probability and the interpretable tumour-axis
label/probability. Returns a DataFrame; the caller decides whether to
.to_csv() it. Frees the GPU before returning so a loop over many models
does not accumulate memory.

Columns:methylbert
    read_id       read name
    dmr_label     contiguous DMR id (0..N-1)
    dmr_ctype     DMR characteristic type, 'T' or 'N'
    sample_ctype  read origin sample, 'T' or 'N'
    y_oriented    int(sample_ctype == dmr_ctype)        # the trained label
    p_oriented    P(y_oriented == 1)                     # model output
    y_tumour      int(sample_ctype == 'T')               # tumour-axis label
    p_tumour      p_oriented        if dmr_ctype == 'T'  # tumour-axis prob,
                  1 - p_oriented    if dmr_ctype == 'N'  # un-rotated per DMR
"""
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.trainer import MethylBertFinetuneTrainer

SEQ_LEN = 511
N_MERS = 3
BATCH = 32
NUM_WORKERS = 8


def _softmax_col1(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def dump_predictions(model_path: str, data_path: str) -> pd.DataFrame:
    tokenizer = MethylVocab(N_MERS)

    # The scored dataset. A finetune trainer also needs a "train" dataset only
    # to size the DMR embedding (num_dmrs); the data file itself supplies that,
    # so we use the same file for both — we never train here, only infer.
    dataset = MethylBertFinetuneDataset(data_path, tokenizer, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH, num_workers=NUM_WORKERS,
                        pin_memory=True, shuffle=False)
    n_dmrs = dataset.num_dmrs()

    trainer = MethylBertFinetuneTrainer(
        len(tokenizer), save_path=model_path,
        train_dataloader=loader, test_dataloader=loader,
        with_cuda=True, loss="bce",
    )
    trainer.load(model_path, n_dmrs=n_dmrs, load_fine_tune=True)

    res, logits = trainer.read_classification(loader, tokenizer, logit=True)

    # --- assemble columns ---
    read_id      = np.asarray(res["name"]).ravel()
    dmr_label    = np.asarray(res["dmr_label"]).ravel().astype(int)
    dmr_ctype    = np.asarray(res["dmr_ctype"]).ravel().astype(str)
    sample_ctype = np.asarray(res["ctype"]).ravel().astype(str)

    # fail loud on unexpected ctype values rather than silently mislabel
    bad = set(np.unique(dmr_ctype)) - {"T", "N"}
    if bad:
        raise ValueError(f"unexpected dmr_ctype values: {bad}")
    bad = set(np.unique(sample_ctype)) - {"T", "N"}
    if bad:
        raise ValueError(f"unexpected sample_ctype values: {bad}")

    p_oriented = _softmax_col1(logits)
    y_oriented = (sample_ctype == dmr_ctype).astype(int)

    y_tumour = (sample_ctype == "T").astype(int)
    is_T_dmr = dmr_ctype == "T"
    p_tumour = np.where(is_T_dmr, p_oriented, 1.0 - p_oriented)

    df = pd.DataFrame({
        "read_id":      read_id,
        "dmr_label":    dmr_label,
        "dmr_ctype":    dmr_ctype,
        "sample_ctype": sample_ctype,
        "y_oriented":   y_oriented,
        "p_oriented":   p_oriented,
        "y_tumour":     y_tumour,
        "p_tumour":     p_tumour,
    })

    # sanity: y_oriented must equal the trainer's own ctype_label, if present
    if "ctype_label" in res:
        ref = np.asarray(res["ctype_label"]).ravel().astype(int)
        if not np.array_equal(ref, y_oriented):
            raise ValueError("y_oriented disagrees with trainer ctype_label — "
                             "check ctype/dmr_ctype column alignment")

    # free GPU before returning (so loops over many models don't accumulate)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return df


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Dump per-read predictions for one model.")
    ap.add_argument("--model", required=True, help="fine-tuned checkpoint dir")
    ap.add_argument("--data",  required=True, help="data CSV (train/eval/test)")
    ap.add_argument("--out",   required=True, help="output CSV path")
    a = ap.parse_args()
    df = dump_predictions(a.model, a.data)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")
    print(df.head().to_string(index=False))