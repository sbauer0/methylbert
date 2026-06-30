#!/usr/bin/env python3
"""
Generate train/eval (run-1, PAU5*) and test (run-2, PAU6*) data
for DMR counts 100/250/500/1000.

  train/eval -> ~/finetuneDatasets/dmr{N}        (split_ratio=0.85)
  test       -> ~/finetuneTestdata/dmr{N}_test   (split_ratio=None)

Writes the sc_dataset TSVs automatically. Run detached:
  nohup python generate_all_data.py > gen.log 2>&1 & ; tail -f gen.log
"""
import os
from functools import partial
from methylbert.data.finetune_data_generate import finetune_data_generate
from methylbert.data.nanopore.finetune_extract import ont_read_extract

HOME    = "/home/bauerste"
DMRS    = f"{HOME}/methylbertDMRs/dmrs_filtered.tsv"
REF     = f"{HOME}/GRCh38_no_alt_analysis_set/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
TRAINDIR = f"{HOME}/finetuneDatasets"
TESTDIR  = f"{HOME}/finetuneTestdata"


# run-1 = PAU5* (train/eval), run-2 = PAU6* (test)
RUN1 = [("/tmp/bauerste/COLO829_BL/colo829/PAU59949.d052sup4305mCG_5hmCGvHg38_pass.bam", "T"),
        ("/tmp/bauerste/COLO829_BL/colo829bl/PAU59807.d052sup4305mCG_5hmCGvHg38_pass.bam", "N")]
RUN2 = [("/tmp/bauerste/COLO829_BL/colo829/PAU61426.d052sup4305mCG_5hmCGvHg38_pass.bam", "T"),
        ("/tmp/bauerste/COLO829_BL/colo829bl/PAU61427.d052sup4305mCG_5hmCGvHg38_pass.bam", "N")]

os.makedirs(TRAINDIR, exist_ok=True)
os.makedirs(TESTDIR, exist_ok=True)

def write_sc(path, bams):
    with open(path, "w") as f:
        for bam, label in bams:
            f.write(f"{bam}\t{label}\n")
    return path

SC_RUN1 = write_sc(f"{TRAINDIR}/run1_bams.tsv", RUN1)
SC_RUN2 = write_sc(f"{TESTDIR}/run2_bams.tsv",  RUN2)

# (total_dmrs, tag, output_dir, sc_dataset, split_ratio); n_dmrs = total//2
PASSES = []
for total in (100, 250, 500, 1000):
    PASSES.append((total, "run1", f"{TRAINDIR}/dmr{total}",      SC_RUN1, 0.85))
    PASSES.append((total, "test", f"{TESTDIR}/dmr{total}_test",  SC_RUN2, None))

for total, tag, out, sc, split in PASSES:
    n_dmrs = total // 2
    print(f"\n=== total={total} ({tag}) n_dmrs={n_dmrs} split={split} -> {out} ===", flush=True)
    finetune_data_generate(
        f_dmr=DMRS,
        output_dir=out,
        f_ref=REF,
        sc_dataset=sc,
        n_mers=3,
        n_dmrs=n_dmrs,
        split_ratio=split,
        ignore_sex_chromo=False,
        methyl_caller="dorado",
        read_extract_sequences_func=partial(ont_read_extract, mm_flag="?"),
    )
    print(f"=== done {out} ===", flush=True)

print("\nALL PASSES COMPLETE")
