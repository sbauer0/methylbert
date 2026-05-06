#!/bin/bash
#SBATCH --job-name=mb_preprocess
#SBATCH --partition=gpu_normal
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err

# Submit with:
#     sbatch slurm_preprocess.sh
#
# The script runs the parallel nanopore preprocessor on the BAMs listed in
# manifest.tsv (in the submission directory). Output shards go to
#     /data/gidb/shared/datasets/MethylBERT/pretrain_shards_4state_v1
# directly — no /tmp staging.
#
# If the job is killed (timeout, node failure, etc.), just resubmit. The
# preprocessor's resumability check skips (BAM, region) pairs whose shards
# are already complete.

set -e

echo "=== Job started: $(date) ==="
echo "Host:           $(hostname)"
echo "Slurm job ID:   $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Mem allocated:  ${SLURM_MEM_PER_NODE:-unset} MB"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo

# ---- Environment ---------------------------------------------------------
# methylbert is pip-installed (`pip install -e .`) inside this venv.
source ~/venvs/5basemethylbert/bin/activate

echo "=== Environment check ==="
which python
python --version
python -c "import methylbert; print('methylbert from:', methylbert.__file__)"
python -c "import pysam; print('pysam version:', pysam.__version__)"
echo

# ---- Inputs/outputs ------------------------------------------------------
MANIFEST="$SLURM_SUBMIT_DIR/manifest.tsv"
OUTPUT_DIR=/data/gidb/shared/datasets/MethylBERT/pretrain_shards_4state_v1

echo "=== Inputs/outputs ==="
echo "Manifest:    $MANIFEST"
echo "Output dir:  $OUTPUT_DIR"
echo
echo "=== Disk space on output filesystem ==="
df -h /data/gidb/shared/datasets/MethylBERT
echo

# ---- Run -----------------------------------------------------------------
echo "=== Starting preprocessing: $(date) ==="

python -m methylbert.data.nanopore.preprocess \
    "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --per-region \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --min-mapq 20 \
    -v

echo
echo "=== Job finished: $(date) ==="
