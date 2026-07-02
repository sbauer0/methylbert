#!/bin/bash
#SBATCH --job-name=mixdmr
#SBATCH --partition=gpu_normal
# optional: delete the next line to let the scheduler pick any gpu_normal node
#SBATCH --nodelist=cc2g05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/bauerste/methylbert_finetune/methylbert/myNotebooks/runs_mixed_dmrs/trainAndDump-%j.log

# stdout+stderr are merged into the --output file above (like nohup ... 2>&1).

set -eo pipefail

RUNDIR=/home/bauerste/methylbert_finetune/methylbert/myNotebooks/runs_mixed_dmrs
VENV=/home/bauerste/venvs/5basemethylbert/bin/activate   # assumes ~/venvs; edit if elsewhere

echo "=== job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# --- environment ---
source "$VENV"
echo "python: $(which python)"

# dump_predictions.py lives with the original run scripts; import it in place
# (with any sibling helpers it may use) rather than copying it around.
export PYTHONPATH="/home/bauerste/methylbert_finetune/methylbert/myNotebooks/runs:${PYTHONPATH:-}"

cd "$RUNDIR"

# --- show resolved paths in the log before committing ---
python train_mixed_dmr.py --dry-run

# --- train both variants (with + wo methylation), then dump test predictions ---
# train_mixed_dmr.py runs each train/dump phase in its own subprocess so the GPU
# is released between training and inference; both variants share this one GPU.
set +e
python train_mixed_dmr.py
rc=$?
set -e

echo "=== finished at $(date) (exit ${rc}) ==="
exit ${rc}