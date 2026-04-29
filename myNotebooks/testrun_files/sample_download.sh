#!/bin/bash
#SBATCH --job-name=sra_download
#SBATCH --output=sra_download_%j.log
#SBATCH --error=sra_download_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpu_normal

# ---- Configuration ----
ACCESSIONS=("SRR28305167" "SRR28305166")
WORKDIR="${HOME}/5BaseTestrun/nanoporeData"
CONDA_ENV="r_env"

# ---- Setup ----
set -euo pipefail  # exit on error, undefined var, or failed pipe

# Activate conda environment
# This finds your conda install regardless of where it lives
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Confirm tools are available
prefetch --version
echo "Working directory: ${WORKDIR}"
cd "${WORKDIR}"

# ---- Download ----
for ACC in "${ACCESSIONS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Starting download: ${ACC} at $(date)"
    echo "=================================================="

    prefetch --type all --max-size u --progress "${ACC}"

    echo "Finished ${ACC} at $(date)"
    echo "Contents:"
    ls -lh "${ACC}/" || echo "WARNING: directory ${ACC} not found"
done

# ---- Summary ----
echo ""
echo "=================================================="
echo "All downloads complete at $(date)"
echo "Total disk usage:"
du -sh "${ACCESSIONS[@]}" 2>/dev/null
echo "=================================================="