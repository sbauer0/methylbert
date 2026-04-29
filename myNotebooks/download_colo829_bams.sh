#!/bin/bash
#SBATCH --job-name=download_colo829_bams
#SBATCH --partition=cpu_normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=download_colo829_bams_%j.out
#SBATCH --error=download_colo829_bams_%j.err

set -euo pipefail

source ~/.bashrc
conda activate r_env

TARGET="/data/gidb/shared/datasets/MethylBERT/tmp"
BUCKET="s3://ont-open-data/colo829_2024.03"

mkdir -p "${TARGET}/colo829_2024.03/basecalls/colo829/sup"
mkdir -p "${TARGET}/colo829_2024.03/basecalls/colo829bl/sup"

echo "Downloading COLO829 tumour SUP BAMs..."
aws s3 sync \
  "${BUCKET}/basecalls/colo829/sup/" \
  "${TARGET}/colo829_2024.03/basecalls/colo829/sup/" \
  --exclude "*" \
  --include "*.bam" \
  --include "*.bam.bai" \
  --no-sign-request

echo "Downloading COLO829BL normal SUP BAMs..."
aws s3 sync \
  "${BUCKET}/basecalls/colo829bl/sup/" \
  "${TARGET}/colo829_2024.03/basecalls/colo829bl/sup/" \
  --exclude "*" \
  --include "*.bam" \
  --include "*.bam.bai" \
  --no-sign-request

echo "Download finished."
echo "Files:"
find "${TARGET}/colo829_2024.03/basecalls" -type f \( -name "*.bam" -o -name "*.bam.bai" \) -lh