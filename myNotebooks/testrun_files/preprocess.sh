#!/bin/bash
#SBATCH --job-name=5base_preprocess
#SBATCH --partition=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bauerste/5BaseTestrun/logs/preprocess_%j.out
#SBATCH --error=/home/bauerste/5BaseTestrun/logs/preprocess_%j.err

set -euo pipefail

# Ensure log directory exists
mkdir -p /home/bauerste/5BaseTestrun/logs

# Activate Python environment
source /home/bauerste/venvs/5basemethylbert/bin/activate

# Make methylbert importable
export PYTHONPATH=/home/bauerste/5BaseTestrun/methylbert/src:${PYTHONPATH:-}

echo "Job started on $(hostname) at $(date)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

python - <<'PYEOF'
from methylbert.data.vocab import MethylVocab
import methylbert.data.genome as genome

vocab = MethylVocab(k=3)

genome.pretrain_data_preprocess_5base_binary(
    f_ref="/home/bauerste/5BaseTestrun/hg38/GCF_000001405.26_GRCh38_genomic.fa",
    sc_dataset="/home/bauerste/5BaseTestrun/bams.txt",
    vocab=vocab,
    output_dir="/home/bauerste/5BaseTestrun/pretrain_data_bin",
    seq_len=150,
    n_cores=6,               # 6 BAMs, 6 workers; leaves 2 cores for I/O overhead
)
PYEOF

echo "Job finished at $(date)"