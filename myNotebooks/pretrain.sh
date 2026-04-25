#!/bin/bash
#SBATCH --job-name=5base_pretrain
#SBATCH --partition=rigs_stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/home/bauerste/5BaseTestrun/logs/pretrain_%j.out
#SBATCH --error=/home/bauerste/5BaseTestrun/logs/pretrain_%j.err

set -euo pipefail

mkdir -p /home/bauerste/5BaseTestrun/logs

# Stage data to node-local scratch
SCRATCH_DIR=/tmp/bauerste_$SLURM_JOB_ID/pretrain_data_bin
mkdir -p $SCRATCH_DIR

echo "Staging data to local scratch at $(date)..."
cp /home/bauerste/5BaseTestrun/pretrain_data_bin/* $SCRATCH_DIR/
echo "Staging complete at $(date). Size: $(du -sh $SCRATCH_DIR)"

# Cleanup on exit (success or failure)
trap "echo 'Cleaning up scratch...'; rm -rf /tmp/bauerste_$SLURM_JOB_ID" EXIT

# Activate environment
source /home/bauerste/venvs/5basemethylbert/bin/activate
export PYTHONPATH=/home/bauerste/5BaseTestrun/methylbert/src:${PYTHONPATH:-}

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Run training, pointing dataset at staged location
python - <<PYEOF
import os
from torch.utils.data import DataLoader, random_split
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertPretrainDatasetBinary
from methylbert.trainer import MethylBertPretrainTrainer

vocab = MethylVocab(k=3)

dataset = MethylBertPretrainDatasetBinary(
    data_dir="$SCRATCH_DIR",
    vocab=vocab,
    seq_len=150
)
print(f"Dataset size: {len(dataset):,}")

train_size = int(0.98 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=8, pin_memory=True,
                          persistent_workers=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                         num_workers=2, pin_memory=True,
                         persistent_workers=True, drop_last=False)

trainer = MethylBertPretrainTrainer(
    vocab_size=len(vocab),
    save_path="/home/bauerste/5BaseTestrun/pretrained_model",
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    lr=4e-4,
    warmup_step=2000,
    decrease_steps=16000,
    eval_freq=500,
    log_freq=100,
    save_freq=2000,
    amp=True,
    gradient_accumulation_steps=8,
)
trainer.create_model(type_vocab_size=1, num_hidden_layers=6)
trainer.train(steps=20000)
PYEOF

echo "Job finished at $(date)"