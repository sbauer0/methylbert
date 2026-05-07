import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from methylbert.data.vocab import MethylVocab
from methylbert.data.dataset import MethylBertPretrainDatasetBinary
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from methylbert import trainer as tr

# DDP setup. torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
print(f"PRE-INIT: LOCAL_RANK env={os.environ.get('LOCAL_RANK')}, "
      f"local_rank={local_rank}, "
      f"current_device={torch.cuda.current_device()}, "
      f"device_count={torch.cuda.device_count()}", flush=True)

dist.init_process_group(
    backend="nccl",
    device_id=torch.device(f"cuda:{local_rank}"),  # PyTorch 2.4+ explicit binding
)
world_size = dist.get_world_size()
rank       = dist.get_rank()
is_master  = (rank == 0)

print(f"POST-INIT: rank={rank}, local_rank={local_rank}, "
      f"current_device={torch.cuda.current_device()}", flush=True)

def log(msg):
    if is_master:
        print(msg, flush=True)

# Make sure non-master ranks don't spam stdout from inside the trainer.
# Errors still flow through stderr.
if not is_master:
    import sys
    sys.stdout = open(os.devnull, "w")

vocab = MethylVocab(k=3)

train_dataset = MethylBertPretrainDatasetBinary(
    data_dir="/tmp/pretrain_shards_4state_v1_balanced/train",
    vocab=vocab,
    seq_len=510,
)
test_dataset = MethylBertPretrainDatasetBinary(
    data_dir="/tmp/pretrain_shards_4state_v1_balanced/test",
    vocab=vocab,
    seq_len=510,
)

print(f"Train dataset size: {len(train_dataset):,}")
print(f"Test dataset size:  {len(test_dataset):,}")

train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42)
test_sampler  = DistributedSampler(test_dataset,  shuffle=False)

train_loader = DataLoader(
    train_dataset, batch_size=128, sampler=train_sampler,
    num_workers=4, pin_memory=True,
    persistent_workers=True, drop_last=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=128, sampler=test_sampler,
    num_workers=4, pin_memory=True,
    persistent_workers=True, drop_last=False,
)

trainer = tr.MethylBertPretrainTrainer(
    vocab_size=len(vocab),
    save_path="/home/bauerste/Methylbert_methylation_encoding/pretrained_model_2gpu",
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    lr=4e-4,
    warmup_step=10000,
    decrease_steps=180000,        # extended from 100k to fit a ~200k step budget
    eval_freq=1000,
    log_freq=100,
    save_freq=5000,               # lowered from 10k for safer checkpointing
    amp=True,
    gradient_accumulation_steps=4,
)

trainer.create_model(type_vocab_size=4, num_hidden_layers=6)
trainer.model = trainer.model.to(f"cuda:{local_rank}")
# Wrap the freshly-created model in DDP. The trainer holds a reference to
# self.model; we replace it with the DDP-wrapped version. Internal forward
# calls delegate transparently.
if world_size > 1:
    trainer.model = DDP(trainer.model, device_ids=[local_rank])

trainer.train(steps=100)   # smoke test; bump to ~200000 for real run

dist.destroy_process_group()