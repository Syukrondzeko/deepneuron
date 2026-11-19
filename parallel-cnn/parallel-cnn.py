"""Multi-GPU CIFAR-10 training with PyTorch DistributedDataParallel (one process per GPU).

Based on the PyTorch DDP tutorial series:
https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import SimpleCNN, cifar10


def ddp_setup(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # DistributedSampler does the shuffling
        sampler=DistributedSampler(dataset),
    )


class Trainer:
    def __init__(self, model, train_data: DataLoader, optimizer, gpu_id: int, save_every: int, ckpt_path: str):
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.ckpt_path = ckpt_path
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])

    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.model(source), targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int):
        self.train_data.sampler.set_epoch(epoch)
        total = 0.0
        for source, targets in self.train_data:
            total += self._run_batch(source.to(self.gpu_id), targets.to(self.gpu_id))

        # Average the epoch loss across all GPUs so rank 0 can report a global number.
        loss = torch.tensor(total / len(self.train_data), device=self.gpu_id)
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        if self.gpu_id == 0:
            print(f"Epoch {epoch + 1} | Steps/GPU: {len(self.train_data)} | Loss: {loss.item():.4f}")

    def _save_checkpoint(self, epoch: int):
        torch.save(self.model.module.state_dict(), self.ckpt_path)
        print(f"Epoch {epoch + 1} | Training checkpoint saved at {self.ckpt_path}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            is_save_epoch = (epoch + 1) % self.save_every == 0 or epoch + 1 == max_epochs
            if self.gpu_id == 0 and is_save_epoch:
                self._save_checkpoint(epoch)


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    try:
        dataset = cifar10(args.data_dir, download=False)  # downloaded once by the parent process
        model = SimpleCNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        trainer = Trainer(model, prepare_dataloader(dataset, args.batch_size), optimizer, rank,
                          args.save_every, args.ckpt_path)
        trainer.train(args.total_epochs)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed (multi-GPU) CIFAR-10 training")
    parser.add_argument("total_epochs", type=int, help="Total epochs to train the model")
    parser.add_argument("save_every", type=int, help="Save a checkpoint every N epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU (default: 32)")
    parser.add_argument("--lr", default=1e-3, type=float, help="SGD learning rate (default: 1e-3)")
    parser.add_argument("--data-dir", default="./data", help="Where CIFAR-10 is stored/downloaded")
    parser.add_argument("--ckpt-path", default="./checkpoints/parallel-cnn.pt", help="Checkpoint file path")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise SystemExit("No CUDA devices found; DDP training with NCCL needs at least one GPU.")

    os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
    # Download once here so the GPU workers don't race to write the same files.
    cifar10(args.data_dir, download=True)

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
