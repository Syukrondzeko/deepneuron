# Parallel CNN training on CIFAR-10

Compares single-GPU training with multi-GPU training via PyTorch `DistributedDataParallel` (DDP) and PyTorch Lightning.

| File | What it does |
|---|---|
| [model.py](model.py) | Shared `SimpleCNN` model and CIFAR-10 loader |
| [cnn.py](cnn.py) | Single-device baseline |
| [parallel-cnn.py](parallel-cnn.py) | DDP training, one process per visible GPU |
| [ddp_lightning.py](ddp_lightning.py) | Same idea using PyTorch Lightning (separate, larger CNN) |
| [evaluate.py](evaluate.py) | Test-set accuracy for a `cnn.py` / `parallel-cnn.py` checkpoint |
| [parallel-cnn.ipynb](parallel-cnn.ipynb) | Runs the experiments and records the timing comparison |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Single GPU: 150 epochs, checkpoint every 50
python cnn.py 150 50 --batch_size 64 --data-dir ./data --ckpt-path ./checkpoints/cnn.pth

# All visible GPUs with DDP
python parallel-cnn.py 150 50 --batch_size 64 --data-dir ./data --ckpt-path ./checkpoints/parallel-cnn.pt

# PyTorch Lightning, 2 GPUs
python ddp_lightning.py --epochs 5 --devices 2 --data-dir ./data

# Evaluate a checkpoint
python evaluate.py ./checkpoints/parallel-cnn.pt --data-dir ./data
```

On a shared cluster, point `--data-dir` and `--ckpt-path` at your scratch space.

## Results (2 GPUs, 150 epochs, batch size 64)

| Run | real | user | sys |
|---|---|---|---|
| Single GPU (`cnn.py`) | 26m 15s | 25m 32s | 0m 54s |
| DDP, 2 GPUs (`parallel-cnn.py`) | 20m 07s | 190m 18s | 1m 12s |

DDP cut wall-clock time by ~23%. CPU (user) time is much higher because DDP runs one process per GPU. The single-GPU timing included the first-time dataset download.

Based on the [PyTorch DDP tutorial series](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py).
