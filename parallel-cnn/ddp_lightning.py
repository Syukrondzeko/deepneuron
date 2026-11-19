"""Multi-GPU CIFAR-10 training with PyTorch Lightning (DDP strategy). Requires Lightning >= 2.0."""

import argparse
import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import cifar10


class LitCNN(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.layer2(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 training with PyTorch Lightning DDP")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument("--devices", default=2, type=int, help="Number of GPUs")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ckpt-path", default="./checkpoints/lightning-cnn.pt")
    args = parser.parse_args()

    cifar10(args.data_dir, download=True)  # download before the GPU processes start
    loader = DataLoader(cifar10(args.data_dir, download=False), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

    model = LitCNN()
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy="ddp", max_epochs=args.epochs)
    trainer.fit(model, train_dataloaders=loader)

    if trainer.is_global_zero:
        os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.ckpt_path)
        print(f"Model saved at {args.ckpt_path}")


if __name__ == "__main__":
    main()
