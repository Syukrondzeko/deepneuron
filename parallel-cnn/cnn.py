"""Single-device CIFAR-10 training (baseline for the DDP comparison)."""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimpleCNN, cifar10


def train(model, train_loader, optimizer, epochs, save_every, ckpt_path, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved at {ckpt_path}")

    print("Finished Training")


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 on a single device")
    parser.add_argument("epochs", type=int, help="Number of epochs to train")
    parser.add_argument("save_every", type=int, help="Save the model every N epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument("--lr", default=1e-3, type=float, help="SGD learning rate (default: 1e-3)")
    parser.add_argument("--data-dir", default="./data", help="Where CIFAR-10 is stored/downloaded")
    parser.add_argument("--ckpt-path", default="./checkpoints/cnn.pth", help="Checkpoint file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_loader = DataLoader(cifar10(args.data_dir), batch_size=args.batch_size, shuffle=True)
    train(model, train_loader, optimizer, args.epochs, args.save_every, args.ckpt_path, device)


if __name__ == "__main__":
    main()
