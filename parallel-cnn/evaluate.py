"""Evaluate a SimpleCNN checkpoint (from cnn.py or parallel-cnn.py) on the CIFAR-10 test set."""

import argparse

import torch
from torch.utils.data import DataLoader

from model import CLASSES, SimpleCNN, cifar10


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a SimpleCNN state_dict (.pt/.pth)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    loader = DataLoader(cifar10(args.data_dir, train=False), batch_size=args.batch_size, shuffle=False)

    correct = total = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        if i == 0:
            print("Sample predictions:", " ".join(CLASSES[p] for p in predicted[:10].tolist()))
            print("Ground truth      :", " ".join(CLASSES[t] for t in labels[:10].tolist()))

    print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
