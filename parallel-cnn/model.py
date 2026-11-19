"""Shared model and data helpers for the CIFAR-10 training scripts."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class SimpleCNN(nn.Module):
    """Four conv blocks followed by three fully connected layers (32x32 RGB -> 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = F.max_pool2d(F.relu(conv(x)), 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def cifar10(data_dir: str, train: bool = True, download: bool = True):
    """Return the CIFAR-10 split stored in (or downloaded to) data_dir."""
    return datasets.CIFAR10(root=data_dir, train=train, download=download, transform=TRANSFORM)
