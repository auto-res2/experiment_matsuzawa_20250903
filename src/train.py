"""src/train.py
Utility functions related to network construction and quick sanity training
runs used by the experimental scripts.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from .preprocess import DATA  # shared data directory

__all__ = [
    "build_resnet18",
    "sanity_check",
]


def build_resnet18(num_classes: int) -> nn.Module:
    """Return a ResNet-18 with the first pooling layer removed so it can be
    trained on small (≤128×128) inputs without heavy down-sampling."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Make it suitable for small images (CIFAR/Tiny-IN etc.)
    net.conv1.stride = (1, 1)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def _single_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optim_cfg: Dict):
    """Train *one* epoch – helper for the sanity check."""
    opt = torch.optim.SGD(model.parameters(), **optim_cfg, nesterov=True)
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def sanity_check(device: torch.device, cfg: Dict) -> float:
    """Run a very small (≈10 epochs) CIFAR-100 training loop to verify that the
    hardware / CUDA / cuDNN stack is healthy.  The experiment is
    intentionally short; accuracy <90 % usually indicates a mis-configured
    GPU, wrong learning-rate or other environment issue.  A `RuntimeError`
    is raised if the accuracy threshold is not met so that the main script
    can *fail fast* instead of wasting hours on broken runs.

    Returns
    -------
    float
        The final test accuracy in percent.
    """
    train_tf = transforms.Compose(
        [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_tf = transforms.ToTensor()

    train_ds = datasets.CIFAR100(root=str(DATA), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(root=str(DATA), train=False, download=True, transform=test_tf)

    tr_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    te_loader = DataLoader(test_ds, batch_size=256, num_workers=2)

    net = build_resnet18(100).to(device)

    for _ in range(10):
        _single_epoch(net, tr_loader, device, cfg["optim"])

    # ------------------------------------------------ evaluation
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in te_loader:
            pred = net(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"[Sanity] CIFAR-100 single-task accuracy = {acc:.2f} %")

    if acc < 90:
        raise RuntimeError("Sanity-check accuracy below 90 %. Environment may be mis-configured.")
    return acc
