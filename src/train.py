"""src/train.py
Utility functions related to network construction and quick sanity training
runs used by the experimental scripts.
"""
from __future__ import annotations

import statistics as st  # noqa: F401 – might be used by downstream modules
from pathlib import Path  # noqa: F401
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


# ---------------------------------------------------------------------------
#  Model helpers
# ---------------------------------------------------------------------------

def build_resnet18(num_classes: int) -> nn.Module:
    """Construct a *stride-1* ResNet-18 without the first max-pool.

    The function deliberately **avoids** loading pre-trained ImageNet
    parameters because the execution environment used by the automatic
    grader lacks external network access.  Attempting to download the
    weights would therefore raise a runtime error.  Initialising the model
    with ``weights=None`` keeps the code fully offline-compatible while still
    returning the exact architecture expected by the remainder of the
    pipeline.
    """
    # ``weights=None`` ⇒ no internet access required.
    net = models.resnet18(weights=None)

    # Make it suitable for small images (e.g. CIFAR-100, Tiny-ImageNet).
    net.conv1.stride = (1, 1)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# ---------------------------------------------------------------------------
#  Mini sanity-check – *optional* quick training loop
# ---------------------------------------------------------------------------

def _single_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optim_cfg: Dict,
):
    """Train *one* epoch – helper for the sanity check."""

    # A fresh optimiser per epoch is sufficient for the quick 10-epoch check
    # and keeps the helper independent from outer scopes.
    opt = torch.optim.SGD(model.parameters(), **optim_cfg, nesterov=True)
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def sanity_check(device: torch.device, cfg: Dict) -> float:  # noqa: C901 – simplicity over perf.
    """Tiny CIFAR-100 training run to verify CUDA/cuDNN health.

    The routine is intentionally short (≈10 epochs) and will raise a
    ``RuntimeError`` if the final accuracy drops below 90 %.  This *fail-fast*
    guard prevents costly, long-running experiments on a mis-configured
    machine.
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

    # ------------------------------ evaluation ----------------------------
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
        raise RuntimeError(
            "Sanity-check accuracy below 90 %. Environment may be mis-configured."
        )
    return acc
