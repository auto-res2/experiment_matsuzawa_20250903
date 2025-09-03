"""src/evaluate.py
Evaluation and metric utilities.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1.  Accuracy on single data loader
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy (%) of *model* on *loader*."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total * 100.0
