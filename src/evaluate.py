"""src/evaluate.py
Evaluation utilities – accuracy metrics, over-smoothing indicators and a
convenience wrapper that returns a dictionary with statistics for each
split as well as APSD, row-diff and col-diff.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .preprocess import DEVICE

# ----------------------------------------------------------------------------
#  METRICS
# ----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return classification accuracy in percent."""
    return (logits.argmax(dim=-1) == labels).float().mean().item() * 100.0


def row_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=1, keepdim=True)).norm(dim=1).mean().item()


def col_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()


def apsd(z: torch.Tensor) -> float:
    z = z.detach() - z.mean(0, keepdim=True)
    return torch.linalg.svdvals(z).mean().item()


# ----------------------------------------------------------------------------
#  EVALUATION WRAPPER
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data) -> Dict[str, float]:
    """Return dict with accuracy for train/val/test & smoothing metrics."""
    model.eval()
    logits, h = model(data.x, data.edge_index)

    res = {
        split: accuracy(logits[getattr(data, f"{split}_mask")], data.y[getattr(data, f"{split}_mask")])
        for split in ("train", "val", "test")
    }
    res.update({"row": row_diff(h), "col": col_diff(h), "apsd": apsd(h)})
    return res
