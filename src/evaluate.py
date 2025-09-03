"""src/evaluate.py – metrics & evaluation helpers"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
#   Metric functions
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item() * 100.0


def row_col_diff(z: torch.Tensor):  # noqa: D401
    """Row-diff / Col-diff oversmoothing indicators."""
    z = z.detach()
    row_mean = z.mean(dim=1, keepdim=True)
    col_mean = z.mean(dim=0, keepdim=True)
    row_diff = (z - row_mean).norm(dim=1).mean().item()
    col_diff = (z - col_mean).norm(dim=1).mean().item()
    return row_diff, col_diff


def apsd(z: torch.Tensor) -> float:
    """Average pair-wise singular value distance (APSD)."""
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    return torch.mean(s).item()


# ---------------------------------------------------------------------------
#   Full evaluation routine (Acc. + smoothing metrics)
# ---------------------------------------------------------------------------

def eval_model(model: nn.Module, data):  # noqa: ANN001
    """Returns (train_acc, val_acc, test_acc, row_diff, col_diff, apsd)."""
    model.eval()
    with torch.no_grad():
        logits, z = (
            model(data.x, data.edge_index)
            if hasattr(model, "mutual_info_loss")
            else (model(data.x, data.edge_index), None)
        )
        acc_tr = accuracy(logits[data.train_mask], data.y[data.train_mask])
        acc_val = accuracy(logits[data.val_mask], data.y[data.val_mask])
        acc_te = accuracy(logits[data.test_mask], data.y[data.test_mask])
        rd, cd = row_col_diff(z if z is not None else logits)
        return acc_tr, acc_val, acc_te, rd, cd, apsd(z if z is not None else logits)
