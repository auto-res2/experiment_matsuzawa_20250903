"""src/evaluate.py – tiny evaluation helpers

The real code base would provide *much* richer functionality (calculation of
additional metrics, logging, etc.).  For the purposes of the automated tests we
only need **accuracy** on the canonical train/val/test node splits.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

__all__ = ["eval_model"]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy (as a *percentage*, not a fraction)."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.numel()


def eval_model(model: torch.nn.Module, data: Data) -> tuple[float, float, float]:
    """Return `(train_acc, val_acc, test_acc)` – the exact tuple shape expected by
    the rest of the repository (see calls in `src/main.py`).
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        train_acc = _accuracy(logits[data.train_mask], data.y[data.train_mask])
        val_acc = _accuracy(logits[data.val_mask], data.y[data.val_mask])
        test_acc = _accuracy(logits[data.test_mask], data.y[data.test_mask])
    return train_acc, val_acc, test_acc
