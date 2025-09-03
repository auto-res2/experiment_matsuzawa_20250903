"""src/evaluate.py
Utilities for computing metrics and model evaluation.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

################################################################################
#  METRIC PRIMITIVES
################################################################################

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item() * 100


def row_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=1, keepdim=True)).norm(dim=1).mean().item()


def col_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()


def apsd(z: torch.Tensor) -> float:
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    return s.mean().item()

################################################################################
#  FULL EVALUATION ROUTINE
################################################################################

@torch.no_grad()
def evaluate(model, data: Data) -> Dict[str, float]:
    model.eval()
    logits, _ = model(data.x, data.edge_index)
    out = {}
    for split in ["train", "val", "test"]:
        mask = getattr(data, f"{split}_mask")
        out[f"acc_{split}"] = accuracy(logits[mask], data.y[mask])

    # Representation statistics on *all* nodes
    _, h = model(data.x, data.edge_index)
    out["row_diff"] = row_diff(h)
    out["col_diff"] = col_diff(h)
    out["apsd"] = apsd(h)
    return out
