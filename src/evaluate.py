from __future__ import annotations
"""
evaluate.py – all metric computations & plotting utilities.
All experiment figures are now saved under the directory
`.research/iteration22/images` as mandated by the task description.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_auc_score  # additional metrics if needed

# -----------------------------------------------------------------------------
# basic metrics used in the paper
# -----------------------------------------------------------------------------

@torch.inference_mode()
def accuracy(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.inference_mode()
def worst_group_acc(model, loader, device: str) -> float:
    """Expects loader that returns (x, y, metadata) where metadata holds
    group id in its first column (WILDS convention)."""
    model.eval()
    correct, counts = {}, {}
    for x, y, meta in loader:
        groups = meta[:, 0].cpu().tolist() if meta.ndim > 1 else meta.cpu().tolist()
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        eq = (preds == y).cpu().numpy()
        for g, ok in zip(groups, eq):
            correct[g] = correct.get(g, 0) + ok
            counts[g] = counts.get(g, 0) + 1
    return min(correct[g] / counts[g] for g in correct)

# -----------------------------------------------------------------------------
# plotting helpers (publication-style bar chart)
# -----------------------------------------------------------------------------

_IMG_DIR = Path(".research/iteration22/images")
_IMG_DIR.mkdir(parents=True, exist_ok=True)

def bar_chart(data: dict, title: str, fname: str):
    keys, vals = list(data.keys()), [data[k] for k in data]
    plt.figure(figsize=(max(len(keys) * 0.6, 4), 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    path = _IMG_DIR / f"{fname}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
