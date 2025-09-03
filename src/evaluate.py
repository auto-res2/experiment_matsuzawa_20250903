"""
evaluate.py – validation/test utilities & plotting helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                           scalar accuracy helper
# -----------------------------------------------------------------------------

def evaluate_accuracy(model: torch.nn.Module, loader, device: str) -> float:
    """Top-1 accuracy on a dataloader (no grad, no AMP)."""
    model.eval()
    num, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            num += y.size(0)
    return correct / num if num > 0 else 0.0


# -----------------------------------------------------------------------------
#                                 plotting
# -----------------------------------------------------------------------------

IMAGES_DIR = Path(".research/iteration6/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def bar_plot(values: Dict[str, float], title: str, fname: str) -> str:
    """Simple bar plot saved to the mandatory images folder."""
    labels = list(values.keys())
    y = list(values.values())
    plt.figure(figsize=(max(4, 0.45 * len(labels)), 4))
    plt.bar(range(len(y)), y, color="tab:blue")
    plt.xticks(range(len(y)), labels, rotation=45, ha="right")
    for i, v in enumerate(y):
        plt.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    path = IMAGES_DIR / f"{fname}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)
