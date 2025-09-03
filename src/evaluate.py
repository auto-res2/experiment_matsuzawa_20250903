"""src/evaluate.py – evaluation utilities & plotting."""
import pathlib
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

__all__ = ["evaluate_model", "lineplot"]

# -----------------------------------------------------------------------------
# plotting helpers
# -----------------------------------------------------------------------------

def lineplot(xs: Sequence, ys: Sequence, *, title: str, xlabel: str, ylabel: str, fname: pathlib.Path):
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# accuracy evaluation
# -----------------------------------------------------------------------------

def evaluate_model(model: torch.nn.Module, dataset, device: str = "cuda") -> float:
    model.eval()
    total = 0
    correct = 0
    loader = DataLoader(dataset, batch_size=256, num_workers=4)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            total += y.numel()
            correct += (pred == y).sum().item()
    model.train()
    return 100.0 * correct / total
