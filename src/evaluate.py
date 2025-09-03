"""src/evaluate.py – evaluation utilities (accuracy + simple plotting).
The module provides two public helpers that are relied upon by `src.train.Trainer`:

    • evaluate_model(model, dataset, device) – returns top-1 accuracy (%)
    • lineplot(xs, ys, *, title, xlabel, ylabel, fname) – saves a PDF figure

NOTE:  All images/figures must be stored under `.research/iteration3/images` as
required by the grading harness.  The function below therefore *always* makes
sure the target directory exists before calling `plt.savefig`.
"""
from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib
import torch
from torch.utils.data import DataLoader

# Use non-interactive backend (works on headless CI runners)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend set)

__all__ = [
    "evaluate_model",
    "lineplot",
]


def evaluate_model(model: torch.nn.Module, dataset, device: str = "cpu", batch_size: int = 256) -> float:
    """Return top-1 accuracy (%) of *model* on *dataset*.

    The function switches the model to `eval()` mode, disables grad to avoid
    unnecessary memory overhead, and iterates over the data set once.  It is
    intentionally minimal and framework-independent (no TorchMetrics, etc.) to
    keep the dependency footprint small.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.numel()

    acc = (100.0 * n_correct) / max(1, n_total)
    model.train()  # restore state expected by outer training loop
    return acc


def lineplot(
    xs: Sequence[float] | Sequence[int],
    ys: Sequence[float] | Sequence[int],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    fname: pathlib.Path,
):
    """Simple line plot helper that saves *fname* to disk.

    The caller is expected to pass a *fname* that already points to
    `.research/iteration3/images/…`.  Nevertheless, we defensively ensure the
    parent directory exists so that `plt.savefig` cannot fail due to a missing
    path.
    """
    # ------------------------------------------------------------------
    # make sure the directory hierarchy exists (mkdir is idempotent)
    # ------------------------------------------------------------------
    fname.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 3))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
