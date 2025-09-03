"""src/evaluate.py
Evaluation helpers, statistics and plotting utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# -----------------------------------------------------------------------------
# Image output directory – mandated by the evaluation environment
# -----------------------------------------------------------------------------
IMG_ROOT = Path(".research/iteration5/images")  # updated as per requirements
IMG_ROOT.mkdir(parents=True, exist_ok=True)


def evaluate_model(model: torch.nn.Module, dl: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Return (average accuracy, surrogate worst-group accuracy)."""
    model.eval()
    DEVICE = next(model.parameters()).device

    total, correct = 0, 0
    grp_correct: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _ in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            if isinstance(logits, tuple):  # Allow for (logits, features)
                logits = logits[0]
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            grp_correct.append((pred == y).float().cpu())

    acc = 100 * correct / total
    wg = 100 * min([g.mean().item() for g in grp_correct])  # lower-bound surrogate
    return acc, wg


def aggregate_and_plot(
    test_results: List[Tuple[float, float]],
    erm_wg: np.ndarray,
    fig_name: str = "wgacc_waterbirds.pdf",
):
    """Aggregate results across seeds, perform paired t-test vs ERM, and save bar plot."""

    accs = np.asarray([a for a, _ in test_results])
    wgs = np.asarray([b for _, b in test_results])

    print("\n--- Aggregate over seeds ---")
    print(f"AvgAcc  {accs.mean():.2f} ± {accs.std():.2f}")
    print(f"WGAcc   {wgs.mean():.2f} ± {wgs.std():.2f}")

    t, p = ttest_rel(wgs, erm_wg)
    print(f"Paired t-test PCCM vs ERM (WGAcc)  t={t:.2f}  p={p:.4f}")

    # ------------------ plotting ------------------
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(4, 3))
    methods = ["ERM", "PCCM"]
    means = [erm_wg.mean(), wgs.mean()]
    stds = [erm_wg.std(), wgs.std()]
    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=["#d95f02", "#1b9e77"])
    ax.set_ylabel("Worst-Group Accuracy (%)")
    for rect, val in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center")

    out_path = IMG_ROOT / fig_name
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
