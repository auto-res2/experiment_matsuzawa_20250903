from __future__ import annotations
"""src/evaluate.py
Evaluation utilities: metrics and plotting.
"""
import pathlib
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr

from .utils import ensure_dir, IMAGE_ROOT

sns.set_style("whitegrid")

################################################################################
#  Metrics                                                                     #
################################################################################

def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """Return accuracy and macro-F1 computed on CPU."""
    pred = logits.argmax(-1).detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()
    return {
        "acc": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
    }


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman correlation helper (converts tensors to CPU numpy)."""
    return float(spearmanr(a.detach().cpu().numpy(), b.detach().cpu().numpy())[0])

################################################################################
#  Plotting                                                                    #
################################################################################

def plot_lines(
    xs: Sequence[int],
    ys: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: pathlib.Path,
) -> None:
    """Draw line plot and save it under .research/iteration33/images/…"""
    # Ensure every visualisation is stored in the mandated directory.
    if not out_path.is_absolute():
        out_path = IMAGE_ROOT / out_path
    ensure_dir(out_path.parent)

    plt.figure(figsize=(6, 4))
    for lbl, series in ys.items():
        plt.plot(xs, series, label=lbl)
        for x, y in zip(xs, series):
            plt.text(x, y, f"{y:.2f}", fontsize=6, ha="center", va="bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
