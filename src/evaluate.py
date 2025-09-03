from __future__ import annotations
"""
evaluate.py – evaluation metrics & visualisation helpers
All plots are saved into .research/iteration19/images to comply with the
project specification.
"""
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
#  Global output directory for all figures
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / ".research" / "iteration19" / "images"  # ← updated path
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
#  Metrics                                                                    #
###############################################################################


def cls_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        "accuracy": accuracy_score(y_np, pred),
        "macro_f1": f1_score(y_np, pred, average="macro"),
    }


def row_diff(h: torch.Tensor) -> float:
    d = h.unsqueeze(0) - h.unsqueeze(1)
    return d.norm(p=2, dim=-1).mean().item()


def col_diff(h: torch.Tensor) -> float:
    d = h.T.unsqueeze(0) - h.T.unsqueeze(1)
    return d.norm(p=2, dim=-1).mean().item()


def eff_rank(h: torch.Tensor, eps: float = 1e-6) -> float:
    _u, s, _v = torch.linalg.svd(h, full_matrices=False)
    p = s / (s.sum() + eps)
    H = -(p * torch.log(p + eps)).sum()
    return torch.exp(H).item()


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return spearmanr(x.cpu().numpy(), y.cpu().numpy())[0]

###############################################################################
#  Plotting helpers                                                           #
###############################################################################

def _annotate(ax):
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
            )


def line(
    xs: List[int],
    ys: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    fname: str | pathlib.Path,
):
    """Line plot that is always redirected into IMAGES_DIR."""

    plt.figure(figsize=(6, 4))
    for k, v in ys.items():
        plt.plot(xs, v, label=k)
    _annotate(plt.gca())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fname = pathlib.Path(fname).with_suffix(".pdf").name  # keep only file name
    out_path = IMAGES_DIR / fname
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
