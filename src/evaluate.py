import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

# All figures must be stored under the dedicated research folder -----------------------
# NOTE: updated to iteration17 according to the instructions
IMG_DIR = pathlib.Path(".research/iteration17/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# --------------------------------------------------------------------------------------
#  Plotting helpers
# --------------------------------------------------------------------------------------

def lineplot(
    xs: List[int],
    ys: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    fname: pathlib.Path | str,
) -> None:
    """Save a PDF line plot under .research/iteration17/images/<fname>."""
    plt.figure(figsize=(6, 4))
    for label, series in ys.items():
        plt.plot(xs, series, label=label)
        for x, y in zip(xs, series):
            plt.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=6,
            )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = IMG_DIR / pathlib.Path(fname).name
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
#  Metrics
# --------------------------------------------------------------------------------------

def row_diff(h: torch.Tensor) -> float:
    diff = h.unsqueeze(0) - h.unsqueeze(1)
    return diff.norm(p=2, dim=-1).mean().item()


def col_diff(h: torch.Tensor) -> float:
    diff = h.t().unsqueeze(0) - h.t().unsqueeze(1)
    return diff.norm(p=2, dim=-1).mean().item()


def effective_rank(h: torch.Tensor, eps: float = 1e-6) -> float:
    # Shannon entropy of singular values  ->  effective rank (Roy & Vetterli 2007)
    _, s, _ = torch.linalg.svd(h, full_matrices=False)
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    return torch.exp(entropy).item()


def classification_metrics(logits: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1).cpu().numpy()
    y = y_true.cpu().numpy()
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return spearmanr(x.cpu().numpy(), y.cpu().numpy())[0]
