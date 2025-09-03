import math
import pathlib
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sns.set(style="whitegrid")

__all__ = [
    "row_diff", "col_diff", "effective_rank", "classification_metrics",
    "evaluate", "save_lineplot"
]

# -----------------------------------------------------------------------------
# Diagnostic & classification metrics
# -----------------------------------------------------------------------------

def _sample_pairwise_dist(x: torch.Tensor, num_samples: int = 10_000) -> float:
    """Return the average L2 distance of *num_samples* random row-pairs.
    This avoids constructing an N×N matrix which would quickly exhaust GPU
    memory (and indeed caused the OOM crash we are fixing).  The estimate is
    unbiased because pairs are drawn uniformly with replacement.
    """
    n = x.size(0)
    if n == 0:
        return float("nan")

    # Always operate on *CPU* to free up GPU VRAM for the actual model.
    # Detaching is cheap and ensures no autograd graph is kept alive.
    x_cpu = x.detach().cpu()

    num_samples = min(num_samples, n * (n - 1))  # safety guard
    idx1 = torch.randint(0, n, (num_samples,))
    idx2 = torch.randint(0, n, (num_samples,))

    # make sure we don't pick identical indices (distance=0 would bias low)
    same = idx1 == idx2
    if same.any():
        idx2[same] = (idx2[same] + 1) % n

    diffs = x_cpu[idx1] - x_cpu[idx2]
    return diffs.norm(p=2, dim=-1).mean().item()


def row_diff(x: torch.Tensor, exact_threshold: int = 4_000) -> float:
    """Average pair-wise row distance.

    For |rows| ≤ *exact_threshold* we compute the exact value; otherwise we
    fall back to a Monte-Carlo estimate to keep memory usage under control.
    """
    n = x.size(0)
    if n <= exact_threshold:
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        return diff.norm(p=2, dim=-1).mean().item()
    # Large tensor – use sampling
    return _sample_pairwise_dist(x)


def col_diff(x: torch.Tensor, exact_threshold: int = 4_000) -> float:
    """Average pair-wise column distance (feature-wise).  Same logic as
    *row_diff* but applied to *xᵀ* whose size is usually modest; still, we keep
    the sampling fallback for completeness.
    """
    m = x.size(1)
    if m <= exact_threshold:
        diff = x.t().unsqueeze(0) - x.t().unsqueeze(1)
        return diff.norm(p=2, dim=-1).mean().item()
    return _sample_pairwise_dist(x.t())


def effective_rank(x: torch.Tensor, eps: float = 1e-6) -> float:
    # limit rank approximation for memory safety
    q = min(100, x.size(1) - 1)
    u, s, v = torch.svd_lowrank(x.detach().cpu(), q=q)
    p = s / (s.sum() + eps)
    H = -(p * torch.log(p + eps)).sum()
    return torch.exp(H).item()


def classification_metrics(logits: torch.Tensor, y_true: torch.Tensor) -> tuple[float, float]:
    pred = logits.argmax(dim=1).cpu().numpy()
    y = y_true.cpu().numpy()
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")

# -----------------------------------------------------------------------------
# Evaluation wrapper – called from training and inference scripts
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, data: "torch_geometric.data.Data") -> Dict[str, Any]:
    model.eval()
    logits, aux = model(data.x, data.edge_index, epoch=0)

    acc, macro = classification_metrics(logits[data.test_mask], data.y[data.test_mask])
    rd = row_diff(logits)
    cd = col_diff(logits)
    er = effective_rank(logits)

    return {
        "accuracy": acc,
        "macro_f1": macro,
        "row_diff": rd,
        "col_diff": cd,
        "eff_rank": er,
        "expected_K": aux.get("expected_K", None),
    }

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def save_lineplot(x: list[int] | np.ndarray,
                  ys: Dict[str, list[float] | np.ndarray],
                  x_label: str,
                  y_label: str,
                  title: str,
                  fname: str):
    path = pathlib.Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for name, series in ys.items():
        plt.plot(x, series, label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()