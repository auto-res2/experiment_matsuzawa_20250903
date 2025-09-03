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

def row_diff(x: torch.Tensor) -> float:
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    return diff.norm(p=2, dim=-1).mean().item()

def col_diff(x: torch.Tensor) -> float:
    diff = x.t().unsqueeze(0) - x.t().unsqueeze(1)
    return diff.norm(p=2, dim=-1).mean().item()

def effective_rank(x: torch.Tensor, eps: float = 1e-6) -> float:
    # limit rank approximation for memory safety
    q = min(100, x.size(1) - 1)
    u, s, v = torch.svd_lowrank(x, q=q)
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
