import torch
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# ---------------- EFFECTIVE RANK ---------------------------------------------

def effective_rank(H: torch.Tensor, thresh: float = 0.99):
    """Zhang et al. effective rank metric (fraction of singular values covering thresh mass)."""
    with torch.no_grad():
        u, s, v = torch.svd(H)
        s_norm = s / s.sum()
        cumsum = torch.cumsum(s_norm, dim=0)
        return (cumsum <= thresh).sum().item() / H.size(1)


# ---------------- EVALUATE ----------------------------------------------------

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        y_pred = logits.argmax(dim=1)
        accs = {}
        for split in ["train", "val", "test"]:
            mask = getattr(data, f"{split}_mask")
            accs[split] = accuracy_score(data.y[mask].cpu(), y_pred[mask].cpu())
    return accs, logits


# ---------------- PLOTTING ----------------------------------------------------

def save_lineplot(xs: List[int], ys_dict: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, fname):
    plt.figure()
    for k, v in ys_dict.items():
        plt.plot(xs, v, label=k, marker="o")
        for x, y in zip(xs, v):
            plt.text(x, y, f"{y:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", format="pdf")
    plt.close()
