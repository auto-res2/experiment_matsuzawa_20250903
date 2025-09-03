from __future__ import annotations
"""src/evaluate.py
Evaluation utilities, statistics and publication-quality plots.
"""
import json  # noqa: F401 – might be useful for future extensions
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless by default – no GUI backend required
import matplotlib.pyplot as plt  # noqa: E402
import torch

# ---------------------------------------------------------------------
# Runtime device is determined once inside main.py and re-used everywhere
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Central directory for all figures – must follow the instructions exactly
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / ".research" / "iteration3" / "images"  # updated as per spec
FIG_DIR.mkdir(parents=True, exist_ok=True)

################################################################################
#                             ───  METRICS ───                                 #
################################################################################

def accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """Top-1 accuracy in percent."""
    model.eval(); correct = n = 0  # noqa: E702 – intentional compact style
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); n += y.size(0)  # noqa: E702
    return 100.0 * correct / max(1, n)

################################################################################
#                             ───  PLOTS ───                                   #
################################################################################

def plot_task_accuracies(acc_per_task: List[float], fname: str = "accuracy_jemb.pdf") -> Path:
    """Line plot with per-task accuracy numbers on top of each marker."""
    plt.figure(figsize=(6, 4))
    plt.plot(acc_per_task, "o-")
    for i, v in enumerate(acc_per_task):
        plt.text(i, v + 0.5, f"{v:.1f}")
    plt.xlabel("Task")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Task Accuracy – JEMB")
    plt.grid(True)
    plt.tight_layout()
    out_path = FIG_DIR / fname
    plt.legend(["JEMB"])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
