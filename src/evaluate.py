"""src/evaluate.py
Evaluation utilities, statistical analysis and plotting helpers.
"""
from __future__ import annotations
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ---------------------------------------------------------------------------
# CONSTANTS – all research images MUST go to .research/iteration3/images
# ---------------------------------------------------------------------------
IMG_DIR = Path(__file__).resolve().parent.parent / ".research" / "iteration3" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _savefig(fname: str):
    path = IMG_DIR / fname
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {path.relative_to(Path.cwd())}")


def line_plot(xs: List[int], ys_dict: Dict[str, List[float]], *,
              title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure(figsize=(8, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    _savefig(filename)


def bar_plot(labels: List[str], values_dict: Dict[str, List[float]], *,
             title: str, ylabel: str, filename: str):
    x = np.arange(len(labels))
    width = 0.15
    plt.figure(figsize=(10, 4))
    for i, (name, vals) in enumerate(values_dict.items()):
        plt.bar(x + i * width, vals, width, label=name)
    plt.xticks(x + width, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    _savefig(filename)
