# src/evaluate.py
# -*- coding: utf-8 -*-
"""Evaluation & visualisation helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")      # headless backend
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Paths
# -----------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
# According to the new project specification all experiment figures must live
# under `.research/iteration2/images`.
IMAGES_DIR   = ROOT / ".research" / "iteration2" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------------

def save_line_plot(xs: Sequence, ys: Sequence, xlabel: str, ylabel: str,
                   title: str, filename: str) -> str:
    """Save a publication-quality line plot into the required images directory."""
    plt.figure(figsize=(4, 3))
    plt.plot(xs, ys, marker="o", label=ylabel)
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}", fontsize=6, ha="center", va="bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = IMAGES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved figure → {out_path.relative_to(ROOT)}")
    return str(out_path)
