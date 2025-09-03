"""src/evaluate.py
Evaluation, statistical analysis & plotting utilities.
All plots are written under `.research/iteration8/images/` as required.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import statsmodels.stats.anova as sm_anova

# ---------------------------------------------------------------------
# directory for figures (created on first import)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NOTE: During the refactor all images must now live under iteration8
IMAGES_DIR = PROJECT_ROOT / ".research" / "iteration8" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

# ------------------------------ plotting -----------------------------

def _annotate_line(x: Sequence[int], y: Sequence[float]):
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.1f}")


def learning_curve(x: Sequence[int], ys_dict: Dict[str, Sequence[float]], title: str, fig_name: str | None = None):
    """Save a PDF learning-curve plot under IMAGES_DIR."""
    if fig_name is None:
        fig_name = title.replace(" ", "_") + ".pdf"
    fig_path = IMAGES_DIR / fig_name
    plt.figure(figsize=(8, 4))
    for lab, ys in ys_dict.items():
        plt.plot(x, ys, label=lab)
        _annotate_line(x, ys)
    plt.xlabel("Task #")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", format="pdf")
    plt.close()
    return fig_path

# ------------------------------ statistics --------------------------

def rm_anova(df: pd.DataFrame, dv: str, within: str, subject: str):
    """Repeated-measures ANOVA (wrapper around statsmodels)."""
    return sm_anova.AnovaRM(df, dv, subject, [within]).fit()


def paired_t(a: np.ndarray, b: np.ndarray):
    """Paired t-test plus Cohen's d effect size."""
    t, p = ss.ttest_rel(a, b)
    d = (np.mean(a) - np.mean(b)) / np.std(a - b, ddof=1)
    return t, p, d
