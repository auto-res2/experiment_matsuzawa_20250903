from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# All figures must be stored under .research/iteration3/images according to the
# project guidelines.  We therefore create (or reuse) that exact directory.
# -----------------------------------------------------------------------------
IMG_DIR = Path(".research/iteration3/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Simple plotting helpers
# -----------------------------------------------------------------------------

def plot_curve(xs, ys, title: str, xlab: str, ylab: str, fname: str) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outfile = IMG_DIR / f"{fname}.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"[FIG SAVED] {outfile}")


def plot_bar(labels, values, fname: str, title: str = "Accuracy") -> None:
    plt.figure(figsize=(max(6, len(labels) * 1.2), 3))
    plt.bar(range(len(values)), values, color="steelblue")
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    for idx, v in enumerate(values):
        plt.text(idx, v + 0.01, f"{v:.2f}", ha="center")
    plt.ylabel(title)
    plt.tight_layout()
    outfile = IMG_DIR / f"{fname}.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"[FIG SAVED] {outfile}")


def report_results(exp_name: str, metrics: Dict[str, float | Dict | list]):
    print("================  EXPERIMENT  ================")
    print(exp_name)
    print("================  RESULTS      ================")
    for k, v in metrics.items():
        print(f"{k}: {v}")
