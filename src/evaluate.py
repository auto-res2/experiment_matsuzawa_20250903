from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

"""
evaluate.py – statistical analysis & visualisation utilities.
All figures are saved to `.research/iteration23/images` as PDF so that
reviewers find them in the predefined directory.
"""

# permanent, reviewer-specified directory -------------------------------------
FIG_DIR = Path('.research/iteration23/images')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

def plot_accuracy(logs: Dict[str, 'train.Log']):
    """Plot per-task average accuracy curves for every method."""

    plt.figure(figsize=(6, 3))
    for name, lg in logs.items():
        plt.plot(lg.acc, label=name)
        for i, v in enumerate(lg.acc):
            plt.text(i, v + 0.3, f"{v:.1f}", fontsize=6)

    plt.xlabel('Task')
    plt.ylabel('Avg acc (%)')
    plt.legend()
    plt.grid(True)

    fname = FIG_DIR / 'training_accuracy.pdf'
    plt.savefig(fname, bbox_inches='tight')
    print('[fig] saved', fname)
