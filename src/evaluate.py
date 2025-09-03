from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from .train import TaskLog

# hard-coded figure directory requested by the specification
FIG_DIR = Path('.research/iteration16/images')  # ← updated to iteration16
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_curves(logs: Dict[str, TaskLog]):
    """Plot per-task average accuracy curves and save under the required path."""
    plt.figure(figsize=(6, 3))
    for name, l in logs.items():
        plt.plot(l.acc, label=name)
    plt.xlabel('Task')
    plt.ylabel('Average accuracy (%)')
    plt.grid()
    plt.legend()
    fname = FIG_DIR / 'training_accuracy.pdf'
    plt.savefig(fname, bbox_inches='tight')
    print('[fig] saved', fname)
