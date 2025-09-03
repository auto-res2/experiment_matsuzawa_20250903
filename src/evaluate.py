from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

# Where camera-ready figures must go -----------------------------------------
# Repository policy (iteration17): all images are stored under the following
# directory.  This path complies with the automated test that asserts every
# figure is saved inside ``.research/iteration17/images``.
FIG_DIR = Path(".research/iteration17/images")
FIG_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# ─── PLOTTING ────────────────────────────────────────────────────────────────
###############################################################################

def bar_chart(data: Dict[str, float], title: str, file_stem: str) -> Path:
    """Save a bar chart as PDF and return the resulting path."""

    labels, vals = list(data.keys()), list(data.values())

    # Ensure a reasonable figure width regardless of number of bars
    width = 0.5 * max(len(labels), 1) + 2
    plt.figure(figsize=(width, 4))
    plt.bar(range(len(vals)), vals, color="#4C72B0")
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")

    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()

    out_path = FIG_DIR / f"{file_stem}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
