from __future__ import annotations
"""
evaluate.py
~~~~~~~~~~~
Contains simple helper utilities for evaluation & visualisation so that the
core *train.py* stays focused.
"""
import pathlib
from typing import List
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: F401 – required for seaborn styling side-effects

# All experiment figures must be saved under this directory (see instructions)
FIG_DIR = (
    pathlib.Path(__file__)  # src/evaluate.py
    .resolve()
    .parent
    .parent
    / ".research"
    / "iteration5"  # ← updated as per specification
    / "images"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#                         TRAINING CURVE PLOTTER
# -----------------------------------------------------------------------------

def save_training_curves(losses: List[float], val_scores: List[float], run_name: str) -> None:
    """Save loss & val-F1 curves for a single run."""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax2 = ax1.twinx()

    ax1.plot(losses, label="Cross-entropy", color="tab:red")
    ax2.plot(val_scores, label="Val F1", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax2.set_ylabel("F1", color="tab:blue")
    ax1.tick_params(axis="y", colors="tab:red")
    ax2.tick_params(axis="y", colors="tab:blue")

    # annotate last points
    ax1.annotate(f"{losses[-1]:.3f}", (len(losses) - 1, losses[-1]))
    ax2.annotate(f"{val_scores[-1]:.3f}", (len(val_scores) - 1, val_scores[-1]))

    fig.tight_layout()
    out_path = FIG_DIR / f"training_loss_{run_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    # Show relative path wrt the root .research directory for concise logging
    print(f"Figure saved – {out_path.relative_to(FIG_DIR.parent.parent)}")