"""src/evaluate.py
Evaluation utilities – accuracy computation and result visualisation.
All heavy dependencies (matplotlib / seaborn) live here so that importing
*train.py* remains light-weight when evaluation is not required.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# create figure directory once ------------------------------------------------
_FIG_DIR = os.path.join(".research", "iteration1", "images")
os.makedirs(_FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
#   1) accuracy helper
# -----------------------------------------------------------------------------

def evaluate_accuracy(model, data, split: str = "test") -> float:
    """Return classification accuracy for the requested split mask."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
    if split == "val":
        mask = data.val_mask
    elif split == "test":
        mask = data.test_mask
    elif split == "train":
        mask = data.train_mask
    else:
        raise ValueError("split needs to be train/val/test")
    return (pred[mask] == data.y[mask]).float().mean().item()


# -----------------------------------------------------------------------------
#   2) aggregation + plotting
# -----------------------------------------------------------------------------

def summarise_and_plot(result_dict: Dict[str, List[Dict[str, Any]]], fig_name: str = "accuracy_depth_stress.pdf") -> None:
    """Turn raw run-wise dictionaries into a bar-plot and console table."""
    df_rows = []
    for exp_name, runs in result_dict.items():
        row = pd.DataFrame(runs).mean().to_dict()
        row.update({"exp": exp_name})
        df_rows.append(row)
    df = pd.DataFrame(df_rows)

    # ---- textual summary ----
    print("\n================  Experimental numerical data  ================")
    print(df.to_string(index=False, float_format="%.4f"))
    print("==============================================================")

    # ---- figure ----
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="exp", y="test_acc", palette="crest")
    plt.xticks(rotation=45, ha="right")
    for i, row in df.iterrows():
        ax.text(i, row["test_acc"] + 0.002, f"{row['test_acc']:.2f}", ha="center")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()

    save_path = os.path.join(_FIG_DIR, fig_name)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
