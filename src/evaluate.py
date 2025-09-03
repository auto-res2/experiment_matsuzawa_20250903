"""
evaluate.py – evaluation utilities & figure helpers
"""
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Group-aware worst-group accuracy (default: 4 groups in Waterbirds)
# -----------------------------------------------------------------------------

def worst_group_acc(preds: torch.Tensor, labels: torch.Tensor, groups: List[Any]) -> float:
    groups_tensor = torch.tensor(groups)
    worst = 1.0
    for g in torch.unique(groups_tensor):
        idx = groups_tensor == g
        correct = (preds[idx] == labels[idx]).float().mean().item()
        worst = min(worst, correct)
    return worst


# -----------------------------------------------------------------------------
# Evaluation routine (no gradient, batched)
# -----------------------------------------------------------------------------

def evaluate_model(model, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x = x.cuda(non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
            all_groups.extend(g)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    wga = worst_group_acc(preds, labels, all_groups)
    acc = (preds == labels).float().mean().item()
    return {"acc": acc, "wga": wga}


# -----------------------------------------------------------------------------
# Simple bar-plot helper – always stores under .research/iteration1/images
# -----------------------------------------------------------------------------

def save_bar(values: Dict[str, float], title: str, fname: Path):
    out_dir = Path(".research/iteration1/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / fname.name

    sns.set(style="whitegrid")
    keys, vals = list(values.keys()), list(values.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color="steelblue")
    ax.set_ylabel(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    plt.savefig(str(fname), bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[Figure] saved → {fname}")
