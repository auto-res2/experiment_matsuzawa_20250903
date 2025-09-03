from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
#   All figures for *this* iteration must be stored under .research/iteration8
# ---------------------------------------------------------------------------
IMG_DIR = Path(".research/iteration8/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#                               METRICS
# ---------------------------------------------------------------------------

def _accuracy(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    tot, ok = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pr = model(xb.to(device)).argmax(1)
            ok += (pr.cpu() == yb).sum().item()
            tot += len(yb)
    return ok / tot * 100.0


# ---------------------------------------------------------------------------
#                           WATERBIRDS EVALUATION
# ---------------------------------------------------------------------------

def _spur_direction_tensor(PCs: Any, spur_idx: List[int], device: str) -> torch.Tensor:
    """Returns the 1-D spurious direction as a torch tensor on *device*."""
    if isinstance(PCs, torch.Tensor):
        vec = PCs[spur_idx[0]].clone()
    else:
        # When stored as a NumPy array
        vec = torch.from_numpy(np.copy(PCs[spur_idx[0]]))
    return vec.to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 1×C×1×1


def evaluate_waterbirds(artefacts: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Computes test accuracy, ∆Prob robustness metric and writes plots."""

    model = artefacts["model"]
    cdg = artefacts["cdg"]
    PCs = artefacts["pcs"]
    spur_idx = artefacts["spur_idx"]
    ds_test = artefacts["ds_test"]
    seed = artefacts["seed"]
    device = cfg["env"]["device"]

    # ------------------------------------------------------------------
    # Accuracy on the *entire* test split
    # ------------------------------------------------------------------
    test_acc = _accuracy(model, DataLoader(ds_test, batch_size=256), device)

    # ------------------------------------------------------------------
    # ∆Prob (counterfactual robustness) on 2 000 random samples
    # ------------------------------------------------------------------
    rand_ds, _ = random_split(
        ds_test,
        [2000, len(ds_test) - 2000],
        generator=torch.Generator().manual_seed(seed),
    )
    dp: List[float] = []
    delta = _spur_direction_tensor(PCs, spur_idx, device)  # fixed across the loop

    with torch.no_grad():
        for xb, _ in DataLoader(rand_ds, batch_size=64):
            xb_d = xb.to(device)
            p_orig = F.softmax(model(xb_d), 1)
            xb_cf = cdg.generate(xb_d, delta)
            p_cf = F.softmax(model(xb_cf.to(device)), 1)
            dp.extend((p_orig - p_cf).abs().max(1)[0].cpu().tolist())
    dprob = float(np.mean(dp))

    # ------------------------------------------------------------------
    # Plot – bar with accuracy annotation
    # ------------------------------------------------------------------
    fig_path = IMG_DIR / f"waterbirds_acc_seed{seed}.pdf"
    plt.figure(figsize=(3, 4))
    sns.barplot(x=["GCDI"], y=[test_acc])
    plt.ylim(0, 100)
    plt.ylabel("Test Acc (%)")
    plt.text(0, test_acc + 1, f"{test_acc:.1f}", ha="center")
    plt.title(f"Waterbirds – seed {seed}")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    return {
        "seed": seed,
        "test_accuracy": test_acc,
        "delta_prob": dprob,
        "figure": str(fig_path.name),
    }


# ---------------------------------------------------------------------------
#                           MULTI-SEED AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs = np.array([r["test_accuracy"] for r in results])
    dpbs = np.array([r["delta_prob"] for r in results])
    return {
        "dataset": "Waterbirds",
        "method": "GCDI",
        "seeds": len(results),
        "test_acc_mean": float(accs.mean()),
        "test_acc_std": float(accs.std(ddof=1)),
        "delta_prob_mean": float(dpbs.mean()),
        "delta_prob_std": float(dpbs.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
#                               OVERVIEW PLOT
# ---------------------------------------------------------------------------

def overview_plot(agg: Dict[str, Any]):
    fig_path = IMG_DIR / "waterbirds_overview.pdf"
    plt.figure(figsize=(4, 4))
    sns.barplot(x=["Acc"], y=[agg["test_acc_mean"]])
    plt.ylim(0, 100)
    plt.title("Waterbirds – mean over seeds")
    plt.text(0, agg["test_acc_mean"] + 1, f"{agg['test_acc_mean']:.1f}±{agg['test_acc_std']:.1f}")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path.name