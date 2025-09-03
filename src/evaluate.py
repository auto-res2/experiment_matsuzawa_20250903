"""src/evaluate.py – evaluation routines & plotting utilities"""
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

IMG_DIR = Path(".research/iteration7/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _accuracy(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    tot, ok = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pr = model(xb.to(device)).argmax(1)
            ok += (pr.cpu() == yb).sum().item()
            tot += len(yb)
    return ok / tot * 100.0


def evaluate_waterbirds(artefacts: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Computes test accuracy, ∆Prob robustness metric and writes plots."""

    model = artefacts["model"]
    cdg = artefacts["cdg"]
    PCs = artefacts["pcs"]
    spur_idx = artefacts["spur_idx"]
    ds_test = artefacts["ds_test"]
    seed = artefacts["seed"]
    device = cfg["env"]["device"]

    # Accuracy
    test_acc = _accuracy(model, DataLoader(ds_test, batch_size=256), device)

    # ∆Prob on 2 000 random test samples
    rand_ds, _ = random_split(
        ds_test,
        [2000, len(ds_test) - 2000],
        generator=torch.Generator().manual_seed(seed),
    )
    dp: List[float] = []
    with torch.no_grad():
        for xb, _ in DataLoader(rand_ds, batch_size=64):
            p_orig = F.softmax(model(xb.to(device)), 1)
            delta = (
                torch.from_numpy(PCs[spur_idx[0]].copy())
                .to(device)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            xb_cf = cdg.generate(xb.to(device), delta)
            p_cf = F.softmax(model(xb_cf.to(device)), 1)
            dp.extend((p_orig - p_cf).abs().max(1)[0].cpu().tolist())
    dprob = float(np.mean(dp))

    # Plot – bar with accuracy annotation
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
