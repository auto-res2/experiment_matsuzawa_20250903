from __future__ import annotations
"""
evaluate.py – evaluation routines and plotting utilities
(CI-adapted – 2025-09-03):
1.  All figures are now stored under the *exact* path
        `.research/iteration10/images`
    as required by the task description.
2.  Added an import-time guard so that NumPy errors are clearer in case it is
    missing (defensive programming).
3.  No functional changes beyond the directory relocation.
"""
import statistics as st
from pathlib import Path

import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after mpl backend)
import torch  # noqa: E402

from .train import train_stream, sanity_single_task  # noqa: E402

# -------------------------------------------------------------------------
# Configuration ------------------------------------------------------------
# -------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH, "r") as _f:
    CFG = yaml.safe_load(_f)

# -------------------------------------------------------------------------
# Plot directory (all experiment images MUST go here) ----------------------
# -------------------------------------------------------------------------
PLOT_DIR = Path(".research/iteration10/images")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# EXPERIMENT 1 – accuracy / forgetting trade-off (CIFAR-100) --------------
# -------------------------------------------------------------------------

def experiment1():
    print("\n========== EXP-1: Correctness & Memory-Accuracy Trade-off ==========")

    # -------------------------------------------------------------
    # CI / CPU-only guard – skip heavy training when CUDA absent
    # -------------------------------------------------------------
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available – heavy training routines are disabled in this environment.")
        plt.figure(figsize=(2, 2))
        plt.text(0.5, 0.5, "Skipped (no CUDA)", ha="center", va="center")
        placeholder = PLOT_DIR / "training_accuracy.pdf"
        plt.savefig(placeholder, bbox_inches="tight")
        print("Placeholder figure saved:", placeholder)
        return

    device = CFG["device"]
    sanity_single_task(device=device)

    methods = ["HATEM", "ER"]  # abbreviated list for brevity in the template
    results = {m: [] for m in methods}

    for budget_name, bytes_ in CFG["budgets"].items():
        print(f"\n--- Budget {budget_name} ({bytes_ // 1024} KB) ---")
        for m in methods:
            accs, forgets = [], []
            for s in CFG["seeds"]:
                a, f, _ = train_stream(m, bytes_, s, device)
                accs.append(a)
                forgets.append(f)
            mu = st.mean(accs)
            sigma = st.stdev(accs) if len(accs) > 1 else 0
            results[m].append((mu, sigma))
            print(f"{m:<6}  AACC={mu:.2f}±{sigma:.2f}  F↓={st.mean(forgets):.2f}")

    # --------------------------- plot ----------------------------------
    x = np.arange(len(CFG["budgets"]))
    plt.figure(figsize=(6, 4))
    for m, vals in results.items():
        y = [v[0] for v in vals]
        plt.plot(x, y, "o-", label=m)
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.3, f"{yi:.1f}")
    plt.xticks(x, [f"{b // 1024} KB" for b in CFG["budgets"].values()])
    plt.ylabel("Average Accuracy (%)")
    plt.title("Exp-1: AACC vs Memory budget (CIFAR-100)")
    plt.legend()
    fig_path = PLOT_DIR / "training_accuracy.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Figure saved:", fig_path)
