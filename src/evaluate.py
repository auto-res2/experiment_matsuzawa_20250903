"""
evaluate.py – evaluation routines and plotting utilities
"""
from __future__ import annotations
import statistics as st
from pathlib import Path
import yaml, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .train import train_stream, sanity_single_task

CFG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
with open(CFG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

PLOT_DIR = Path('.research/iteration2/images')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# EXPERIMENT 1 – accuracy / forgetting trade-off (CIFAR-100)
# -------------------------------------------------------------------------

def experiment1():
    print('\n========== EXP-1: Correctness & Memory-Accuracy Trade-off ==========')
    device = CFG['device']
    sanity_single_task(device=device)

    methods = ['HATEM', 'ER']  # abbreviated list for brevity
    results = {m: [] for m in methods}

    for budget_name, bytes_ in CFG['budgets'].items():
        print(f"\n--- Budget {budget_name} ({bytes_ // 1024} KB) ---")
        for m in methods:
            accs, forgets = [], []
            for s in CFG['seeds']:
                a, f, _ = train_stream(m, bytes_, s, device)
                accs.append(a); forgets.append(f)
            mu, sigma = st.mean(accs), (st.stdev(accs) if len(accs) > 1 else 0)
            results[m].append((mu, sigma))
            print(f"{m:<6}  AACC={mu:.2f}±{sigma:.2f}  F↓={st.mean(forgets):.2f}")

    # --------------------------- plot ----------------------------------
    x = np.arange(len(CFG['budgets']))
    plt.figure(figsize=(6, 4))
    for m, vals in results.items():
        y = [v[0] for v in vals]
        plt.plot(x, y, 'o-', label=m)
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.3, f"{yi:.1f}")
    plt.xticks(x, [f"{b // 1024} KB" for b in CFG['budgets'].values()])
    plt.ylabel('Average Accuracy (%)')
    plt.title('Exp-1: AACC vs Memory budget (CIFAR-100)')
    plt.legend()
    fig_path = PLOT_DIR / 'training_accuracy.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    print('Figure saved:', fig_path)
