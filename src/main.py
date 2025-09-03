"""
main.py – orchestrates the experiment.  Run with `python -m src.main`.
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, List

from .train import Trainer
from .evaluate import bar_plot

# -----------------------------------------------------------------------------
#                          load configuration YAML
# -----------------------------------------------------------------------------
CFG_PATH = Path("config/config.yaml")
if not CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found.  Please ensure the repository layout is correct.")

with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# -----------------------------------------------------------------------------
#                        minimal demo  – matches original
# -----------------------------------------------------------------------------
print("================  EXPERIMENT DESCRIPTION  =================", flush=True)
print("Experiment 1 – FULL ROBUSTNESS GRID (subset demo: Waterbirds, ResNet-50, 2 methods)")
print("===========================================================", flush=True)

results: Dict[str, List[float]] = {}

for dataset, backbone, method in [
    ("waterbirds", "resnet50", "erm"),
    ("waterbirds", "resnet50", "irm"),
]:
    for seed in CFG["random_seeds"]:
        trainer = Trainer(CFG, dataset, method, backbone, seed)
        metrics = trainer.fit()
        key = f"{dataset}_{backbone}_{method}"
        results.setdefault(key, []).append(metrics["AccID"])

mean_acc = {k: sum(v) / len(v) for k, v in results.items()}
fig_path = bar_plot(mean_acc, "AccID – Waterbirds (demo)", "accuracy_waterbirds_demo")

print("================  EXPERIMENTAL DATA  =====================", flush=True)
for k, v in mean_acc.items():
    print(f"{k}: {v:.4f}")
print("================  FIGURE FILENAMES  =====================", flush=True)
print(fig_path, flush=True)
