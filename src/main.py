"""src/main.py – experiment orchestrator
Usage:   python -m src.main
Updated: • only run lightweight ERM by default to avoid heavy diffusion
          downloads; set env ENABLE_DICA=1 to enable DiCA.
        • Adapt figure save-dir to .research/iteration3/images via evaluate.py
"""
from __future__ import annotations

import itertools, os, random, time, yaml
from pathlib import Path
from typing import Dict, Any

import torch

from .train import Trainer
from .evaluate import report_results, plot_bar

# --------------------------------------------------
# helper: seed
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------------------------------
# load YAML config
# --------------------------------------------------

CONFIG_PATH = Path("config/config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please ensure it exists.")
with open(CONFIG_PATH, "r") as fp:
    cfg: Dict[str, Any] = yaml.safe_load(fp)

# --------------------------------------------------
# run a minimal sweep (Waterbirds / ResNet-50 / ERM only by default)
# --------------------------------------------------

def now():
    return time.strftime("%Y-%m-%d@%H:%M:%S")


def main():
    exp_metrics: Dict[str, float] = {}

    print(f"\n>>> Starting Experiment – {now()}")

    # By default we skip the extremely heavy DiCA runs which require downloading
    # multi-GB Stable-Diffusion weights.  Set the environment variable
    #   ENABLE_DICA=1
    # to include DiCA in the sweep.
    methods_to_run = ["erm"]
    if os.getenv("ENABLE_DICA", "0") == "1":
        methods_to_run.append("dica")

    for dataset_name in ["waterbirds"]:
        for backbone, method in itertools.product(["resnet50"], methods_to_run):
            key = f"{dataset_name}_{backbone}_{method}"
            test_acc_runs = []
            for seed in cfg["training"]["seeds"]:
                set_seed(seed)
                trainer = Trainer(cfg, dataset_name, method, backbone)
                trainer.train()
                test_loader = torch.utils.data.DataLoader(
                    trainer.test_ds, batch_size=64, shuffle=False
                )
                acc = trainer.evaluate(test_loader)
                test_acc_runs.append(acc)
            exp_metrics[key] = sum(test_acc_runs) / len(test_acc_runs)

    # ---------- plotting ----------
    labels = list(exp_metrics.keys())
    values = list(exp_metrics.values())
    plot_bar(labels, values, fname="waterbirds_accuracy", title="Accuracy")

    report_results("Waterbirds demo", {"overall_acc": exp_metrics})


if __name__ == "__main__":
    main()
