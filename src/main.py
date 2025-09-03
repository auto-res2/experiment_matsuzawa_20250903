"""src/main.py
Entry-point that orchestrates the full experimental workflow.
Run via  →   python -m src.main
"""
from __future__ import annotations

import os
import random
import warnings
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

from .preprocess import load_dataset
from .train import DeepGCN, Trainer
from .evaluate import evaluate_accuracy, summarise_and_plot

warnings.filterwarnings("ignore", category=UserWarning)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
#   Main
# -----------------------------------------------------------------------------

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    common_cfg = {k: v for k, v in cfg.items() if k != "experiments"}

    device = torch.device(common_cfg["device"] if torch.cuda.is_available() else "cpu")

    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # For demonstration we only run the *depth_stress* experiment to keep run-time short.
    exp_cfg = cfg["experiments"]["depth_stress"]
    trainer = Trainer(device=device, common_cfg=common_cfg)

    for dspec in exp_cfg["datasets"]:
        data = load_dataset(dspec)

        hidden_dim = exp_cfg["backbone"]["hidden_dim"]
        depths = exp_cfg["backbone"].get("depths", [exp_cfg["backbone"].get("layers", 1)])

        for depth in depths:
            for norm_cfg in exp_cfg["normalisers"]:
                exp_name = f"{dspec['name']}_L{depth}_{norm_cfg['name']}"

                # reproducibility – single seed to keep example snappy
                _set_seed(0)

                model = DeepGCN(
                    in_dim=data.x.size(-1),
                    hidden=hidden_dim,
                    out_dim=int(data.y.max().item()) + 1,
                    layers=depth,
                    normaliser_cfg=norm_cfg,
                    dropout=float(common_cfg.get("dropout", 0.5)),
                )

                lr = common_cfg["optimizer"]["lr_grid"][1]
                wd = common_cfg["optimizer"]["weight_decay"].get(dspec.get("name", "default"), 5e-4)
                epochs = (
                    exp_cfg.get("epochs", {}).get(dspec.get("loader", "Planetoid"), 200)
                    if isinstance(exp_cfg.get("epochs"), dict)
                    else exp_cfg.get("epochs", 200)
                )

                stat = trainer.train(
                    model,
                    data,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=wd,
                    eval_fn=evaluate_accuracy,
                )
                results[exp_name].append(stat)
                print(f"Finished {exp_name}: acc={stat['test_acc']:.4f}")

    summarise_and_plot(results)


if __name__ == "__main__":
    main()
