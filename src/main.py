"""
main.py (entry point)
~~~~~~~~~~~~~~~~~~~~~
Coordinates the whole experimental pipeline:
  1. loads YAML configuration,
  2. iterates over experiments,
  3. uses *Trainer* to train individual models,
  4. aggregates & persists results.
Run via:   python -m src.main
"""
from __future__ import annotations
import os, json, math, pathlib
from typing import Dict, Any, List
import numpy as np

import torch

from .preprocess import DatasetFactory
from .train import Trainer

# -----------------------------------------------------------------------------
#                               CONFIG HANDLING
# -----------------------------------------------------------------------------
import yaml

CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
CFG_PATH.parent.mkdir(exist_ok=True, parents=True)

_DEFAULT_YAML = {
    "common": {
        "device": "cuda:0",
        "seeds": list(range(1, 11)),
        "max_epochs": 400,
        "early_stop": 50,
        "optimiser": {"lr": 0.01, "wd": 5e-4, "betas": [0.9, 0.999], "eps": 1e-8},
        "ada": {
            "tau0": 2.5,
            "tau_final": 0.4,
            "tau_anneal_portion": 0.7,
            "lambda_reg": 1e-3,
            "uniformity_weight": 0.05,
            "phi_hidden": 32,
        },
    },
    "experiment1": {
        "datasets": ["Cora", "PubMed", "Chameleon", "Squirrel", "ogbn-arxiv"],
        "depths": [2, 8, 32, 64],
        "backbones": ["gcn", "sage"],
    },
}


def _ensure_cfg() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        with open(CFG_PATH, "w", encoding="utf-8") as fp:
            yaml.safe_dump(_DEFAULT_YAML, fp)
        print(f"Default config written to {CFG_PATH}")
    with open(CFG_PATH, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# -----------------------------------------------------------------------------
#                               EXPERIMENT 1
# -----------------------------------------------------------------------------

def experiment1(cfg: Dict[str, Any]):
    print("\n================  EXPERIMENT 1  – Depth-vs-Accuracy Stress-Test  ================")
    datasets: List[str] = cfg["experiment1"]["datasets"]
    depths: List[int] = cfg["experiment1"]["depths"]
    backbones: List[str] = cfg["experiment1"]["backbones"]

    seeds: List[int] = cfg["common"]["seeds"]
    optimiser = cfg["common"]["optimiser"]
    ada_cfg = cfg["common"]["ada"]

    results = {}
    for dname in datasets:
        ds = DatasetFactory.get(dname)
        for backbone in backbones:
            for depth in depths:
                for variant in ["vanilla", "adasmooth"]:
                    use_ada = variant == "adasmooth"
                    tag = f"{dname}_{backbone}_{depth}L_{variant}"
                    print(f"\n----------------  Running: {tag}  ----------------")
                    trainer = Trainer(cfg["common"], ds, run_name=tag, device=_device())
                    metrics_seed = []
                    for seed in seeds:
                        out = trainer.run_single(
                            seed=seed,
                            backbone=backbone,
                            depth=depth,
                            use_ada=use_ada,
                            optim_cfg=optimiser,
                            ada_cfg=ada_cfg if use_ada else None,
                        )
                        metrics_seed.append(out)
                    results[tag] = metrics_seed
                    mean_test = np.mean([m["test"] for m in metrics_seed])
                    ci95 = 1.96 * np.std([m["test"] for m in metrics_seed]) / math.sqrt(len(seeds))
                    print(f"{tag} – test F1 = {mean_test:.4f} ± {ci95:.4f}")

    out_file = pathlib.Path(__file__).resolve().parent.parent / "exp1_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Numerical results written to {out_file}")


# -----------------------------------------------------------------------------
#                               DEVICE UTILITY
# -----------------------------------------------------------------------------

def _device() -> str:
    cfg_dev = _CFG["common"].get("device", "cuda:0")
    if cfg_dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available – falling back to CPU.")
        return "cpu"
    return cfg_dev


# -----------------------------------------------------------------------------
#                                   MAIN
# -----------------------------------------------------------------------------

def main():
    global _CFG  # noqa: PLW0603
    _CFG = _ensure_cfg()

    # limit visible GPU (important on multi-GPU server)
    os.environ["CUDA_VISIBLE_DEVICES"] = _CFG["common"].get("device", "cuda:0").split(":")[-1]

    experiment1(_CFG)


if __name__ == "__main__":
    main()
