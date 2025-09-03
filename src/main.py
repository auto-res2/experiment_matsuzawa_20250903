"""Entry point – run with
    python -m src.main
Everything is organised via relative imports so that the project can be
installed as an editable package if desired (``pip install -e .``).
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import time
from types import SimpleNamespace
from typing import Dict, Any

import yaml
import torch

# -----------------------------------------------------------------------------
# Local package imports
# -----------------------------------------------------------------------------
from .preprocess import load_dataset
from .train import full_train
from .evaluate import evaluate  # used for quick test runs

# models live in a sibling directory specified by the challenge statement
from models import DeepGCN, DeepGAT, APDWrapper  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please ensure it is available.")

CONFIG: Dict[str, Any] = yaml.safe_load(_CONFIG_PATH.read_text())

# -----------------------------------------------------------------------------
# Model registry for convenience
# -----------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "gcn_deep":   lambda in_d, hid, out_d, _: DeepGCN(in_d, hid, out_d, num_layers=128),
    "gat_deep":   lambda in_d, hid, out_d, _: DeepGAT(in_d, hid, out_d, heads=8, num_layers=128),
    "apd_gcn":    lambda in_d, hid, out_d, _: APDWrapper("gcn", in_d, hid, out_d, num_layers=128),
    "apd_gat":    lambda in_d, hid, out_d, _: APDWrapper("gat", in_d, hid, out_d, num_layers=128),
}

# -----------------------------------------------------------------------------
# Experiment helpers
# -----------------------------------------------------------------------------

def _fabricate_masks(data: "torch_geometric.data.Data", train: float = 0.6, val: float = 0.2):
    n = data.num_nodes
    idx = torch.randperm(n)
    tr_end, va_end = int(train * n), int((train + val) * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[idx[:tr_end]] = True
    data.val_mask[idx[tr_end:va_end]] = True
    data.test_mask[idx[va_end:]] = True


def run_experiment_1(cfg: Dict[str, Any]):
    print("\n===== EXPERIMENT 1 – Synthetic depth adaptivity =====")
    data = load_dataset("synthetic_chain_core", **cfg["dataset"])

    # fabricate train / val / test splits
    _fabricate_masks(data)

    hidden = 128
    results = []

    for model_name in cfg["models"]:
        if model_name not in _MODEL_REGISTRY:
            print(f"Model {model_name} not implemented – skipping.")
            continue
        print(f"\n--- {model_name} ---")
        model = _MODEL_REGISTRY[model_name](data.num_node_features, hidden, len(torch.unique(data.y)), cfg)
        metrics = full_train(model, data, CONFIG, pathlib.Path("runs/exp1") / model_name, f"{model_name}_exp1")
        results.append((model_name, metrics))
        print(json.dumps(metrics, indent=2))

    print("\nSUMMARY – Experiment 1")
    for name, m in results:
        print(f"{name:<12}  acc={m['accuracy']:.3f}  rowDiff={m['row_diff']:.3f}  effRank={m['eff_rank']:.1f}")
    print("Figures written to .research/iteration1/images/")


def run_experiment_2(cfg: Dict[str, Any]):
    print("\n===== EXPERIMENT 2 – Real-world benchmark suite =====")
    hidden = 128
    for ds_name in cfg["datasets"]:
        print(f"\nDataset: {ds_name}")
        dataset = load_dataset(ds_name)
        data = dataset[0] if hasattr(dataset, "num_classes") else dataset
        num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else len(torch.unique(data.y))

        if not hasattr(data, "train_mask"):
            _fabricate_masks(data)

        for model_name in cfg["models"]:
            if model_name not in _MODEL_REGISTRY:
                continue
            tag = f"{model_name}_{ds_name}"
            model = _MODEL_REGISTRY[model_name](data.num_node_features, hidden, num_classes, cfg)
            metrics = full_train(model, data, CONFIG, pathlib.Path("runs/exp2") / ds_name / model_name, tag)
            print(f"{tag:<25} – acc {metrics['accuracy']:.3f}")


def main():
    start = time.time()
    run_experiment_1(CONFIG["experiments"]["exp1"])
    run_experiment_2(CONFIG["experiments"]["exp2"])
    elapsed = (time.time() - start) / 60
    print(f"All experiments completed in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
