"""
main.py – entry point (`python -m src.main`)
Loads YAML configuration, prepares data, model and starts training.
"""
import argparse
import time
import random
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .preprocess import get_waterbirds_splits
from .train import AutoCFDiffClassifier, ERMClassifier, Trainer
from .evaluate import save_bar


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def _dict_to_ns(obj):
    """Recursively converts dicts/lists into SimpleNamespace for convenient dot access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_ns(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_experiment(cfg):
    print("========== EXPERIMENT ==========")
    print("ID      :", cfg.exp_id)
    print("Backbone:", cfg.model.backbone, "   AutoCF-Diff:", cfg.autocf.enabled)
    print("Datasets:", [d.name for d in cfg.datasets])
    print("================================\n")

    # Disable AutoCF-Diff if no CUDA is available to avoid heavy diffusion downloads
    if cfg.autocf.enabled and not torch.cuda.is_available():
        print("[Warning] CUDA not available – disabling AutoCF-Diff (set autocf.enabled=false in config to suppress).")
        cfg.autocf.enabled = False

    # Currently only Waterbirds is implemented.
    train_ds, val_ds, test_ds = get_waterbirds_splits()
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)

    model = AutoCFDiffClassifier(cfg) if cfg.autocf.enabled else ERMClassifier(cfg)
    trainer = Trainer(cfg, model, (train_loader, val_loader, test_loader))
    test_stats = trainer.fit()

    print("---- FINAL RESULTS ----")
    print(test_stats)

    # Store bar plot under the mandated directory
    save_bar({"Worst-Group-Acc": test_stats["wga"], "Avg-Acc": test_stats["acc"]},
             title="Waterbirds performance", fname=Path("accuracy.pdf"))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser(description="AutoCF-Diff experimental runner")
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file")
    p.add_argument("--fast_dev_run", action="store_true", help="1 batch / 1 epoch smoke test")
    args = p.parse_args()

    # ---------------------------------------------------------------------
    # Load config YAML → SimpleNamespace
    # ---------------------------------------------------------------------
    with open(args.config, "r") as fp:
        cfg_dict = yaml.safe_load(fp)
    if args.fast_dev_run:
        cfg_dict.setdefault("train", {})["fast_dev_run"] = True
    cfg = _dict_to_ns(cfg_dict)

    # seed & output dir
    set_seed(cfg.train.seed)
    Path(cfg.runtime.save_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run_experiment(cfg)
    print(f"Total runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    cli()
