"""src/main.py
Light-weight launcher that orchestrates the three experiments.  The heavy
computations are *optional* – if one of the extra dependencies cannot be
imported in the current environment the script degrades gracefully and only
prints a warning instead of crashing the whole program.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import yaml

from .evaluate import run_exp1, run_exp2, run_exp3


CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def _load_cfg() -> dict:
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CFG_PATH}")
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


def main():
    torch.backends.cudnn.benchmark = True  # speed-up for fixed input sizes

    cfg = _load_cfg()

    # Ensure plot directory exists (even if the real experiments are skipped).
    plot_dir = Path(cfg["plots"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # ----------------  Run experiments (may be stubs)  -------------------
    run_exp1(cfg)
    run_exp2(cfg)
    run_exp3(cfg)

    elapsed_h = (time.time() - start) / 3600
    print(f"\nDone. Total wall-clock time: {elapsed_h:.2f} h")


if __name__ == "__main__":
    main()
