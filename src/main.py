"""src/main.py
Entry-point that orchestrates the three experiments described in the paper. It
loads the YAML configuration, ensures output directories exist, sets sensible
cuDNN flags and defers the heavy lifting to `src.evaluate`.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml

from .evaluate import run_exp1, run_exp2, run_exp3


def _load_cfg() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("Configuration file not found: " + str(cfg_path))
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def main():
    torch.backends.cudnn.benchmark = True  # speed-up for fixed input sizes

    cfg = _load_cfg()

    # Ensure plot directory exists (helps when running on cluster nodes).
    plot_dir = Path(cfg["plots"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    run_exp1(cfg)
    run_exp2(cfg)
    run_exp3(cfg)

    total_h = (time.time() - t0) / 3600
    print(f"\nAll experiments finished in {total_h:.2f} h")


if __name__ == "__main__":
    main()
