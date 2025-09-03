"""src/main.py – entry-point that orchestrates the experiment using the
refactored project structure. Configuration is loaded from ../config/config.yaml
(relative to *this* file).
"""
from __future__ import annotations

import json
import pathlib
import sys
import traceback

import torch
import yaml

# -----------------------------------------------------------------------------
# import project modules *after* adding src/ to sys.path
# -----------------------------------------------------------------------------
SRC_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

sys.path.insert(0, str(SRC_DIR))  # ensures `import src.*` works in notebooks too

from src.preprocess import download_and_prepare, split_cifar100
from src.train import Trainer


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing configuration at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    print("================ Experiment – LoRA-DiffMem vs baselines ================")
    print(json.dumps(cfg["experiment_1"], indent=2))

    try:
        download_and_prepare()
        tasks, testset = split_cifar100()

        trainer = Trainer(cfg, device="cuda" if torch.cuda.is_available() else "cpu")
        out_dir = PROJECT_ROOT / ".research" / "iteration1" / "images"
        trainer.train_stream(tasks, testset, method="lora_diffmem", seed=42, out_dir=out_dir)
    except Exception as e:  # pylint: disable=broad-except
        print("\n*** FATAL: experiment aborted ***")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
