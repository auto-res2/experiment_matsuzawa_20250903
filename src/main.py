"""src/main.py
Entry-point orchestrating the experimental workflow.
Call   python -m src.main
"""
from __future__ import annotations
import os
import json
import pathlib
import yaml

from .preprocess import build_synthetic_cc
from .train import build_model, train_single_run
from .utils import ensure_dir

################################################################################
#  Configuration                                                               #
################################################################################
CFG_PATH = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
CONFIG = yaml.safe_load(CFG_PATH.read_text())

################################################################################
#  Experiment-1 : Synthetic core-vs-chain                                     #
################################################################################

def experiment_1(cfg: dict) -> None:
    print("\n=====================  EXPERIMENT 1 – Synthetic  =====================")
    exp_name = "exp1"
    for seed in CONFIG["common"]["seeds"]:
        data = build_synthetic_cc(seed=seed, **cfg["dataset"])
        print(f"Seed {seed:02d} | Nodes={data.num_nodes} | Edges={data.num_edges}")
        for model_tag in cfg["models"]:
            metrics_path = pathlib.Path("runs") / exp_name / model_tag / f"seed{seed}" / "metrics.json"
            if metrics_path.exists():
                print(metrics_path.read_text())
                continue
            model = build_model(model_tag, data.num_node_features, cfg["hidden"], 10)
            train_single_run(model, data, CONFIG, seed, exp_name, model_tag)

################################################################################
#  Main                                                                        #
################################################################################

def main() -> None:
    ensure_dir(pathlib.Path("runs"))
    experiment_1(CONFIG["experiments"]["exp1"])
    # Real-world experiments are heavy; guard them with an env-flag.
    if os.getenv("RUN_EXP2") == "1":
        from .run_real import run_all  # pragma: no cover – heavy optional import
        run_all(CONFIG)


if __name__ == "__main__":
    main()
