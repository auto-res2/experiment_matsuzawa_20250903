"""Main entry-point: *python -m src.main*"""

from __future__ import annotations

import json
import pathlib
from typing import Dict

import yaml
import torch_geometric

from . import preprocess as P
from . import models as M
from . import train as T

# --------------------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------------------
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
CONFIG: Dict = yaml.safe_load(CONFIG_PATH.read_text())


# --------------------------------------------------------------------------------------
#  Model factory
# --------------------------------------------------------------------------------------

def _build(model_name: str, in_dim: int, hidden: int, out_dim: int):
    if model_name == "gcn_deep":
        return M.DeepGCN(in_dim, hidden, out_dim)
    if model_name == "gat_deep":
        return M.DeepGAT(in_dim, hidden, out_dim)
    if model_name == "gcnii":
        return M.GCNII(in_dim, hidden, out_dim)
    if model_name == "pairnorm_si":
        return M.PairNormGCN(in_dim, hidden, out_dim)
    if model_name == "apd_gcn":
        return M.APD("gcn", in_dim, hidden, out_dim)
    if model_name == "apd_gat":
        return M.APD("gat", in_dim, hidden, out_dim)
    raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------------------------------------------
#  Experiments
# --------------------------------------------------------------------------------------

def experiment1(cfg):
    print(cfg["description"])

    runs_root = pathlib.Path("runs/exp1")

    for seed in CONFIG["common"]["seeds"]:
        # synthetic benchmark -----------------------------------------------------------
        data = P.build_synthetic_cc(seed=seed, **cfg["dataset"])
        print(f"\nSeed {seed}")

        for model_name in cfg["models"]:
            model = _build(model_name, data.num_features, cfg["hidden"], 10)
            run_dir = runs_root / model_name / f"seed{seed}"

            metrics = T.fit(model, data, CONFIG, run_dir, seed)
            print(json.dumps({"model": model_name, **metrics}, indent=2))

    print(
        "Figures saved under .research/iteration7/images – files follow the naming "
        "convention <topic>.pdf"
    )


def main() -> None:
    experiment1(CONFIG["experiments"]["exp1"])


if __name__ == "__main__":
    main()