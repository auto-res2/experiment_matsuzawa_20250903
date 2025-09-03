"""
main.py – top-level orchestration
Run   $ python -m src.main   from project root.
"""
from __future__ import annotations
import json
import pathlib
from typing import Dict, List

import yaml
import torch

from . import preprocess as P
from . import train as T

# ---------------------------------------------------------------------------
#  Load configuration                                                         #
# ---------------------------------------------------------------------------
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
CFG = yaml.safe_load(CONFIG_PATH.read_text())


###############################################################################
#  Utilities                                                                  #
###############################################################################

def _build_model(name: str, f_in: int, hid: int, f_out: int):
    """Factory that delegates to the correct ctor in train.py (where models live)."""
    from .train import (
        APD,
        DeepGAT,
        DeepGCN,
        GCNII,
        PairNormGCN,
    )

    factories = {
        "gcn_deep": lambda: DeepGCN(f_in, hid, f_out),
        "gat_deep": lambda: DeepGAT(f_in, hid, f_out),
        "gcnii": lambda: GCNII(f_in, hid, f_out),
        "pairnorm_si": lambda: PairNormGCN(f_in, hid, f_out),
        "apd_gcn": lambda: APD("gcn", f_in, hid, f_out),
        "apd_gat": lambda: APD("gat", f_in, hid, f_out),
    }
    return factories[name]()


###############################################################################
#  Experiment-1 – synthetic benchmark                                         #
###############################################################################

def experiment1(cfg: Dict):
    print("\n==== Experiment-1 – Synthetic core-vs-chain benchmark ====")

    for seed in CFG["common"]["seeds"]:
        data = P.build_synthetic_cc(seed=seed, **cfg["dataset"])
        print(f"\nSeed {seed}")

        for m_name in cfg["models"]:
            run_dir = pathlib.Path(f"runs/exp1/{m_name}/seed{seed}")
            if run_dir.exists():
                res = torch.load(run_dir / "checkpoint.pt")["metrics"]
            else:
                model = _build_model(m_name, data.num_features, cfg["hidden"], 10)
                res = T.fit(model, data, CFG, run_dir, seed)
            print(json.dumps({"model": m_name, **res}, indent=2))

    print(
        "Figures have been saved under .research/iteration18/images – see training_loss.pdf & accuracy.pdf."
    )


###############################################################################
#  Main                                                                       #
###############################################################################

def main():
    experiment1(CFG["experiments"]["exp1"])
    # experiment2() & experiment3() would follow a similar pattern – omitted for brevity


if __name__ == "__main__":
    main()
