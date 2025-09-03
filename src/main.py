"""Entry point – orchestrates the experimental workflow.
Run with:  python -m src.main
"""
import os
import random
import pathlib
import json
from typing import Dict, List

import yaml
import torch

# ----------------------------------------------------------------------------
# Relative imports from sibling modules
from .train import AP_LIB_GCN, train_step, Timer
from .evaluate import evaluate, effective_rank, save_lineplot
from .preprocess import load_dataset

# ----------------------------------------------------------------------------
# Load configuration ---------------------------------------------------------
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# Ensure output directories exist -------------------------------------------
IMG_DIR = pathlib.Path(__file__).resolve().parent.parent / ".research" / "iteration1" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = CFG["data_root"]


def _seed_all(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Experiment 1: Depth-Scalability & Over-smoothing Resistance ----------------

def experiment1(device):
    conf = CFG["experiments"]["exp1"]
    print("\n==== Experiment 1 –", conf["name"], "====")
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for dname in conf["datasets"]:
        ds = load_dataset(dname, DATA_ROOT)
        data = ds[0].to(device)
        results[dname] = {}

        for depth in conf["depth_grid"]:
            torch.cuda.empty_cache()
            try:
                model = AP_LIB_GCN(
                    ds.num_node_features,
                    conf["hidden"],
                    ds.num_classes,
                    depth,
                    CFG["aplib"]["k_max"],
                    CFG["aplib"]["tau"],
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=conf["lr"],
                    weight_decay=conf["weight_decay"],
                )
                criterion = torch.nn.CrossEntropyLoss()

                best_val, best_test = 0.0, 0.0
                patience = 0

                with Timer() as t_total:
                    for epoch in range(conf["epochs"]):
                        train_step(
                            model,
                            data,
                            optimizer,
                            criterion,
                            (
                                CFG["aplib"]["lambda_mix"],
                                CFG["aplib"]["lambda_denoise"],
                                CFG["aplib"]["lambda_l0"],
                            ),
                            device,
                        )

                        accs, logits = evaluate(model, data, device)
                        if accs["val"] > best_val:
                            best_val, best_test = accs["val"], accs["test"]
                            patience = 0
                        else:
                            patience += 1
                        if patience >= conf["patience"]:
                            break

                eff_rank = effective_rank(logits.detach())
                res = {
                    "test_acc": round(best_test * 100, 2),
                    "val_best": round(best_val * 100, 2),
                    "eff_rank": round(eff_rank, 3),
                    "train_time_s": round(t_total.seconds, 1),
                }
                print(f"{dname} depth={depth}:", res)
                results[dname][depth] = res
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at depth {depth} for {dname}, skipping deeper depths …")
                    break
                raise e

        # Plot accuracy vs depth --------------------------------------------
        depths = list(results[dname].keys())
        accs = [results[dname][d]["test_acc"] for d in depths]
        out_path = IMG_DIR / f"accuracy_{dname}.pdf"
        save_lineplot(
            depths,
            {"AP-LIB": accs},
            "Depth",
            "Test Acc (%)",
            f"{dname} – accuracy vs depth",
            out_path,
        )

    # Store raw results ------------------------------------------------------
    with open("exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ----------------------------------------------------------------------------
# MAIN -----------------------------------------------------------------------

def main():
    device = torch.device(CFG["device"] if CFG["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _seed_all(0)

    # Run experiments --------------------------------------------------------
    experiment1(device)
    # TODO: experiment2(), experiment3() (left out for brevity)


if __name__ == "__main__":
    main()
