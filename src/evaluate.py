from __future__ import annotations
"""
evaluate.py – training/evaluation orchestration + plotting helpers
"""
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 (kept for styling consistency)

from .train import (
    DEVICE,
    JEMBModel,
    InfLoRA,
    AQM_ER,
)
from .preprocess import build_task_stream

# ---------------------------------------------------------------------
#  Directories (created on import)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
# --------- UPDATED TO ITERATION 11 AS REQUIRED -----------------------
IMG_DIR = ROOT / ".research/iteration11/images"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
#  Light log container
# ---------------------------------------------------------------------
@dataclass
class Log:
    acc: List[float] = field(default_factory=list)
    bytesA: List[int] = field(default_factory=list)
    bytesB: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------
#  Core training-and-evaluation routine for *one* method
# ---------------------------------------------------------------------

def run_method(
    name: str,
    ModelCls,
    stream,
    cfg_train: dict,
    seed: int = 0,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ModelCls(num_cls=cfg_train["classes_per_task"], cap_bytes=cfg_train["budget_bytes"]).to(DEVICE)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg_train["lr"],
        momentum=cfg_train["momentum"],
        weight_decay=cfg_train["weight_decay"],
    )

    log: Log = Log()
    test_loaders = []

    for tid, (tr, va, te) in enumerate(stream, 1):
        trL = DataLoader(tr, batch_size=cfg_train["batch_size"], shuffle=True, num_workers=2)
        vaL = DataLoader(va, batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2)
        teL = DataLoader(te, batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2)
        test_loaders.append(teL)

        for _ in range(cfg_train["epochs_per_task"]):
            model.train_epoch(trL, opt)
            model.realloc(vaL)

        # ------ evaluation on accumulated tasks -----------------------
        accs = [model.eval_loader(l) for l in test_loaders]
        log.acc.append(sum(accs) / len(accs))
        log.bytesA.append(model.ledger.A)
        log.bytesB.append(model.ledger.B)
        print(f"{name:8s} task {tid:02d}  avg-acc={log.acc[-1]:5.2f}  A={model.ledger.A}  B={model.ledger.B}")

    summary = {
        "method": name,
        "A_T": log.acc[-1],
        "bytes_adapter": log.bytesA[-1],
        "bytes_buffer": log.bytesB[-1],
    }
    return summary, log


# ---------------------------------------------------------------------
#  Plot helper (publication quality)
# ---------------------------------------------------------------------

def plot_curve(logs: Dict[str, Log]):
    plt.figure(figsize=(6, 3))
    for name, l in logs.items():
        plt.plot(l.acc, label=name)
        for i, v in enumerate(l.acc):
            plt.text(i, v + 0.3, f"{v:.1f}", fontsize=6)
    plt.xlabel("Task")
    plt.ylabel("Average Accuracy (%)")
    plt.legend()
    plt.grid(True)
    fname = IMG_DIR / "accuracy_curve_ci.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print("[fig] saved →", fname)


# ---------------------------------------------------------------------
#  High-level experiment wrapper used by main.py
# ---------------------------------------------------------------------

def run_experiment(cfg: dict):
    """Top-level experiment driver invoked from src.main."""
    from .train import run_unit_tests  # local import to avoid circularity

    print("\n" + cfg["experiment"]["description"] + "\n")
    run_unit_tests()
    start = time.time()

    # ---------- build task stream -------------------------------------
    stream = build_task_stream(
        data_dir=Path(cfg["dataset"]["data_root"]),
        mean=cfg["dataset"]["mean"],
        std=cfg["dataset"]["std"],
        num_tasks=cfg["dataset"]["num_tasks"],
        classes_per_task=cfg["dataset"]["classes_per_task"],
        seed=cfg["experiment"]["seed"],
    )

    logs: Dict[str, Log] = {}
    results: List[dict] = []

    name2cls = {
        "jemb": JEMBModel,
        "inflora": InfLoRA,
        "aqm_er": AQM_ER,
    }

    for m in cfg["experiment"]["methods"]:
        res, lg = run_method(m, name2cls[m], stream, cfg["training"], seed=cfg["experiment"]["seed"])
        logs[m] = lg
        results.append(res)

    # --------------- persist raw metrics ------------------------------
    out = {
        "description": cfg["experiment"]["description"],
        "per_task": {k: lg.__dict__ for k, lg in logs.items()},
        "summary": results,
        "wall_clock_s": round(time.time() - start, 2),
    }
    (RUNS_DIR / "run_ci.json").write_text(json.dumps(out, indent=2))
    print("\n[results]", json.dumps(results, indent=2))
    print("[saved] runs/run_ci.json")

    # plot -------------------------------------------------------------
    plot_curve(logs)
