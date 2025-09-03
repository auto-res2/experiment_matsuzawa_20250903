"""src/evaluate.py
All functions that *evaluate* learners, run the three experiments and produce
publication-ready plots live here.
"""
from __future__ import annotations

import csv
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Fail-fast dependency check.  Users are informed how to install missing
#  packages instead of being greeted by an undecipherable `ImportError` later.
# ---------------------------------------------------------------------------
REQ_PKGS = {
    "avalanche": "pip install avalanche-lib",
    "advertorch": "pip install advertorch",
    "taming": "pip install git+https://github.com/CompVis/taming-transformers.git",
    "hatem": "pip install git+https://github.com/HATEM-project/hatem-cl.git",
    "derpp": "pip install git+https://github.com/aimagelab/mammoth.git",
    "adaptive_quantization": "pip install git+https://github.com/pclucas14/adaptive-quantization-modules.git",
    "rar": "pip install git+https://github.com/YaqianZhang/RepeatedAugmentedRehearsal.git",
}
for _pkg, _hint in REQ_PKGS.items():
    try:
        __import__(_pkg)
    except ImportError as e:
        raise RuntimeError(f"Required package '{_pkg}' missing – install via `{_hint}`") from e

from avalanche.benchmarks.utils import SplitStrategy  # noqa: F401  – used by third-party learners
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics  # noqa: F401
from avalanche.training.plugins import EvaluationPlugin  # noqa: F401
from avalanche.logging import InteractiveLogger  # noqa: F401

from hatem import HATEMBuffer, HATEMLearner  # noqa: F401 – used at run-time
from derpp import DERPlusPlusLearner  # noqa: F401 – used at run-time
from adaptive_quantization import AQMLearner  # noqa: F401 – used at run-time
from rar import RARLearner  # noqa: F401 – used at run-time

from .train import build_resnet18, sanity_check

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _dump_csv(fname: str, rows: List[Dict]):
    """Write a list-of-dicts to *fname* (CSV)."""
    if not rows:
        return
    with open(fname, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", fname)


# ---------------------------------------------------------------------------
#  EXP-1 – Accuracy & Forgetting
# ---------------------------------------------------------------------------

def _instantiate_learner(tag: str, cls, seed: int, device: torch.device, budget: int, dspec: Dict):
    """Factory that wraps the multiple *ER* modes so callers do not have to
    remember the magic keyword arguments each time."""
    if tag == "HATEM":
        return HATEMLearner(seed=seed, device=device, budget=budget, dspec=dspec)
    if tag == "ER":
        return DERPlusPlusLearner(seed=seed, device=device, budget=budget, dspec=dspec, der_mode="raw")
    if tag == "VQ-VAE":
        return DERPlusPlusLearner(seed=seed, device=device, budget=budget, dspec=dspec, der_mode="vqvae")
    # regular class instantiation
    return cls(seed=seed, device=device, budget=budget, dspec=dspec)


def _plot_exp1(results: Dict[str, List[float]], budget: int, dataset: str, plot_dir: Path):
    xs = np.arange(len(results))
    plt.figure(figsize=(7, 4))
    for i, (m, vals) in enumerate(results.items()):
        y = vals[0]
        plt.bar(i, y, label=m)
        plt.text(i, y + 0.5, f"{y:.1f}", ha="center")
    plt.xticks(xs, results.keys(), rotation=45)
    plt.ylabel("Average Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(f"AACC @ {budget // 1024} KB – {dataset}")
    plt.legend()
    fn = plot_dir / f"accuracy_{dataset}_{budget // 1024}KB.pdf"
    plt.savefig(fn, bbox_inches="tight")
    print("Figure saved:", fn)


def run_exp1(cfg: Dict):
    print("\n================ EXP-1: Accuracy & Forgetting ================")
    device = torch.device(cfg["device"]) if cfg["device"] != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # quick sanity-check to prevent wasting GPU time on a broken env
    sanity_check(device, cfg)

    METHODS = {
        "HATEM": HATEMLearner,
        "ER": "raw",  # handled in _instantiate_learner
        "DER++": DERPlusPlusLearner,
        "AQM": AQMLearner,
        "RAR": RARLearner,
        "VQ-VAE": "vqvae",  # handled in _instantiate_learner
    }

    plot_dir = Path(cfg["plots"])
    _ensure_dir(plot_dir)

    for dataset_name, dspec in cfg["datasets"].items():
        for budget in cfg["budgets_B"]:
            print(f"\nDataset={dataset_name}, Budget={budget // 1024} KB")
            results: Dict[str, List[float]] = {m: [] for m in METHODS}
            for m, cls in METHODS.items():
                aacc, f_vals = [], []
                for seed in cfg["seeds"]:
                    learner = _instantiate_learner(m, cls, seed, device, budget, dspec)
                    acc, forgetting = learner.run()  # type: ignore – third-party API
                    aacc.append(acc)
                    f_vals.append(forgetting)
                mu, sigma = st.mean(aacc), st.stdev(aacc)
                results[m] = [mu, sigma, st.mean(f_vals)]
                print(f"{m:<7}  AACC={mu:.2f}±{sigma:.2f}  F↓={results[m][2]:.2f}")
            _plot_exp1(results, budget, dataset_name, plot_dir)


# ---------------------------------------------------------------------------
#  EXP-2 – Throughput & Memory
# ---------------------------------------------------------------------------

def _plot_exp2(log: List[Dict], plot_dir: Path):
    metrics: Dict[str, List[float]] = {}
    for v in log:
        metrics.setdefault(v["variant"], []).append(v["write_ms"])

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(metrics))
    for i, (var, arr) in enumerate(metrics.items()):
        mu, sigma = st.mean(arr), st.stdev(arr)
        plt.bar(i, mu, yerr=sigma, capsize=5, label=var)
        plt.text(i, mu + 0.2, f"{mu:.2f}")
    plt.xticks(xs, metrics.keys())
    plt.ylabel("Write latency (ms)")
    plt.title("Exp-2: Mean write latency")
    plt.legend()
    fn = plot_dir / "write_latency.pdf"
    plt.savefig(fn, bbox_inches="tight")
    print("Figure saved:", fn)


def run_exp2(cfg: Dict):
    print("\n================ EXP-2: Throughput & Memory ================")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for timing study (Exp-2)")
    device = torch.device("cuda")

    variants = {"HATEM": HATEMLearner, "ER": "raw", "VQ-VAE": "vqvae"}
    budget = 524_288  # fixed 0.5 MB
    timing_log: List[Dict] = []

    for m, cls in variants.items():
        for seed in cfg["seeds"]:
            learner = _instantiate_learner(m, cls, seed, device, budget, cfg["datasets"]["CIFAR100"])
            timing_csv = learner.run_timing()  # type: ignore – external API
            timing_log.extend(timing_csv)

    _dump_csv("exp2_timing.csv", timing_log)
    _plot_exp2(timing_log, Path(cfg["plots"]))


# ---------------------------------------------------------------------------
#  EXP-3 – Drift & Robustness
# ---------------------------------------------------------------------------

def _make_variant(name, obj, seed, device, budget, dspec):
    if name == "ER":
        return DERPlusPlusLearner(seed=seed, device=device, budget=budget, dspec=dspec, der_mode="raw")
    if isinstance(obj, tuple):
        cls, kw = obj
        return cls(seed=seed, device=device, budget=budget, dspec=dspec, **kw)
    return obj(seed=seed, device=device, budget=budget, dspec=dspec)


def _plot_exp3(res: Dict[str, Tuple[float, float, float]], plot_dir: Path):
    xs = np.arange(len(res))
    plt.figure(figsize=(7, 4))
    clean = [v[0] for v in res.values()]
    drift = [v[1] for v in res.values()]
    pgd = [v[2] for v in res.values()]
    plt.bar(xs - 0.25, clean, width=0.25, label="Clean")
    plt.bar(xs, drift, width=0.25, label="Drift Δ")
    plt.bar(xs + 0.25, pgd, width=0.25, label="PGD-10")
    for i, (c, d, p) in enumerate(zip(clean, drift, pgd)):
        plt.text(i - 0.25, c + 0.3, f"{c:.1f}")
        plt.text(i, d + 0.3, f"{d:.1f}")
        plt.text(i + 0.25, p + 0.3, f"{p:.1f}")
    plt.xticks(xs, res.keys())
    plt.ylabel("Accuracy (%) / Δpp")
    plt.title("Exp-3: Robustness Ablations")
    plt.legend()
    fn = plot_dir / "robustness_tinyIN.pdf"
    plt.savefig(fn, bbox_inches="tight")
    print("Figure saved:", fn)


def run_exp3(cfg: Dict):
    print("\n================ EXP-3: Drift & Robustness ================")
    device = torch.device(cfg["device"]) if cfg["device"] != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    VARIANTS: Dict[str, Union[type, Tuple]] = {
        "Full": HATEMLearner,
        "–Tier2": (HATEMLearner, {"disable_proto": True}),
        "EncTune": (HATEMLearner, {"finetune_encoder": True}),
        "Token512": (HATEMLearner, {"vocab": 512}),
        "ER": "raw",
    }

    dspec = cfg["datasets"]["TinyIN"]
    budget = 524_288
    all_results: Dict[str, Tuple[float, float, float]] = {}

    for v, obj in VARIANTS.items():
        drops, pgd_acc, final_acc = [], [], []
        for seed in cfg["seeds"]:
            learner = _make_variant(v, obj, seed, device, budget, dspec)
            out = learner.run_drift_and_pgd()  # type: ignore – external API
            drops.append(out[0])
            pgd_acc.append(out[1])
            final_acc.append(out[2])
        all_results[v] = (st.mean(final_acc), st.mean(drops), st.mean(pgd_acc))
        print(
            f"{v:<9}  Clean={all_results[v][0]:.2f}  DriftΔ={all_results[v][1]:.2f}  PGD={all_results[v][2]:.2f}"
        )
    _plot_exp3(all_results, Path(cfg["plots"]))
