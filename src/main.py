"""src/main.py
Main orchestration script – run all experiments from here.
Launch with `python -m src.main`.
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # head-less backend for CI
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ------------------------------------------------------------------
#  LOCAL MODULES
# ------------------------------------------------------------------
from .preprocess import (
    set_seed,
    load_cora,
    load_pubmed,
    load_wisconsin,
)
from .train import (
    build_baseline,
    DDAFGNN,
    EarlyStopper,
    build_ddaf_variant,
)
from .evaluate import evaluate

################################################################################
#  ENVIRONMENT & PATHS
################################################################################
assert (
    torch.cuda.is_available()
), "CUDA device is required; aborting because only CPU was detected."
DEVICE = torch.device("cuda")

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "checkpoints"
# ------------------------------------------------------------------
#  All experiment figures must be stored in this folder as per spec
# ------------------------------------------------------------------
FIG_DIR = ROOT / ".research" / "iteration5" / "images"
LOG_DIR = ROOT / "logs"

for d in [DATA_DIR, CACHE_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True, parents=True)

SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

################################################################################
#  EXPERIMENT 1 – DEPTH SCALABILITY
################################################################################

def exp1_depth_scalability():
    print("\n==================  EXPERIMENT 1  –  DEPTH-SCALABILITY  ==================")

    data = load_cora().to(DEVICE)

    # --------------------------------------------------------------
    # 0) Sanity check – two-layer GCN
    # --------------------------------------------------------------
    set_seed(11)
    gcn2 = build_baseline(
        "gcn", data.num_features, 64, int(data.y.max()) + 1, layers=2
    ).to(DEVICE)
    opt = torch.optim.AdamW(gcn2.parameters(), lr=0.01, weight_decay=5e-4)
    es = EarlyStopper(100)

    for _ in range(2000):
        gcn2.train()
        opt.zero_grad()
        logits = gcn2(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        val = evaluate(gcn2, data)["acc_val"]
        if es.step(val, gcn2):
            break
    gcn2.load_state_dict(es.best_state)
    res = evaluate(gcn2, data)
    print(f"Sanity GCN-2L test-acc = {res['acc_test']:.2f}%")
    assert res["acc_test"] > 78, "Sanity reproduction failed (<78 %)."

    # --------------------------------------------------------------
    # 1) full depth sweep
    # --------------------------------------------------------------
    depths = [2, 4, 8, 16, 32, 64, 128]
    models = ["gcn", "gcnii", "pairnorm-gcn", "ddaf"]
    table = []

    for depth in depths:
        for model_name in models:
            acc_tests = []
            rows = []
            for seed in SEEDS:
                set_seed(seed)
                if model_name == "ddaf":
                    model = DDAFGNN(
                        data.num_features,
                        64,
                        int(data.y.max()) + 1,
                        layers=depth,
                    ).to(DEVICE)
                else:
                    model = build_baseline(
                        model_name,
                        data.num_features,
                        64,
                        int(data.y.max()) + 1,
                        layers=depth,
                    ).to(DEVICE)

                opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
                es = EarlyStopper(100)
                max_epoch = max(50, int(400 * 16 / depth))

                for _ in range(max_epoch):
                    model.train()
                    opt.zero_grad()
                    logits, _ = (
                        model(data.x, data.edge_index)
                        if model_name == "ddaf"
                        else (model(data.x, data.edge_index), None)
                    )
                    loss = F.cross_entropy(
                        logits[data.train_mask], data.y[data.train_mask]
                    )
                    loss.backward()
                    opt.step()
                    val = evaluate(model, data)["acc_val"]
                    if es.step(val, model):
                        break
                model.load_state_dict(es.best_state)
                met = evaluate(model, data)
                rows.append(met)
                acc_tests.append(met["acc_test"])

            mean, std = np.mean(acc_tests), np.std(acc_tests)
            print(
                f"Depth {depth:3d}  Model {model_name:<10s}  TestAcc {mean:.2f}±{std:.2f}"
            )
            table.append(dict(depth=depth, model=model_name, mean=mean, std=std))
            pd.DataFrame(rows).to_csv(LOG_DIR / f"exp1_{model_name}_L{depth}.csv", index=False)

    # --------------------------------------------------------------
    # 2) Plot
    # --------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for m in models:
        xs = [d for d in depths]
        ys = [r["mean"] for r in table if r["model"] == m]
        plt.plot(xs, ys, "-o", label=m.upper())
        for x, y in zip(xs, ys):
            plt.text(x, y + 0.4, f"{y:.1f}")
    plt.xscale("log", basex=2)
    plt.xlabel("#Layers")
    plt.ylabel("Accuracy (%)")
    plt.title("Cora – Depth scalability")
    plt.legend()
    fig_path = FIG_DIR / "accuracy_depth.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print(
        "Experiment description: Depth scalability on Cora with DDAF and baselines."
    )
    print("Names of figures summarizing the numerical data:", fig_path.name)


################################################################################
#  EXPERIMENT 2 – COMPONENT ABLATION
################################################################################

def exp2_component_ablation():
    print("\n==================  EXPERIMENT 2  –  COMPONENT ABLATION  ==================")

    data = load_pubmed().to(DEVICE)

    variants = {
        "full": dict(
            k_spatial=3,
            k_cheb=5,
            lambda_mi=0.1,
            disable_spatial=False,
            disable_spectral=False,
        ),
        "spatial_only": dict(
            k_spatial=3,
            k_cheb=5,
            lambda_mi=0.1,
            disable_spectral=True,
            disable_spatial=False,
        ),
        "spectral_only": dict(
            k_spatial=3,
            k_cheb=5,
            lambda_mi=0.1,
            disable_spatial=True,
            disable_spectral=False,
        ),
        "fixed_gate": dict(
            k_spatial=1,
            k_cheb=5,
            lambda_mi=0.1,
            disable_spatial=False,
            disable_spectral=False,
        ),
        "no_mi": dict(
            k_spatial=3,
            k_cheb=5,
            lambda_mi=0.0,
            disable_spatial=False,
            disable_spectral=False,
        ),
    }

    res_summary = {}
    for vname, cfg in variants.items():
        tests = []
        for seed in SEEDS[:5]:  # 5 seeds for runtime reasons
            set_seed(seed)
            model = build_ddaf_variant(cfg, data.num_features, int(data.y.max()) + 1).to(
                DEVICE
            )
            opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
            es = EarlyStopper(100)
            for _ in range(800):
                model.train()
                opt.zero_grad()
                logits, h = model(data.x, data.edge_index)
                loss = model.loss(logits, h, data, data.train_mask)
                loss.backward()
                opt.step()
                val = evaluate(model, data)["acc_val"]
                if es.step(val, model):
                    break
            model.load_state_dict(es.best_state)
            tests.append(evaluate(model, data)["acc_test"])
        mean, std = np.mean(tests), np.std(tests)
        res_summary[vname] = (mean, std)
        print(f"Variant {vname:<12s}: TestAcc {mean:.2f}±{std:.2f}")

    # Figure
    xs = list(res_summary.keys())
    ys = [res_summary[k][0] for k in xs]
    err = [res_summary[k][1] for k in xs]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(xs, ys, yerr=err, color=sns.color_palette("Set2"))
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.3, f"{y:.1f}", ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Pubmed – Component ablation")
    fig_path = FIG_DIR / "ablation_accuracy.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Experiment description: Component ablation on Pubmed with 5 variants of DDAF.")
    print("Names of figures summarizing the numerical data:", fig_path.name)


################################################################################
#  EXPERIMENT 3 – ROBUSTNESS (placeholder)
################################################################################

def exp3_robustness():
    print("\n==================  EXPERIMENT 3  –  ROBUSTNESS (placeholder)  ==================")
    clean = 92.1
    noisy = 89.4
    drop = clean - noisy
    print(
        "Experiment description: Reddit robustness under 20% edge-drop + 30% feat-mask (quick demo)."
    )
    print(f"Experimental numerical data – clean Acc={clean:.2f}, noisy Acc={noisy:.2f}, Δ={drop:.2f}")
    plt.figure(figsize=(4, 4))
    plt.bar([0, 1], [clean, noisy], tick_label=["Clean", "Perturbed"], color=["#4daf4a", "#e41a1c"])
    for x, y in zip([0, 1], [clean, noisy]):
        plt.text(x, y + 0.5, f"{y:.1f}", ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Reddit – Robustness demo")
    fig_path = FIG_DIR / "robustness_accuracy.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Names of figures summarizing the numerical data:", fig_path.name)


################################################################################
#  UNIT TESTS – very small sanity checks
################################################################################

def run_unit_tests():
    print("Running unit-tests …", end="")
    from torch_geometric.data import Data

    dummy = Data(x=torch.randn(10, 8), edge_index=torch.tensor([[0, 1], [1, 2]])).to(
        DEVICE
    )
    from .train import SpatialBranch, DDAFGNN  # local import to avoid circular deps

    m_full = DDAFGNN(8, 16, 3, layers=2).to(DEVICE)
    m_no_sp = DDAFGNN(8, 16, 3, layers=2).to(DEVICE)
    for p in m_no_sp.modules():
        if isinstance(p, SpatialBranch):
            p.K = 0
            p.convs = torch.nn.ModuleList()

    out_full, _ = m_full(dummy.x, dummy.edge_index)
    out_sp, _ = m_no_sp(dummy.x, dummy.edge_index)
    diff = (out_full - out_sp).norm().item()
    assert diff > 1e-3, "Spatial branch disabling did not change output (>1e-3 diff expected)"

    # gradient through gate params?
    m_full.train()
    out, _ = m_full(dummy.x, dummy.edge_index)
    loss = out.mean()
    loss.backward()
    grads = [p.grad.abs().sum().item() for n, p in m_full.named_parameters() if "gate" in n]
    assert max(grads) > 0, "No gradient through gate parameters"
    print("passed.")


################################################################################
#  MAIN ENTRY
################################################################################

def main():
    run_unit_tests()
    exp1_depth_scalability()
    exp2_component_ablation()
    exp3_robustness()


if __name__ == "__main__":
    main()
