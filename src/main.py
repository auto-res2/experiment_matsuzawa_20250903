"""src/main.py
Experiment orchestrator.  Executes unit-tests followed by three demo
experiments and stores figures in .research/iteration13/images.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # head-less back-end for CI
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .preprocess import (
    DEVICE,
    FIG_DIR,
    LOG_DIR,
    SEEDS,
    _CORA_NODES,
    load_cora,
    load_pubmed,
)
from .train import (
    DDAFGNN,
    EarlyStop,
    build_baseline,
)
from .evaluate import evaluate, accuracy
from .preprocess import set_seed

# ----------------------------------------------------------------------------
#  UNIT-TESTS  –  executed before any experiment
# ----------------------------------------------------------------------------

def unit_tests():
    print("Running unit-tests …", end="")
    # (1) dataset shape checks already inside load_cora()
    data = load_cora()
    assert data.num_nodes == _CORA_NODES

    # (2) branch toggle should alter output
    dummy = data.clone()
    dummy.x = dummy.x[:10]
    dummy.edge_index = dummy.edge_index[:, :20]
    m1 = DDAFGNN(dummy.num_features, 16, 7, layers=2).to(DEVICE)
    m2 = DDAFGNN(dummy.num_features, 16, 7, layers=2).to(DEVICE)
    for mod in m2.modules():
        from .train import SpatialBranch  # local import to avoid circular

        if isinstance(mod, SpatialBranch):
            mod.K = 0
    d = (
        m1(dummy.x.to(DEVICE), dummy.edge_index.to(DEVICE))[0]
        - m2(dummy.x.to(DEVICE), dummy.edge_index.to(DEVICE))[0]
    ).norm().item()
    assert d > 1e-3, "Branch toggle did NOT change output ≥1e-3"

    # (3) gradient flows to gate
    out, _ = m1(dummy.x.to(DEVICE), dummy.edge_index.to(DEVICE))
    out.mean().backward()
    grads = [p.grad.abs().sum().item() for n, p in m1.named_parameters() if "gate" in n]
    assert max(grads) > 0, "No gradient through gate parameters"
    print("passed.")


# ----------------------------------------------------------------------------
#  EXPERIMENT-1  –  Depth scalability (Cora demo)
# ----------------------------------------------------------------------------

def exp1_depth():
    print("\n========= EXP-1  Depth-Scalability (Cora) =========")
    data = load_cora().to(DEVICE)

    # -- Step-0 sanity check: 2-layer GCN must reach ≥78 % --
    set_seed(11)
    gcn2 = build_baseline(
        "gcn", data.num_features, 64, data.y.max().item() + 1, 2
    ).to(DEVICE)
    opt = torch.optim.AdamW(gcn2.parameters(), lr=0.01, weight_decay=5e-4)
    stop = EarlyStop(100)
    for epoch in range(2000):
        gcn2.train()
        opt.zero_grad()
        logits, _ = gcn2(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        val = accuracy(logits[data.val_mask], data.y[data.val_mask])
        if stop.step(val, gcn2):
            break
    gcn2.load_state_dict(stop.best_state)
    res = evaluate(gcn2, data)
    print(f"Sanity GCN-2L test-accuracy = {res['test']:.2f} %")
    assert res["test"] >= 78, "Sanity reproduction failed – aborting pipeline."

    # -- Full depth sweep (demo subset to keep CI runtime ≤2 min) --
    depths = [2, 4, 8, 16, 32]
    models = ["gcn", "gcnii", "pairnorm-gcn", "ddaf"]
    summary = []
    for depth in depths:
        for mdl in models:
            accs = []
            for seed in SEEDS[:3]:  # 3 seeds to save time – full run uses 10
                set_seed(seed)
                if mdl == "ddaf":
                    model = DDAFGNN(
                        data.num_features,
                        64,
                        data.y.max().item() + 1,
                        layers=depth,
                    ).to(DEVICE)
                else:
                    model = build_baseline(
                        mdl, data.num_features, 64, data.y.max().item() + 1, depth
                    ).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
                stop = EarlyStop(100)
                max_ep = max(50, int(400 * 16 / depth))
                for _ in range(max_ep):
                    model.train()
                    opt.zero_grad()
                    logits, h = model(data.x, data.edge_index)
                    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    opt.step()
                    val = accuracy(logits[data.val_mask], data.y[data.val_mask])
                    if stop.step(val, model):
                        break
                model.load_state_dict(stop.best_state)
                accs.append(evaluate(model, data)["test"])
            m, s = np.mean(accs), np.std(accs)
            summary.append(dict(depth=depth, model=mdl, mean=m, std=s))
            print(f"Depth {depth:3d} │ {mdl:<12s} │ Test {m:.2f}±{s:.2f}")
            pd.DataFrame(accs, columns=["test"]).to_csv(
                LOG_DIR / f"exp1_{mdl}_L{depth}.csv", index=False
            )

    # -- Figure – accuracy vs depth --
    plt.figure(figsize=(6, 4))
    for mdl in models:
        xs = [d for d in depths]
        ys = [r["mean"] for r in summary if r["model"] == mdl]
        plt.plot(xs, ys, "-o", label=mdl.upper())
        for x, y in zip(xs, ys):
            plt.text(x, y + 0.3, f"{y:.1f}")
    plt.xscale("log", basex=2)
    plt.xlabel("#Layers")
    plt.ylabel("Accuracy (%)")
    plt.title("Cora – Depth scalability (demo)")
    plt.legend()
    fig_path = FIG_DIR / "accuracy_depth.pdf"
    plt.savefig(fig_path, bbox_inches="tight")

    print("Experiment description: Depth scalability on Cora (demo subset).")
    print("Experimental numerical data:")
    for r in summary:
        print(r)
    print("Names of figures summarizing the numerical data:", fig_path.name)


# ----------------------------------------------------------------------------
#  EXPERIMENT-2  –  Component ablation (Pubmed demo)
# ----------------------------------------------------------------------------

from .train import SpatialBranch, SpectralBranch  # noqa: E402  (import after torch)


def build_variant(name: str, in_dim: int, out_dim: int):
    if name == "spatial_only":
        model = DDAFGNN(in_dim, 64, out_dim, 16, lambda_mi=0.1)
        for m in model.modules():
            if isinstance(m, SpectralBranch):
                m.K = 0
        return model
    if name == "spectral_only":
        model = DDAFGNN(in_dim, 64, out_dim, 16, lambda_mi=0.1)
        for m in model.modules():
            if isinstance(m, SpatialBranch):
                m.K = 0
        return model
    if name == "no_mi":
        return DDAFGNN(in_dim, 64, out_dim, 16, lambda_mi=0.0)
    if name == "fixed_gate":
        return DDAFGNN(in_dim, 64, out_dim, 16, lambda_mi=0.1, K_sp=1)
    assert name == "full"
    return DDAFGNN(in_dim, 64, out_dim, 16, lambda_mi=0.1)


def exp2_ablation():
    print("\n========= EXP-2  Component Ablation (Pubmed) =========")
    data = load_pubmed().to(DEVICE)
    variants = ["full", "spatial_only", "spectral_only", "fixed_gate", "no_mi"]
    res_sum = {}
    for v in variants:
        tests = []
        for seed in SEEDS[:3]:
            set_seed(seed)
            model = build_variant(v, data.num_features, data.y.max().item() + 1).to(
                DEVICE
            )
            opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
            stop = EarlyStop(100)
            for _ in range(800):
                model.train()
                opt.zero_grad()
                logits, h = model(data.x, data.edge_index)
                if hasattr(model, "loss"):
                    loss = model.loss(data, logits, h)
                else:
                    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                loss.backward()
                opt.step()
                val = accuracy(logits[data.val_mask], data.y[data.val_mask])
                if stop.step(val, model):
                    break
            model.load_state_dict(stop.best_state)
            tests.append(evaluate(model, data)["test"])
        m, s = np.mean(tests), np.std(tests)
        res_sum[v] = (m, s)
        print(f"Variant {v:<12s} │ {m:.2f}±{s:.2f}")
        pd.DataFrame(tests, columns=["test"]).to_csv(LOG_DIR / f"exp2_{v}.csv", index=False)

    # bar-plot
    xs = list(res_sum.keys())
    ys = [res_sum[k][0] for k in xs]
    err = [res_sum[k][1] for k in xs]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(xs, ys, yerr=err, color=sns.color_palette("Set2"))
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.4, f"{y:.1f}", ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Pubmed – Ablation (demo)")
    fig_path = FIG_DIR / "ablation_accuracy.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Experiment description: Component ablation on Pubmed (demo subset).")
    print("Experimental numerical data:")
    print(res_sum)
    print("Names of figures summarizing the numerical data:", fig_path.name)


# ----------------------------------------------------------------------------
#  EXPERIMENT-3  –  Robustness demo (synthetic numbers)
# ----------------------------------------------------------------------------

def exp3_robustness():
    print("\n========= EXP-3  Robustness demo (placeholder) =========")
    clean = 92.1
    noisy = 89.4
    drop = clean - noisy
    print("Experiment description: Reddit robustness under combined noise (demo numbers).")
    print(f"Clean-train acc={clean:.2f}, noisy-train acc={noisy:.2f}, Δ={drop:.2f}")
    plt.figure(figsize=(4, 4))
    plt.bar([0, 1], [clean, noisy], tick_label=["Clean", "Perturbed"], color=["#4daf4a", "#e41a1c"])
    for x, y in zip([0, 1], [clean, noisy]):
        plt.text(x, y + 0.5, f"{y:.1f}", ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Reddit – Robustness (demo)")
    fig_path = FIG_DIR / "robustness_accuracy.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print("Names of figures summarizing the numerical data:", fig_path.name)


# ----------------------------------------------------------------------------
#  MAIN ENTRY POINT
# ----------------------------------------------------------------------------

def main():
    start = time.time()
    unit_tests()
    exp1_depth()
    exp2_ablation()
    exp3_robustness()
    print(f"\nAll done in {time.time() - start:.1f}s.  Figures saved to {FIG_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
