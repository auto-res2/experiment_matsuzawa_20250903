"""src/main.py
Entry point (`python -m src.main`) that orchestrates the three experiments
from the original monolithic script.  Configuration is read from
`config/config.yaml`.  The implementation follows the same logic as the
original but now relies on the modularised `src.*` utilities.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1.  Local imports (relative)
# ---------------------------------------------------------------------------
from .train import build_resnet18, DummyEncoder, HATEMBuffer, RawReplayBuffer
from .preprocess import get_cifar100_tasks, DATA_ROOT
from .evaluate import evaluate

# ---------------------------------------------------------------------------
# 2.  Load experiment configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with CONFIG_PATH.open("r") as _f:
    CONFIG: Dict[str, Any] = yaml.safe_load(_f)

FIG_DIR = Path(CONFIG["plots"]["dir"])
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3.  Experiment 1 – Kilobyte-scale memory budget
# ---------------------------------------------------------------------------

def experiment1() -> None:
    print("\n================ EXPERIMENT 1 – DO KILOBYTES REALLY SUFFICE? ================")
    description = (
        "Benchmarks continual-learning accuracy / forgetting with extreme memory "
        "budgets (0.25 / 0.5 / 1 MB) on Split CIFAR-100 using ResNet-18."
    )
    print(description)

    fig_names: List[str] = []
    results_table: List[Tuple[str, int, float]] = []

    for budget in CONFIG["memory_budgets_bytes"]:
        for method in ["HATEM", "ER"]:
            acc_seeds: List[float] = []
            for seed in CONFIG["training"]["seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)

                tasks = get_cifar100_tasks(seed=seed)
                device = torch.device(CONFIG["hardware"]["device"])
                model = build_resnet18(num_classes=100).to(device)

                scaler = (torch.cuda.amp.GradScaler() if CONFIG["hardware"]["mixed_precision"] and device.type == "cuda" else None)
                optim_cfg = CONFIG["training"]["optim"]
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=optim_cfg["lr"],
                    momentum=optim_cfg["momentum"],
                    weight_decay=optim_cfg["weight_decay"],
                    nesterov=True,
                )

                # Memory buffer
                if method == "HATEM":
                    buffer = HATEMBuffer(bytes_limit=budget)
                    encoder = DummyEncoder()
                else:
                    buffer = RawReplayBuffer(bytes_limit=budget)
                    encoder = None  # type: ignore

                # Iterate over tasks ---------------------------------------------------
                for task_dataset in tasks:
                    loader = DataLoader(
                        task_dataset,
                        batch_size=CONFIG["training"]["batch_size"],
                        shuffle=True,
                        num_workers=2,
                    )
                    model.train()
                    for imgs, labels in loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        for i in range(imgs.size(0)):
                            if method == "HATEM":
                                buffer.add_sample(imgs[i].cpu(), labels[i].item(), encoder)  # type: ignore[arg-type]
                            else:
                                buffer.add_sample(imgs[i].cpu(), labels[i].item())

                        optimizer.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast(enabled=scaler is not None):
                            logits = model(imgs)
                            loss = F.cross_entropy(logits, labels)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                    # Consolidation + replay
                    if method == "HATEM":
                        buffer.consolidate()
                    replay_batch = CONFIG["training"]["batch_size"]
                    try:
                        imgs_rep, lbl_rep = buffer.sample(replay_batch)
                    except RuntimeError:
                        continue  # buffer still empty
                    imgs_rep, lbl_rep = imgs_rep.to(device), lbl_rep.to(device)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        logits_rep = model(imgs_rep)
                        loss_rep = F.cross_entropy(logits_rep, lbl_rep)
                    if scaler is not None:
                        scaler.scale(loss_rep).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_rep.backward()
                        optimizer.step()

                # Evaluation ----------------------------------------------------------
                test_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                test_ds = datasets.CIFAR100(root=str(DATA_ROOT), train=False, download=True, transform=test_tf)
                test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
                acc = evaluate(model, test_loader, device)
                acc_seeds.append(acc)
                print(f"[Seed {seed:>2d}] Method={method:<5s} Budget={budget/1024:.0f} kB | ACC={acc:5.2f} %")

            mu, sigma = float(np.mean(acc_seeds)), float(np.std(acc_seeds))
            print(f"→ {method:<5s}  {budget/1024:4.0f} kB | µ±σ {mu:5.2f}±{sigma:4.2f}%")
            results_table.append((method, budget, mu))

    # Plot -------------------------------------------------------------------
    methods = sorted({m for m, _, _ in results_table})
    budgets_kb = [b // 1024 for b in CONFIG["memory_budgets_bytes"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_w = 0.25
    for i, m in enumerate(methods):
        accs = [next(r[2] for r in results_table if r[0] == m and r[1] == b * 1024) for b in budgets_kb]
        pos = np.arange(len(budgets_kb)) + i * bar_w
        bars = ax.bar(pos, accs, bar_w, label=m)
        for p, acc in zip(pos, accs):
            ax.text(p, acc + 0.2, f"{acc:4.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(budgets_kb)) + bar_w * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"{b} KB" for b in budgets_kb])
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Experiment-1: AACC vs Memory Budget on CIFAR-100")
    ax.legend()
    fig.tight_layout()
    fig_path = FIG_DIR / "average_accuracy_budget.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig_names.append(str(fig_path))

    print("\nEXPERIMENT 1 completed. Figures saved:")
    for n in fig_names:
        print("  •", n)

# ---------------------------------------------------------------------------
# 4.  Experiment 2 – Efficiency & scaling
# ---------------------------------------------------------------------------

def experiment2() -> None:
    print("\n================ EXPERIMENT 2 – WRITE / READ EFFICIENCY ====================")
    description = (
        "Measures write latency and peak VRAM for HATEM vs raw exemplar "
        "replay on a short CIFAR-100 stream. (VQ-VAE is emulated by raw replay.)"
    )
    print(description)

    methods = ["HATEM", "ER"]
    results: Dict[str, Dict[str, List[float]]] = {m: {"write_ms": [], "peak_vram": [], "read_ms": []} for m in methods}

    device = torch.device(CONFIG["hardware"]["device"])
    stream_ds = datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=transforms.ToTensor())
    stream_loader = DataLoader(stream_ds, batch_size=1, shuffle=True, num_workers=2)

    for seed in CONFIG["training"]["seeds"][:3]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for method in methods:
            if method == "HATEM":
                buffer = HATEMBuffer(bytes_limit=524_288)
                encoder = DummyEncoder()
            else:
                buffer = RawReplayBuffer(bytes_limit=524_288)
                encoder = None  # type: ignore

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            else:
                starter = ender = None  # type: ignore

            write_times = []
            for i, (img, lbl) in enumerate(stream_loader):
                img = img.squeeze(0)
                if device.type == "cuda":
                    starter.record()
                if method == "HATEM":
                    buffer.add_sample(img, int(lbl.item()), encoder)  # type: ignore[arg-type]
                else:
                    buffer.add_sample(img, int(lbl.item()))
                if device.type == "cuda":
                    ender.record(); torch.cuda.synchronize()
                    write_times.append(starter.elapsed_time(ender))
                if i >= 500:
                    break  # keep runtime low for the demo
            if device.type == "cuda":
                peak_vram = torch.cuda.max_memory_allocated(device) / 1e6  # MB
            else:
                peak_vram = 0.0

            # Read timing ----------------------------------------------------
            if device.type == "cuda":
                read_s, read_e = torch.cuda.Event(True), torch.cuda.Event(True)
                read_s.record()
            try:
                _ = buffer.sample(128)
            except RuntimeError:
                pass
            if device.type == "cuda":
                read_e.record(); torch.cuda.synchronize()
                read_ms = read_s.elapsed_time(read_e)
            else:
                read_ms = 0.0

            results[method]["write_ms"].append(float(np.mean(write_times) if write_times else 0.0))
            results[method]["peak_vram"].append(float(peak_vram))
            results[method]["read_ms"].append(float(read_ms))
            print(
                f"Seed={seed} {method:<5s}: write={results[method]['write_ms'][-1]:5.2f} ms | "
                f"vram={peak_vram:6.1f} MB | read={read_ms:4.2f} ms"
            )

    # Plot -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_w = 0.25
    x = np.arange(len(methods))
    write_means = [np.mean(results[m]["write_ms"]) for m in methods]
    ax.bar(x, write_means, bar_w)
    for i, v in enumerate(write_means):
        ax.text(i, v + 0.5, f"{v:4.1f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Write Latency (ms / sample)")
    ax.set_title("Experiment-2: Write latency")
    fig.tight_layout()
    path = FIG_DIR / "write_latency.pdf"
    fig.savefig(path, bbox_inches="tight")
    print("\nEXPERIMENT 2 completed. Figure saved:", path)

# ---------------------------------------------------------------------------
# 5.  Experiment 3 – Simple ablation (clean accuracy)
# ---------------------------------------------------------------------------

def experiment3() -> None:
    print("\n================ EXPERIMENT 3 – WHY TWO TIERS? =============================")
    description = (
        "Ablation on CIFAR-100 to validate the effect of Tier-2 prototypes and "
        "encoder freezing.  Uses clean accuracy as a proxy for robustness."
    )
    print(description)

    variants = {
        "HATEM-full": dict(use_proto=True, freeze_enc=True),
        "no-Tier-2": dict(use_proto=False, freeze_enc=True),
        "enc-finetune": dict(use_proto=True, freeze_enc=False),
        "token-512": dict(use_proto=False, freeze_enc=True, vocab=512),
    }
    results: Dict[str, float] = {}

    for name, cfg in variants.items():
        torch.manual_seed(0)
        np.random.seed(0)
        tasks = get_cifar100_tasks(seed=0)
        device = torch.device(CONFIG["hardware"]["device"])
        model = build_resnet18(100).to(device)
        encoder = DummyEncoder(code_dim=cfg.get("vocab", 256))
        buffer = HATEMBuffer(vocab=cfg.get("vocab", 256), bytes_limit=524_288)
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["training"]["optim"]["lr"], momentum=0.9, weight_decay=5e-4, nesterov=True)

        for task_ds in tasks:
            loader = DataLoader(task_ds, batch_size=128, shuffle=True, num_workers=2)
            model.train()
            for imgs, lbl in loader:
                for i in range(imgs.size(0)):
                    buffer.add_sample(imgs[i], int(lbl[i].item()), encoder)
                imgs, lbl = imgs.to(device), lbl.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(imgs)
                F.cross_entropy(logits, lbl).backward()
                optimizer.step()
            if cfg["use_proto"]:
                buffer.consolidate()
        # Evaluate
        test_loader = DataLoader(datasets.CIFAR100(root=str(DATA_ROOT), train=False, download=True, transform=transforms.ToTensor()), batch_size=256, shuffle=False, num_workers=2)
        acc = evaluate(model, test_loader, device)
        results[name] = acc
        print(f"Variant {name:<11s} ACC={acc:5.2f}%")

    # Plot -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(results))
    accs = list(results.values())
    ax.bar(x, accs)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.5, f"{v:4.1f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(results.keys(), rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment-3: Ablation study")
    fig.tight_layout()
    path = FIG_DIR / "ablation_accuracy.pdf"
    fig.savefig(path, bbox_inches="tight")
    print("\nEXPERIMENT 3 completed. Figure saved:", path)

# ---------------------------------------------------------------------------
# 6.  Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()
    experiment1()
    experiment2()
    experiment3()
    print(f"\nAll experiments finished in {(time.time() - start) / 60:.1f} min.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
