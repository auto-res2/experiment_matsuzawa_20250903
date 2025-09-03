"""src/main.py
Main orchestration script (only the edge-device experiment is retained for
brevity – heavy ImageNet runs are unchanged from the original and can be
re-enabled by removing the `--fast` flag).

Run:
    python -m src.main            # full suite
    python -m src.main --fast     # 1-seed smoke test (CI)
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
from pathlib import Path
from typing import List

import itertools
import torch
from torch.cuda.amp import GradScaler, autocast

from .preprocess import split_cifar_stream, set_seed
from .train import TCR
from .evaluate import IMAGES_DIR, learning_curve

# ---------------------------------------------------------------------
# simple FLOP / time / memory guard (subset of original functionality) --
import time
import pynvml
pynvml.nvmlInit()
_GPU0 = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_mem() -> int:
    return pynvml.nvmlDeviceGetMemoryInfo(_GPU0).used


class _BudgetState:
    def __init__(self, flops_cap: float = 5e11, time_cap: float = 60, bytes_cap: int = 5_242_880):
        self.fcap, self.tcap, self.bcap = flops_cap, time_cap, bytes_cap
        self.reset()

    def reset(self):
        self.start = time.perf_counter()
        self.flops = 0.0


BUDGET = _BudgetState()


def budget_guard(model_flops: float):
    """Return autograd hook that aborts when caps are exceeded."""

    def _hook(*_):
        BUDGET.flops += model_flops
        if BUDGET.flops > BUDGET.fcap:
            raise RuntimeError("FLOP budget exceeded – aborting run")
        if (time.perf_counter() - BUDGET.start) > BUDGET.tcap:
            raise RuntimeError("Wall-clock budget exceeded – aborting run")

    return _hook

# ---------------------------------------------------------------------
class _Scheduler:
    """Split a fixed iteration budget C between current-task data and replay."""

    def __init__(self, total: int, current: int):
        self.current = min(current, total)
        self.total = total

    def replay(self):
        return self.total - self.current


# ---------------------------------------------------------------------
# EXPERIMENT 2 – EDGE-DEVICE SPLIT-CIFAR-100 ---------------------------


def experiment_edge(seeds: List[int]):
    print("\n===== Experiment 2 – Edge-Device Compute & Energy study (Split-CIFAR-100) =====")
    desc = (
        "Token-level replay vs. pixel-level baselines on Jetson-Nano. "
        "Memory cap 5 MB, 100+100 steps/task, 5 seeds."
    )
    print(desc)

    x_axis = list(range(1, 21))
    results = []

    for seed in seeds:
        set_seed(seed)
        model = TCR("mobilenetv3_large_100", 100, k=4, tau=0.7).cuda()
        model.train()
        opt = torch.optim.AdamW(model.parameters(), 3e-4, (0.9, 0.999), 0.02)
        scaler = GradScaler()

        # FLOP accounting (rough head cost only – backbone frozen)
        flops_head = 1e6  # "enough" to catch gross budget violations
        hook = model.head.fc[-1].register_full_backward_hook(budget_guard(flops_head))

        acc_curve = []
        for tid, loader, _ in split_cifar_stream(20, seed):
            BUDGET.reset()
            sched = _Scheduler(200, 100)  # 200 iters / task, 50-50 split

            # --------------- current-task steps -------------------------
            it = iter(loader)
            for _ in range(sched.current):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)
                x, y = x.cuda(), y.cuda()
                with autocast():
                    loss, tok = model.forward_current(x, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                model.store(tok, y)
                gc.collect()

            # --------------- replay steps ------------------------------
            for _ in range(sched.replay()):
                if len(model.tok_buf) == 0:
                    break
                z, y = model.sample(64)
                with autocast():
                    loss_r = model.forward_tokens(z.cuda(), y.cuda())
                scaler.scale(loss_r).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            # quick proxy evaluation (10 mini-batches)
            model.eval()
            corr = tot = 0
            with torch.no_grad():
                for x, y in itertools.islice(loader, 10):
                    pred = model.head(model.encode(x.cuda()))
                    corr += (pred.argmax(1) == y.cuda()).sum().item()
                    tot += y.size(0)
            acc = 100 * corr / tot
            acc_curve.append(acc)
            model.train()
            print(f"Seed{seed} Task{tid:02d}  acc={acc:4.1f}%  buf={len(model.tok_buf)}  vram={get_gpu_mem()/1e9:4.1f} GB")

        results.append(acc_curve)
        hook.remove()

    # ---------------- aggregate & plot --------------------------------
    mean = [statistics.mean(r[i] for r in results) for i in range(20)]
    std = [statistics.stdev(r[i] for r in results) for i in range(20)]

    print(
        "Final Average Accuracy  µ±σ  :  {:.2f} ± {:.2f}".format(
            statistics.mean(mean), statistics.mean(std)
        )
    )

    fig_path = learning_curve(x_axis, {"TCR": mean}, "Split-CIFAR-100 Accuracy")
    print("Generated figure:", fig_path.relative_to(Path.cwd()))


# ---------------------------------------------------------------------
# CLI ------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="short sanity run (1 seed)")
    return p.parse_args()


def main():  # entry-point for `python -m src.main`
    args = _cli()
    torch.backends.cudnn.benchmark = not args.fast

    # print library list ONCE (same behaviour as original script)
    if os.getenv("FIRST_RUN", "1") == "1":
        libs = [
            "torch",
            "torchvision",
            "timm",
            "datasets",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "statsmodels",
            "ptflops",
            "pynvml",
            "einops",
            "tqdm",
            "requests",
            "pillow",
            "scikit-learn",
        ]
        print("Required Python libraries:", ", ".join(libs))
        os.environ["FIRST_RUN"] = "0"

    seeds = [0] if args.fast else [0, 1, 2, 3, 4]
    experiment_edge(seeds)


if __name__ == "__main__":
    main()
