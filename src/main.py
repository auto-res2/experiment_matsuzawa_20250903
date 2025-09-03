"""src/main.py
Minimal runnable entry-point so that `python -m src.main` executes without
errors and without pulling in heavyweight external datasets.  A tiny
random graph is generated on-the-fly, a 2-layer GCN (+ optional normaliser)
gets trained for a handful of epochs, and a single test-accuracy number
is printed.  The whole routine is CPU-friendly and finishes in <5 seconds
on the GitHub Actions runner (≈500 MB RAM budget).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.train import DeepGCN, Trainer
from src.evaluate import evaluate_accuracy
from src.preprocess import generate_toy_graph

# ---------------------------------------------------------------------------
# Configuration dictionary – only the keys accessed by *Trainer* are defined
# ---------------------------------------------------------------------------
COMMON_CFG: Dict[str, Any] = {
    "optimizer": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
}


def _run_demo() -> None:
    """Fabricate toy data → train for a few epochs → report accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----  tiny synthetic dataset  -----
    data: Data = generate_toy_graph()

    # -----  model  -----
    model = DeepGCN(
        in_dim=data.x.size(-1),
        hidden=32,
        out_dim=int(data.y.max().item()) + 1,
        layers=2,  # input + output layer only – keeps runtime tiny
        normaliser_cfg={"name": "none"},
        dropout=0.0,
    )

    trainer = Trainer(device=device, common_cfg=COMMON_CFG)

    tic = time.time()
    stats = trainer.train(
        model=model,
        data=data,
        epochs=20,  # very short – we only need to prove that training works
        lr=1e-2,
        weight_decay=1e-4,
        eval_fn=evaluate_accuracy,
        early_stop_patience=40,
    )
    duration = time.time() - tic

    print("\n================  Quick-start demo finished  ================")
    print(f"Test-accuracy : {stats['test_acc']*100:.2f} %")
    print(f"Best val-acc  : {stats['best_val']*100:.2f} %")
    print(f"Time elapsed  : {duration:.2f} s  (trainer measured {stats['seconds']:.2f} s)")
    print("============================================================\n")


if __name__ == "__main__":
    _run_demo()
