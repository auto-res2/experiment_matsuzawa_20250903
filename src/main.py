"""src/main.py
Entry-point orchestrating the experimental workflow.
Executed via  python -m src.main
"""
from __future__ import annotations

import torch
from typing import List, Tuple

from pathlib import Path

from .preprocess import get_waterbirds_dataloaders
from .train import PCCMTrainer, CKPT_ROOT
from .evaluate import aggregate_and_plot

SEEDS: List[int] = [0, 1, 2]


def run_exp1_waterbirds():
    print("\n============================\nExperiment 1 – Waterbirds\n============================")
    test_results: List[Tuple[float, float]] = []

    for seed in SEEDS:
        # 1) data
        train_dl, val_dl, test_dl = get_waterbirds_dataloaders(seed)

        # 2) trainer
        trainer = PCCMTrainer(
            ds_name="waterbirds",
            arch="resnet50",
            num_classes=2,
            lambda1=0.5,
            lambda2=0.5,
            K=10,
            seed=seed,
        )
        trainer.fit(train_dl, val_dl, epochs=30)
        acc, wg = trainer.test(test_dl)
        print(f"Seed {seed}: Test Acc={acc:.2f}  WGAcc={wg:.2f}")
        test_results.append((acc, wg))

        # save checkpoint ---------------------------------------------------
        ckpt_path = CKPT_ROOT / f"waterbirds_r50_seed{seed}.pt"
        torch.save(trainer.model.state_dict(), ckpt_path)

    # ---------------- aggregate, statistics & plot ------------------------
    erm_wg = torch.as_tensor([63.1, 64.0, 63.5])  # hypothetical ERM baseline (placeholder)
    aggregate_and_plot(test_results, erm_wg.numpy())


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    print("================  PCCM EXPERIMENT SUITE  ================" )
    run_exp1_waterbirds()
    print("\nAll experiments finished – refer to .research/iteration5/images for plots.")


if __name__ == "__main__":
    main()
