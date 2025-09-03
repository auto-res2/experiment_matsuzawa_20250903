"""src/main.py – Entry-point (python -m src.main)
Coordinates data-loading, training and evaluation for the edge-device study.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

import torch

from .train import TCRModel, Trainer
from .evaluate import line_plot
from .preprocess import build_split_cifar100, set_seed


################################################################################
# GLOBALS
################################################################################

def human_readable(num_bytes: int) -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


################################################################################
# EXPERIMENT – EDGE DEVICE (Split-CIFAR-100)
################################################################################

def run_edge_experiment():
    print("\n================ EDGE-DEVICE STUDY – Split-CIFAR-100 ================")
    seeds = [0, 1]  # shortened for demo – set to 5 for full paper results

    all_results = defaultdict(list)

    for seed in seeds:
        print(f"\n----- Seed {seed} -----")
        set_seed(seed)

        # 1.  Build continual stream ----------------------------------------
        stream = list(build_split_cifar100(tasks=20, seed=seed))

        # 2.  Prepare model & trainer ---------------------------------------
        model   = TCRModel(backbone_name="mobilenetv3_large_100", k_tokens=4,
                           num_classes=100, codebook_temp=0.5)
        trainer = Trainer(model)

        acc_per_task = []

        # 3.  Iterate over tasks --------------------------------------------
        for task_id, loader in stream:
            trainer.train_task(loader)
            acc = trainer.eval_loader(loader)
            acc_per_task.append(acc)
            print(f"Task {task_id:02d} | Accuracy {acc:5.2f}% | Buffer size {human_readable(model.replay_tokens.numel())}")
        all_results['TCR'].append(acc_per_task)

    # 4. Aggregate over seeds ----------------------------------------------
    avg_acc = [statistics.mean([seed_vec[t] for seed_vec in all_results['TCR']]) for t in range(20)]
    print("\nFinal Average Accuracy (TCR) :", statistics.mean(avg_acc))

    # 5. Plot ---------------------------------------------------------------
    line_plot(list(range(1, 21)), {"TCR": avg_acc},
              title="Accuracy over Tasks – Split-CIFAR-100",
              xlabel="Task #", ylabel="Accuracy (%)",
              filename="acc_split_cifar100_TCR.pdf")


################################################################################
# main -----------------------------------------------------------------------
################################################################################

def main():
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU required for these experiments.")
    run_edge_experiment()


if __name__ == "__main__":
    main()
