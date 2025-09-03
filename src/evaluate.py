"""src/evaluate.py
Experiment orchestration, statistical analysis and plotting utilities.
It builds dataset streams, invokes *ContinualLearner* from train.py and
creates accuracy plots that are stored under `.research/iteration1/images`.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms as T
import torchvision

from .train import ContinualLearner
from .preprocess import data_dir, set_global_seed

# Make sure the images directory exists.
images_dir = Path(".research/iteration1/images")
images_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
#  Dataset stream builders
# -----------------------------------------------------------------------------

def _cifar_transform(train: bool):
    if train:
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandAugment(2, 9),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )


def build_stream(exp_cfg, shared_cfg):
    """Return list of (train_loader,val_loader,test_loader) triples."""

    name = exp_cfg["dataset"]
    if name == "split_cifar100":
        full = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=False,
            transform=_cifar_transform(train=True),
        )
        test = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=False,
            transform=_cifar_transform(train=False),
        )
        order = list(range(100))
        random.seed(shared_cfg["seed"])
        random.shuffle(order)
        tasks = [order[i : i + 5] for i in range(0, 100, 5)]
        stream = []
        for cls in tasks:
            idx = [i for i, (_, label) in enumerate(full) if label in cls]
            sub = Subset(full, idx)
            len_val = int(0.1 * len(sub))
            train_set, val_set = random_split(sub, [len(sub) - len_val, len_val])

            test_idx = [i for i, (_, label) in enumerate(test) if label in cls]
            test_loader = DataLoader(
                Subset(test, test_idx),
                batch_size=256,
                shuffle=False,
                num_workers=4,
            )
            stream.append(
                (
                    DataLoader(
                        train_set,
                        batch_size=shared_cfg["batch_sz"],
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                    ),
                    DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4),
                    test_loader,
                )
            )
        return stream

    raise NotImplementedError(f"Unknown dataset stream: {name}")


# -----------------------------------------------------------------------------
#  Experiment Engine
# -----------------------------------------------------------------------------

class ExperimentEngine:
    def __init__(self, exp_cfg, shared_cfg):
        self.exp = exp_cfg
        self.shared = shared_cfg
        set_global_seed(shared_cfg["seed"])

    # ---------------------------------------------------------------------
    def run(self):
        print("=" * 80)
        print(f"Experiment description: {self.exp['name']}")
        stream = build_stream(self.exp, self.shared)
        learner = ContinualLearner(self.exp, self.shared)

        acc_per_task: List[float] = []
        for t, (ldr_tr, ldr_val, ldr_test) in enumerate(stream):
            tic = time.time()
            learner.train_task(t, ldr_tr, ldr_val)
            toc = time.time()
            acc = learner.evaluate(ldr_test)
            acc_per_task.append(acc)
            print(f"Task {t:02d}  ACC={acc:.4f}  time={toc - tic:.1f}s")

        avg_acc = float(np.mean(acc_per_task))
        print(f"Final Average Accuracy: {avg_acc:.4f}")

        # ------------- Plot ------------------------------------------------
        fig = plt.figure(figsize=(6, 3))
        sns.lineplot(x=list(range(len(acc_per_task))), y=acc_per_task, marker="o")
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        for i, a in enumerate(acc_per_task):
            plt.text(i, a + 0.005, f"{a * 100:.1f}")
        fname = images_dir / "accuracy_stream.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print("Names of figures summarizing the numerical data:", fname.name)
