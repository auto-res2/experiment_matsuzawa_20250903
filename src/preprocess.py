"""
preprocess.py – data downloading, preprocessing & task-stream builder
"""
from __future__ import annotations
import random, tarfile, urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split

# CIFAR-100 constants ---------------------------------------------------
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _download_cifar_if_missing(data_dir: Path):
    tgt = data_dir / "cifar-100-python"
    if tgt.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "cifar100.tgz"
    if not tgz.exists():
        print("[DL] CIFAR-100 …")
        urllib.request.urlretrieve(CIFAR_URL, tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(data_dir)


def _build_transforms(mean: List[float], std: List[float]):
    train_tf = T.Compose([
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_task_stream(
    data_dir: Path,
    mean: List[float],
    std: List[float],
    num_tasks: int,
    classes_per_task: int,
    seed: int = 0,
):
    """Returns list[(train, val, test)] for Split-CIFAR100."""
    _download_cifar_if_missing(data_dir)

    train_tf, val_tf = _build_transforms(mean, std)
    full_train = torchvision.datasets.CIFAR100(data_dir, train=True,  download=False, transform=train_tf)
    full_test  = torchvision.datasets.CIFAR100(data_dir, train=False, download=False, transform=val_tf)

    cls: List[int] = list(range(100))
    random.Random(seed).shuffle(cls)
    tasks = [cls[i : i + classes_per_task] for i in range(0, num_tasks * classes_per_task, classes_per_task)]

    def _subset(ds, cls_ids):
        idx = [i for i, (_, y) in enumerate(ds) if y in cls_ids]
        return torch.utils.data.Subset(ds, idx)

    stream = []
    for t in tasks:
        tr = _subset(full_train, t)
        te = _subset(full_test,  t)
        v  = int(0.05 * len(tr))
        tr, va = random_split(tr, [len(tr) - v, v], generator=torch.Generator().manual_seed(seed))
        stream.append((tr, va, te))
    return stream
