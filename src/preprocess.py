"""
preprocess.py – data loading & task-stream generation.
Separate from train.py to keep dataset-specific logic contained.
"""
from __future__ import annotations

import os, tarfile, random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split

# -----------------------------------------------------------------------------
# helpers ----------------------------------------------------------------------

def _subset(ds, cls_ids):
    idx = [i for i, (_, y) in enumerate(ds) if y in cls_ids]
    return torch.utils.data.Subset(ds, idx)


# -----------------------------------------------------------------------------
# public API -------------------------------------------------------------------

def make_stream(cfg: Dict) -> List[Tuple]:
    """Download CIFAR-100 if absent and create the continual-learning stream."""

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / 'data'
    DATA.mkdir(parents=True, exist_ok=True)

    # ---- download -----------------------------------------------------
    cifar_dir = DATA / 'cifar-100-python'
    if not cifar_dir.exists():
        print('[download] CIFAR-100 …')
        tmp = DATA / 'cifar100.tgz'
        import urllib.request
        urllib.request.urlretrieve(cfg['dataset']['url'], tmp)
        tarfile.open(tmp).extractall(DATA)

    # ---- transforms ---------------------------------------------------
    mean, std = cfg['dataset']['mean'], cfg['dataset']['std']
    train_tf = T.Compose([T.RandAugment(2, 9), T.ToTensor(), T.Normalize(mean, std)])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    _train = torchvision.datasets.CIFAR100(DATA, train=True, download=False, transform=train_tf)
    _test = torchvision.datasets.CIFAR100(DATA, train=False, download=False, transform=val_tf)

    # ---- build task list ---------------------------------------------
    classes = list(range(100))
    random.Random(cfg['dataset']['permutation_seed']).shuffle(classes)
    tasks = [classes[i:i + cfg['dataset']['classes_per_task']] for i in range(0, 100, cfg['dataset']['classes_per_task'])]

    if cfg['quick_ci']['enabled']:
        tasks = tasks[: cfg['quick_ci']['tasks']]

    stream = []
    for cls_ids in tasks:
        tr = _subset(_train, cls_ids)
        te = _subset(_test, cls_ids)

        v = int(0.05 * len(tr))
        tr, va = random_split(tr, [len(tr) - v, v], generator=torch.Generator().manual_seed(0))
        stream.append((tr, va, te, cls_ids))

    return stream
