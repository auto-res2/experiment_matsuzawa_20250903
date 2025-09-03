"""src/preprocess.py – data download / transforms / stream builder"""
from __future__ import annotations
import random, tarfile, urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T


# ────────────────────────────────────────────────────────────────────────
# Build data stream for Split-CIFAR100                                   
# ────────────────────────────────────────────────────────────────────────


def _ensure_cifar_download(url: str, data_dir: Path):
    cifar_dir = data_dir / 'cifar-100-python'
    if cifar_dir.exists():
        return
    print('[download] CIFAR-100 …')
    tmp = data_dir / 'cifar100.tgz'
    urllib.request.urlretrieve(url, tmp)
    tarfile.open(tmp).extractall(data_dir)


def _subset(ds, cls_ids):
    idx = [i for i, (_, y) in enumerate(ds) if y in cls_ids]
    return torch.utils.data.Subset(ds, idx)


def build_stream(cfg) -> List[Tuple[torch.utils.data.Dataset, ...]]:
    """Return list[(train,val,test)] for every task defined in the cfg."""

    # directories ------------------------------------------------------
    root = Path(__file__).resolve().parent.parent  # project root
    data_dir = root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # dataset download -------------------------------------------------
    _ensure_cifar_download(cfg['dataset']['url'], data_dir)

    mean, std = cfg['dataset']['mean'], cfg['dataset']['std']
    train_tf = T.Compose([
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    full_train = torchvision.datasets.CIFAR100(data_dir, train=True, download=False, transform=train_tf)
    full_test = torchvision.datasets.CIFAR100(data_dir, train=False, download=False, transform=val_tf)

    # fixed 20 × 5 split ------------------------------------------------
    classes = list(range(100))
    random.Random(cfg['dataset']['seed_order']).shuffle(classes)
    step = cfg['dataset']['classes_per_task']
    tasks = [classes[i:i + step] for i in range(0, 100, step)]

    stream = []
    for cls in tasks:
        tr = _subset(full_train, cls)
        te = _subset(full_test, cls)
        v = int(0.05 * len(tr))
        tr, va = random_split(tr, [len(tr) - v, v], generator=torch.Generator().manual_seed(0))
        stream.append((tr, va, te))
    return stream
