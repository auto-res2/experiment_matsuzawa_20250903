"""src/preprocess.py – data download / transforms / stream builder (iteration-19)
Fixes:
1. Ensure class labels are remapped to a contiguous incremental range so that they
   match the dynamically expanding classifier head. This prevents the CUDA
   assert "t < n_classes" that occurred when original CIFAR-100 labels (0–99)
   were fed to a head that only had, e.g., 5 outputs.
2. No other behavioural change – download logic, augmentation, etc. stay the
   same.
"""
from __future__ import annotations
import random, tarfile, urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T


# ────────────────────────────────────────────────────────────────────────
# Helper – download dataset                                               
# ────────────────────────────────────────────────────────────────────────

def _ensure_cifar_download(url: str, data_dir: Path):
    cifar_dir = data_dir / 'cifar-100-python'
    if cifar_dir.exists():
        return
    print('[download] CIFAR-100 …')
    tmp = data_dir / 'cifar100.tgz'
    urllib.request.urlretrieve(url, tmp)
    tarfile.open(tmp).extractall(data_dir)


# ────────────────────────────────────────────────────────────────────────
# Dataset wrappers                                                        
# ────────────────────────────────────────────────────────────────────────

class RemapDataset(torch.utils.data.Dataset):
    """Wrap a PyTorch Dataset/Subset and remap its labels with a dictionary."""

    def __init__(self, base_ds: torch.utils.data.Dataset, label_map: Dict[int, int]):
        self.base_ds = base_ds
        self.label_map = label_map

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, self.label_map[int(y)]


def _subset(ds: torch.utils.data.Dataset, cls_ids: List[int]):
    idx = [i for i, (_, y) in enumerate(ds) if int(y) in cls_ids]
    return torch.utils.data.Subset(ds, idx)


# ────────────────────────────────────────────────────────────────────────
# Public API – build_stream                                               
# ────────────────────────────────────────────────────────────────────────

def build_stream(cfg) -> List[Tuple[torch.utils.data.Dataset, ...]]:
    """Return list[(train,val,test)] for every task defined in the cfg.
    Labels are remapped so that classes that appear for the first time get the
    next available contiguous id. This matches the dynamic head expansion done
    in `src/train.py`.
    """

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

    # fixed 20 × 5 split ----------------------------------------------
    classes = list(range(100))
    random.Random(cfg['dataset']['seed_order']).shuffle(classes)
    step = cfg['dataset']['classes_per_task']
    tasks = [classes[i : i + step] for i in range(0, 100, step)]

    # global label remapping: first class encountered ⇒ label 0, …
    flat_order = [c for task in tasks for c in task]
    label_map = {orig: new for new, orig in enumerate(flat_order)}

    stream = []
    for cls_ids in tasks:
        # create class-restricted subsets
        tr_sub = _subset(full_train, cls_ids)
        te_sub = _subset(full_test, cls_ids)

        # remap labels to contiguous ids
        tr_remap = RemapDataset(tr_sub, label_map)
        te_remap = RemapDataset(te_sub, label_map)

        # validation split (5 % of training)
        v = int(0.05 * len(tr_remap))
        tr_split, va_split = random_split(
            tr_remap,
            [len(tr_remap) - v, v],
            generator=torch.Generator().manual_seed(0),
        )
        stream.append((tr_split, va_split, te_remap))

    return stream
