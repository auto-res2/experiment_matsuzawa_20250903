"""
preprocess.py – all data handling & augmentation logic.
Only a subset of the original datasets (Waterbirds) is kept to keep the example
light-weight.  Additional datasets can be added following the same template.
"""
from __future__ import annotations
import requests, tarfile, zipfile
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# WILDS provides Waterbirds & CelebA splits ready-made -----------------------
from wilds import get_dataset

# -----------------------------------------------------------------------------
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
#                              transforms
# -----------------------------------------------------------------------------

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _tfm_train():
    return T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def _tfm_test():
    return T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    )


# -----------------------------------------------------------------------------
#                           dataset specific loaders
# -----------------------------------------------------------------------------

def _waterbirds(batch_size: int, num_workers: int):
    dataset = get_dataset("waterbirds", root_dir=str(DATA_ROOT / "waterbirds"), download=True)
    dl_train = dataset.get_subset("train", transform=_tfm_train())
    dl_val = dataset.get_subset("val", transform=_tfm_test())
    dl_test = dataset.get_subset("test", transform=_tfm_test())

    loaders = {
        "train": DataLoader(dl_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(dl_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(dl_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, dataset.n_classes


LOADER_REGISTRY = {"waterbirds": _waterbirds}


# -----------------------------------------------------------------------------
#                              public API
# -----------------------------------------------------------------------------

def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int):
    if dataset_name not in LOADER_REGISTRY:
        raise RuntimeError(f"Dataset '{dataset_name}' is not implemented in preprocess.py")
    return LOADER_REGISTRY[dataset_name](batch_size, num_workers)
