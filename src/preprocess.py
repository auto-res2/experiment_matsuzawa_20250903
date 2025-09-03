"""src/preprocess.py
Dataset downloading and DataLoader preparation utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Directories (created here to ensure availability before downloads)
# -----------------------------------------------------------------------------
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose(
    [T.Resize(256), T.RandomResizedCrop(224), T.RandAugment(), T.ToTensor(), T.Normalize(MEAN, STD)]
)
val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

# -----------------------------------------------------------------------------
# Dataset definitions
# -----------------------------------------------------------------------------


class Waterbirds(Dataset):
    """Waterbirds spurious-correlation benchmark (CUB-Waterbirds)."""

    def __init__(self, split: str, transform=None):
        assert split in {"train", "val", "test"}
        self.ds = load_dataset("grodino/waterbirds", split=split, cache_dir=str(DATA_ROOT))
        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img, y = sample["image"], sample["label"]
        img = self.transform(img) if self.transform else img
        return img, y, idx  # idx acts as a unique key


# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------


def get_waterbirds_dataloaders(
    seed: int,
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoaders for Waterbirds given a seed."""

    full_train = Waterbirds("train", transform=train_tf)
    labels: List[int] = [y for _, y, _ in full_train]
    idx_train, idx_val = train_test_split(
        list(range(len(full_train))), test_size=0.2, random_state=seed, stratify=labels
    )

    train_ds = Subset(full_train, idx_train)
    val_ds = Subset(full_train, idx_val)
    test_ds = Waterbirds("test", transform=val_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dl, val_dl, test_dl
