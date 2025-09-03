"""src/preprocess.py
Dataset download helpers, task-split generator and basic preprocessing
(transforms, normalisation, etc.).  All data lives in ROOT/data by default.
"""
from __future__ import annotations
import tarfile, zipfile, random, shutil, os
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, random_split
import torchvision
import torchvision.transforms as T

# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

################################################################################
#                               ───  DATASETS ───                               #
################################################################################
class SplitDataset(Dataset):
    """Restrict a base dataset to the given class indices and remap labels."""

    def __init__(self, base_ds: Dataset, class_ids: List[int]):
        self.base = base_ds
        self.cls_ids = set(class_ids)
        self.idxs = [i for i, (_, y) in enumerate(base_ds) if y in self.cls_ids]
        self.label_map = {old: i for i, old in enumerate(sorted(self.cls_ids))}

    def __len__(self) -> int:  # noqa: D401 – obvious return
        return len(self.idxs)

    def __getitem__(self, idx):
        x, y = self.base[self.idxs[idx]]
        return x, self.label_map[y]

################################################################################
#                        ───  DOWNLOAD  HELPERS ───                             #
################################################################################

def _safe_download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    try:
        print(f"[DL] {dest.name} …")
        urlretrieve(url, dest)
    except Exception as e:  # pragma: no cover – network errors
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading {url}: {e}") from e


def download_and_prepare_cifar100(cfg):
    _safe_download(cfg["data"]["cifar100"]["url"], DATA_DIR / "cifar-100-python.tar.gz")
    tgt = DATA_DIR / "cifar-100-python"
    if not tgt.exists():
        tarfile.open(DATA_DIR / "cifar-100-python.tar.gz").extractall(DATA_DIR)


def download_and_prepare_miniimagenet(cfg):
    _safe_download(cfg["data"]["miniimagenet"]["url"], DATA_DIR / "miniImageNet.zip")
    tgt = DATA_DIR / "miniImageNet"
    if not tgt.exists():
        zipfile.ZipFile(DATA_DIR / "miniImageNet.zip").extractall(DATA_DIR)


def download_and_prepare_rotmnist(cfg):
    _safe_download(cfg["data"]["rotmnist"]["url"], DATA_DIR / "rot_mnist.npy")


def download_and_prepare_speechcommands(cfg):
    _safe_download(cfg["data"]["speech"]["url"], DATA_DIR / "speech_commands_v0.02.tar.gz")
    tgt = DATA_DIR / "SpeechCommands"
    if not tgt.exists():
        tarfile.open(DATA_DIR / "speech_commands_v0.02.tar.gz").extractall(tgt)


def ensure_all_datasets(cfg):
    """Public utility called by main.py – downloads all datasets once."""
    download_and_prepare_cifar100(cfg)
    download_and_prepare_miniimagenet(cfg)
    download_and_prepare_rotmnist(cfg)
    download_and_prepare_speechcommands(cfg)

################################################################################
#                       ───  SPLIT-CIFAR-100  (20×5) ───                       #
################################################################################

def build_cifar_splits(cfg, *, seed: int) -> List[Tuple[Dataset, Dataset, Dataset]]:
    """Return (train,val,test) datasets for each of the 20 tasks with 5 classes."""
    tr = T.Compose(
        [T.ToTensor(), T.Normalize(cfg["data"]["cifar100"]["mean"], cfg["data"]["cifar100"]["std"])]
    )
    full_train = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=True, download=False, transform=tr
    )
    full_test = torchvision.datasets.CIFAR100(
        root=DATA_DIR, train=False, download=False, transform=tr
    )

    random.seed(seed)
    classes = list(range(100))
    random.shuffle(classes)
    tasks = [classes[i : i + 5] for i in range(0, 100, 5)]

    task_datasets = []
    for cls in tasks:
        train_ds = SplitDataset(full_train, cls)
        test_ds = SplitDataset(full_test, cls)
        # 90/10 split for validation
        val_sz = int(0.1 * len(train_ds))
        train_ds, val_ds = random_split(
            train_ds,
            [len(train_ds) - val_sz, val_sz],
            generator=torch.Generator().manual_seed(seed),
        )
        task_datasets.append((train_ds, val_ds, test_ds))
    return task_datasets
