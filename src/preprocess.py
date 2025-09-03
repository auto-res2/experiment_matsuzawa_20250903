"""src/preprocess.py – dataset download & preprocessing"""
from __future__ import annotations

import subprocess, tarfile, zipfile, os, requests, random, time
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Dict, Any

import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CelebA
from torch.utils.data import random_split

# --------------------------------------------------
# Utility timer (local)
# --------------------------------------------------

@contextmanager
def timing(msg: str):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print(f"[TIMER] {msg}: {toc - tic:.2f}s")

# --------------------------------------------------
# Data root and helpers
# --------------------------------------------------

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True, parents=True)


def _download(url: str, target: Path) -> Path:
    if target.exists():
        return target
    print(f"[DOWNLOAD] {url} -> {target}")
    with timing(f"download {url}"):
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return target

# --------------------------------------------------
# Dataset-specific preparation routines
# --------------------------------------------------

def prepare_waterbirds(cfg: Dict[str, Any]) -> Path:
    root = DATA_ROOT / "waterbirds"
    if root.exists():
        return root
    print("[PREP] Clone Waterbirds dataset …")
    code = subprocess.call(["git", "clone", "--depth", "1", cfg["url"], str(root)])
    if code != 0:
        raise RuntimeError("Waterbirds clone failed")
    return root


def prepare_celeba(cfg: Dict[str, Any]) -> Path:
    root = DATA_ROOT / "celeba"
    if root.exists():
        return root
    print("[PREP] Download CelebA via torchvision …")
    CelebA(root=str(DATA_ROOT), split="all", download=True)
    return root

PREP_FN = {
    "waterbirds": prepare_waterbirds,
    "celeba": prepare_celeba,
}

# --------------------------------------------------
# Public dataset retrieval API
# --------------------------------------------------

def get_dataset(name: str, cfg: Dict[str, Any]):
    """Return (train,val,test) datasets for given name."""
    tfms_train = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tfms_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if name not in PREP_FN:
        raise KeyError(f"Dataset {name} not supported in this demo.")

    root = PREP_FN[name](cfg["datasets"][name])

    if name == "waterbirds":
        full = ImageFolder(root / "train")
        n = len(full)
        n_val = int(0.1 * n)
        train_ds, val_ds = random_split(full, [n - n_val, n_val])
        test_ds = ImageFolder(root / "test")
        # attach transforms
        for ds in (train_ds, val_ds):
            ds.dataset.transform = tfms_train  # type: ignore
        test_ds.transform = tfms_test
        return train_ds, val_ds, test_ds

    if name == "celeba":
        celeba_all = CelebA(root=str(DATA_ROOT), split="all")
        # For brevity we treat CelebA as single split demo
        celeba_all.transform = tfms_train
        n = len(celeba_all)
        n_val = int(0.1 * n)
        n_test = int(0.1 * n)
        train_ds, val_ds, test_ds = random_split(
            celeba_all, [n - n_val - n_test, n_val, n_test]
        )
        return train_ds, val_ds, test_ds

    # fallback
    raise KeyError(f"Dataset {name} not implemented")
