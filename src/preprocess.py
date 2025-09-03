"""src/preprocess.py
Data loading & continual-stream builders.
Implements:
• split_cifar_stream – Split-CIFAR-100 continual-learning benchmark.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Generator, Iterable, Tuple

import datasets
import numpy as np
import requests
import torch
from einops import rearrange  # noqa: F401 – used by some torchvision ops
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# ---------------------------------------------------------------------
# paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# misc helpers (very small subset of the original utils.py) -------------
_CHUNK = 1 << 15  # 32 KiB


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha1(fp: Path):
    h = hashlib.sha1()
    with open(fp, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, expected_sha1: str | None = None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_sha1 and _sha1(target) != expected_sha1:
            print(f"Checksum mismatch for {target.name} – re-downloading …")
            target.unlink()
        else:
            return target
    print(f"Downloading {url} → {target}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tot = int(r.headers.get("content-length", 0))
        with tqdm(total=tot, unit="B", unit_scale=True, desc=target.name) as bar, open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    if expected_sha1:
        assert _sha1(target) == expected_sha1, "SHA-1 mismatch after download"
    return target


def extract(archive: Path, dst: Path):
    if dst.exists():
        return dst
    import tarfile, zipfile

    print(f"Extracting {archive.name} …")
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            tar.extractall(dst)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
    else:
        raise ValueError("Unsupported archive format")
    return dst

# ---------------------------------------------------------------------
# 1) Split-CIFAR-100 stream (edge-device scenario) ----------------------
_aug_cifar = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

_CIFAR_ARCHIVE = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
_CIFAR_PATH = DATA_ROOT / "cifar100"


def split_cifar_stream(tasks: int = 20, seed: int = 0):
    """Yield `(task_id, dataloader, class_subset)` for Split-CIFAR-100."""
    set_seed(seed)
    if not _CIFAR_PATH.exists():
        tar = download(_CIFAR_ARCHIVE, DATA_ROOT / "cifar100.tar.gz")
        extract(tar, _CIFAR_PATH)

    ds = CIFAR100(root=str(_CIFAR_PATH), train=True, download=False, transform=_aug_cifar)
    cls_order = list(range(100))
    random.shuffle(cls_order)
    per_task = 100 // tasks

    for t in range(tasks):
        cls_subset = set(cls_order[t * per_task : (t + 1) * per_task])
        idx = [i for i, (_, y) in enumerate(ds) if y in cls_subset]
        loader = DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        yield t, loader, cls_subset
