"""
preprocess.py – dataset download/extraction and DataLoader construction.
Only the datasets explicitly referenced in the original script are supported.
"""
from __future__ import annotations
import tarfile, zipfile, shutil, requests, os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import yaml

# -----------------------------------------------------------------------------
# configuration & constants
# -----------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CFG_PATH) as f:
    CONFIG = yaml.safe_load(f)

_DATA_ROOT = Path("data"); _DATA_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# helper – download with resume & sanity check
# -----------------------------------------------------------------------------

def _download(url: str, dest: Path, chunk: int = 1024 * 1024):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        if not r.ok:
            raise RuntimeError(f"Failed to download {url}")
        with open(dest, "wb") as f:
            for chunk_data in r.iter_content(chunk):
                if chunk_data:
                    f.write(chunk_data)
    return dest

# -----------------------------------------------------------------------------
# torchvision transforms
# -----------------------------------------------------------------------------
_mean, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_train_tf = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandAugment(2, 9),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_mean, _std),
])
_test_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_mean, _std),
])

# -----------------------------------------------------------------------------
# individual dataset loaders ---------------------------------------------------
# -----------------------------------------------------------------------------

def _load_wilds_dataset(name: str):
    from wilds import get_dataset as _get_wilds
    ds = _get_wilds(name, root_dir=str(_DATA_ROOT / name), download=True)
    tr = ds.get_subset("train", transform=_train_tf)
    va = ds.get_subset("val", transform=_test_tf)
    te = ds.get_subset("test", transform=_test_tf)
    return {"train": tr, "val": va, "test": te}, ds.n_classes, True


def _load_imagenet9():
    url = CONFIG["datasets"]["imagenet9"]["url"]
    tar_path = _download(url, _DATA_ROOT / "imagenet9.tar.gz")
    if not (_DATA_ROOT / "imagenet9").exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(_DATA_ROOT / "imagenet9")
    from robustness_bench.datasets.imagenet9 import ImageNet9
    tr = ImageNet9(_DATA_ROOT / "imagenet9", split="train", transform=_train_tf)
    va = ImageNet9(_DATA_ROOT / "imagenet9", split="val", transform=_test_tf)
    te = ImageNet9(_DATA_ROOT / "imagenet9", split="test", transform=_test_tf)
    return {"train": tr, "val": va, "test": te}, 9, False


def _load_ninco():
    url = CONFIG["datasets"]["ninco"]["url"]
    tar_path = _download(url, _DATA_ROOT / "ninco.tar.gz")
    if not (_DATA_ROOT / "ninco").exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(_DATA_ROOT / "ninco")
    from robustness_bench.datasets.ninco import NINCO
    tr = NINCO(_DATA_ROOT / "ninco", split="train", transform=_train_tf)
    va = NINCO(_DATA_ROOT / "ninco", split="val", transform=_test_tf)
    te = NINCO(_DATA_ROOT / "ninco", split="test", transform=_test_tf)
    return {"train": tr, "val": va, "test": te}, tr.num_classes, False

# registry ------------------------------------------------------
_LOADERS: Dict[str, callable] = {
    "waterbirds": lambda: _load_wilds_dataset("waterbirds"),
    "celeba": lambda: _load_wilds_dataset("celebA"),
    "imagenet9": _load_imagenet9,
    "ninco": _load_ninco,
}

# -----------------------------------------------------------------------------
# public API – returns ready DataLoaders
# -----------------------------------------------------------------------------

def get_loaders(name: str, batch: int, workers: int):
    if name not in _LOADERS:
        raise RuntimeError(f"Dataset {name} not implemented")
    dsets, n_cls, has_groups = _LOADERS[name]()
    loaders = {
        split: DataLoader(ds, batch_size=batch, shuffle=(split == "train"), num_workers=workers, pin_memory=True)
        for split, ds in dsets.items()
    }
    return loaders, n_cls, has_groups
