"""
preprocess.py – data downloading, preprocessing and DataLoader assembly
All dataset-specific code is concentrated here so that extending the project to
new datasets is straightforward.
"""
from __future__ import annotations
import tarfile, zipfile, subprocess, shutil, tempfile
from pathlib import Path
from typing import Dict, Callable

import requests
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

###########################################################################
# ─── CONFIG ───────────────────────────────────────────────────────────────
###########################################################################
import yaml
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CFG_PATH, "r") as _f:
    cfg = yaml.safe_load(_f)

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)

###########################################################################
# ─── DOWNLOAD HELPERS ─────────────────────────────────────────────────────
###########################################################################

def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Failed fetching {url}")

    with open(dest, "wb") as f:
        for chunk_bytes in r.iter_content(chunk):
            if chunk_bytes:
                f.write(chunk_bytes)
    return dest

###########################################################################
# ─── TRANSFORMS ───────────────────────────────────────────────────────────
###########################################################################
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TFM_TRAIN = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandAugment(2, 9),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

TFM_EVAL = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

###########################################################################
# ─── DATASET REGISTRY ─────────────────────────────────────────────────────
###########################################################################

try:
    from wilds import get_dataset
except ImportError:  # pragma: no cover
    get_dataset = None


def _wilds_loader(name: str):
    if get_dataset is None:
        raise RuntimeError("`wilds` package missing – required for dataset loading")
    ds = get_dataset(name, root_dir=str(DATA_ROOT / name), download=True)
    return ds

# Only Waterbirds & CelebA are wired in this public snippet -----------------
_LOADERS: Dict[str, Callable] = {
    "waterbirds": lambda: _wilds_loader("waterbirds"),
    "celeba":     lambda: _wilds_loader("celebA"),
}

###########################################################################
# ─── PUBLIC API ───────────────────────────────────────────────────────────
###########################################################################

def get_loaders(ds_name: str, batch: int, workers: int):
    """Return train/val/test DataLoaders, number of classes and group flag."""
    if ds_name not in _LOADERS:
        raise NotImplementedError(f"Dataset {ds_name} not implemented – aborting as per fail-fast policy.")

    dataset = _LOADERS[ds_name]()
    tr = dataset.get_subset("train", transform=TFM_TRAIN)
    va = dataset.get_subset("val",   transform=TFM_EVAL)
    te = dataset.get_subset("test",  transform=TFM_EVAL)

    dls = {
        "train": DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True),
        "val":   DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True),
        "test":  DataLoader(te, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True),
    }
    n_cls = dataset.n_classes
    has_groups = hasattr(dataset, "_group_array")
    return dls, n_cls, has_groups
