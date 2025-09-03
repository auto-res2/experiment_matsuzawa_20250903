"""src/preprocess.py – dataset download & preprocessing (revised)
Fixes:
1. Robust Waterbirds preparation – rely on WILDS helper instead of a shallow Git
   clone that misses Git-LFS objects.  This guarantees the presence of image
   files and avoids FileNotFoundError on `data/waterbirds/train`.
2. Provide a lightweight wrapper around WILDS subsets so that the rest of the
   pipeline (which expects `(image, label)` pairs and a `.classes` attribute)
   remains unchanged.
"""
from __future__ import annotations

import subprocess, requests, time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Tuple

import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CelebA
from torch.utils.data import random_split, Dataset

# NEW: WILDS for reliable Waterbirds download ----------------------------------
from wilds import get_dataset as _wilds_get_dataset  # type: ignore

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


def _download(url: str, target: Path) -> Path:  # retained for potential future use
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

# NOTE: The previous implementation attempted to `git clone` a HuggingFace repo
# which uses Git-LFS – large binary objects are *not* fetched by default, hence
# the missing `train/` directory.  We now leverage the authoritative WILDS
# helper which downloads a tarball and extracts images properly.


def prepare_waterbirds(cfg: Dict[str, Any]) -> Path:
    """Download Waterbirds via WILDS if not cached, return root path."""
    root = DATA_ROOT / "waterbirds_wilds"
    if root.exists():
        return root

    print("[PREP] Download Waterbirds via WILDS …")
    # WILDS will place files inside `root` – we pass the path explicitly.
    _wilds_get_dataset(dataset="waterbirds", download=True, root_dir=str(root))
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

# -----------------------------------------------------------------------------
# Wrapper around WILDS subsets so that they behave like standard torchvision
# datasets (return (image, label) and expose `.classes`).  This avoids touching
# the rest of the training pipeline.
# -----------------------------------------------------------------------------

class _WildsSubsetWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset
        # Waterbirds has two classes (0=landbird, 1=waterbird)
        self.classes = list(range(subset.dataset.n_classes))  # type: ignore[attr-defined]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y, _ = self.subset[idx]
        return x, y

# --------------------------------------------------
# Public dataset retrieval API
# --------------------------------------------------

def get_dataset(name: str, cfg: Dict[str, Any]):
    """Return (train,val,test) datasets for given *name*."""
    tfms_train = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tfms_test = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if name not in PREP_FN:
        raise KeyError(f"Dataset {name} not supported in this demo.")

    root = PREP_FN[name](cfg["datasets"][name])

    # -------------------- Waterbirds (via WILDS) --------------------
    if name == "waterbirds":
        from wilds import get_dataset as _get_ds  # local import to keep global deps minimal

        ds = _get_ds(dataset="waterbirds", root_dir=str(root))
        train_ds = _WildsSubsetWrapper(ds.get_subset("train", transform=tfms_train))
        val_ds = _WildsSubsetWrapper(ds.get_subset("val", transform=tfms_test))
        test_ds = _WildsSubsetWrapper(ds.get_subset("test", transform=tfms_test))
        return train_ds, val_ds, test_ds

    # -------------------- CelebA (torchvision) ----------------------
    if name == "celeba":
        celeba_all = CelebA(root=str(DATA_ROOT), split="all")
        celeba_all.transform = tfms_train
        n = len(celeba_all)
        n_val = int(0.1 * n)
        n_test = int(0.1 * n)
        train_ds, val_ds, test_ds = random_split(
            celeba_all, [n - n_val - n_test, n_val, n_test]
        )
        return train_ds, val_ds, test_ds

    # ----------------------------------------------------------------
    raise KeyError(f"Dataset {name} not implemented")
