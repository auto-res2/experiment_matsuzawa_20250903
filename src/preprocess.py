"""src/preprocess.py
Data-related helper utilities: deterministic seeding, download & extraction
of raw dataset archives and directory bookkeeping.
"""
from __future__ import annotations

import hashlib
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np

# -----------------------------------------------------------------------------
#  Directory setup (updated figure path for iteration-3) -----------------------
# -----------------------------------------------------------------------------
root_dir = Path(".")
data_dir = root_dir / "data"
checkpoints_dir = root_dir / "checkpoints"
images_dir = root_dir / ".research/iteration3/images"

for _d in [data_dir, checkpoints_dir, images_dir]:
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Global utilities
# -----------------------------------------------------------------------------

def set_global_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#  Hash helpers (SHA-256 & MD5 fallback) ---------------------------------------
# -----------------------------------------------------------------------------

def _sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):  # noqa: B023
            h.update(chunk)
    return h.hexdigest()


def _md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):  # noqa: B023
            h.update(chunk)
    return h.hexdigest()


_DATASETS: Dict[str, Dict[str, str]] = {
    "mnist": {
        # original `ylecun/mnist` repo changed – torchvision can download on-demand
        "url": "https://huggingface.co/datasets/mnist/resolve/main/train-images-idx3-ubyte.gz",
        "sha256": "25ae6c6c5e834b631c7ad8886e5fcd9b21ca01943ceafe68137e77cce14700b5",
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        # keep MD5 for backward compatibility – handled explicitly in _download()
        "sha256": "eb9058c3a382ffc7106e4002c42a8d85",
    },
    "miniimagenet": {
        "url": "https://huggingface.co/datasets/timm/mini-imagenet/resolve/main/mini-imagenet.tar.gz",
        "sha256": "c00de98ce179cf32292b01937db114d9",
    },
    "tinyimagenet": {
        "url": "https://huggingface.co/datasets/slegroux/tiny-imagenet-200-clean/resolve/main/tiny-imagenet-200.tar.gz",
        "sha256": "2a3cb29d0e2031fb2d6c1ba4019a4517",
    },
}

# -----------------------------------------------------------------------------
#  Internal helpers ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _download(url: str, dest: Path, expected_hash: str):
    """Download *url* to *dest* and verify integrity (SHA-256 or MD5)."""

    import tempfile

    # Already present and valid – nothing to do.
    if dest.exists():
        if (len(expected_hash) == 64 and _sha256sum(dest) == expected_hash) or (
            len(expected_hash) == 32 and _md5sum(dest) == expected_hash
        ):
            return
        # hash mismatch – re-download
        dest.unlink()

    print(f"Downloading {url} → {dest}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        try:
            urllib.request.urlretrieve(url, tmp.name)
        except Exception as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        tmp_path = Path(tmp.name)
        # verify
        if len(expected_hash) == 64:  # SHA-256
            valid = _sha256sum(tmp_path) == expected_hash
        elif len(expected_hash) == 32:  # MD5
            valid = _md5sum(tmp_path) == expected_hash
        else:
            raise ValueError("Expected hash must be MD5 (32 hex) or SHA-256 (64 hex)")
        if not valid:
            raise RuntimeError(f"Checksum mismatch for {dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(dest)


# -----------------------------------------------------------------------------
#  Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _map_stream_to_raw(name: str) -> str:
    """Map high-level stream names (e.g. *split_cifar100*) to raw dataset keys."""
    if name.startswith("split_cifar100"):
        return "cifar100"
    if name.startswith("split_miniimagenet") or name.startswith("miniimagenet"):
        return "miniimagenet"
    if name.startswith("tinyimagenet"):
        return "tinyimagenet"
    if name.startswith("permuted_mnist") or name.startswith("mnist"):
        return "mnist"
    raise NotImplementedError(f"Unrecognised dataset stream '{name}'")


def acquire_datasets(cfg):  # noqa: D401 – public API
    """Download (if necessary) and extract all datasets referenced in *cfg*."""

    # Determine which raw datasets are required by the set of experiments
    required = { _map_stream_to_raw(exp["dataset"]) for exp in cfg["experiments"] }

    for name in required:
        meta = _DATASETS[name]
        url, hash_val = meta["url"], meta["sha256"]
        file_name = Path(url).name
        out_file = data_dir / file_name
        _download(url, out_file, hash_val)

        # ---------------- extraction ---------------------------------
        if tarfile.is_tarfile(out_file):
            with tarfile.open(out_file) as tar:
                top_level = tar.getmembers()[0].name.split("/")[0]
            marker_dir = data_dir / top_level
            if marker_dir.exists():
                continue  # already extracted

            print(f"Extracting {out_file} → {data_dir}")
            with tarfile.open(out_file) as tar:
                tar.extractall(data_dir)
        # .gz files for MNIST stay compressed – torchvision handles them.
