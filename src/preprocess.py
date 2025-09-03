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
#  Directory setup (updated figure path for iteration-2) -----------------------
# -----------------------------------------------------------------------------
root_dir = Path(".")
data_dir = root_dir / "data"
checkpoints_dir = root_dir / "checkpoints"
images_dir = root_dir / ".research/iteration2/images"

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
#  Download helpers
# -----------------------------------------------------------------------------

def _sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


_DATASETS: Dict[str, Dict[str, str]] = {
    "mnist": {
        "url": "https://huggingface.co/datasets/ylecun/mnist/resolve/main/data/train-images-idx3-ubyte.gz",
        "sha256": "179d0897fc5f609cfddc0229a20f5fe5a5c6a42688829ae84b5117d22fa98ff9",
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
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

def _download(url: str, dest: Path, expected_sha256: str):
    """Download *url* to *dest* and verify SHA-256 integrity."""

    import tempfile

    # Already present and valid – nothing to do.
    if dest.exists() and _sha256sum(dest) == expected_sha256:
        return

    print(f"Downloading {url} → {dest}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        tmp_path = Path(tmp.name)
        if _sha256sum(tmp_path) != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.replace(dest)


# -----------------------------------------------------------------------------
#  Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def acquire_datasets(cfg):  # noqa: D401 – public API
    """Download (if necessary) and extract all datasets referenced in *cfg*."""

    for name, meta in _DATASETS.items():
        url, sha256 = meta["url"], meta["sha256"]
        file_name = Path(url).name
        out_file = data_dir / file_name
        _download(url, out_file, sha256)

        # ---------------- extraction ---------------------------------
        # CIFAR-100 / miniImageNet / TinyImageNet come as TAR.GZ, MNIST as GZ.
        # We only handle archives that are *not* already extracted.
        if tarfile.is_tarfile(out_file):
            # Heuristic: the first member's top-level directory is the marker.
            with tarfile.open(out_file) as tar:
                top_level = tar.getmembers()[0].name.split("/")[0]
            marker_dir = data_dir / top_level
            if marker_dir.exists():
                continue  # already extracted

            print(f"Extracting {out_file} → {data_dir}")
            with tarfile.open(out_file) as tar:
                tar.extractall(data_dir)
        # Individual .gz files (e.g. MNIST) are kept compressed – torchvision
        # will handle them directly – so no further action is required.
