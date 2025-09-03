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

# Make sure that *all* project-wide directories exist early.
root_dir = Path(".")
data_dir = root_dir / "data"
checkpoints_dir = root_dir / "checkpoints"
images_dir = root_dir / ".research/iteration1/images"

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

def _download(url: str, dest: Path, expected_sha256: str):
    """Download *url* to *dest* and check SHA-256 integrity."""

    import tempfile

    if dest.exists() and _sha256sum(dest) == expected_sha256:
        return  # already downloaded & verified

    print(f"Downloading {url} → {dest}")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = Path(tmp.name)
            if _sha256sum(tmp_path) != expected_sha256:
                raise RuntimeError(f"Checksum mismatch for {dest.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------

def acquire_datasets(cfg):
    """Download (if necessary) and extract all datasets referenced in *cfg*."""

    for name, meta in _DATASETS.items():
        url, sha256 = meta["url"], meta["sha256"]
        out_file = data_dir / Path(url).name
        _download(url, out_file, sha256)

        # ---------------- extraction ---------------------------------
        if out_file.suffix in {".gz", ".tar"}:
            target_dir = data_dir / name
            if target_dir.exists():
                continue  # already extracted
            print(f"Extracting {out_file} → {target_dir}")
            target_dir.mkdir(exist_ok=True)
            try:
                if tarfile.is_tarfile(out_file):
                    with tarfile.open(out_file) as tar:
                        tar.extractall(target_dir)
                else:
                    import gzip, shutil

                    with gzip.open(out_file, "rb") as f_in, open(target_dir / name, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                raise RuntimeError(f"Failed to extract {out_file}: {e}")
