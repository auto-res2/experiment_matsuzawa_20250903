"""src/preprocess.py – data downloading, transforms and task creation."""
from __future__ import annotations

import io
import pathlib
import zipfile
from typing import List, Tuple

import requests
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms as T

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

DATA_DIR = pathlib.Path("data").absolute()
DATA_DIR.mkdir(exist_ok=True)

# Mean/std need three channels for RGB datasets (e.g. CIFAR, Tiny-ImageNet)
_RGB_MEAN = (0.5, 0.5, 0.5)
_RGB_STD = (0.5, 0.5, 0.5)

TRANSFORMS = {
    "train": T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_RGB_MEAN, _RGB_STD),
        ]
    ),
    "test": T.Compose(
        [
            T.ToTensor(),
            T.Normalize(_RGB_MEAN, _RGB_STD),
        ]
    ),
}

# -----------------------------------------------------------------------------
# download helpers
# -----------------------------------------------------------------------------

def download_and_prepare() -> None:
    """Download Tiny-ImageNet and ensure other datasets are present.
    Torchvision will take care of CIFAR-100 automatically. We implement
    Tiny-ImageNet manually because it isn't available via torchvision.
    """

    tiny_root = DATA_DIR / "tiny-imagenet-200"
    if tiny_root.exists():
        return  # already present

    url = "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1"
    print("Downloading Tiny-ImageNet … (could take a moment)")
    r = requests.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed to download Tiny-ImageNet – HTTP %d" % r.status_code)

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(DATA_DIR)
    print("Tiny-ImageNet extracted to", tiny_root)

# -----------------------------------------------------------------------------
# task splits (CIFAR-100 example used in Experiment-1)
# -----------------------------------------------------------------------------

def split_cifar100() -> Tuple[List[torch.utils.data.Dataset], torch.utils.data.Dataset]:
    ds_train = torchvision.datasets.CIFAR100(
        DATA_DIR,
        train=True,
        download=True,
        transform=TRANSFORMS["train"],
    )
    ds_test = torchvision.datasets.CIFAR100(
        DATA_DIR,
        train=False,
        download=True,
        transform=TRANSFORMS["test"],
    )

    # build {class → [indices]} mapping
    cls_to_idx = {cls: [] for cls in range(100)}
    for idx, (_, y) in enumerate(ds_train):
        cls_to_idx[y].append(idx)

    tasks: List[Subset] = []
    for t in range(20):
        classes = list(range(t * 5, (t + 1) * 5))
        indices = sum([cls_to_idx[c] for c in classes], [])
        tasks.append(Subset(ds_train, indices))

    return tasks, ds_test
