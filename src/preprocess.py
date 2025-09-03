"""src/preprocess.py – data acquisition and stream construction"""
from __future__ import annotations
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, random_split
import torchvision
import torchvision.transforms as T

# -------------------------------------------------------------
#                     Project-wide paths
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
#                     Helper: safe download
# -------------------------------------------------------------

def _download(url: str, dest: Path):
    """Download URL → dest if *dest* does not exist."""
    if dest.exists():
        return
    print(f"[DL] {url.split('/')[-1]} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


# -------------------------------------------------------------
#                     CIFAR-100 continual stream
# -------------------------------------------------------------
class _Split(Dataset):
    """Subset of *base* containing only classes in *cls_ids* (labels re-indexed)."""

    def __init__(self, base: Dataset, cls_ids: List[int]):
        self.base = base
        self.map = {c: i for i, c in enumerate(sorted(cls_ids))}
        self.idxs = [i for i, (_, y) in enumerate(base) if y in self.map]

    # -------------------- standard dataset API -------------------------
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        x, y = self.base[self.idxs[idx]]
        return x, self.map[y]


# ----------------------------------------------------------------------
#                            public API
# ----------------------------------------------------------------------

def cifar_stream(cfg, seed: int) -> List[Tuple[Dataset, Dataset, Dataset]]:
    """Return list of (train, val, test) per task – 20 tasks × 5 classes."""

    # --- make sure raw data is available ----------------------------------
    cifar_dir = DATA / "cifar-100-python"
    if not cifar_dir.exists():
        tgz = DATA / "cifar100.tar.gz"
        _download(cfg.data.cifar100["url"], tgz)
        tarfile.open(tgz).extractall(DATA)

    # ---------------------- torchvision datasets -------------------------
    tr = T.Compose(
        [T.ToTensor(), T.Normalize(cfg.data.cifar100["mean"], cfg.data.cifar100["std"])]
    )
    train_set = torchvision.datasets.CIFAR100(DATA, train=True, transform=tr, download=False)
    test_set = torchvision.datasets.CIFAR100(DATA, train=False, transform=tr, download=False)

    cls_ids = list(range(100))
    random.Random(seed).shuffle(cls_ids)
    tasks = [cls_ids[i : i + cfg.model.num_classes_per_task] for i in range(0, 100, cfg.model.num_classes_per_task)]

    out = []
    for t in tasks:
        tr_ds = _Split(train_set, t)
        te_ds = _Split(test_set, t)
        val_size = int(0.05 * len(tr_ds))
        tr_ds, val_ds = random_split(
            tr_ds,
            [len(tr_ds) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        out.append((tr_ds, val_ds, te_ds))
    return out
