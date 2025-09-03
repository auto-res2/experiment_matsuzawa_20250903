"""
preprocess.py – data download / preparation utilities
"""
from __future__ import annotations
import hashlib, urllib.request, tarfile, pathlib, os, random
from typing import Dict, Any

from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

DATA_ROOT = pathlib.Path("data"); DATA_ROOT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
#                Download helpers
# ----------------------------------------------------------------------------

def _md5(path: pathlib.Path) -> str:
    m = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def download_url(url: str, dst: pathlib.Path, expected_md5: str | None = None):
    if dst.exists() and (expected_md5 is None or _md5(dst) == expected_md5):
        return
    if dst.exists():
        dst.unlink()
    print(f"[DL] {url} → {dst}")
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            raise RuntimeError("gdown required for Google-Drive download: pip install gdown")
        gdown.download(url, str(dst), quiet=False)
    else:
        urllib.request.urlretrieve(url, dst)
    if expected_md5 and _md5(dst) != expected_md5:
        raise RuntimeError("MD5 checksum mismatch – download corrupted.")

# ----------------------------------------------------------------------------
#                Dataset wrappers
# ----------------------------------------------------------------------------
class WaterbirdsDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.transform = transform
        meta_file = os.path.join(root, f"waterbird_complete95_forest2water2/{split}.csv")
        if not os.path.exists(meta_file):
            raise FileNotFoundError("Waterbirds metadata CSV missing – did extraction succeed?")
        import pandas as pd
        df = pd.read_csv(meta_file)
        self.samples = [(os.path.join(root, row["img_filename"]), int(row["y"])) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


class CelebAMakeup(Dataset):
    """CelebA binary classification (Heavy_Makeup)."""

    def __init__(self, root: str, split: str, transform=None, downsample: int | None = None):
        self.dataset = datasets.CelebA(root, split=split, target_type=["attr"], download=False, transform=transform)
        self.indices = list(range(len(self.dataset)))
        if downsample and split == "train":
            random.shuffle(self.indices)
            self.indices = self.indices[:downsample]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, attr = self.dataset[real_idx]
        y = int(attr[30].item())  # Heavy_Makeup
        return x, y
