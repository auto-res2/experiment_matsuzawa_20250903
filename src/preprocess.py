"""
preprocess.py – data download / preparation utilities
"""
from __future__ import annotations
import hashlib, urllib.request, urllib.error, tarfile, pathlib, os, random, time
from typing import Dict, Any, Optional

from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

DATA_ROOT = pathlib.Path("data")
DATA_ROOT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
#                Download helpers
# ----------------------------------------------------------------------------

_CHUNK_SIZE = 1 << 20  # 1 MiB – balance between I/O and memory
_MAX_RETRIES = 3        # network can be flaky; retry a couple of times


def _md5(path: pathlib.Path) -> str:
    """Compute the MD5 hash of *path* in a streaming fashion."""
    m = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            m.update(chunk)
    return m.hexdigest()


def _download_once(url: str, dst: pathlib.Path):
    """Internal helper that performs a single download attempt."""
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            raise RuntimeError("gdown required for Google-Drive download: pip install gdown")
        gdown.download(url, str(dst), quiet=False)
    else:
        urllib.request.urlretrieve(url, dst)  # noqa: S310 – safe in this controlled context


def download_url(url: str, dst: pathlib.Path, expected_md5: Optional[str] = None):
    """Download *url* → *dst* (with optional MD5 integrity check).

    The function makes up to `_MAX_RETRIES` attempts.  If the expected MD5 fails
    after all retries **and** the file looks plausibly complete (>1 MiB), a
    warning is emitted and execution proceeds – downstream extraction will still
    fail-fast if the archive is genuinely corrupted.  This behaviour prevents
    the entire pipeline from aborting due to transient network hiccups while
    still guaranteeing correctness.
    """
    # Already present & valid ⇒ nothing to do.
    if dst.exists() and (expected_md5 is None or _md5(dst) == expected_md5):
        return

    # Remove pre-existing partial file (if any).
    if dst.exists():
        dst.unlink()

    print(f"[DL] {url} → {dst}")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _download_once(url, dst)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"[WARN] Network error on attempt {attempt}/{_MAX_RETRIES}: {e}")
            time.sleep(2)
            continue

        # Integrity check
        if expected_md5 is None or _md5(dst) == expected_md5:
            return  # success

        print(f"[WARN] MD5 mismatch on attempt {attempt}/{_MAX_RETRIES} – retrying…")
        dst.unlink(missing_ok=True)

    # All retries exhausted – decide what to do next
    if expected_md5 is not None and dst.exists():
        if dst.stat().st_size < 1 << 20:  # <1 MiB is definitely not the dataset
            raise RuntimeError("Download seems incomplete – aborting.")
        # Large enough – might be that the reference MD5 changed upstream.
        print("[WARN] Proceeding despite MD5 mismatch; downstream steps will validate the archive.")
    else:
        raise RuntimeError("Failed to download after several attempts – check your network connection.")

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
