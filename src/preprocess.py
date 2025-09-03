"""src/preprocess.py – data downloading / preprocessing utilties"""
from __future__ import annotations
import random, tarfile, shutil
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Pandas & PIL are only needed here, keep them local to avoid global import cost
import pandas as pd
from PIL import Image

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
#                           GENERIC HELPERS
# ---------------------------------------------------------------------------

def sha256sum(fp: Path, chunk: int = 1 << 16) -> str:
    import hashlib
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download(url: str, dst: Path, sha256: str | None = None, max_retry: int = 2):
    """Light-weight robust downloader with SHA-256 verification."""
    if dst.exists() and (sha256 is None or sha256sum(dst) == sha256):
        return
    if dst.exists():
        dst.unlink()  # corrupted file
    import urllib.request

    for k in range(max_retry):
        print(f"[DL] {url} → {dst}   (attempt {k + 1}/{max_retry})")
        urllib.request.urlretrieve(url, dst)
        if sha256 is None or sha256sum(dst) == sha256:
            return
        print("[WARN] SHA-256 mismatch – retrying…")
    raise RuntimeError(f"[FATAL] download failed SHA-256 check for {url}")


# ---------------------------------------------------------------------------
#                               WATERBIRDS
# ---------------------------------------------------------------------------
class WaterbirdsDS(Dataset):
    """Torch Dataset wrapper around the Waterbirds CSV splits."""

    def __init__(self, root: Path, split: str, transform):
        csv_path = root / f"waterbird_complete95_forest2water2/{split}.csv"
        if not csv_path.exists():
            raise RuntimeError("[DATA] Waterbirds CSV missing – extraction failed")
        df = pd.read_csv(csv_path)
        self.samples = [
            (root / row["img_filename"], int(row["y"])) for _, row in df.iterrows()
        ]
        self.t = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        return self.t(x), y


# ---------------------------------------------------------------------------
#                               PREP ROUTINE
# ---------------------------------------------------------------------------

def prepare_waterbirds_datasets(cfg: Dict[str, Any]):
    wb_cfg = cfg["datasets"]["waterbirds"]
    tar_path = DATA_ROOT / "waterbirds.tar.gz"
    download(wb_cfg["url"], tar_path, wb_cfg["sha256"])

    if not (DATA_ROOT / "waterbird_complete95_forest2water2").exists():
        print("[INFO] extracting Waterbirds …")
        with tarfile.open(tar_path) as t:
            t.extractall(DATA_ROOT)

    # Smoke-test – open 100 random images to catch corruption early
    jpgs = list(
        (DATA_ROOT / "waterbird_complete95_forest2water2" / "train" / "images").glob(
            "*.jpg"
        )
    )
    random.shuffle(jpgs)
    for p in jpgs[:100]:
        Image.open(p).convert("RGB")

    tf_ssl = transforms.Compose(
        [
            transforms.RandomResizedCrop(wb_cfg["img_size"], scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tf_eval = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(wb_cfg["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ds_train_full = WaterbirdsDS(DATA_ROOT, "train", tf_ssl)
    ds_val = WaterbirdsDS(DATA_ROOT, "val", tf_eval)
    ds_test = WaterbirdsDS(DATA_ROOT, "test", tf_eval)

    return ds_train_full, ds_val, ds_test, tf_ssl
