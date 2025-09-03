# src/preprocess.py
# -*- coding: utf-8 -*-
"""Data loading & preprocessing utilities."""
from __future__ import annotations
import tarfile, urllib.request as urlreq, random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as tvf

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Dataset classes
# -----------------------------------------------------------------------------

class WaterbirdsDataset(Dataset):
    """Stanford Waterbirds dataset (pre-processed 224×224)."""
    base_dir = DATA_DIR / "waterbirds"

    def __init__(self, split: str, transform):
        assert split in {"train", "val", "test"}
        self.transform = transform
        self._ensure_download()
        meta = pd.read_csv(self.base_dir / "metadata.csv")
        split_idx = {"train": 0, "val": 1, "test": 2}[split]
        self.samples = meta[meta["split"] == split_idx]
        self.img_paths = self.samples["img_filename"].tolist()
        self.labels = self.samples["y"].values.astype(np.int64)
        self.groups = self.samples["place"].values.astype(np.int64)  # land=0, water=1

    # ------------------------------------------------------------------
    #  standard Dataset stuff
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.base_dir / self.img_paths[idx]
        with open(img_path, "rb") as _:
            img = torchvision.io.read_image(str(img_path))  # CHW uint8
        img = img.float() / 255.0
        img = self.transform(img) if self.transform else img
        return img, self.labels[idx], self.groups[idx]

    # ------------------------------------------------------------------
    #  helper
    # ------------------------------------------------------------------
    @classmethod
    def _ensure_download(cls):
        if (cls.base_dir / "metadata.csv").exists():
            return
        print("Downloading Waterbirds dataset …")
        tar_path = DATA_DIR / "waterbirds.tar.gz"
        url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
        try:
            urlreq.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(DATA_DIR)
        except Exception as e:
            raise RuntimeError("Failed to download Waterbirds dataset") from e
        finally:
            if tar_path.exists():
                tar_path.unlink(missing_ok=True)
        print("Waterbirds download complete.")

# -----------------------------------------------------------------------------
#  CelebA single-attribute (binary) dataset
# -----------------------------------------------------------------------------

class CelebABinaryAttr(Dataset):
    def __init__(self, split: str, attr: str, transform):
        assert split in {"train", "val", "test"}
        self.transform = transform
        self.ds = torchvision.datasets.CelebA(root=DATA_DIR, split=split, download=True,
                                              target_type="attr")
        self.attr_idx = self.ds.attr_names.index(attr)
        self.male_idx = self.ds.attr_names.index("Male")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, attrs = self.ds[idx]
        y = int(attrs[self.attr_idx].item())
        group = int(attrs[self.male_idx].item())  # use Male attribute as pseudo group
        img = self.transform(img) if self.transform else img
        return img, y, group

# -----------------------------------------------------------------------------
#  Mini-ImageNet-9 tiny benchmark via HF datasets
# -----------------------------------------------------------------------------

class MiniImageNet9(Dataset):
    def __init__(self, split: str, transform):
        assert split in {"train", "val", "test"}
        from datasets import load_dataset
        raw = load_dataset("timm/mini-imagenet", split="train")
        raw = raw.filter(lambda ex: ex["label"] < 9)  # keep first 9 classes
        train_size = int(0.9 * len(raw))
        indices = list(range(len(raw)))
        self.sel_ids = indices[:train_size] if split == "train" else indices[train_size:]
        self.data = [raw[i] for i in self.sel_ids]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import io, PIL.Image as Image
        ex = self.data[idx]
        img = Image.open(io.BytesIO(ex["img"]))
        img = self.transform(img) if self.transform else img
        return img, int(ex["label"]), 0  # dummy group

# -----------------------------------------------------------------------------
#  Generic transform getter (ImageNet style normalisation)
# -----------------------------------------------------------------------------

def get_transforms(img_size: int, *, train: bool):
    if train:
        aug = tvf.Compose([
            tvf.ToPILImage() if train else tvf.Lambda(lambda x: x),
            tvf.Resize(int(img_size * 256 / 224)),
            tvf.RandomResizedCrop(img_size) if train else tvf.CenterCrop(img_size),
            tvf.RandomHorizontalFlip(),
            tvf.ToTensor(),
            tvf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        aug = tvf.Compose([
            tvf.ToPILImage(),
            tvf.Resize(int(img_size * 256 / 224)),
            tvf.CenterCrop(img_size),
            tvf.ToTensor(),
            tvf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return aug
