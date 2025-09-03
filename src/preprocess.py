"""src/preprocess.py – data acquisition & loaders"""
from __future__ import annotations

import shutil
import tarfile
import urllib.request as urllib
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F

# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#   Generic downloader helpers
# ---------------------------------------------------------------------------

def download_url(url: str, save_path: Path):
    if save_path.exists():
        return
    print(f"Downloading {url} → {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.urlopen(url) as resp, open(save_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)
    assert save_path.exists() and save_path.stat().st_size > 0, "Download failed!"


def fetch_and_extract(url: str, extract_to: Path):
    filename = url.split("/")[-1]
    tmp_path = extract_to / filename
    download_url(url, tmp_path)
    print(f"Extracting {tmp_path} …")
    if filename.endswith((".tgz", ".tar.gz")):
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(extract_to)
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(extract_to)
    else:
        raise RuntimeError(f"Unknown archive type: {filename}")

# ---------------------------------------------------------------------------
#   Built-in loaders (extend as needed)
# ---------------------------------------------------------------------------

def load_cora() -> Data:
    dataset = Planetoid(root=str(DATA_DIR / "Cora"), name="Cora", transform=NormalizeFeatures())
    return dataset[0]


def load_chameleon() -> Data:  # noqa: D401
    npz_path = DATA_DIR / "chameleon.npz"
    if not npz_path.exists():
        download_url(
            "https://github.com/bingzhewei/geom-gcn/raw/master/new_data/chameleon.npz",
            npz_path,
        )
    npz = np.load(npz_path, allow_pickle=True)
    x = torch.from_numpy(npz["features"]).float()
    edge_index = torch.from_numpy(npz["edge_index"]).long()
    y = torch.from_numpy(npz["labels"]).long()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    data = Data(x=x, edge_index=edge_index, y=y)
    # build 60/20/20 splits
    masks = []
    for _ in range(10):
        idx = torch.randperm(data.num_nodes)
        n = data.num_nodes
        tr, va, te = idx[: int(0.6 * n)], idx[int(0.6 * n) : int(0.8 * n)], idx[int(0.8 * n) :]
        mtr = torch.zeros(n, dtype=torch.bool)
        mva = mtr.clone()
        mte = mtr.clone()
        mtr[tr] = True
        mva[va] = True
        mte[te] = True
        masks.append((mtr, mva, mte))
    data.masks = masks  # type: ignore[attr-defined]
    return data


def load_ogbn_arxiv():  # noqa: D401
    dataset = PygNodePropPredDataset("ogbn-arxiv", root=str(DATA_DIR / "ogbn_arxiv"))
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.x = F.normalize(data.x, p=1, dim=-1)
    return data, split_idx["train"], split_idx["valid"], split_idx["test"]
