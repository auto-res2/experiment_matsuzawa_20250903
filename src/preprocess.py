"""src/preprocess.py
Dataset download & loading utilities.  Encapsulates all logic that touches
external storage/network so that the rest of the pipeline stays I/O-free.
"""
from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

# -----------------------------------------------------------------------------
#   1)  GEOM-GCN heterophily datasets
# -----------------------------------------------------------------------------
_GEOM_URL_BASE = "https://github.com/bingzhewei/geom-gcn/raw/master/new_data/"
_GDIR = os.path.join("data", "geom")
os.makedirs(_GDIR, exist_ok=True)


def _download_geom_npz(name: str) -> str:
    fpath = os.path.join(_GDIR, f"{name}.npz")
    if os.path.exists(fpath):
        return fpath
    url = _GEOM_URL_BASE + f"{name}.npz"
    print(f"[preprocess] Downloading {name} ↘ {url}")
    urllib.request.urlretrieve(url, fpath)
    return fpath


def load_geom_dataset(name: str):
    """Return *torch_geometric.data.Data* for a GEOM-GCN small heterophilic graph."""
    npz_path = _download_geom_npz(name.lower())
    obj = np.load(npz_path, allow_pickle=True)

    A = obj["adjacency_matrix"].item() if "adjacency_matrix" in obj else obj["adj"].item()
    X = obj["node_features"] if "node_features" in obj else obj["feat"]
    Y = obj["node_labels"] if "node_labels" in obj else obj["label"]

    edge_index, _ = from_scipy_sparse_matrix(A)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, num_nodes=x.shape[0])
    # masks will be set later (same splits as the paper)
    return data

# -----------------------------------------------------------------------------
#   2)  Generic dataset loader that the Trainer will call
# -----------------------------------------------------------------------------

def load_dataset(spec: Dict[str, Any]):
    """Dispatcher that wraps Planetoid / OGB / GEOM loaders behind one API."""
    name = spec["name"]
    loader = spec["loader"]

    if loader == "Planetoid":
        from torch_geometric.datasets import Planetoid

        root = os.path.join("data", name)
        ds = Planetoid(root=root, name=name)
        return ds[0]

    if loader == "OGB":
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name=name, root=os.path.join("data", name))
        data = ds[0]
        data.edge_index = data.edge_index.t().contiguous()  # OGB uses [E, 2]
        return data

    if loader == "Geom":
        return load_geom_dataset(name)

    raise ValueError(f"Unknown loader → {loader}")
