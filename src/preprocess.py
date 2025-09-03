"""
preprocess.py
~~~~~~~~~~~~~
Handles dataset download / loading and light pre-processing.  Each benchmark is
returned as a PyG *InMemory* dataset.  Fails fast if a download error occurs –
no synthetic fall-backs are used.
"""
from __future__ import annotations
import pathlib, os
from typing import Any

from torch_geometric.datasets import Planetoid, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class DatasetFactory:
    """Utility factory that abstracts away dataset construction details."""

    @staticmethod
    def get(name: str):
        name_l = name.lower()
        try:
            if name_l in {"cora", "citeseer", "pubmed"}:
                # Planetoid names are capitalised inside the loader
                return Planetoid(str(DATA_DIR), name=name_l.capitalize())
            if name_l in {"chameleon", "squirrel"}:
                return WikipediaNetwork(str(DATA_DIR), name=name_l, geom_gcn_preprocess=True)
            if name_l.startswith("ogbn"):
                return PygNodePropPredDataset(name=name_l, root=str(DATA_DIR))
        except Exception as e:
            # download failures / throttled network, etc.
            raise RuntimeError(f"Could not download or load dataset '{name}': {e}") from e
        raise ValueError(f"Dataset '{name}' is not implemented in DatasetFactory.")
