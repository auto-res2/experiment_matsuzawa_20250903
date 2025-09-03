import os
from typing import Dict
from pathlib import Path

import torch_geometric as tg
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import add_self_loops  # noqa: F401  (side-effect import)

# ogb is optional – only required for ogbn-arxiv
try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    PygNodePropPredDataset = None  # type: ignore


DATASET_URLS: Dict[str, str] = {
    "Cora": "https://github.com/kimiyoung/planetoid/raw/master/data",
    "Citeseer": "https://github.com/kimiyoung/planetoid/raw/master/data",
    "PubMed": "https://github.com/kimiyoung/planetoid/raw/master/data",
    "Chameleon": "https://github.com/graphdml-uiuc-jlu/geom-gcn/raw/master/new_data",
    "Squirrel": "https://github.com/graphdml-uiuc-jlu/geom-gcn/raw/master/new_data",
    "ogbn-arxiv": "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
}


def load_dataset(name: str, data_root: str):
    """Download / load dataset using PyG helpers."""
    root = os.path.join(data_root, name)
    if name in ["Cora", "Citeseer", "PubMed"]:
        ds = Planetoid(root, name)
    elif name in ["Chameleon", "Squirrel"]:
        ds = WikipediaNetwork(root, name, geom_gcn_split=True)
    elif name == "ogbn-arxiv":
        if PygNodePropPredDataset is None:
            raise ImportError("ogb is not installed – required for ogbn-arxiv")
        ds = PygNodePropPredDataset(name, root)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds
