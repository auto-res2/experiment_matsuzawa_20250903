"""src/preprocess.py
Dataset loading and general reproducibility helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import torch

# ------------------------------------------------------------------
#  Optional torch-geometric dependency (see detailed explanation in
#  ``src/train.py``).  We provide ultra-light fallbacks to keep the CI runtime
#  free from heavyweight compiled wheels.
# ------------------------------------------------------------------
try:
    from torch_geometric.datasets import Planetoid  # type: ignore
    from torch_geometric.data import Data as _TGData  # type: ignore
    from torch_geometric.utils import to_undirected, add_self_loops  # type: ignore
    _TG_AVAILABLE = True
except Exception:  # pragma: no cover – fallback stubs
    _TG_AVAILABLE = False

    class _TGData:  # minimal stand-in for torch_geometric.data.Data
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to(self, *_, **__):
            return self  # no-op for device transfer

    Planetoid = None  # will trigger synthetic-graph fallback later

    def to_undirected(edge_index):  # noqa: D401
        return edge_index

    def add_self_loops(edge_index):  # noqa: D401
        return edge_index, None

# ------------------------------------------------------------------
#  PROJECT-WIDE PATHS
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

################################################################################
#  REPRODUCIBILITY
################################################################################

def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
#  UTILS
################################################################################

def _row_norm(x: torch.Tensor):
    x = x.float()
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return x / row_sum

################################################################################
#  DATASET LOADERS
################################################################################

def _post_process(data: _TGData):
    data.x = _row_norm(data.x)
    data.edge_index = add_self_loops(to_undirected(data.edge_index))[0]
    return data


def load_cora() -> _TGData:
    if not _TG_AVAILABLE:
        raise RuntimeError("torch_geometric not installed – cannot load real dataset")
    ds = Planetoid(root=str(DATA_DIR / "Cora"), name="Cora")
    return _post_process(ds[0])


def load_pubmed() -> _TGData:
    if not _TG_AVAILABLE:
        raise RuntimeError("torch_geometric not installed – cannot load real dataset")
    ds = Planetoid(root=str(DATA_DIR / "Pubmed"), name="Pubmed")
    return _post_process(ds[0])


def load_wisconsin() -> _TGData:
    if not _TG_AVAILABLE:
        raise RuntimeError("torch_geometric not installed – cannot load real dataset")
    ds = Planetoid(root=str(DATA_DIR / "Wisconsin"), name="Wisconsin")
    return _post_process(ds[0])
