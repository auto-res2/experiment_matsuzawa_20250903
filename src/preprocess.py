"""src/preprocess.py
Data loading utilities, global paths, reproducibility helpers and hard
environment checks.  Other modules import datasets & constants from here.
NOTE:  All experiment figures are now stored under
    .research/iteration14/images
as required by the CI instructions.
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import Tuple

import numpy as np
import torch
from torch_geometric.utils import add_self_loops, to_undirected

try:
    from torch_geometric.datasets import Planetoid
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torch_geometric or one of its CUDA extensions failed to import – "
        "make sure torch-scatter/sparse wheels match *exactly* torch==2.1.* "
        "and CUDA 11.8.") from e

# ----------------------------------------------------------------------------
#  ENVIRONMENT & GLOBAL PATHS
# ----------------------------------------------------------------------------

assert torch.cuda.is_available(), (
    "CUDA device is required – aborting because only CPU is visible.  "
    "Please install cudatoolkit-11.8 or ensure the CI runner exposes a GPU.")

DEVICE = torch.device("cuda")

ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
# -------------------------------------------------------------------------
#  IMPORTANT: All figures must go to .research/iteration14/images as per CI
# -------------------------------------------------------------------------
FIG_DIR = ROOT / ".research" / "iteration14" / "images"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"

for p in (DATA_DIR, FIG_DIR, LOG_DIR, CKPT_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
_CORA_NODES = 2708
_CORA_EDGES = 10556

torch.use_deterministic_algorithms(True)

# ----------------------------------------------------------------------------
#  UTILS – reproducibility
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------------------------------
#  DATA HELPERS
# ----------------------------------------------------------------------------

def _row_norm(x: torch.Tensor):
    x = x.float()
    row_sum = x.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return x / row_sum


def _prepare_pyg(ds):
    data = ds[0]
    data.x = _row_norm(data.x)
    data.edge_index = add_self_loops(to_undirected(data.edge_index))[0]
    return data


def load_cora():
    ds = Planetoid(root=str(DATA_DIR / "Cora"), name="Cora")
    data = _prepare_pyg(ds)
    # ---------- shape asserts (unit-test #1) ----------
    assert data.num_nodes == _CORA_NODES, "Cora node count mismatch – corrupted download?"
    assert (
        data.edge_index.size(1) == _CORA_EDGES
    ), "Cora edge count mismatch – corrupted download?"
    return data


def load_pubmed():
    ds = Planetoid(root=str(DATA_DIR / "Pubmed"), name="Pubmed")
    return _prepare_pyg(ds)


def load_wisconsin():
    ds = Planetoid(root=str(DATA_DIR / "Wisconsin"), name="Wisconsin")
    return _prepare_pyg(ds)
