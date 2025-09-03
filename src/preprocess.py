"""src/preprocess.py
Dataset loading / random seed utilities.  The implementation is deliberately
minimal: if `torch_geometric` is available we load the canonical Cora dataset;
otherwise we raise an exception so that the caller can fall back to the tiny
synthetic graph defined in ``src/main.py``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["set_seed", "load_cora"]

################################################################################
#  REPRODUCIBILITY UTILITIES
################################################################################

def set_seed(seed: int = 42) -> None:  # noqa: D401
    """Seed *all* RNGs (Python, NumPy, PyTorch, and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

################################################################################
#  DATA LOADING
################################################################################

def load_cora():  # noqa: D401
    """Load the Cora citation network via *torch-geometric*.

    We keep the import within the function to avoid adding a hard dependency:
    if the package is missing the caller can still recover gracefully.
    """
    try:
        from torch_geometric.datasets import Planetoid  # type: ignore
        from torch_geometric.transforms import NormalizeFeatures  # type: ignore
    except Exception as e:  # pragma: no cover – optional dependency
        raise RuntimeError("`torch_geometric` not available – cannot load Cora") from e

    root = Path(os.getenv("TORCH_GEO_DATA", "/tmp")) / "cora"
    dataset = Planetoid(root=str(root), name="Cora", transform=NormalizeFeatures())
    return dataset[0]
