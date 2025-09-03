"""src/preprocess.py – dataset loading & light pre-processing."""
from __future__ import annotations

from pathlib import Path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

__all__ = ["load_cora"]


_DATA_ROOT = Path("data")  # one central directory keeps things tidy


def load_cora() -> "torch_geometric.data.Data":  # pragma: no cover – small helper
    """Load/we download the *Cora* citation network via PyG's `Planetoid` class.

    The function returns the *single* `Data` object contained in the dataset so
    that callers can directly move it to the desired device (CPU/GPU) via
    `.to(device)`.
    """
    dataset = Planetoid(root=_DATA_ROOT.as_posix(), name="Cora", transform=NormalizeFeatures())
    return dataset[0]
