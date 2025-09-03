"""src/preprocess.py
Small, self-contained utilities to fabricate a toy graph so that the
training / evaluation pipeline can be exercised during CI without the
need to download any real benchmark dataset from the internet.

The helper returns a *torch_geometric.data.Data* object with
• 100 nodes, undirected Erdős–Rényi style connectivity (|E|≈300)
• 16-dimensional random features (standard Normal)
• 3 random classes with a 60 / 20 / 20 train–val–test split

The toy graph is deliberately tiny so that the whole unit-test suite
finishes within a few seconds on a CPU-only runner.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data

__all__ = ["generate_toy_graph"]


def _erdos_renyi_edges(num_nodes: int, num_edges: int, seed: int | None = None) -> torch.Tensor:
    """Return a bidirectional *edge_index* tensor with shape [2, 2·num_edges]."""
    rng = np.random.default_rng(seed)
    # sample unordered pairs (i<j) without replacement
    pairs = set()
    while len(pairs) < num_edges:
        i, j = int(rng.integers(0, num_nodes)), int(rng.integers(0, num_nodes))
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    src, dst = zip(*pairs)
    # make edges undirected  (i,j) and (j,i)
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return edge_index


def generate_toy_graph(
    num_nodes: int = 100,
    num_edges: int = 300,
    num_feats: int = 16,
    num_classes: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Data:
    """Fabricate a small random graph for quick smoke-tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x = torch.randn(num_nodes, num_feats, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    edge_index = _erdos_renyi_edges(num_nodes, num_edges, seed)

    # boolean split masks ----------------------------------------------------
    indices = np.random.permutation(num_nodes)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data
