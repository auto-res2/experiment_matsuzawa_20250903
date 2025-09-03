from __future__ import annotations
"""
preprocess.py – data loading & synthetic dataset generation
"""
import random
import pathlib
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import (
    add_self_loops,
    erdos_renyi_graph,
    to_undirected,
    degree,
)
from ogb.nodeproppred import PygNodePropPredDataset

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Internal helpers                                                           #
# ---------------------------------------------------------------------------

def _bucket(values: np.ndarray, k: int = 10) -> np.ndarray:
    """Map a continuous vector into k balanced buckets."""
    q = np.quantile(values, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(values, q)


# ---------------------------------------------------------------------------
#  Synthetic core-vs-chain dataset                                            #
# ---------------------------------------------------------------------------

def build_synthetic_cc(
    *,
    core_nodes: int,
    core_p: float,
    num_chains: int,
    chain_len: int,
    noise_nodes: int,
    seed: int,
    **_: Any,  # ignore any extra keys (e.g. "name") coming from config
) -> Data:
    """Generate the synthetic benchmark used in Experiment-1."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # A) Erdős-Rényi core -------------------------------------------------
    # erdos_renyi_graph already returns an edge_index with shape (2, E)
    edge_index = erdos_renyi_graph(core_nodes, core_p).long().contiguous()

    # B) Append chains ----------------------------------------------------
    chain_offsets = []
    for c in range(num_chains):
        head = core_nodes + c * chain_len
        chain_offsets.append(head)

        # linear chain edges (u -> v)
        chain_edges = [[head + i, head + i + 1] for i in range(chain_len - 1)]
        # attach head to a random core node
        attach = torch.randint(0, core_nodes, (1,)).item()
        chain_edges.append([attach, head])

        # Convert to shape (2, N_edges) to match PyG format
        chain_ei = torch.tensor(chain_edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, chain_ei], dim=1)

    # C) make undirected & add self-loops --------------------------------
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    # D) Node features ----------------------------------------------------
    total_nodes = core_nodes + num_chains * chain_len + noise_nodes
    x = torch.randn(total_nodes, 2)
    z0 = x[:, 0].numpy()

    # E) Labels with **depth** rule --------------------------------------
    labels = np.zeros(total_nodes, dtype=int)

    # core: mean of 1-hop neighbours' z0 bucketed into 10 classes
    for v in range(core_nodes):
        neigh = edge_index[1][edge_index[0] == v].cpu().numpy()
        labels[v] = _bucket(z0[neigh].mean())

    # chains: node t >= 10 uses value of node t-10
    for head in chain_offsets:
        for t in range(chain_len):
            node = head + t
            if t < 10:
                labels[node] = np.random.randint(0, 10)  # will be discarded
            else:
                labels[node] = _bucket(z0[node - 10])

    # noise nodes: random labels
    labels[core_nodes + num_chains * chain_len :] = np.random.randint(
        0, 10, noise_nodes
    )

    y = torch.from_numpy(labels).long()

    # F) Train / val / test masks with no leakage ------------------------
    mask_tr = torch.zeros(total_nodes, dtype=torch.bool)
    mask_va = torch.zeros_like(mask_tr)
    mask_te = torch.zeros_like(mask_tr)

    rng = np.random.default_rng(seed)

    def _split(idx):
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        mask_tr[idx[: int(0.6 * n)]] = True
        mask_va[idx[int(0.6 * n) : int(0.8 * n)]] = True
        mask_te[idx[int(0.8 * n) :]] = True

    # core
    _split(np.arange(core_nodes))
    # chains (skip first 10 nodes)
    for head in chain_offsets:
        _split(np.arange(head + 10, head + chain_len))
    # noise
    _split(np.arange(core_nodes + num_chains * chain_len, total_nodes))

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=mask_tr,
        val_mask=mask_va,
        test_mask=mask_te,
    )


# ---------------------------------------------------------------------------
#  Real datasets loader                                                       #
# ---------------------------------------------------------------------------

def load_real(name: str):
    name = name.lower()
    root = _DATA_DIR / name

    if name in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root=root, name=name.capitalize())[0]
    if name in {"cornell", "texas", "wisconsin"}:
        return WebKB(root=root, name=name.capitalize())[0]
    if name in {"chameleon", "squirrel"}:
        return WikipediaNetwork(root=root, name=name.capitalize(), geom_gcn_preprocess=True)[
            0
        ]
    if name in {"ogbn-arxiv", "ogbn-products"}:
        return PygNodePropPredDataset(root=root, name=name)[0]
    raise ValueError(f"Unknown dataset {name}")
