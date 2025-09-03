import pathlib
from typing import Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import erdos_renyi_graph, add_self_loops, to_undirected, degree
from ogb.nodeproppred import PygNodePropPredDataset

__all__ = ["load_dataset"]

_DATA_ROOT = pathlib.Path("data")
_DATA_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Real-world datasets
# -----------------------------------------------------------------------------

def _get_planetoid(name: str):
    return Planetoid(root=_DATA_ROOT / name, name=name)

def _get_webkb(name: str):
    return WebKB(root=_DATA_ROOT / name, name=name)

def _get_wikipedia(name: str):
    return WikipediaNetwork(root=_DATA_ROOT / name, name=name, geom_gcn_preprocess=True)

def _get_ogb(name: str):
    return PygNodePropPredDataset(name=name, root=_DATA_ROOT / name)

# -----------------------------------------------------------------------------
# Synthetic benchmark graph used in Experiment-1
# -----------------------------------------------------------------------------

def _synthetic_chain_core(num_chains: int = 50,
                          chain_len: int = 800,
                          core_nodes: int = 5000,
                          p_core: float = 0.024,
                          noise_nodes: int = 25000) -> Data:
    """Generate the synthetic graph composed of a dense core and sparse chains.

    The function returns a ``torch_geometric.data.Data`` object whose structure
    mirrors the benchmark described in the paper.  Implementation is kept
    simple – it does *not* attempt to be GPU-efficient since the graph is small
    enough for CPU construction.
    """

    # 1) Core – Erdős-Rényi graph
    core_edge_index = erdos_renyi_graph(core_nodes, p_core)

    # Convert tensor edge list → Python list so it can be concatenated
    core_edges: list[list[int]] = core_edge_index.t().cpu().tolist()

    # 2) Chains attached to the core
    chain_edges: list[list[int]] = []
    for c in range(num_chains):
        start = core_nodes + c * chain_len
        # linear chain edges
        for i in range(chain_len - 1):
            chain_edges.append([start + i, start + i + 1])
        # attach head to random core node
        head_target = torch.randint(0, core_nodes, (1,))
        chain_edges.append([head_target.item(), start])

    # 3) Noise nodes – remain isolated (no edges)
    total_nodes = core_nodes + num_chains * chain_len + noise_nodes

    # Combine all edges and convert back to tensor
    all_edges = core_edges + chain_edges
    edge_index = torch.tensor(np.array(all_edges).T, dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    # Dummy features & labels (real experiment computes labels on-the-fly)
    x = torch.randn(total_nodes, 2)
    y = torch.randint(0, 10, (total_nodes,))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = total_nodes
    return data

# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------

def load_dataset(name: str, **kwargs) -> Any:
    """Factory that returns a *torch_geometric* dataset or ``Data`` instance."""
    name_low = name.lower()

    if name_low in {"cora", "citeseer", "pubmed"}:
        return _get_planetoid(name.capitalize())
    if name_low in {"cornell", "texas", "wisconsin"}:
        return _get_webkb(name.capitalize())
    if name_low in {"chameleon", "squirrel"}:
        return _get_wikipedia(name.capitalize())
    if name_low in {"ogbn-arxiv", "ogbn-products"}:
        return _get_ogb(name)
    if name_low == "synthetic_chain_core":
        return _synthetic_chain_core(**kwargs)

    raise ValueError(f"Unknown dataset identifier: {name}")
