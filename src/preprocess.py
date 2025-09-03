"""Data loading and synthetic-dataset construction utilities."""

import pathlib
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import add_self_loops, erdos_renyi_graph, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
#  Synthetic benchmark – core & chains
# --------------------------------------------------------------------------------------

def _bucket(values: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Quantile bucketisation so that classes are (roughly) balanced."""
    quantiles = np.quantile(values, np.linspace(0, 1, num_classes + 1)[1:-1])
    return np.digitize(values, quantiles)


def build_synthetic_cc(
    *,
    num_chains: int,
    chain_len: int,
    core_nodes: int,
    core_p: float,
    noise_nodes: int,
    seed: int = 0,
) -> Data:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -----------------------------------------------------------------------
    # 1) Core – Erdős–Rényi graph
    # -----------------------------------------------------------------------
    core_edge_index = erdos_renyi_graph(core_nodes, core_p)

    # -----------------------------------------------------------------------
    # 2) Chains attached to random core nodes
    # -----------------------------------------------------------------------
    chains = []
    chain_heads = []
    offset = core_nodes
    for c in range(num_chains):
        start = offset + c * chain_len
        # linear chain
        for i in range(chain_len - 1):
            chains.append([start + i, start + i + 1])
        # connect chain head to a random core node
        head_target = torch.randint(0, core_nodes, (1,)).item()
        chains.append([head_target, start])
        chain_heads.append(start)

    edge_index = torch.tensor(core_edge_index + chains, dtype=torch.long).t()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)

    # -----------------------------------------------------------------------
    # 3) Features – 2-d Gaussian
    # -----------------------------------------------------------------------
    total_nodes = core_nodes + num_chains * chain_len + noise_nodes
    x = torch.randn(total_nodes, 2)

    # -----------------------------------------------------------------------
    # 4) Labels following the *depth rule*
    # -----------------------------------------------------------------------
    z0 = x[:, 0].numpy()
    labels = np.zeros(total_nodes, dtype=int)

    # (a) core – function of 1-hop neighbourhood mean
    for v in range(core_nodes):
        neigh = edge_index[1][edge_index[0] == v]
        labels[v] = _bucket(z0[neigh].mean(), 10)

    # (b) chains – node t uses feature of node t-10 in the same chain
    for head in chain_heads:
        for t in range(10, chain_len):
            v = head + t
            labels[v] = _bucket(z0[v - 10], 10)
        # first 10 nodes get random labels (excluded from training)
        for t in range(10):
            labels[head + t] = np.random.randint(0, 10)

    # (c) noise nodes – random labels
    labels[core_nodes + num_chains * chain_len :] = np.random.randint(0, 10, noise_nodes)

    y = torch.from_numpy(labels).long()

    # -----------------------------------------------------------------------
    # 5) Train/val/test masks (60/20/20) – per component
    # -----------------------------------------------------------------------
    mask_tr = torch.zeros(total_nodes, dtype=torch.bool)
    mask_va = torch.zeros_like(mask_tr)
    mask_te = torch.zeros_like(mask_tr)

    rng = np.random.default_rng(seed)

    def _split(indices: np.ndarray) -> None:
        rng.shuffle(indices)
        n = len(indices)
        mask_tr[indices[: int(0.6 * n)]] = True
        mask_va[indices[int(0.6 * n) : int(0.8 * n)]] = True
        mask_te[indices[int(0.8 * n) :]] = True

    # core indices
    _split(np.arange(core_nodes))

    # chain indices – skip the first 9 nodes of each chain
    for head in chain_heads:
        _split(np.arange(head + 10, head + chain_len))

    # noise indices
    _split(np.arange(core_nodes + num_chains * chain_len, total_nodes))

    return Data(x=x, edge_index=edge_index, y=y, train_mask=mask_tr, val_mask=mask_va, test_mask=mask_te)


# --------------------------------------------------------------------------------------
#  Real-world datasets (Planetoid, WebKB, WikipediaNetwork, OGB)
# --------------------------------------------------------------------------------------

def load_real(name: str):
    name_lc = name.lower()

    if name_lc in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root=DATA_DIR / name, name=name)[0]

    if name_lc in {"cornell", "texas", "wisconsin"}:
        return WebKB(root=DATA_DIR / name, name=name.capitalize())[0]

    if name_lc in {"chameleon", "squirrel"}:
        return WikipediaNetwork(
            root=DATA_DIR / name, name=name.capitalize(), geom_gcn_preprocess=True
        )[0]

    if name_lc in {"ogbn-arxiv", "ogbn-products"}:
        return PygNodePropPredDataset(root=DATA_DIR / name, name=name)[0]

    raise ValueError(f"Unknown real dataset: {name}")