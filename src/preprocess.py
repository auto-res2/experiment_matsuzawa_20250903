[UPDATED CONTENT WITH inline set_seed]
```
"""src/preprocess.py
Synthetic core-vs-chain graph generation and deterministic utilities.
"""
from __future__ import annotations
import random
import pathlib
import typing as T

import numpy as np
import torch
from torch_geometric.utils import erdos_renyi_graph, add_self_loops, degree
from torch_geometric.data import Data

################################################################################
#  Local deterministic helper (avoids dependency on missing utils module)     #
################################################################################

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

__all__ = ["build_synthetic_cc"]

################################################################################
#  Helper                                                                      #
################################################################################

def _bucket(z: np.ndarray, k: int = 10) -> np.ndarray:
    """Discretise *z* into *k* quantile buckets (0-based labels)."""
    qs = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(z, qs)

################################################################################
#  Synthetic dataset                                                           #
################################################################################

def build_synthetic_cc(
    *,
    core_nodes: int,
    core_p: float,
    num_chains: int,
    chain_len: int,
    noise_nodes: int,
    seed: int = 0,
) -> Data:
    """Construct the *core-vs-chain* synthetic benchmark used in Exp-1."""
    set_seed(seed)
    # core graph -------------------------------------------------------
    core_edges = erdos_renyi_graph(core_nodes, core_p)
    ei = torch.tensor(core_edges, dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)  # undirected

    # chains -----------------------------------------------------------
    head_offsets = []
    for c in range(num_chains):
        head = core_nodes + c * chain_len
        head_offsets.append(head)
        chain = torch.arange(head, head + chain_len)
        edges = torch.stack([chain[:-1], chain[1:]], dim=0)
        ei = torch.cat([ei, edges, edges.flip(0)], dim=1)
        # attach chain head to a random core node
        attach = random.randrange(core_nodes)
        ei = torch.cat([
            ei,
            torch.tensor([[attach], [head]]),
            torch.tensor([[head], [attach]]),
        ], dim=1)

    ei, _ = add_self_loops(ei)

    # features & labels ----------------------------------------------
    N = core_nodes + num_chains * chain_len + noise_nodes
    x = torch.randn(N, 2)
    z0 = x[:, 0].numpy()
    y = np.zeros(N, dtype=int)

    row, col = ei
    for v in range(core_nodes):
        neigh = col[row == v].cpu().numpy()
        y[v] = _bucket(z0[neigh].mean())

    # chain labels – need 10 hops                                    
    for head in head_offsets:
        for t in range(chain_len):
            idx = head + t
            y[idx] = _bucket(z0[idx - 10]) if t >= 10 else random.randint(0, 9)

    # noise labels – random
    y[core_nodes + num_chains * chain_len :] = np.random.randint(0, 10, noise_nodes)
    y = torch.from_numpy(y).long()

    # -----------------------------------------------------------------
    def _split(indices):
        idxs = list(indices)
        random.shuffle(idxs)
        a, b = int(0.6 * len(idxs)), int(0.8 * len(idxs))
        return set(idxs[:a]), set(idxs[a:b]), set(idxs[b:])

    train, val, test = set(), set(), set()
    # core
    a, b, c = _split(range(core_nodes))
    train |= a; val |= b; test |= c
    # chains (skip first 10)
    for head in head_offsets:
        a, b, c = _split(range(head + 10, head + chain_len))
        train |= a; val |= b; test |= c
    # noise
    a, b, c = _split(range(core_nodes + num_chains * chain_len, N))
    train |= a; val |= b; test |= c

    def _mask(s):
        return torch.tensor([i in s for i in range(N)])

    return Data(
        x=x,
        edge_index=ei,
        y=y,
        train_mask=_mask(train),
        val_mask=_mask(val),
        test_mask=_mask(test),
    )
```