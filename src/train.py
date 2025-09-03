"""src/train.py
Model definitions, baseline builders and training utilities.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv, GCN2Conv, ChebConv, PairNorm
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torch_geometric or one of its CUDA extensions failed to import – "
        "make sure torch-scatter/sparse wheels match *exactly* torch==2.1.* "
        "and CUDA 11.8.") from e

from .preprocess import DEVICE

# ----------------------------------------------------------------------------
#  MODEL BUILDING BLOCKS  –  DDAF and baselines
# ----------------------------------------------------------------------------

class SpatialBranch(nn.Module):
    """K-hop GCN with node-wise Gumbel-sigmoid gates."""
    def __init__(self, in_dim: int, out_dim: int, K: int = 3, tau: float = 0.6):
        super().__init__()
        self.K = K
        self.tau = tau
        self.convs = nn.ModuleList([
            GCNConv(in_dim if k == 0 else out_dim, out_dim)
            for k in range(K)
        ])
        self.gate = nn.Linear(in_dim, K)

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        g = torch.sigmoid(self.gate(x) / self.tau)  # (N,K)
        outs = []
        h = x
        for k, conv in enumerate(self.convs):
            h = F.relu(conv(h, edge_index))
            outs.append(h * g[:, k : k + 1])
        return torch.stack(outs, 0).sum(0)


class SpectralBranch(nn.Module):
    """Chebyshev polynomial with coefficients produced by an MLP."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 5):
        super().__init__()
        self.K = K
        self.cheb = ChebConv(in_dim, out_dim, K)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 128), nn.ReLU(), nn.Linear(128, K)
        )

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        stats = torch.cat([x.mean(0), x.var(0)], 0)
        theta = self.mlp(stats).tanh()  # (K,)
        return F.relu(self.cheb(x, edge_index, theta))


class DDAFLayer(nn.Module):
    def __init__(self, in_dim: int, h: int, K_sp: int = 3, K_ch: int = 5):
        super().__init__()
        self.spatial = SpatialBranch(in_dim, h, K_sp)
        self.spectral = SpectralBranch(in_dim, h, K_ch)
        self.alpha = nn.Linear(in_dim, 2)  # fusion weights
        self.bn = nn.BatchNorm1d(h * 2)

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        xs = self.spatial(x, edge_index)
        xf = self.spectral(x, edge_index)
        a = torch.softmax(self.alpha(x), -1)  # (N,2)
        out = torch.cat([xs * a[:, 0:1], xf * a[:, 1:2]], 1)
        return self.bn(out)


class MIProj(nn.Module):
    def __init__(self, in_dim: int, proj: int = 128):
        super().__init__()
        self.l = nn.Linear(in_dim, proj)

    def forward(self, x):  # pylint: disable=arguments-differ
        return F.normalize(self.l(x), dim=-1)


def info_nce(z1: torch.Tensor, z2: torch.Tensor):
    N = z1.size(0)
    pos = (z1 * z2).sum(-1, keepdim=True)  # (N,1)
    sim = z1 @ z2.t()  # (N,N)
    logits = torch.cat([pos, sim], 1)
    labels = torch.zeros(N, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)


class DDAFGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        layers: int = 16,
        K_sp: int = 3,
        K_ch: int = 5,
        lambda_mi: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DDAFLayer(in_dim if i == 0 else hidden * 2, hidden, K_sp, K_ch)
            for i in range(layers)
        ])
        self.cls = nn.Linear(hidden * 2, out_dim)
        self.mi_in = MIProj(in_dim)
        self.mi_out = MIProj(hidden * 2)
        self.lmbd = lambda_mi

    def forward(self, x, edge_index):  # pylint: disable=arguments-differ
        for layer in self.layers:
            x = F.dropout(layer(x, edge_index), 0.5, self.training)
        h = x
        logits = self.cls(h)
        return logits, h

    # unified loss for convenience
    def loss(self, data, logits, h):  # pylint: disable=arguments-differ
        ce = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if self.lmbd > 0:
            mi = info_nce(self.mi_in(data.x), self.mi_out(h))
            return ce + self.lmbd * mi
        return ce


# --------------------------  Baseline builders  ---------------------------


def build_baseline(
    name: str,
    in_dim: int,
    hidden: int,
    out_dim: int,
    layers: int,
):
    name = name.lower()

    if name == "gcn":
        convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(layers - 1)]
        )
        lin = nn.Linear(hidden, out_dim)

        class Net(nn.Module):
            def forward(self, x, eidx):  # pylint: disable=arguments-differ
                for c in convs:
                    x = F.relu(c(x, eidx))
                    x = F.dropout(x, 0.5, self.training)
                return lin(x), x

        return Net()

    if name == "gcnii":
        proj = nn.Linear(in_dim, hidden)
        convs = nn.ModuleList(
            [GCN2Conv(hidden, alpha=0.3, theta=1.0, layer=i + 1) for i in range(layers)]
        )
        lin = nn.Linear(hidden, out_dim)

        class Net(nn.Module):
            def forward(self, x, eidx):  # pylint: disable=arguments-differ
                h0 = F.relu(proj(x))
                h = h0
                for c in convs:
                    h = F.relu(c(h, h0, eidx))
                    h = F.dropout(h, 0.5, self.training)
                return lin(h), h

        return Net()

    if name == "pairnorm-gcn":
        convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(layers - 1)]
        )
        pn = PairNorm("PN")
        lin = nn.Linear(hidden, out_dim)

        class Net(nn.Module):
            def forward(self, x, eidx):  # pylint: disable=arguments-differ
                for c in convs:
                    x = F.relu(c(x, eidx))
                    x = pn(x)
                    x = F.dropout(x, 0.5, self.training)
                return lin(x), x

        return Net()

    raise ValueError(f"Unknown baseline {name}")


# ----------------------------------------------------------------------------
#  TRAIN / EVAL helpers shared between experiments
# ----------------------------------------------------------------------------

class EarlyStop:
    """Simple early-stopping utility."""

    def __init__(self, patience: int = 100):
        self.pat = patience
        self.best = -1
        self.best_state = None
        self.cnt = 0

    def step(self, val_acc: float, model: nn.Module):
        if val_acc > self.best:
            self.best = val_acc
            # store only on CPU to avoid unnecessary GPU memory pressure
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.cnt = 0
        else:
            self.cnt += 1
        return self.cnt > self.pat
