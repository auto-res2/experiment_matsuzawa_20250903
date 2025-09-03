"""src/train.py – model architectures, training utilities"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GCN2Conv,
    ChebConv,
)

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
SEEDS: List[int] = [11, 13, 17, 19, 23]


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Model definitions (DDAF-GNN and baselines)
# ---------------------------------------------------------------------------
class SpatialBranch(nn.Module):
    """K_max-hop message-passing with differentiable per-node gates."""

    def __init__(self, in_dim: int, out_dim: int, k_max: int = 3):
        super().__init__()
        self.k_max = k_max
        self.convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else out_dim, out_dim) for i in range(k_max)]
        )
        self.gate_fc = nn.Linear(in_dim, k_max)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: N802
        gates = torch.sigmoid(self.gate_fc(x))  # (N,k_max) in [0,1]
        outs = []
        h = x
        for hop, conv in enumerate(self.convs):
            h = F.relu(conv(h, edge_index))
            outs.append(h * gates[:, hop : hop + 1])
        return torch.stack(outs, dim=0).sum(dim=0)


class SpectralBranch(nn.Module):
    """Chebyshev polynomial filter with on-the-fly coefficients."""

    def __init__(self, in_dim: int, out_dim: int, k_order: int = 3):
        super().__init__()
        self.k_order = k_order
        self.cheb = ChebConv(in_dim, out_dim, K=k_order)
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 128), nn.ReLU(), nn.Linear(128, k_order)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: N802
        stats = torch.cat([x.mean(dim=0), x.var(dim=0)], dim=0)
        theta = self.coeff_mlp(stats).tanh()  # (K,) in (−1,1)
        return F.relu(self.cheb(x, edge_index, theta))


class DDAFGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k_max: int = 3, k_order: int = 3):
        super().__init__()
        self.s_branch = SpatialBranch(in_dim, out_dim, k_max)
        self.f_branch = SpectralBranch(in_dim, out_dim, k_order)
        self.alpha_fc = nn.Linear(in_dim, 2)
        self.norm = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: N802
        xs = self.s_branch(x, edge_index)
        xf = self.f_branch(x, edge_index)
        alphas = torch.softmax(self.alpha_fc(x), dim=-1)  # (N,2)
        out = torch.cat([xs * alphas[:, 0:1], xf * alphas[:, 1:2]], dim=-1)
        return self.norm(out)


class MIHead(nn.Module):
    """Lightweight projector for InfoNCE mutual-information regulariser."""

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)

    def forward(self, x):  # noqa: D401,N802
        return F.normalize(self.proj(x), dim=-1)


class DDAFGNN(nn.Module):
    """Dual-Domain Adaptive Filtering GNN (proposed model)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 16,
        k_max: int = 3,
        k_order: int = 3,
        lambda_mi: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DDAFGNNLayer(
                    in_dim if i == 0 else hidden_dim * 2, hidden_dim, k_max, k_order
                )
                for i in range(num_layers)
            ]
        )
        self.cls_head = nn.Linear(hidden_dim * 2, out_dim)
        self.mi_head_in = MIHead(in_dim)
        self.mi_head_out = MIHead(hidden_dim * 2)
        self.lambda_mi = lambda_mi

    # ---------------------------------------------------------------------
    def forward(self, x, edge_index):  # noqa: D401,N802
        for layer in self.layers:
            x = F.dropout(layer(x, edge_index), p=0.5, training=self.training)
        z = x
        logits = self.cls_head(z)
        return logits, z

    # ------------------------------------------------------------------
    def mutual_info_loss(self, x0, z):
        proj0 = self.mi_head_in(x0)
        projL = self.mi_head_out(z)
        pos = (proj0 * projL).sum(dim=-1)
        neg = torch.matmul(proj0, projL.t())  # (N×N)
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        labels = torch.zeros(x0.size(0), dtype=torch.long, device=x0.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
#  Baseline model factory (GCN, GAT, GCNII, fallback MLP)
# ---------------------------------------------------------------------------

def build_baseline(name: str, in_dim: int, hidden: int, out_dim: int, num_layers: int):
    name = name.lower()

    class MLPClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fcs = nn.ModuleList()
            self.fcs.append(nn.Linear(in_dim, hidden))
            for _ in range(num_layers - 2):
                self.fcs.append(nn.Linear(hidden, hidden))
            self.fcs.append(nn.Linear(hidden, out_dim))
            self.bn = nn.BatchNorm1d(hidden)

        def forward(self, x, *args):  # noqa: N802
            for fc in self.fcs[:-1]:
                x = F.relu(self.bn(fc(x)))
            return self.fcs[-1](x)

    # ---------------------------  GCN  ----------------------------------
    if name == "gcn":
        convs = nn.ModuleList([
            GCNConv(in_dim if i == 0 else hidden, hidden) for i in range(num_layers - 1)
        ])
        lin = nn.Linear(hidden, out_dim)

        class GCN(nn.Module):
            def forward(self, x, edge_index):  # noqa: N802
                for conv in convs:
                    x = F.relu(conv(x, edge_index))
                    x = F.dropout(x, p=0.5, training=self.training)
                return lin(x)

        return GCN()

    # ---------------------------  GAT  ----------------------------------
    if name == "gat":
        heads = 8
        convs = nn.ModuleList([
            GATConv(in_dim if i == 0 else hidden * heads, hidden, heads=heads)
            for i in range(num_layers - 1)
        ])
        lin = nn.Linear(hidden * heads, out_dim)

        class GAT(nn.Module):
            def forward(self, x, edge_index):  # noqa: N802
                for conv in convs:
                    x = F.relu(conv(x, edge_index))
                return lin(x)

        return GAT()

    # ---------------------------  GCNII  --------------------------------
    if name == "gcnii":
        layers = nn.ModuleList([
            GCN2Conv(hidden, alpha=0.3, theta=1.0, layer=i + 1) for i in range(num_layers)
        ])
        proj = nn.Linear(in_dim, hidden)
        lin = nn.Linear(hidden, out_dim)

        class GCNII(nn.Module):
            def forward(self, x, edge_index):  # noqa: N802
                h0 = F.dropout(proj(x), p=0.5, training=self.training)
                h = h0
                for layer in layers:
                    h = F.relu(layer(h, h0, edge_index))
                return lin(h)

        return GCNII()

    # -----------------------  Fallback MLP  -----------------------------
    print(f"[Warning] Baseline '{name}' not implemented – falling back to MLP.")
    return MLPClassifier()


# ---------------------------------------------------------------------------
#  Optimisation utilities
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, data, optimizer):  # noqa: ANN001
    """Single optimisation step. Handles MI loss when model is DDAFGNN."""
    model.train()
    optimizer.zero_grad()
    logits, z = (
        model(data.x, data.edge_index)
        if isinstance(model, DDAFGNN)
        else (model(data.x, data.edge_index), None)
    )
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    if isinstance(model, DDAFGNN) and model.lambda_mi > 0:
        mi_loss = model.mutual_info_loss(data.x, z)
        loss = loss + model.lambda_mi * mi_loss
    loss.backward()
    optimizer.step()
    return loss.item()
