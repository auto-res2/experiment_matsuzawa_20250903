"""src/train.py
Logic related to model definition and the end-to-end training loop.
This variant removes the hard dependency on *torch_sparse* so that the
project installs cleanly on vanilla PyPI without having to compile CUDA
extensions.  If the real package is present we use it, otherwise a very
light-weight fallback (based on torch.sparse_coo_tensor) is registered
under the same import path so that third-party libraries – in particular
PyG – continue to import successfully.
"""
from __future__ import annotations

import os
import time
import copy
import math
import random
import types
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Create a minimal stub of `torch_sparse.SparseTensor` if the real package
#  is absent.  This avoids the build-time failure of torch_sparse (which needs
#  a full PyTorch/CUDA tool-chain) while still providing the small subset of
#  functionality required by the SCNR layer and by PyG's type annotations.
# -----------------------------------------------------------------------------
try:
    from torch_sparse import SparseTensor  # pylint: disable=import-error
except ModuleNotFoundError:  # pragma: no cover – only executed in CPU-only CI

    class _SparseTensor:  # noqa: D401, pylint: disable=too-few-public-methods
        """Very small wrapper around *torch.sparse* tensors.

        Only implements `matmul`, subtraction and `.device` property which are
        the operations required by *SCNR* as well as PyG's internal checks.
        It is **not** a full replacement for `torch_sparse.SparseTensor` – it
        merely keeps the demo code functional on small/medium graphs.
        """

        def __init__(self, row: torch.Tensor, col: torch.Tensor, value: torch.Tensor,
                     sparse_sizes: Tuple[int, int]):
            indices = torch.stack([row, col], dim=0)
            self._tensor = torch.sparse_coo_tensor(indices, value, sparse_sizes,
                                                   dtype=value.dtype,
                                                   device=value.device).coalesce()

        # ----------------------------- basic ops -----------------------------
        def matmul(self, other: torch.Tensor) -> torch.Tensor:  # noqa: D401
            """Sparse-dense matrix multiplication."""
            return torch.sparse.mm(self._tensor, other)

        def __sub__(self, other):  # noqa: D401
            if isinstance(other, _SparseTensor):
                result = self._tensor - other._tensor
            elif torch.is_tensor(other):  # dense → convert to sparse
                result = self._tensor - other.to_sparse()
            else:
                raise TypeError("Unsupported operand type for subtraction")
            out = self.__class__.__new__(self.__class__)
            out._tensor = result.coalesce()
            return out

        # ----------------------------- helpers ------------------------------
        def to(self, device):  # noqa: D401
            out = self.__class__.__new__(self.__class__)
            out._tensor = self._tensor.to(device)
            return out

        @property
        def device(self):  # noqa: D401
            return self._tensor.device

    # Expose the stub as a bona-fide module so that `import torch_sparse` works
    _mod = types.ModuleType("torch_sparse")
    _mod.SparseTensor = _SparseTensor
    sys.modules["torch_sparse"] = _mod
    SparseTensor = _SparseTensor  # type: ignore  # noqa: N816

# Now that *torch_sparse* is guaranteed to import, we can safely pull in PyG
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils import add_self_loops

# ---------------------------------------------------------------------------
#   1)  SCNR – Spectral-Contrastive Node Rebalancing
# ---------------------------------------------------------------------------
class SCNR(nn.Module):
    """Spectral-Contrastive Node Rebalancing normalisation layer."""

    def __init__(
        self,
        in_dim: int,
        K: int = 16,
        lambda1: float = 0.05,
        lambda2: float = 0.05,
        p_high: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p_high = p_high

        self._eta = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32))
        self.ln = nn.LayerNorm(in_dim, eps=eps)
        self.extra_loss: torch.Tensor | float = 0.0  # populated during fwd

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalised_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> SparseTensor:
        """Return Â = D⁻¹ᐟ² (A+I) D⁻¹ᐟ² as *SparseTensor*."""
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float().to(device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))

    @torch.no_grad()
    def _spectral_sketch(self, adj: SparseTensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Power-iteration-with-deflation to approximate K leading eigen-vectors."""
        vecs: List[torch.Tensor] = []
        residual = adj
        for _ in range(self.K):
            v = torch.randn(num_nodes, 1, device=device)
            for _ in range(3):
                v = residual.matmul(v)
                v = F.normalize(v, dim=0)
            vecs.append(v)
            # rank-1 deflation – *dense* outer-product is small for the graphs
            residual = residual - residual.matmul(v).matmul(v.t())  # type: ignore[arg-type]
        return torch.cat(vecs, dim=1)  # [N, K]

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if not self.training:
            return self.ln(x)  # inference-time fast path

        N = x.size(0)
        device = x.device
        adj = self._normalised_adj(edge_index, N, device)
        V = self._spectral_sketch(adj, N, device)  # [N, K]

        # projections
        H_low = V @ (V.T @ x)
        H_high = x - H_low

        # optional high-frequency dropout
        if self.p_high > 0.0:
            mask = torch.rand_like(H_high).lt(self.p_high)
            H_high = H_high.masked_fill(mask, 0.0)

        # ---------------- Contrastive re-balancing loss ----------------
        def _bt_loss(h: torch.Tensor) -> torch.Tensor:  # Barlow-Twins helper
            h = h - h.mean(0)
            c = h.T @ h / max(N - 1, 1)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
            return on_diag + 0.005 * off_diag

        L_uni = _bt_loss(H_low) + _bt_loss(H_high)
        H_low_c = H_low - H_low.mean(0)
        H_high_c = H_high - H_high.mean(0)
        L_dec = (H_low_c.T @ H_high_c / max(N - 1, 1)).pow(2).sum()

        self.extra_loss = self.lambda1 * L_uni + self.lambda2 * L_dec

        # one folded gradient step (see §4 in paper)
        x_tilde = x - self._eta * torch.autograd.grad(
            outputs=self.extra_loss,
            inputs=x,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0].detach()
        return self.ln(x_tilde)


# ---------------------------------------------------------------------------
#   2)  Backbone – deep, plain GCN
# ---------------------------------------------------------------------------
class DeepGCN(nn.Module):
    """Plain Graph Convolutional Network of arbitrary depth."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        layers: int,
        normaliser_cfg: Dict[str, Any],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert layers >= 2, "Need at least 2 layers (1 hidden + 1 output)"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(in_dim, hidden, cached=False, add_self_loops=False))
        self.norms.append(self._make_norm(hidden, normaliser_cfg))

        # hidden layers
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden, cached=False, add_self_loops=False))
            self.norms.append(self._make_norm(hidden, normaliser_cfg))

        # output layer – no norm afterwards
        self.convs.append(GCNConv(hidden, out_dim, cached=False, add_self_loops=False))
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_norm(hidden_dim: int, cfg: Dict[str, Any]) -> nn.Module:
        name = cfg["name"].lower()
        if name == "none":
            return nn.Identity()
        if name == "pairnorm":
            return PairNorm(scale=1.0, mode="PN-S")
        if name == "contranorm":
            lam = cfg.get("lambda", 0.1)
            return PairNorm(scale=1.0 + lam, mode="PN-S")
        if name == "scnr":
            return SCNR(
                hidden_dim,
                K=cfg.get("K", 16),
                lambda1=cfg.get("lambda1", 0.05),
                lambda2=cfg.get("lambda2", 0.05),
                p_high=cfg.get("p_high", 0.0),
            )
        raise ValueError(f"Unknown normaliser {name}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:  # noqa: D401
        extra_losses: List[torch.Tensor] = []
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x, edge_index) if isinstance(norm, SCNR) else norm(x)
            if isinstance(norm, SCNR):
                extra_losses.append(norm.extra_loss)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x, extra_losses


# ---------------------------------------------------------------------------
#   3)  Single-seed trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Self-contained helper that trains *one* model on *one* dataset for *one* seed."""

    def __init__(
        self,
        device: torch.device,
        common_cfg: Dict[str, Any],
    ) -> None:
        self.device = device
        self.common_cfg = common_cfg

    # ------------------------------------------------------------------
    def train(
        self,
        model: nn.Module,
        data,
        epochs: int,
        lr: float,
        weight_decay: float,
        eval_fn,  # callable from evaluate.py
        early_stop_patience: int = 100,
    ) -> Dict[str, Any]:
        """Core optimisation loop with simple early-stopping on val-accuracy."""

        data = data.to(self.device)
        model = model.to(self.device)

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(self.common_cfg["optimizer"].get("betas", (0.9, 0.999))),
            eps=float(self.common_cfg["optimizer"].get("eps", 1e-8)),
        )

        best_val, best_state, patience = 0.0, None, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        tic = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            optimiser.zero_grad(set_to_none=True)
            out, extra_losses = model(data.x, data.edge_index)

            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            if extra_losses:
                loss = loss + sum(extra_losses)
            loss.backward()
            optimiser.step()

            # ---------------- validation -------------------
            if epoch % 10 == 0 or epoch == epochs:
                val_acc = eval_fn(model, data, split="val")
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 10
                if patience >= early_stop_patience:
                    break

        # ---------------- test + bookkeeping -------------
        model.load_state_dict(best_state)  # type: ignore[arg-type]
        test_acc = eval_fn(model, data, split="test")
        duration = time.time() - tic
        mem_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1e6
            if self.device.type == "cuda" else 0.0
        )
        return {"test_acc": test_acc, "best_val": best_val, "seconds": duration, "memMB": mem_mb}
