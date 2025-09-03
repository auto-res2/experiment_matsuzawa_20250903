
"""
train.py
~~~~~~~~
Model definitions (GCN / GraphSAGE & AdaSmooth wrapper) **and** the generic
training loop that is re-used by all experiments live here.
"""
from __future__ import annotations
import time, math, pathlib
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree
import numpy as np

# -----------------------------------------------------------------------------
#                               MODEL BUILDING BLOCKS
# -----------------------------------------------------------------------------
class PhiMLP(nn.Module):
    """Small MLP that outputs one scalar per node (the Φ_ℓ in the paper)."""
    def __init__(self, in_feats: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:  # (N, in_feats)
        return self.net(stats).squeeze(-1)  # (N,)


class AdaSmoothLayer(nn.Module):
    """One message-passing layer equipped with the AdaSmooth gate."""

    def __init__(self, in_dim: int, out_dim: int, conv_kind: str, phi_hidden: int):
        super().__init__()
        if conv_kind == "gcn":
            self.conv = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        elif conv_kind == "sage":
            self.conv = SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv kind: {conv_kind}")
        self.phi = PhiMLP(in_feats=6, hidden=phi_hidden)
        self.cached_g: torch.Tensor | None = None  # filled during fwd

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        deg: torch.Tensor,
        feat_var: torch.Tensor,
        grad_norm: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msg = self.conv(x, edge_index)
        with torch.no_grad():
            # simple statistical vector composed of raw + log-scaled versions
            stats = torch.stack(
                [deg, feat_var, grad_norm, deg.log1p(), feat_var.log1p(), grad_norm.log1p()], dim=-1
            )
        logits = self.phi(stats)
        g = torch.sigmoid(logits / tau)  # (N,)
        self.cached_g = g.detach().cpu()
        out = g.unsqueeze(-1) * msg + (1 - g).unsqueeze(-1) * x
        return out, g


class GNNModel(nn.Module):
    """Backbone network.  If *ada_cfg* is *None* a vanilla GNN is built."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        backbone: str,
        ada_cfg: Dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.backbone = backbone
        self.use_ada = ada_cfg is not None
        self.dropout = 0.5
        if self.use_ada:
            self.layers = nn.ModuleList(
                [
                    AdaSmoothLayer(
                        in_dim if i == 0 else hidden,
                        hidden if i != depth - 1 else out_dim,
                        conv_kind=backbone,
                        phi_hidden=ada_cfg["phi_hidden"],
                    )
                    for i in range(depth)
                ]
            )
            self.ada_cfg = ada_cfg
        else:
            conv_cls = GCNConv if backbone == "gcn" else SAGEConv
            self.layers = nn.ModuleList(
                [
                    conv_cls(
                        in_dim if i == 0 else hidden,
                        hidden if i != depth - 1 else out_dim,
                        add_self_loops=(backbone == "gcn"),
                    )
                    for i in range(depth)
                ]
            )

    # .........................................................................
    #   forward helpers (split to keep the main *forward* tidy)
    # .........................................................................
    def _forward_vanilla(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != self.depth - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _forward_ada(
        self,
        data,
        deg: torch.Tensor,
        feat_var: torch.Tensor,
        grad_norm: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        g_all = []
        for i, layer in enumerate(self.layers):
            x, g = layer(x, edge_index, deg, feat_var, grad_norm, tau)
            g_all.append(g)
            if i != self.depth - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # store for logging/penalties
        self.cached_g_all = torch.stack(g_all)
        return x

    def forward(self, data, tau: float = 0.0):
        if not self.use_ada:
            return self._forward_vanilla(data.x, data.edge_index)
        # statistics for the AdaSmooth gate
        with torch.no_grad():
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(data.x.device)
            feat_var = data.x.var(dim=1)
            if not hasattr(data, "grad_norm"):
                data.grad_norm = torch.zeros_like(deg)
            grad_norm = data.grad_norm
        return self._forward_ada(data, deg, feat_var, grad_norm, tau)


# -----------------------------------------------------------------------------
#                               TRAINER CLASS
# -----------------------------------------------------------------------------
class Trainer:
    """Thin wrapper that trains one model (given seed / depth / backbone)."""

    def __init__(self, cfg_common: Dict[str, Any], dataset, run_name: str, device: str):
        self.cfg = cfg_common
        self.dataset = dataset
        self.data = dataset[0].to(device)
        self.device = device
        self.run_name = run_name

        # handle splits (works for both OGB & Planetoid / Wiki datasets)
        self.train_idx = self._split("train")
        self.val_idx = self._split("valid")
        self.test_idx = self._split("test")

    # .........................................................................
    def _split(self, part: str):
        """Return indices for the requested split.

        Supports both:
          • OGB loaders – `get_idx_split()` returns dict with keys 'train'/'valid'/'test'.
          • Planetoid / WikiNetwork loaders – boolean masks stored on the *Data* object.
        """
        # -------- OGB style --------------------------------------------------
        if hasattr(self.dataset, "get_idx_split"):
            split = self.dataset.get_idx_split()
            # direct hit
            if part in split:
                return split[part]
            # aliasing ('valid' ↔ 'val')
            if part == "valid" and "val" in split:
                return split["val"]

        # -------- Mask style --------------------------------------------------
        # Build list of candidate mask names (account for aliasing)
        mask_candidates = [f"{part}_mask"]
        if part == "valid":  # Planetoid uses 'val_mask'
            mask_candidates.append("val_mask")
        for mname in mask_candidates:
            if hasattr(self.data, mname):
                return getattr(self.data, mname).nonzero(as_tuple=False).view(-1)

        # If we reach here, we could not resolve the split
        raise RuntimeError("Dataset does not contain recognised split information.")

    # ---------------------------------------------------------------------
    #                    PUBLIC  –  train one model instance
    # ---------------------------------------------------------------------
    def run_single(
        self,
        seed: int,
        backbone: str,
        depth: int,
        use_ada: bool,
        optim_cfg: Dict[str, Any],
        ada_cfg: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GNNModel(
            in_dim=self.data.x.size(-1),
            hidden=128,
            out_dim=int(self.dataset.num_classes),
            depth=depth,
            backbone=backbone,
            ada_cfg=ada_cfg if use_ada else None,
        ).to(self.device)

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=optim_cfg["lr"],
            weight_decay=optim_cfg["wd"],
            betas=tuple(optim_cfg["betas"]),
            eps=optim_cfg["eps"],
        )

        best_val, best_test = -1.0, -1.0
        patience = 0
        max_patience = self.cfg["early_stop"]
        losses, val_scores = [], []
        max_epochs = self.cfg["max_epochs"]
        tic = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimiser.zero_grad()
            tau = self._tau_schedule(epoch, max_epochs, ada_cfg) if use_ada else 0.0
            logits = model(self.data, tau=tau)
            loss = F.cross_entropy(logits[self.train_idx], self.data.y[self.train_idx])

            # AdaSmooth – add spectral regulariser
            if use_ada:
                g_all = model.cached_g_all.to(logits.device).flatten()
                loss = loss + ada_cfg["lambda_reg"] * g_all.var()

            loss.backward()
            # store grad-norm for next forward pass (one representative param)
            if use_ada:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None and p.grad.ndim == 2 and p.size(0) == self.data.num_nodes:
                            self.data.grad_norm.copy_(p.grad.norm(p=2, dim=1))
                            break
            optimiser.step()

            # evaluation ----------------------------------------------------
            train_f1 = self._f1(logits[self.train_idx], self.data.y[self.train_idx])
            val_f1 = self._f1(logits[self.val_idx], self.data.y[self.val_idx])
            test_f1 = self._f1(logits[self.test_idx], self.data.y[self.test_idx])
            losses.append(loss.item())
            val_scores.append(val_f1)

            # early stopping ------------------------------------------------
            if val_f1 > best_val:
                best_val, best_test = val_f1, test_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                break
            if epoch == 1 or epoch % 20 == 0:
                print(f"[{self.run_name}] epoch {epoch:03d}  loss={loss.item():.4f}  valF1={val_f1:.3f}")

        toc = time.time() - tic
        model.load_state_dict(best_state)
        logits = model(self.data, tau=0.0)
        final_test = self._f1(logits[self.test_idx], self.data.y[self.test_idx])

        # save curves --------------------------------------------------------
        from .evaluate import save_training_curves  # local import to avoid circular
        save_training_curves(losses, val_scores, self.run_name)

        return {
            "best_val": best_val,
            "test": final_test,
            "epochs": len(losses),
            "duration": toc,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _tau_schedule(epoch: int, max_epochs: int, ada_cfg: Dict[str, Any]) -> float:
        pct = epoch / max_epochs
        if pct >= ada_cfg["tau_anneal_portion"]:
            return ada_cfg["tau_final"]
        ratio = pct / ada_cfg["tau_anneal_portion"]
        return ada_cfg["tau0"] + (ada_cfg["tau_final"] - ada_cfg["tau0"]) * ratio

    @staticmethod
    def _f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == labels).sum().item() / labels.numel()
