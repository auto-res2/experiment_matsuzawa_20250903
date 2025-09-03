"""
train.py – models and training utilities
"""
from __future__ import annotations
import json
import math
import pathlib
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCN2Conv,
    GCNConv,
    PairNorm,
)
import torch_geometric.utils as pyg_utils

from .evaluate import (
    IMAGES_DIR,
    cls_metrics,
    col_diff,
    eff_rank,
    line,
    row_diff,
    spearman,
)

###############################################################################
#  Re-usable helpers                                                          #
###############################################################################

def _set_seed(seed: int) -> None:
    """Make all relevant libraries deterministic."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # be deterministic – will fall back gracefully if CUDA < 10.2
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


###############################################################################
#  Models                                                                     #
###############################################################################


class PairNormSI(nn.Module):
    """Scale-invariant PairNorm (Zhao & Akoglu, 2020)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x - x.mean(0, keepdim=True)
        # avoid division by 0
        denom = x.norm(p=2, dim=1, keepdim=True).mean().clamp_min(1e-12)
        return x / denom


class DeepGCN(nn.Module):
    def __init__(self, f_in: int, hid: int, f_out: int, layers: int = 128):
        super().__init__()
        self.dp = 0.5
        self.convs = nn.ModuleList(
            [
                GCNConv(f_in if i == 0 else hid, hid if i < layers - 1 else f_out)
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, ei: torch.Tensor):  # type: ignore[override]
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.convs[-1](x, ei), x


class DeepGAT(nn.Module):
    def __init__(
        self, f_in: int, hid: int, f_out: int, layers: int = 128, heads: int = 8
    ):
        super().__init__()
        self.dp = 0.5
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(f_in, hid // heads, heads=heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid, hid // heads, heads=heads))
        self.convs.append(GATConv(hid, f_out, heads=1, concat=False))

    def forward(self, x: torch.Tensor, ei: torch.Tensor):  # type: ignore[override]
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, ei))
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.convs[-1](x, ei), x


class GCNII(nn.Module):
    def __init__(
        self,
        f_in: int,
        hid: int,
        f_out: int,
        layers: int = 64,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        super().__init__()
        self.dp = 0.5
        self.ff_in = nn.Linear(f_in, hid)
        self.convs = nn.ModuleList(
            [GCN2Conv(hid, alpha, theta, layer=i + 1) for i in range(layers)]
        )
        self.head = nn.Linear(hid, f_out)

    def forward(self, x: torch.Tensor, ei: torch.Tensor):  # type: ignore[override]
        x0 = F.relu(self.ff_in(x))
        h = x0
        for conv in self.convs:
            h = F.dropout(h, p=self.dp, training=self.training)
            h = F.relu(conv(h, x0, ei))
        h = F.dropout(h, p=self.dp, training=self.training)
        return self.head(h), h


class PairNormGCN(nn.Module):
    """Deep GCN backbone with scale-invariant PairNorm after every layer."""

    def __init__(self, f_in: int, hid: int, f_out: int, layers: int = 128):
        super().__init__()
        self.dp = 0.5
        self.pn = PairNormSI()
        self.convs = nn.ModuleList(
            [GCNConv(f_in if i == 0 else hid, hid) for i in range(layers - 1)]
        )
        self.out = GCNConv(hid, f_out)

    def forward(self, x: torch.Tensor, ei: torch.Tensor):  # type: ignore[override]
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = self.pn(x)
            x = F.dropout(x, p=self.dp, training=self.training)
        return self.out(x, ei), x


###############################################################################
#  Adaptive-Propagation-Depth GNN                                             #
###############################################################################


def _gumbel(p: torch.Tensor, tau: float, eps: float = 1e-9) -> torch.Tensor:
    g = -torch.empty_like(p).exponential_().log()
    return torch.sigmoid((torch.log(p + eps) + g) / tau)


class GateMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.net(z))


class APD(nn.Module):
    """Adaptive Propagation Depth wrapper – supports GCN / GAT backbones."""

    def __init__(
        self,
        backbone: str,
        f_in: int,
        hid: int,
        f_out: int,
        layers: int = 128,
        lambda_depth: float = 0.1,
        lambda_div: float = 1e-4,
        k_target: int = 8,
    ):
        super().__init__()
        self.L = layers
        self.lambda_depth = lambda_depth
        self.lambda_div = lambda_div
        self.k_target = k_target
        self.tau0, self.tauF = 1.0, 0.2

        if backbone == "gcn":
            self.convs = nn.ModuleList(
                [GCNConv(f_in if i == 0 else hid, hid) for i in range(layers)]
            )
        elif backbone == "gat":
            self.convs = nn.ModuleList(
                [GATConv(f_in if i == 0 else hid, hid // 8, heads=8) for _ in range(layers)]
            )
        else:
            raise ValueError(backbone)

        # gate gets [h_i^l , degree , layer_index]
        self.gates = nn.ModuleList([GateMLP(hid + 2) for _ in range(layers)])
        self.classifier = nn.Linear(hid, f_out)

    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ei: torch.Tensor,
        *,
        epoch: int = 0,
        eval_exit: bool = False,
    ):
        deg = pyg_utils.degree(ei[0], num_nodes=x.size(0)).unsqueeze(1)
        tau = max(self.tauF, self.tau0 - (self.tau0 - self.tauF) * epoch / 200)

        halted = torch.zeros(x.size(0), device=x.device)
        prob_sum = torch.zeros_like(halted)
        accum = torch.zeros(x.size(0), self.classifier.in_features, device=x.device)
        Kexp = torch.zeros_like(halted)

        for l, conv in enumerate(self.convs):
            x = torch.relu(conv(x, ei))
            gate_in = torch.cat([x, deg, torch.full_like(deg, l)], dim=1)
            p = self.gates[l](gate_in).squeeze()
            z = _gumbel(p, tau) if self.training else p

            cont = (1 - halted) * z  # expected prob of continuing
            prob_sum += cont
            accum += cont.unsqueeze(1) * x
            Kexp += cont

            if eval_exit:
                halted = halted + (cont > 0.5).float()

        h_hat = accum / (prob_sum.unsqueeze(1) + 1e-6)
        logits = self.classifier(h_hat)
        return logits, h_hat, Kexp

    # ------------------------------------------------------------------
    def reg_loss(self, Kexp: torch.Tensor) -> torch.Tensor:
        """Depth regulariser that keeps expected K close to k_target."""
        return self.lambda_depth * ((Kexp.mean() - self.k_target) ** 2)


###############################################################################
#  Training Loop                                                              #
###############################################################################


def fit(
    model: nn.Module,
    data: "torch_geometric.data.Data",
    cfg: Dict[str, Any],
    run_dir: pathlib.Path,
    seed: int,
) -> Dict[str, float]:
    """Generic node-classification training loop with early stopping."""

    _set_seed(seed)
    device = torch.device(cfg["common"]["device"])
    model, data = model.to(device), data.to(device)

    opt = torch.optim.AdamW(model.parameters(), **cfg["common"]["optimiser"])

    best_val, patience, best_state = 0.0, 0, None
    tr_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    for epoch in range(cfg["common"]["epochs"]):
        model.train()
        opt.zero_grad()
        out, h, k = model(data.x, data.edge_index, epoch=epoch)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        if hasattr(model, "reg_loss"):
            loss = loss + model.reg_loss(k)
        loss.backward()
        opt.step()

        # validation every `print_every` epochs
        if epoch % cfg["common"]["print_every"] == 0:
            model.eval()
            with torch.no_grad():
                v_out, *_ = model(data.x, data.edge_index, epoch=epoch)
                v_acc = cls_metrics(v_out[data.val_mask], data.y[data.val_mask])["accuracy"]
            tr_loss_hist.append(loss.item())
            val_acc_hist.append(v_acc)

            if v_acc > best_val:
                best_val, best_state, patience = v_acc, model.state_dict(), 0
            else:
                patience += 1

            if patience > cfg["common"]["early_stop_patience"]:
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    #  Plot training curves → .research/iteration18/images
    # ------------------------------------------------------------------
    xs = list(range(0, len(tr_loss_hist) * cfg["common"]["print_every"], cfg["common"]["print_every"]))
    line(xs, {"train_loss": tr_loss_hist}, "epoch", "loss", f"Train-loss seed{seed}", "training_loss.pdf")
    line(xs, {"val_acc": val_acc_hist}, "epoch", "acc", f"Val-acc seed{seed}", "accuracy.pdf")

    # ------------------------------------------------------------------
    #  Test evaluation & metrics
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits, h_final, k_final = model(data.x, data.edge_index, epoch=999)

    metrics: Dict[str, float] = cls_metrics(logits[data.test_mask], data.y[data.test_mask])
    metrics.update({
        "row_diff": row_diff(h_final),
        "col_diff": col_diff(h_final),
        "eff_rank": eff_rank(h_final),
    })

    if hasattr(model, "reg_loss"):
        deg = pyg_utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
        metrics.update(
            {
                "mean_K": k_final.mean().item(),
                "var_K": k_final.var().item(),
                "spearman": spearman(k_final, deg),
            }
        )

    # save artefacts – model + metrics
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, run_dir / "checkpoint.pt")

    return metrics
