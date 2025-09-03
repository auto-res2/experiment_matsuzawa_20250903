"""src/train.py
Model definitions and training utilities.
"""
from __future__ import annotations
import json
import time
import pathlib
import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GCN2Conv
from torch_geometric.utils import degree

from .evaluate import classification_metrics, spearman_corr, plot_lines
from .utils import set_seed, ensure_dir, IMAGE_ROOT

################################################################################
#  Model architectures                                                          #
################################################################################

class DeepGCN(nn.Module):
    """Vanilla deep GCN baseline (128-layer)."""

    def __init__(self, fi: int, hid: int, fo: int, layers: int = 128):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(fi if i == 0 else hid, hid if i < layers - 1 else fo) for i in range(layers)]
        )

    def forward(self, x: torch.Tensor, ei: torch.Tensor):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, ei)


class DeepGAT(nn.Module):
    """Deep GAT baseline (128-layer / 8 heads)."""

    def __init__(self, fi: int, hid: int, fo: int, layers: int = 128):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(fi, hid // 8, heads=8))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid, hid // 8, heads=8))
        self.convs.append(GATConv(hid, fo, heads=1, concat=False))

    def forward(self, x: torch.Tensor, ei: torch.Tensor):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, ei))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, ei)


class GCNII(nn.Module):
    """GCNII baseline (64-layer)."""

    def __init__(self, fi: int, hid: int, fo: int, layers: int = 64, alpha: float = 0.1, theta: float = 0.5):
        super().__init__()
        self.x0_lin = nn.Linear(fi, hid)
        self.convs = nn.ModuleList([GCN2Conv(hid, alpha, theta, layer=i) for i in range(layers)])
        self.out_lin = nn.Linear(hid, fo)

    def forward(self, x: torch.Tensor, ei: torch.Tensor):
        x0 = F.relu(self.x0_lin(x))
        h = x0
        for conv in self.convs:
            h = F.dropout(h, p=0.5, training=self.training)
            h = F.relu(conv(h, x0, ei))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.out_lin(h)


class PairNormGCN(nn.Module):
    """PairNorm-SI baseline (128-layer)."""

    def __init__(self, fi: int, hid: int, fo: int, layers: int = 128):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(fi if i == 0 else hid, hid) for i in range(layers - 1)])
        self.final = GCNConv(hid, fo)

    def forward(self, x: torch.Tensor, ei: torch.Tensor):
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            # PairNorm-SI
            x = x - x.mean(dim=0, keepdim=True)
            x = F.normalize(x, p=2, dim=1)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.final(x, ei)


# -----------------------------------------------------------------------------
# Adaptive Propagation Depth (APD)
# -----------------------------------------------------------------------------
class _Gate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N, 1)
        return torch.sigmoid(self.mlp(z))


def _gumbel_sigmoid(p: torch.Tensor, tau: float, eps: float = 1e-9) -> torch.Tensor:
    g = -torch.empty_like(p).exponential_().log()
    return torch.sigmoid((torch.log(p + eps) + g) / tau)


class APD(nn.Module):
    """Backbone-agnostic Adaptive Propagation Depth wrapper."""

    def __init__(
        self,
        backbone: str,
        fi: int,
        hid: int,
        fo: int,
        layers: int = 128,
        lambda_depth: float = 0.1,
        lambda_div: float = 1e-4,
        k_target: int = 8,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList
        if backbone == "gcn":
            self.layers = nn.ModuleList([GCNConv(fi if i == 0 else hid, hid) for i in range(layers)])
        elif backbone == "gat":
            self.layers = nn.ModuleList([GATConv(fi if i == 0 else hid, hid // 8, heads=8) for _ in range(layers)])
        else:
            raise ValueError(backbone)

        self.gates = nn.ModuleList([_Gate(hid + 2) for _ in range(layers)])
        self.classifier = nn.Linear(hid, fo)

        # regulariser hyper-parameters
        self.lambda_depth = lambda_depth
        self.lambda_div = lambda_div
        self.k_target = k_target
        self.t0, self.tF = 1.0, 0.2  # temperature schedule for Gumbel-Sigmoid

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, ei: torch.Tensor, *, epoch: int = 0):
        N = x.size(0)
        deg = degree(ei[0], num_nodes=N).unsqueeze(1)
        tau = max(self.tF, self.t0 - (self.t0 - self.tF) * epoch / 200)

        halted = torch.zeros(N, device=x.device)  # indicator for finished nodes
        weight_sum = torch.zeros_like(halted)
        rep_sum = torch.zeros(N, self.classifier.in_features, device=x.device)
        k_expected = torch.zeros_like(halted)

        # message-passing with per-layer gating
        for l, layer in enumerate(self.layers):
            x = torch.relu(layer(x, ei))
            p = self.gates[l](torch.cat([x, deg, torch.full_like(deg, l)], dim=1)).squeeze()
            z = _gumbel_sigmoid(p, tau) if self.training else p  # differentiable sample
            cont = (1 - halted) * z  # nodes still continuing

            weight_sum += cont
            rep_sum += cont.unsqueeze(1) * x
            k_expected += cont
            # update halted mask (only matters for inference/analysis)
            halted = torch.where((halted == 0) & (cont > 0.5), torch.ones_like(halted), halted)

        h_final = rep_sum / (weight_sum.unsqueeze(1) + 1e-6)
        return self.classifier(h_final), k_expected

    # ------------------------------------------------------------------
    def depth_regulariser(self, k: torch.Tensor) -> torch.Tensor:
        """Expected-depth regulariser (Equation 4 in the paper)."""
        return self.lambda_depth * (k.mean() - self.k_target).pow(2)

################################################################################
#  Training helper                                                             #
################################################################################

def train_single_run(
    model: nn.Module,
    data,
    cfg: dict,
    seed: int,
    exp_name: str,
    model_tag: str,
) -> dict:
    """Train *model* on *data* for one random seed and return evaluation metrics."""

    # ------------------------------------------------------------------
    set_seed(seed)
    device = torch.device("cuda" if (cfg["common"].get("device") in {"cuda", "auto"} and torch.cuda.is_available()) else "cpu")

    model, data = model.to(device), data.to(device)

    # optimiser --------------------------------------------------------
    opt_cfg = cfg["common"]["optimiser"]
    optimiser = getattr(torch.optim, opt_cfg.pop("name"))(model.parameters(), **opt_cfg)

    # bookkeeping ------------------------------------------------------
    losses, val_accs = [], []
    best_val, patience, best_state = -1.0, 0, None
    t_start = time.time()

    for epoch in range(cfg["common"]["epochs"]):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        logits, k = model(data.x, data.edge_index, epoch=epoch)

        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if isinstance(model, APD):
            loss = loss + model.depth_regulariser(k)
        loss.backward()
        optimiser.step()

        if epoch % cfg["common"]["print_every"] == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(data.x, data.edge_index, epoch=epoch)
            val_acc = classification_metrics(val_logits[data.val_mask], data.y[data.val_mask])["acc"]
            losses.append(loss.item())
            val_accs.append(val_acc)
            # early-stopping --------------------------------------
            if val_acc > best_val:
                best_val, best_state, patience = val_acc, model.state_dict(), 0
            else:
                patience += 1
            if patience > cfg["common"]["early_stop"]:
                break

    # restore best ------------------------------------------------------
    model.load_state_dict(best_state)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # plots
    xs = list(range(0, len(losses) * cfg["common"]["print_every"], cfg["common"]["print_every"]))
    fig_dir = IMAGE_ROOT / exp_name / model_tag / f"seed{seed}"
    ensure_dir(fig_dir)
    plot_lines(xs, {"train_loss": losses}, "epoch", "loss", f"Train-loss-{seed}", fig_dir / "training_loss.pdf")
    plot_lines(xs, {"val_acc": val_accs}, "epoch", "accuracy", f"Val-acc-{seed}", fig_dir / "accuracy.pdf")

    # ------------------------------------------------------------------
    # final test evaluation
    model.eval()
    with torch.no_grad():
        test_logits, k = model(data.x, data.edge_index, epoch=999)

    metrics = classification_metrics(test_logits[data.test_mask], data.y[data.test_mask])

    if isinstance(model, APD):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        metrics.update({
            "mean_K": float(k.mean()),
            "var_K": float(k.var()),
            "rho": spearman_corr(k, deg),
        })

    # save numeric metrics next to figures --------------------------------
    metrics_path = fig_dir / "metrics.json"
    with metrics_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Finished seed={seed:02d} | model={model_tag:<10} | acc={metrics['acc']:.4f} | runtime={time.time()-t_start:.1f}s")
    return metrics

################################################################################
#  Factory helper                                                              #
################################################################################

def build_model(tag: str, fi: int, hid: int, fo: int) -> nn.Module:
    """Utility that maps *tag* ⇢ instantiated nn.Module."""

    factories = {
        "gcn_deep": lambda: DeepGCN(fi, hid, fo),
        "gat_deep": lambda: DeepGAT(fi, hid, fo),
        "gcnii": lambda: GCNII(fi, hid, fo),
        "pairnorm_si": lambda: PairNormGCN(fi, hid, fo),
        "apd_gcn": lambda: APD("gcn", fi, hid, fo),
        "apd_gat": lambda: APD("gat", fi, hid, fo),
    }
    if tag not in factories:
        raise KeyError(f"Unknown model tag '{tag}'. Available: {list(factories)}")
    return factories[tag]()
