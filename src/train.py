"""
train.py – models and training utilities (patched)
"""
from __future__ import annotations
# NOTE: the `from __future__` import has to come right after the (optional)
# module doc-string; any stray text (e.g. the previous `[UPDATED]` marker)
# would break the rule and raise `SyntaxError`.  The marker has therefore
# been removed.

import json
import math
import pathlib
import time
from typing import Dict, Any, List, Union

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
#  Device helper (FIX)                                                       #
###############################################################################


def _resolve_device(device_cfg: Union[str, torch.device]) -> torch.device:
    """Robustly resolve the `device` field coming from config.yaml.

    The original YAML contained a literal Python expression
        "cuda if torch.cuda.is_available() else cpu"
    which is *not* a valid torch.device string.  We interpret this in the
    intended way rather than failing with RuntimeError.
    """
    if isinstance(device_cfg, torch.device):
        return device_cfg

    if not isinstance(device_cfg, str):
        raise ValueError(f"Unsupported type for device spec: {type(device_cfg)}")

    device_cfg = device_cfg.strip().lower()

    # 1) Literal expression pattern from the YAML -------------------------
    if "if torch.cuda.is_available()" in device_cfg:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Common aliases ----------------------------------------------------
    if device_cfg in {"auto", "cuda"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg in {"cpu", "gpu"}:
        return torch.device(
            "cuda" if (device_cfg == "gpu" and torch.cuda.is_available()) else "cpu"
        )

    # 3) Fallback – hope it's a valid torch.device string ------------------
    return torch.device(device_cfg)


###############################################################################
#  Models (unchanged)                                                        #
###############################################################################

# ... [rest of file unchanged, omitted here for brevity] ...

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

    # ------------------------------------------------------------------
    #  DEVICE SELECTION (patched)
    # ------------------------------------------------------------------
    device = _resolve_device(cfg["common"]["device"])
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
    #  Plot training curves → .research/iteration26/images
    # ------------------------------------------------------------------
    xs = list(
        range(0, len(tr_loss_hist) * cfg["common"]["print_every"], cfg["common"]["print_every"])
    )
    line(xs, {"train_loss": tr_loss_hist}, "epoch", "loss", f"Train-loss seed{seed}", "training_loss.pdf")
    line(xs, {"val_acc": val_acc_hist}, "epoch", "acc", f"Val-acc seed{seed}", "accuracy.pdf")

    # ------------------------------------------------------------------
    #  Test evaluation & metrics
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits, h_final, k_final = model(data.x, data.edge_index, epoch=999)

    metrics: Dict[str, float] = cls_metrics(logits[data.test_mask], data.y[data.test_mask])
    metrics.update(
        {
            "row_diff": row_diff(h_final),
            "col_diff": col_diff(h_final),
            "eff_rank": eff_rank(h_final),
        }
    )

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
