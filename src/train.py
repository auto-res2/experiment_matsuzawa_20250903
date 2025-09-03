import time
import pathlib
import random
import numpy as np
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.utils import degree

from .evaluate import (
    classification_metrics,
    row_diff,
    col_diff,
    effective_rank,
    spearman,
    lineplot,
)

# --------------------------------------------------------------------------------------
#  Utility
# --------------------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Make experiment fully deterministic (CUDA-deterministic kernels may be slower)."""
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # torch.use_deterministic_algorithms is available from 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------------------
#  Single-run training routine
# --------------------------------------------------------------------------------------

def fit(
    model: torch.nn.Module,
    data,
    cfg: Dict[str, Any],
    run_dir: pathlib.Path,
    seed: int = 0,
) -> Dict[str, float]:
    """Train *model* on *data* according to *cfg* and return a dictionary of metrics."""

    _set_seed(seed)
    device = torch.device(cfg["common"]["device"])

    data = data.to(device)
    model = model.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,
        weight_decay=cfg["common"]["optimiser"]["weight_decay"],
        eps=cfg["common"]["optimiser"]["eps"],
    )
    scaler = GradScaler(enabled=not cfg["common"]["fp16"])  # AMP disabled by default

    best_val, best_state, patience = 0.0, None, 0
    loss_hist, val_hist = [], []
    print_every = cfg["common"]["print_every"]

    for epoch in range(cfg["common"]["epochs"]):
        model.train()
        optimiser.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["common"]["fp16"]):
            if hasattr(model, "L"):
                # APD-GNN branch ---------------------------------------------------------
                logits, h, k_exp = model(data.x, data.edge_index, epoch)
                loss_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                loss_reg = model.lambda_depth * (k_exp.mean() - model.k_target).pow(2)
                loss = loss_cls + loss_reg
            else:
                # vanilla GNN -----------------------------------------------------------
                logits, h = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # ------------------------ validation -----------------------------------------
        if epoch % print_every == 0:
            model.eval()
            with torch.no_grad():
                val_logits, *_ = model(data.x, data.edge_index)
                val_acc = classification_metrics(
                    val_logits[data.val_mask], data.y[data.val_mask]
                )["accuracy"]

            loss_hist.append(loss.item())
            val_hist.append(val_acc)

            if val_acc > best_val:
                best_val, best_state, patience = val_acc, model.state_dict(), 0
            else:
                patience += 1

            if patience > cfg["common"]["early_stop_patience"]:
                break

    # Restore best model ----------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------------------------------------------------------------------
    #  Diagnostics – save learning curves
    # -----------------------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs_axis = list(range(0, len(loss_hist) * print_every, print_every))

    lineplot(
        epochs_axis,
        {"train_loss": loss_hist},
        xlabel="epoch",
        ylabel="loss",
        title=f"Training loss – seed {seed}",
        fname="training_loss.pdf",
    )
    lineplot(
        epochs_axis,
        {"val_acc": val_hist},
        xlabel="epoch",
        ylabel="accuracy",
        title=f"Validation accuracy – seed {seed}",
        fname="accuracy.pdf",
    )

    # -----------------------------------------------------------------------------------
    #  Final evaluation on the test split
    # -----------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        if hasattr(model, "L"):
            logits, h_final, k_final = model(data.x, data.edge_index, epoch=999)
        else:
            logits, h_final = model(data.x, data.edge_index)

    metrics: Dict[str, float] = classification_metrics(
        logits[data.test_mask], data.y[data.test_mask]
    )

    # Structural/oversmoothing metrics --------------------------------------------------
    metrics.update(
        {
            "row_diff": row_diff(h_final),
            "col_diff": col_diff(h_final),
            "eff_rank": effective_rank(h_final),
        }
    )

    if hasattr(model, "L"):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        metrics.update(
            {
                "mean_K": k_final.mean().item(),
                "var_K": k_final.var().item(),
                "spearman": spearman(k_final, deg),
            }
        )

    # Persist checkpoint and metrics ----------------------------------------------------
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, run_dir / "checkpoint.pt")

    return metrics
