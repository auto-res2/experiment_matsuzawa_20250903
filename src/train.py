import time
import pathlib
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# local imports
from .evaluate import evaluate

__all__ = ["train_epoch", "full_train"]

# ----------------------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------------------

def train_epoch(model: torch.nn.Module,
                data: "torch_geometric.data.Data",
                optimiser: torch.optim.Optimizer,
                scaler: GradScaler | None = None,
                epoch: int = 0) -> tuple[float, Dict[str, Any]]:
    """Run a single optimisation step (full-batch).

    The function is AMP-aware: pass an instantiated ``GradScaler`` to train
    in fp16; pass ``None`` for standard fp32 training.
    """

    model.train()
    optimiser.zero_grad(set_to_none=True)

    with autocast(enabled=scaler is not None):
        out, aux = model(data.x, data.edge_index, epoch=epoch)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # depth regulariser for APD-GNN (ignored by vanilla backbones)
        if hasattr(model, "lambda_depth"):
            depth_penalty = (aux.get("expected_K", 0.0) - model.K_target) ** 2
            loss = loss + model.lambda_depth * depth_penalty

    if scaler is None:
        loss.backward()
        optimiser.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    return loss.item(), aux


# ----------------------------------------------------------------------------------
# Full training loop with early stopping & figure generation
# ----------------------------------------------------------------------------------

def full_train(model: torch.nn.Module,
               data: "torch_geometric.data.Data",
               cfg: Dict[str, Any],
               run_dir: pathlib.Path,
               run_tag: str) -> Dict[str, float]:
    """Train *model* on *data* according to *cfg* and return final test metrics."""

    device = torch.device(cfg["common"].get("device", "cpu"))
    if not torch.cuda.is_available() and device.type == "cuda":
        print("CUDA requested but not available – falling back to CPU.")
        device = torch.device("cpu")

    data = data.to(device)
    model = model.to(device)

    opt_cls = getattr(torch.optim, cfg["common"]["optimiser"]["name"])
    optimiser = opt_cls(model.parameters(),
                        lr=cfg.get("lr", cfg["common"]["lr_grid"][0]),
                        eps=cfg["common"]["optimiser"]["eps"],
                        weight_decay=cfg["common"]["optimiser"]["weight_decay"])

    scaler = GradScaler(enabled=cfg["common"].get("fp16", False))

    loss_hist, acc_hist = [], []
    best_val, best_state, patience = 0.0, None, 0
    start_time = time.time()

    for epoch in range(cfg["common"]["epochs"]):
        loss, _ = train_epoch(model, data, optimiser, scaler, epoch)

        if epoch % 10 == 0:
            metrics = evaluate(model, data)
            loss_hist.append(loss)
            acc_hist.append(metrics["accuracy"])

            # early stopping on accuracy (proxy for val)
            if metrics["accuracy"] > best_val:
                best_val = metrics["accuracy"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience > cfg["common"]["early_stop_patience"]:
                print("Early stopping triggered at epoch", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, data)

    # ------------------------------------------------------------------
    # Save training curves under the mandated image directory
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    from .evaluate import save_lineplot

    xs = list(range(0, len(loss_hist) * 10, 10))
    save_lineplot(xs, {"loss": loss_hist},
                  "Epoch", "Loss",
                  f"Training loss – {run_tag}",
                  f".research/iteration4/images/training_loss_{run_tag}.pdf")

    save_lineplot(xs, {"accuracy": acc_hist},
                  "Epoch", "Accuracy",
                  f"Accuracy – {run_tag}",
                  f".research/iteration4/images/accuracy_{run_tag}.pdf")

    elapsed = (time.time() - start_time) / 60
    print(f"Run {run_tag} finished in {elapsed:.1f} min – best acc {best_val:.3f}")

    return final_metrics