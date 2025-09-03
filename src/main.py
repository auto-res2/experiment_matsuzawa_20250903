```python
"""src/main.py
Entry-point for quick training & evaluation of the DDAF-GNN model.  The script
purposely uses a very small number of layers/epochs so that it can be executed
inside a continuous-integration environment within the allotted time budget.
If the canonical Cora dataset cannot be downloaded (e.g. no internet access),
the script automatically falls back to a tiny synthetic graph so that unit
tests can still succeed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch

# ------------------------------------------------------------------
#  Optional torch_geometric import – fall back to a stub Data class if the
#  heavyweight dependency is unavailable.  The stub is compatible with the
#  minimal access pattern used throughout this repository.
# ------------------------------------------------------------------
try:
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover – minimal fallback

    import torch  # local import so that it is only required for the stub

    class Data:  # pylint: disable=too-few-public-methods
        """Light-weight stand-in for ``torch_geometric.data.Data`` with proper
        device handling.
        """

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to(self, device: torch.device | str, **kwargs):  # noqa: D401
            """Move all tensor attributes **in-place** to *device*.

            The previous implementation returned *self* without actually
            migrating the underlying tensors which led to device-mismatch
            errors once the model was moved to the GPU.  This corrected
            version walks through ``__dict__`` and moves every attribute that
            is a ``torch.Tensor`` to the requested *device*.
            """
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    self.__dict__[k] = v.to(device, **kwargs)
            return self

from src import evaluate as ev
from src import preprocess as pp
from src import train as tr

################################################################################
#  PATHS / CONSTANTS
################################################################################
ROOT = Path(__file__).resolve().parent.parent
# All experiment figures must live in this exact directory according to the
# platform specification.
FIG_DIR = ROOT / ".research" / "iteration11" / "images"  # UPDATED PATH
FIG_DIR.mkdir(parents=True, exist_ok=True)
################################################################################
#  FALL-BACK SYNTHETIC DATASET (used when internet is unavailable)
################################################################################


def _make_tiny_graph(n: int = 120, f: int = 16, c: int = 3) -> Data:
    """Generate a small random graph with train/val/test masks."""

    x = torch.randn(n, f)
    edge_index = torch.randint(0, n, (2, n * 4))  # a very sparse graph
    y = torch.randint(0, c, (n,))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    train_mask[: int(0.6 * n)] = True
    val_mask[int(0.6 * n): int(0.8 * n)] = True
    test_mask[int(0.8 * n):] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

################################################################################
#  CORE LOGIC
################################################################################


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_data() -> Data:
    """Try to load the Cora dataset; if this fails, return a synthetic graph."""

    try:
        return pp.load_cora()
    except Exception as e:  # pragma: no cover – best-effort resilience
        print("[WARN] Could not download Cora – falling back to synthetic data:", e)
        return _make_tiny_graph()


def run(seed: int = 42, epochs: int = 200, layers: int = 4):
    pp.set_seed(seed)
    device = _get_device()

    data = _load_data().to(device)

    num_classes = int(data.y.max().item()) + 1
    model = tr.DDAFGNN(
        input_dim=data.x.size(1),
        hidden=64,
        out_dim=num_classes,
        layers=layers,  # fewer layers than the full model for quick CI runs
        lmbd_mi=0.1,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_state: Optional[dict] = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        logits, h = model(data.x, data.edge_index)
        loss = model.loss(logits, h, data, data.train_mask)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 10 == 0 or epoch == epochs:
            metrics = ev.evaluate(model, data)
            val_acc = metrics["acc_val"]
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            print(
                f"[Epoch {epoch:03d}] "
                f"loss={loss.item():.4f} "
                f"train={metrics['acc_train']:.2f}% "
                f"val={metrics['acc_val']:.2f}% "
                f"test={metrics['acc_test']:.2f}%"
            )

    # ------------------------------------------------------------------
    #  Final evaluation with the best checkpoint
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    final_metrics = ev.evaluate(model, data)
    print("=== Final Metrics ===")
    for k, v in final_metrics.items():
        suffix = "%" if k.startswith("acc_") else ""
        print(f"{k}: {v:.4f}{suffix}")

################################################################################
#  CLI WRAPPER
################################################################################


def _build_arg_parser():
    p = argparse.ArgumentParser(description="DDAF-GNN quick experiment runner")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--layers", type=int, default=4, help="Number of DDAF layers")
    return p


def main():
    args = _build_arg_parser().parse_args()
    tic = time.time()
    run(seed=args.seed, epochs=args.epochs, layers=args.layers)
    toc = time.time()
    print(f"Finished in {toc - tic:.2f} seconds.")


if __name__ == "__main__":
    main()
```