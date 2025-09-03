```python
"""src/evaluate.py
Utilities for computing metrics and model evaluation.
The original implementation required the heavyweight ``torch_geometric``
package to obtain the ``Data`` class that is *only* used for static type
annotations.  To keep the dependency list minimal and avoid large binary
wheels, we replicate the graceful-degradation strategy used throughout the
code-base: attempt to import the real class first and fall back to a very
light-weight stub when the import fails.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Optional torch-geometric dependency (see ``src/train.py`` for details)
# -----------------------------------------------------------------------------
try:
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover – lightweight stub

    class Data:  # pylint: disable=too-few-public-methods
        """Minimal stand-in for ``torch_geometric.data.Data`` used only for
        type annotations and the ``.to(device)`` helper.
        """

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        # Preserve the interface expected by the rest of the code base
        def to(self, device: torch.device | str, **kwargs):  # noqa: D401
            """Move all tensor attributes **in-place** to *device*.

            The original stub was a no-op which resulted in tensors remaining
            on the CPU even when the model resided on the GPU, ultimately
            leading to a device-mismatch runtime error.  We now iterate over
            all attributes and move those that are ``torch.Tensor``s so that
            the behaviour matches the real ``torch_geometric.data.Data``
            implementation closely enough for our use-case.
            """
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    self.__dict__[k] = v.to(device, **kwargs)
            return self

################################################################################
#  METRIC PRIMITIVES
################################################################################

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item() * 100


def row_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=1, keepdim=True)).norm(dim=1).mean().item()


def col_diff(z: torch.Tensor) -> float:
    z = z.detach()
    return (z - z.mean(dim=0, keepdim=True)).norm(dim=1).mean().item()


def apsd(z: torch.Tensor) -> float:
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    return s.mean().item()

################################################################################
#  FULL EVALUATION ROUTINE
################################################################################

@torch.no_grad()
def evaluate(model, data: Data) -> Dict[str, float]:
    model.eval()
    logits, _ = model(data.x, data.edge_index)
    out: Dict[str, float] = {}
    for split in ["train", "val", "test"]:
        mask = getattr(data, f"{split}_mask")
        out[f"acc_{split}"] = accuracy(logits[mask], data.y[mask])

    # Representation statistics on *all* nodes
    _, h = model(data.x, data.edge_index)
    out["row_diff"] = row_diff(h)
    out["col_diff"] = col_diff(h)
    out["apsd"] = apsd(h)
    return out
```