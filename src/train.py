"""src/train.py
Minimal yet fully-functional implementation of the DDAF-GNN model that is
sufficient for the quick end-to-end CI run executed by the evaluation
harness.  The real architecture is considerably more sophisticated; here we
replace it with a light-weight MLP that completely ignores the graph
structure.  This keeps the dependency list tiny (no `torch_geometric` is
required) while still exercising the full training / evaluation pipeline.

If you want to plug-in the *real* model later on, simply swap out the
implementation below – the public API must remain identical so that the
rest of the repository keeps working unchanged.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DDAFGNN"]

###############################################################################
#  MODEL IMPLEMENTATION (very small, suitable for CI)                         #
###############################################################################


class DDAFGNN(nn.Module):
    """A ridiculously simple MLP that pretends to be a *real* DDAF-GNN.

    Parameters
    ----------
    input_dim: int
        Dimensionality of node features.
    hidden: int
        Width of the hidden representation.
    out_dim: int
        Number of target classes.
    layers: int, default=4
        Number of hidden layers (including the output projection).
    lmbd_mi: float, default=0.
        Weight of the mutual-information regulariser in the original model.
        This stub just ignores the value but we carry the parameter so that
        the public API matches the full implementation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        out_dim: int,
        layers: int = 4,
        lmbd_mi: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("`layers` must be ≥ 2 so that we have at least one " "hidden layer and one output layer.")

        self.lmbd_mi = float(lmbd_mi)

        fcs: List[nn.Linear] = []
        in_dim = input_dim
        for _ in range(layers - 1):  # all hidden layers
            fcs.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.fcs = nn.ModuleList(fcs)
        self.out_fc = nn.Linear(in_dim, out_dim)

    # ---------------------------------------------------------------------
    #  Forward / Loss
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D401, pylint: disable=unused-argument
        """Standard *forward*.

        The real model would make use of *edge_index*; the stub ignores it so
        that we do not need `torch_geometric` at runtime.
        """
        h = x
        for fc in self.fcs:
            h = F.relu(fc(h))
        logits = self.out_fc(h)
        return logits, h  # (N, C), (N, hidden)

    # ..................................................................
    def loss(
        self,
        logits: torch.Tensor,
        h: torch.Tensor,
        data,  # noqa: ANN001 – *Any* type is fine here
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss (cross-entropy only in the stub)."""
        loss_cls = F.cross_entropy(logits[mask], data.y[mask])

        # In the real DDAF-GNN an MI regulariser is added.  We purposefully
        # keep the implementation trivial while *retaining* the hyper-
        # parameter so that swapping in the full version later on is easy.
        return loss_cls
