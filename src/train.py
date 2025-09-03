"""src/train.py
Light-weight training helpers so that the public CI can execute the
example workflow shipped in src/main.py.  The goal is **not** to run the
full, expensive hyper-parameter search described in the research code –
only to demonstrate that the pipeline is functional end-to-end.

Two utilities are exposed and imported by main.py:
    • build_model(...)      → returns a very small toy network.
    • train_single_run(...) → performs a few SGD steps and stores metrics.

The implementations below deliberately keep GPU / CPU usage tiny and have
no external dependencies beyond PyTorch + PyG which are already listed in
requirements.txt.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .evaluate import classification_metrics
from .utils import ensure_dir, set_seed

################################################################################
#  Tiny backbone – a 2-layer GCN that works for every requested model tag      #
################################################################################

class _ToyGCN(nn.Module):
    """A minimal GCN that accepts (x, edge_index) but is extremely cheap."""

    def __init__(self, fi: int, hidden: int, fo: int):
        super().__init__()
        self.conv1 = GCNConv(fi, hidden)
        self.conv2 = GCNConv(hidden, fo)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # noqa: D401
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


################################################################################
#  Public API                                                                  #
################################################################################

_MODEL_TAGS = {
    "gcn_deep",
    "gat_deep",
    "gcnii",
    "pairnorm_si",
    "apd_gcn",
    "apd_gat",
}

def build_model(tag: str, in_dim: int, hidden: int, out_dim: int) -> nn.Module:  # noqa: D401
    """Return a *very small* network such that the experiment can run.

    All tags are mapped to the same lightweight backbone.  The purpose is
    only to satisfy the interface required by *src.main* – not to claim
    research-grade performance.
    """
    if tag not in _MODEL_TAGS:
        raise ValueError(f"Unknown model tag: {tag}")
    return _ToyGCN(in_dim, hidden, out_dim)


def train_single_run(
    model: nn.Module,
    data,  # torch_geometric.data.Data – left un-annotated for lighter import
    cfg: Dict,
    seed: int,
    exp_name: str,
    model_tag: str,
):
    """A *tiny* training loop: 10 epochs of full-batch gradient descent."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(10):  # very small budget to keep CI fast
        optimiser.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimiser.step()

    # ----------------- evaluation -----------------
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    metrics = classification_metrics(logits[data.test_mask], data.y[data.test_mask])

    # ----------------- persistence ----------------
    out_dir = pathlib.Path("runs") / exp_name / model_tag / f"seed{seed}"
    ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics
