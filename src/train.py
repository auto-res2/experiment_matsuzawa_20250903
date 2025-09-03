"""src/train.py
Model architectures, memory-allocation controller and task-training
routines live here.  All heavy tensor operations stay in this file so that the
remaining modules have minimal dependencies on PyTorch.
"""
from __future__ import annotations
import math, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# device is determined once in main.py and re-exported here during run-time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
#                              ───  BACKBONE  ───                              #
################################################################################
class FrozenBackbone(nn.Module):
    """ResNet-18 truncated after layer-3 – parameters are frozen."""

    def __init__(self) -> None:
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        # everything except the last two residual layers + FC / pooling
        self.features = nn.Sequential(*list(base.children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False
        self.out_dim: int = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean([-2, -1])  # global average-pool –> 512-d

################################################################################
#                        ───  LOW-RANK  ADAPTER (LoRA) ───                     #
################################################################################
# ---------------------------------------------------------------------------
#  The original implementation depended on an external GitHub repository that
#  cannot be installed in the execution environment because it lacks standard
#  packaging metadata (setup.py/pyproject.toml).  To remove this hard
#  dependency we provide a *very small* in-house alternative that replicates
#  the essential behaviour: a learnable linear projection implemented as the
#  product of two low-rank matrices (B @ A).  This keeps the parameter count
#  and memory footprint comparable to a LoRA head while avoiding any external
#  requirements.
# ---------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    """Linear layer implemented as the product of two low-rank matrices.

    y = x · (B @ A) where  A: [in_dim, r]  and  B: [r, out_dim].
    """

    def __init__(self, in_dim: int, out_dim: int, r: int, bias: bool = False):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, out_dim) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        weight = self.A @ self.B  # [in_dim, out_dim]
        y = x @ weight  # [..., out_dim]
        if self.bias is not None:
            y = y + self.bias
        return y

class SimpleAdapter(nn.Module):
    """Low-rank linear head as a drop-in replacement for InfLoRA.LinearLoRA."""

    def __init__(self, in_dim: int, rank: int, num_classes: int):
        super().__init__()
        self.adapter = LowRankLinear(in_dim, num_classes, r=rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.adapter(x)

################################################################################
#                   ───  AQM  –  VQ-VAE-2  LATENT COMPRESSOR ───               #
################################################################################
# ---------------------------------------------------------------------------
#  The reference code relies on `rosinality/vq-vae-2-pytorch`, another GitHub
#  repository that is not a proper Python package.  For the purposes of the
#  demo the compressor is *never used for a forward pass* in the main
#  experiment.  We therefore provide a minimal stub that fulfils the interface
#  without any heavy dependencies.
# ---------------------------------------------------------------------------
class _VQVAETwoStub(nn.Module):
    def __init__(self, img_size: int, num_layers: int, codebook_dim: int, num_codebook_vectors: int):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_codebook_vectors = num_codebook_vectors

    # return value signature mimics the real implementation
    def encode(self, x):
        B = x.size(0)
        codes = torch.zeros(B, dtype=torch.long, device=x.device)
        return None, None, codes  # dummy outputs

    def decode(self, codes):  # noqa: D401 – dummy recon
        B = codes.size(0)
        return torch.zeros(B, 3, 32, 32, device=codes.device)

class AQM(nn.Module):
    """Vector-quantised latent compressor (stubbed)."""

    def __init__(self, latent_dim: int, codebook_size: int):
        super().__init__()
        # Use the lightweight stub
        self.vqvae = _VQVAETwoStub(
            img_size=32, num_layers=2, codebook_dim=latent_dim, num_codebook_vectors=codebook_size
        )
        for p in self.vqvae.parameters():
            p.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # uint16 indices
        _, _, codes = self.vqvae.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.vqvae.decode(codes)

################################################################################
#                          ───  MEMORY  LEDGER  ───                            #
################################################################################
class KnapsackController:
    """Single-step greedy allocator that moves bytes between replay buffer and
    low-rank adapter subject to a global memory budget M_max.
    """

    def __init__(self, M_max: int):
        self.M_max = int(M_max)
        self.bytes_adapter: int = 0
        self.bytes_buffer: int = 0

    # ---------------------------------------------------------------------
    def update_ledger(self, bytes_adapter: int, bytes_buffer: int) -> None:  # noqa: D401
        self.bytes_adapter = int(bytes_adapter)
        self.bytes_buffer = int(bytes_buffer)

    # ---------------------------------------------------------------------
    def decide(self, util_adapter: float, util_buffer: float) -> Tuple[str, int, int]:
        """Reallocate one percent of the total budget towards the higher marginal
        utility consumer.  Returns (action, new_bytes_adapter, new_bytes_buffer).
        A tiny heuristic but sufficient for the demo script.
        """

        if self.bytes_adapter + self.bytes_buffer > self.M_max:
            raise RuntimeError("Memory ledger overflow – check accounting logic.")

        shift: int = max(1, int(0.01 * self.M_max))  # at least one byte
        action = "keep"
        if util_buffer > util_adapter and self.bytes_adapter >= shift:
            self.bytes_adapter -= shift
            self.bytes_buffer += shift
            action = "A→B"
        elif util_adapter >= util_buffer and self.bytes_buffer >= shift:
            self.bytes_buffer -= shift
            self.bytes_adapter += shift
            action = "B→A"

        return action, self.bytes_adapter, self.bytes_buffer

################################################################################
#                              ───  JEMB  ───                                  #
################################################################################
class JEMBModel(nn.Module):
    """Full continual-learning system composed of a frozen CNN backbone, a
    learnable low-rank adapter and a latent-compression VQ-VAE (AQM).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = FrozenBackbone()
        self.adapter = SimpleAdapter(self.backbone.out_dim, cfg["model"]["adapter"]["rank_init"], cfg["model"]["num_classes"])
        self.aqm = AQM(cfg["model"]["aqm"]["latent_dim"], cfg["model"]["aqm"]["codebook_size"])
        self.controller = KnapsackController(cfg["memory"]["M_max_bytes"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.util_adapter: float = 0.0  # initial utility estimates
        self.util_buffer: float = 0.0

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        feat = self.backbone(x)
        return self.adapter(feat)

    # ------------------------------------------------------------------
    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: D401
        logits = self(x)
        return self.loss_fn(logits, y)

    # ------------------------------------------------------------------
    def estimate_marginal_utilities(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Very rough influence-function proxy: current accuracy divided by bytes"""
        self.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = self(x).argmax(1)
                correct += (pred == y).sum().item()
                n += y.size(0)
        acc = correct / max(1, n)
        util_a = acc / (self.controller.bytes_adapter + 1)
        util_b = acc / (self.controller.bytes_buffer + 1)
        return util_a, util_b

    # ------------------------------------------------------------------
    def after_epoch(self, val_loader):
        self.util_adapter, self.util_buffer = self.estimate_marginal_utilities(val_loader)

    # ------------------------------------------------------------------
    def allocate_memory(self) -> None:  # noqa: D401
        action, ba, bb = self.controller.decide(self.util_adapter, self.util_buffer)
        print(f"[ALLOC] action={action}  bytes_adapter={ba}  bytes_buffer={bb}")

################################################################################
#                         ───  TRAINING  LOOPS ───                             #
################################################################################

def train_one_task(
    model: JEMBModel,
    loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    epochs: int,
) -> None:
    """Mini-batch SGD for a single task."""

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = model.training_step(x, y)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item() * y.size(0)
        ep_loss /= len(loader.dataset)
        print(f"  epoch {ep+1}/{epochs}  loss={ep_loss:.4f}")
        model.after_epoch(val_loader)
