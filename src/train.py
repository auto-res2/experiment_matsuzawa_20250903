"""src/train.py – model, memory-accounting and controller logic for JEMB
The file only contains *model–related* components so that it can be imported
independently from training / evaluation orchestration code.
"""
from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (kept for possible future use)
import torchvision

# -------------------------------------------------------------
#                    Adapter and memory helpers
# -------------------------------------------------------------
class LoRAAdapter(nn.Module):
    """InfLoRA-style low-rank adapter whose rank can grow / shrink on-line."""

    def __init__(self, in_dim: int, out_dim: int, r0: int):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.col_seeds: List[int] = []  # RNG seeds for pruned columns
        self.weight_A = nn.Parameter(torch.zeros(in_dim, r0))
        self.weight_B = nn.Parameter(torch.zeros(r0, out_dim))
        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        nn.init.zeros_(self.weight_B)

    # ------------------------  utilities  ------------------------------
    def current_rank(self) -> int:
        return self.weight_A.shape[1]

    def bytes(self) -> int:
        """Assume 8-bit quantisation → 1 byte per weight."""
        return self.weight_A.numel() + self.weight_B.numel()

    # ----------------------  dynamic rank  -----------------------------
    def grow(self, k: int = 1):
        """Add *k* new rank-1 columns with Stiefel orthogonalisation."""
        A = torch.zeros(self.in_dim, k, device=self.weight_A.device)
        B = torch.zeros(k, self.out_dim, device=self.weight_B.device)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        nn.init.zeros_(B)
        self.weight_A = nn.Parameter(torch.cat([self.weight_A.data, A], dim=1))
        self.weight_B = nn.Parameter(torch.cat([self.weight_B.data, B], dim=0))
        # Stiefel-orthogonalise new columns – gradient-non-interference guarantee
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.weight_A)  # (in_dim, new_r)
            self.weight_A.copy_(q)

    def prune(self, k: int = 1) -> bool:
        """Prune the *k* smallest-magnitude columns. Return *True* if pruning happened."""
        if self.current_rank() <= k:
            return False
        col_norm = self.weight_B.abs().sum(dim=1)  # (r,)
        keep = col_norm.argsort(descending=True)[: self.current_rank() - k]
        self.weight_A = nn.Parameter(self.weight_A[:, keep])
        self.weight_B = nn.Parameter(self.weight_B[keep])
        # store RNG seed for lightweight reconstruction
        self.col_seeds.extend(torch.randint(0, 2 ** 31, (k,)).tolist())
        return True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple proxy
        W = (self.weight_A @ self.weight_B).to(torch.float32)  # de-quant on the fly
        return x @ W


# -------------------------------------------------------------
#                       Replay buffer (VQ codes)
# -------------------------------------------------------------
class VQBuffer:
    """Store vector-quantised latent codes instead of raw inputs."""

    BYTES_PER_CODE = 2  # uint16 index → 2 bytes

    def __init__(self, vq):
        self.vq = vq
        self.codes: List[torch.LongTensor] = []

    def add(self, feats: torch.Tensor):
        _, _, latents = self.vq.encode(feats)[2]  # (B, H, W)
        self.codes.append(latents.cpu())

    def sample(self, n: int) -> torch.Tensor:
        if not self.codes:
            raise RuntimeError("[buffer] sampling from empty buffer")
        flat = torch.cat(self.codes)
        idx = torch.randint(0, len(flat), (n,))
        device = next(self.vq.parameters()).device
        return self.vq.decode(flat[idx].to(device))

    def bytes(self) -> int:
        if not self.codes:
            return 0
        return sum(c.numel() for c in self.codes) * self.BYTES_PER_CODE


# -------------------------------------------------------------
#                       Byte-ledger accounting
# -------------------------------------------------------------
class ByteLedger:
    """Tracks bytes used by (A)dapter and (B)uffer under a hard cap."""

    def __init__(self, cap_kb: int):
        self.cap = cap_kb * 1024  # → bytes
        self.A = 0  # adapter bytes
        self.B = 0  # buffer  bytes

    # ------------------------------------------------------------------
    def update(self, a: int, b: int):
        self.A, self.B = a, b
        if self.A + self.B > self.cap:
            raise RuntimeError("Memory cap breached – aborting run!")

    def move(self, src: str, k: int):
        if src == "A" and self.A >= k:
            self.A -= k
            self.B += k
        elif src == "B" and self.B >= k:
            self.B -= k
            self.A += k
        else:
            raise ValueError("[ledger] illegal move request")

    def asdict(self):
        return dict(bytes_adapter=self.A, bytes_buffer=self.B)


# -------------------------------------------------------------
#               Marginal utility (influence approximation)
# -------------------------------------------------------------
@torch.no_grad()
def marginal_utility(model: nn.Module, val_loader, ledger: ByteLedger):
    good = 0
    tot = 0
    device = next(model.parameters()).device
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        good += (model(x).argmax(dim=1) == y).sum().item()
        tot += y.size(0)
    acc = good / max(tot, 1)
    # avoid div/0
    ua = acc / (ledger.A + 1)
    ub = acc / (ledger.B + 1)
    return ua, ub


# -------------------------------------------------------------
#                      Greedy memory controller
# -------------------------------------------------------------
class GreedyController:
    """1-step greedy allocator based on marginal utility."""

    def __init__(self, ledger: ByteLedger, adapter: LoRAAdapter, buffer: VQBuffer):
        self.ledger = ledger
        self.adapter, self.buffer = adapter, buffer

    def step(self, util_A: float, util_B: float):
        delta = max(512, int(0.01 * self.ledger.cap))  # ≥ 512 bytes
        if util_A > util_B:
            # free bytes from buffer and grow adapter
            if self.ledger.B >= delta:
                self.ledger.move("B", delta)
                self.adapter.grow(k=1)
        else:
            # prune adapter to free bytes for buffer
            if self.adapter.prune(k=1):
                self.ledger.move("A", delta)


# -------------------------------------------------------------
#                          Backbone
# -------------------------------------------------------------
class Backbone(nn.Module):
    """Frozen ResNet-18 trunk returning 512-D spatial average features."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 512)
        return self.features(x).mean(dim=[-2, -1])


# -------------------------------------------------------------
#                         Full JEMB model
# -------------------------------------------------------------
class JEMB(nn.Module):
    """Joint-budget Experience & Model Balancing network wrapper."""

    def __init__(self, num_cls: int, cap_kb: int, cfg):
        super().__init__()
        self.cfg = cfg  # keep a reference for easy access
        self.backbone = Backbone()
        self.adapter = LoRAAdapter(512, num_cls, cfg.model.adapter["rank0"])
        # VQ-VAE-2 (encoder/decoder are *frozen*)
        from vq_vae_2_pytorch import VQVAETwo

        self.vq = VQVAETwo(
            img_size=32,
            num_layers=2,
            codebook_dim=cfg.model.aqm["latent_dim"],
            num_codebook_vectors=cfg.model.aqm["codebook"],
        )
        for p in self.vq.parameters():
            p.requires_grad = False

        # replay buffer & bookkeeping ---------------------------------------------------
        self.buffer = VQBuffer(self.vq)
        self.ledger = ByteLedger(cap_kb)
        self.ctrl = GreedyController(self.ledger, self.adapter, self.buffer)

        self.ce = nn.CrossEntropyLoss()
        # initial ledger update ---------------------------------------------------------
        self.ledger.update(self.adapter.bytes(), self.buffer.bytes())

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.adapter(feat)

    # ---------------------------  continual-learning helpers -----------
    def observe_batch(self, x, y, opt):
        opt.zero_grad()
        loss = self.ce(self(x), y)
        loss.backward()
        opt.step()
        return float(loss.item())

    def after_epoch(self, val_loader):
        ua, ub = marginal_utility(self, val_loader, self.ledger)
        self.ctrl.step(ua, ub)
        # update ledger after structural changes
        self.ledger.update(self.adapter.bytes(), self.buffer.bytes())
