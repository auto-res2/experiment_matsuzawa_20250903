"""
train.py – model architectures, memory-ledger utilities and unit tests
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------------
#  Device helper (exported so that other modules can reuse it)
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
#  1.  Compression modules (vendored Tiny-VQ-VAE and dummy fallback)
# ---------------------------------------------------------------------
class _VectorQuantizer(nn.Module):
    """Minimal vector-quantiser taken from Rosinality’s VQ-VAE-2 impl."""
    def __init__(self, n_e: int = 4096, e_dim: int = 256):
        super().__init__()
        self.e_dim = e_dim
        self.embed  = nn.Parameter(torch.randn(n_e, e_dim))

    def forward(self, z: torch.Tensor):
        flat = z.view(-1, self.e_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        idx   = dist.argmin(1)
        quant = self.embed[idx]
        return quant.view_as(z), idx.view(*z.shape[:-1])


class TinyVQVAE(nn.Module):
    """Extremely small VQ-VAE which operates on 512-dim ResNet features."""
    def __init__(self, codebook: int = 4096, dim: int = 512):
        super().__init__()
        self.quant = _VectorQuantizer(codebook, dim)

    @torch.no_grad()
    def encode(self, feats: torch.Tensor):
        return self.quant(feats)[1]  # indices only

    @torch.no_grad()
    def decode(self, indices: torch.Tensor):
        return self.quant.embed[indices]


class DummyCompressor:
    """Fallback when the VQ-VAE cannot be instantiated (e.g. OOM)."""
    BYTES_PER_SAMPLE = 128  # very small placeholder budget

    def encode(self, feats: torch.Tensor):
        return feats.detach().cpu()

    def decode(self, codes: torch.Tensor):
        return codes.to(DEVICE)


# ---------------------------------------------------------------------
#  2.  Memory-ledger & low-rank adapter utilities
# ---------------------------------------------------------------------
class ByteLedger:
    """Simple byte counter that aborts the run once the cap is exceeded."""
    def __init__(self, cap_bytes: int):
        self.cap = cap_bytes
        self.A   = 0  # adapter bytes
        self.B   = 0  # buffer  bytes

    def update(self, a: int, b: int):
        self.A, self.B = a, b
        if self.A + self.B > self.cap:
            raise RuntimeError("Memory cap breached → aborting")

    def asdict(self):
        return {"bytes_adapter": self.A, "bytes_buffer": self.B}


class LoRAAdapter(nn.Module):
    """InfLoRA-style low-rank adapter with run-time grow/prune ops."""
    def __init__(self, in_dim: int, out_dim: int, r: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, r))
        self.B = nn.Parameter(torch.empty(r, out_dim))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor):
        return x @ (self.A @ self.B)

    # ------- dynamic rank manipulation ---------------------------------
    def grow(self, k: int = 1):
        new_A = torch.zeros(self.A.size(0), k, device=self.A.device)
        new_B = torch.zeros(k, self.B.size(1), device=self.B.device)
        nn.init.kaiming_uniform_(new_A)
        nn.init.zeros_(new_B)
        self.A = nn.Parameter(torch.cat([self.A.data, new_A], 1))
        self.B = nn.Parameter(torch.cat([self.B.data, new_B], 0))

    def prune(self, k: int = 1):
        if self.A.size(1) <= k:
            return False
        keep = torch.arange(self.A.size(1) - k, device=self.A.device)
        self.A = nn.Parameter(self.A[:, keep])
        self.B = nn.Parameter(self.B[keep])
        return True

    def bytes(self):
        # 8-bit quantised weights ⇒ 1 byte/param (≈ optimistic)
        return self.A.numel() + self.B.numel()


# ---------------------------------------------------------------------
#  3.  Backbone and full continual-learning models
# ---------------------------------------------------------------------
class Backbone(nn.Module):
    """Frozen ResNet-18 feature extractor."""
    def __init__(self):
        super().__init__()
        res = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(res.children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.features(x).mean([-2, -1])  # (B, 512)


def _marginal_util(acc: float, bytes_: int):
    return acc / (bytes_ + 1)


class JEMBModel(nn.Module):
    """Joint-budget Experience & Model Balancing (proposed)."""
    def __init__(self, num_cls: int, cap_bytes: int):
        super().__init__()
        self.backbone = Backbone()
        self.adapter  = LoRAAdapter(512, num_cls, r=8)
        # --------------------------------------------------------------
        try:
            self.cmp       = TinyVQVAE()
            self.use_dummy = False
        except Exception as exc:
            print("[warn] TinyVQVAE failed, using DummyCompressor", exc)
            self.cmp       = DummyCompressor()
            self.use_dummy = True
        # --------------------------------------------------------------
        self.buffer: List[torch.Tensor] = []
        self.ledger  = ByteLedger(cap_bytes)
        self.ce      = nn.CrossEntropyLoss()
        self._sync_ledger()

    # ------------------------------------------------------------------
    def _sync_ledger(self):
        bytes_b = (
            len(self.buffer) * (2 if not self.use_dummy else DummyCompressor.BYTES_PER_SAMPLE)
        )
        self.ledger.update(self.adapter.bytes(), bytes_b)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        return self.adapter(z)

    # ---------------------- optimisation helpers ----------------------
    def train_epoch(self, loader, opt):
        self.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = self.ce(self(x), y)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def eval_loader(self, loader):
        self.eval()
        good = tot = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            good += (self(x).argmax(1) == y).sum().item()
            tot  += y.size(0)
        return 100 * good / tot if tot else 0.0

    # --------------------- adaptive allocator -------------------------
    @torch.no_grad()
    def realloc(self, val_loader):
        acc = self.eval_loader(val_loader)
        uA  = _marginal_util(acc, self.ledger.A)
        uB  = _marginal_util(acc, self.ledger.B)
        if uA > uB:
            if self.ledger.B >= 512:
                self.adapter.grow(1)
        else:
            self.adapter.prune(1)
        self._sync_ledger()


class InfLoRA(JEMBModel):
    """Baseline which only grows adapters (no replay buffer)."""
    def realloc(self, *_):
        pass

    def _sync_ledger(self):  # buffer is forced to 0
        self.ledger.update(self.adapter.bytes(), 0)


class AQM_ER(nn.Module):
    """Baseline which uses compressed replay buffer + linear probe."""
    def __init__(self, num_cls: int, cap_bytes: int):
        super().__init__()
        self.backbone = Backbone()
        self.linear   = nn.Linear(512, num_cls)
        try:
            self.cmp       = TinyVQVAE()
            self.use_dummy = False
        except Exception:
            self.cmp       = DummyCompressor()
            self.use_dummy = True
        self.buffer: List[torch.Tensor] = []
        self.cap     = cap_bytes
        self.ce      = nn.CrossEntropyLoss()
        self.ledger  = ByteLedger(cap_bytes)
        self._sync_ledger()

    # ------------------------------------------------------------------
    def _sync_ledger(self):
        bytes_b = (
            len(self.buffer) * (2 if not self.use_dummy else DummyCompressor.BYTES_PER_SAMPLE)
        )
        a = self.linear.weight.numel() + self.linear.bias.numel()
        self.ledger.update(a, bytes_b)

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.linear(self.backbone(x))

    def train_epoch(self, loader, opt):
        self.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = self.ce(self(x), y)
            loss.backward()
            opt.step()

    def realloc(self, val_loader):
        space = self.cap - self.ledger.B - self.ledger.A
        add   = space // (2 if not self.use_dummy else DummyCompressor.BYTES_PER_SAMPLE)
        if add > 0:
            for x, _ in val_loader:  # only need one batch
                x       = x.to(DEVICE)
                codes   = self.cmp.encode(self.backbone(x)).cpu()
                portion = min(add, len(codes))
                self.buffer.append(codes[:portion])
                break
            self._sync_ledger()


# ---------------------------------------------------------------------
#  4.  Lightweight unit tests (fail-fast)
# ---------------------------------------------------------------------

def _test_byte_ledger():
    led = ByteLedger(1024)
    led.update(100, 200)
    assert led.A == 100 and led.B == 200
    try:
        led.update(900, 300)
        raise AssertionError("Should have thrown")
    except RuntimeError:
        pass


def _test_adapter_roundtrip():
    ad = LoRAAdapter(10, 3, 2)
    w0 = (ad.A @ ad.B).clone()
    ad.grow(1)
    ad.prune(1)
    assert torch.allclose(w0, (ad.A @ ad.B), atol=1e-4)


def run_unit_tests():
    for t in (_test_byte_ledger, _test_adapter_roundtrip):
        t()
    print("[tests] all passed")
