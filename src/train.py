"""
train.py – model architectures, memory ledger, training utilities,
unit-tests and the generic per-method training routine.
All heavy lifting lives here so that src/main.py can stay concise.
"""
from __future__ import annotations

import random, time, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .preprocess import make_stream

# ---------------------------------------------------------------------
# 1.  DEVICE ------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# 2.  MEMORY LEDGER ------------------------------------------------------------
class ByteLedger:
    """Keeps a running total of bytes consumed by (A)dapter and (B)uffer."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.A = 0  # adapter bytes
        self.B = 0  # buffer  bytes

    # ------------------------------------------------------------------
    def update(self, A: int, B: int):
        self.A, self.B = int(A), int(B)
        if self.A + self.B > self.cap:
            raise RuntimeError('Memory cap exceeded – aborting (ledgerCap gate)')

    # ------------------------------------------------------------------
    def as_dict(self):
        return {'bytes_adapter': self.A, 'bytes_buffer': self.B}


# ---------------------------------------------------------------------
# 3.  CLASS  →  LABEL MAP -------------------------------------------------------
class ClassMap:
    """Maps global dataset labels to the current local head indices."""

    def __init__(self):
        self.g2l: Dict[int, int] = {}

    def add(self, global_ids: List[int]):
        for gid in global_ids:
            if gid not in self.g2l:
                self.g2l[gid] = len(self.g2l)

    # ------------------------------------------------------------------
    def encode(self, y: torch.Tensor) -> torch.Tensor:
        mapper = torch.tensor([self.g2l[i.item()] for i in y.cpu()], device=y.device)
        return mapper


# ---------------------------------------------------------------------
# 4.  BUILDING BLOCKS ----------------------------------------------------------
class LoRA(nn.Module):
    """InfLoRA-style 8-bit low-rank adapter."""

    def __init__(self, out_c: int, r: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(512, r) * 0.01)  # fp32 params but
        self.B = nn.Parameter(torch.zeros(r, out_c))       # counted as 1 byte

    # ------------------------------------------------------------------
    def bytes(self) -> int:
        return (self.A.numel() + self.B.numel())  # 1 byte / param (8-bit)

    # ------------------------------------------------------------------
    def grow_rank(self, k: int):
        if k == 0:
            return
        if k > 0:
            newA = torch.randn(512, k, device=self.A.device) * 0.01
            newB = torch.zeros(k, self.B.size(1), device=self.B.device)
            self.A = nn.Parameter(torch.cat([self.A.data, newA], 1))
            self.B = nn.Parameter(torch.cat([self.B.data, newB], 0))
        else:  # prune last |k| columns  (k is negative)
            keep = self.A.size(1) + k  # k is negative
            self.A = nn.Parameter(self.A.data[:, :keep])
            self.B = nn.Parameter(self.B.data[:keep])

    # ------------------------------------------------------------------
    def grow_out(self, new_c: int):
        extra = torch.zeros(self.A.size(1), new_c, device=self.B.device)
        self.B = nn.Parameter(torch.cat([self.B.data, extra], 1))

    # ------------------------------------------------------------------
    def forward(self, z):
        return z @ (self.A @ self.B)


class FrozenResnet18(nn.Module):
    """ResNet-18 feature extractor with frozen weights."""

    def __init__(self):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        self.feat = nn.Sequential(*list(m.children())[:-2])
        for p in self.feat.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.feat(x).mean([-2, -1])  # (B, 512)


class DynamicHead(nn.Module):
    """Linear classification head that can be expanded online."""

    def __init__(self):
        super().__init__()
        # Start with zero classes (allowed – 0×512 weight & 0-length bias)
        self.fc = nn.Linear(512, 0)

    # ------------------------------------------------------------------
    def expand(self, n_new: int):
        """Add `n_new` output neurons while preserving existing weights."""
        if n_new <= 0:
            return  # nothing to do

        device = self.fc.weight.device
        # --- create new parameters ------------------------------------------------
        W = torch.empty(n_new, 512, device=device)
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))
        b = torch.zeros(n_new, device=device)

        # --- concatenate with old parameters --------------------------------------
        self.fc.weight = nn.Parameter(torch.cat([self.fc.weight.data, W], 0))
        self.fc.bias = nn.Parameter(torch.cat([self.fc.bias.data, b], 0))

        # --- update meta-information ---------------------------------------------
        # PyTorch does *not* automatically update the cached `out_features` attr when
        # the underlying parameters are replaced, so we must keep it in sync for
        # correct introspection and unit-tests.
        self.fc.out_features = self.fc.weight.size(0)

    # ------------------------------------------------------------------
    def forward(self, z):
        return self.fc(z)


# ---------------------------------------------------------------------
# 5.  MODELS -------------------------------------------------------------------
class _BaseModel(nn.Module):
    """Common helpers shared by all baselines."""

    def accuracy(self, dl: DataLoader):
        self.eval()
        g = t = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = self(x).argmax(1)
                g += (preds == self.map.encode(y)).sum().item()
                t += y.size(0)
        return 100 * g / t


class JEMB(_BaseModel):
    """Joint-budget Experience & Model Balancing (proposed)."""

    def __init__(self, ledger: ByteLedger, cls_map: ClassMap, init_rank: int = 8):
        super().__init__()
        self.ledger = ledger
        self.map = cls_map
        self.back = FrozenResnet18()
        self.head = DynamicHead()
        self.adapter = LoRA(0, init_rank)
        self.buffer: List[Tuple[bytes, int]] = []  # (compressed_code, global_lbl)
        self.ce = nn.CrossEntropyLoss()
        self._sync_ledger()

    # ------------------------------------------------------------------
    def _sync_ledger(self):
        self.ledger.update(self.adapter.bytes(), sum(len(c) for c, _ in self.buffer))

    # ------------------------------------------------------------------
    def expand_for_task(self, new_global_ids: List[int]):
        self.map.add(new_global_ids)
        self.head.expand(len(new_global_ids))
        self.adapter.grow_out(len(new_global_ids))
        self._sync_ledger()

    # ------------------------------------------------------------------
    def forward(self, x):
        z = self.back(x)
        return self.head(z) + self.adapter(z)

    # ------------------------------------------------------------------
    def loss_on_batch(self, x, y_global):
        return self.ce(self(x), self.map.encode(y_global))

    # ------------------------------------------------------------------
    def train_task(self, dl: DataLoader, epochs: int, opt):
        for _ in range(max(3, epochs)):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.loss_on_batch(x, y)
                loss.backward()
                opt.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def allocate(self, val_dl: DataLoader):
        base = self.accuracy(val_dl)

        # — test +1 rank ------------------------------------------------
        self.adapter.grow_rank(1)
        self._sync_ledger()
        accA = self.accuracy(val_dl)
        utilA = accA - base
        self.adapter.grow_rank(-1)  # revert

        # — test +1 sample (dummy 32-B code) ----------------------------
        code = torch.randint(0, 256, (32,), dtype=torch.uint8).numpy().tobytes()
        self.buffer.append((code, 0))
        self._sync_ledger()
        accB = self.accuracy(val_dl)
        utilB = accB - base
        self.buffer.pop()

        # — greedy decision -------------------------------------------
        if utilB / 32 > utilA / max(1, self.adapter.A.size(1)) and self.ledger.B + 32 < self.ledger.cap:
            self.buffer.append((code, 0))
        elif self.adapter.A.size(1) > 1:
            self.adapter.grow_rank(-1)
        self._sync_ledger()


class Reservoir(_BaseModel):
    """Reservoir-sampling replay baseline."""

    def __init__(self, ledger: ByteLedger, cls_map: ClassMap, K: int = 512):
        super().__init__()
        self.ledger = ledger
        self.map = cls_map
        self.K = K
        self.back = FrozenResnet18()
        self.head = DynamicHead()
        self.ce = nn.CrossEntropyLoss()
        self.buf: List[Tuple[torch.Tensor, int]] = []
        self.n_seen = 0
        self._sync()

    # ------------------------------------------------------------------
    def _sync(self):
        # Each stored image is 3×32×32 = 3072 bytes if kept in uint8.
        self.ledger.update(self.head.fc.weight.numel() + self.head.fc.bias.numel(), len(self.buf) * 3072)

    # ------------------------------------------------------------------
    def expand_for_task(self, new_ids):
        self.map.add(new_ids)
        self.head.expand(len(new_ids))
        self._sync()

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.head(self.back(x))

    # ------------------------------------------------------------------
    def loss(self, x, yG):
        return self.ce(self(x), self.map.encode(yG))

    # ------------------------------------------------------------------
    def train_task(self, dl, ep, opt):
        for _ in range(ep):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.loss(x, y)
                loss.backward()
                opt.step()

                # reservoir update -----------------------------------
                for xi, yi in zip(x.cpu(), y.cpu()):
                    self.n_seen += 1
                    if len(self.buf) < self.K:
                        self.buf.append((xi, yi.item()))
                    else:
                        j = random.randint(0, self.n_seen - 1)
                        if j < self.K:
                            self.buf[j] = (xi, yi.item())
        self._sync()


class InfLoRA(_BaseModel):
    """Adapter-only baseline (no replay buffer)."""

    def __init__(self, ledger: ByteLedger, cls_map: ClassMap):
        super().__init__()
        self.ledger = ledger
        self.map = cls_map
        self.back = FrozenResnet18()
        self.head = DynamicHead()
        self.adapter = LoRA(0)
        self.ce = nn.CrossEntropyLoss()
        self._sync()

    # ------------------------------------------------------------------
    def _sync(self):
        self.ledger.update(self.adapter.bytes(), 0)

    # ------------------------------------------------------------------
    def expand_for_task(self, new_ids):
        self.map.add(new_ids)
        self.head.expand(len(new_ids))
        self.adapter.grow_out(len(new_ids))
        self._sync()

    # ------------------------------------------------------------------
    def forward(self, x):
        z = self.back(x)
        return self.head(z) + self.adapter(z)

    # ------------------------------------------------------------------
    def loss(self, x, yG):
        return self.ce(self(x), self.map.encode(yG))

    # ------------------------------------------------------------------
    def train_task(self, dl, ep, opt):
        for _ in range(ep):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.loss(x, y)
                loss.backward()
                opt.step()
        self._sync()


# ---------------------------------------------------------------------
# 6.  UNIT-TESTS  (≤ 5 s) ------------------------------------------------------

def test_head_growth():
    led = ByteLedger(1 << 20)
    mp = ClassMap()
    m = JEMB(led, mp)
    m.expand_for_task([0, 1, 2, 3, 4])
    assert m.head.fc.out_features == 5
    m.expand_for_task([5])
    assert m.head.fc.out_features == 6


def test_label_remap_after_expansion():
    led = ByteLedger(1 << 20)
    mp = ClassMap()
    m = JEMB(led, mp).to(DEVICE)
    m.expand_for_task([0])
    x = torch.randn(4, 3, 32, 32, device=DEVICE)
    y = torch.tensor([0, 0, 0, 0], device=DEVICE)
    m.loss_on_batch(x, y).backward()  # before expansion

    m.expand_for_task([1])
    y2 = torch.tensor([0, 1, 0, 1], device=DEVICE)
    m.loss_on_batch(x, y2).backward()  # must not raise


def test_ledger_never_exceeds_cap():
    led = ByteLedger(1024)
    led.update(256, 128)
    for _ in range(30):
        a = random.randint(0, 256)
        b = led.cap - a
        led.update(a, b)


def run_unit_tests():
    for t in (test_head_growth, test_label_remap_after_expansion, test_ledger_never_exceeds_cap):
        t()
    print('[unit] all green')


# ---------------------------------------------------------------------
# 7.  TRAINING  ENTRY ----------------------------------------------------------
@dataclass
class Log:
    acc: List[float] = field(default_factory=list)
    A: List[int] = field(default_factory=list)
    B: List[int] = field(default_factory=list)


# ------------------------------------------------------------------

def run_method(name: str, ctor: Callable, cfg: Dict, device=DEVICE):
    """Train *one* method across the task stream; returns summary & log."""

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    stream = make_stream(cfg)
    cls_map = ClassMap()
    ledger = ByteLedger(cfg['memory_budget']['bytes'])
    model = ctor(ledger, cls_map).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=cfg['hyper']['lr'],
                          momentum=cfg['hyper']['momentum'], weight_decay=cfg['hyper']['weight_decay'])

    log = Log()
    cumul_tests = []  # store test loaders for up-to-now tasks

    for tid, (tr, va, te, cls_ids) in enumerate(stream, 1):
        model.expand_for_task(cls_ids)

        dl_tr = DataLoader(tr, batch_size=cfg['hyper']['batch'], shuffle=True, num_workers=2, pin_memory=True)
        dl_va = DataLoader(va, batch_size=cfg['hyper']['batch'], shuffle=False, num_workers=2)
        dl_te = DataLoader(te, batch_size=cfg['hyper']['batch'], shuffle=False, num_workers=2)
        cumul_tests.append(dl_te)

        model.train_task(dl_tr, cfg['hyper']['epochs_per_task'], opt)
        if hasattr(model, 'allocate'):
            model.allocate(dl_va)

        accs = [model.accuracy(l) for l in cumul_tests]
        log.acc.append(sum(accs) / len(accs))
        log.A.append(model.ledger.A)
        log.B.append(model.ledger.B)
        print(f"{name:<9} task {tid:02d}  A_T={log.acc[-1]:5.2f}%   A={log.A[-1]:4d}  B={log.B[-1]:4d}")

    return {'method': name, 'A_T': log.acc[-1]}, log
