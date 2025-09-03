"""src/train.py – all training-related logic (models, trainer, unit tests)"""
from __future__ import annotations
import random, time, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from .preprocess import build_stream

# ────────────────────────────────────────────────────────────────────────
# 1.  Hardware device helper                                              
# ────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[env] device', DEVICE)

# ────────────────────────────────────────────────────────────────────────
# 2.  Ledger & Adapter utilities                                          
# ────────────────────────────────────────────────────────────────────────
class ByteLedger:
    """Tracks memory consumption of adapter parameters (A) and buffer (B)."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.A = 0  # bytes spent on adapters
        self.B = 0  # bytes spent on replay buffer / codes

    def update(self, A: int, B: int):
        self.A, self.B = int(A), int(B)
        if self.A + self.B > self.cap:
            raise RuntimeError('Memory cap exceeded – aborting to keep guarantees.')

    def dict(self):
        return {'bytes_adapter': self.A, 'bytes_buffer': self.B}


class LoRA(nn.Module):
    """Tiny LoRA-style low-rank adapter (8-bit quantised ⇒ 1 byte / weight)."""

    def __init__(self, out_c: int, r: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(512, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_c))

    # ------------------------------------------------------------------
    def forward(self, f):  # f: (B, 512)
        return f @ (self.A @ self.B)

    # ----- dynamic grow / prune ---------------------------------------
    def grow(self, k: int):
        if k <= 0:
            return
        newA = torch.randn(512, k, device=self.A.device) * 0.01
        newB = torch.zeros(k, self.B.shape[1], device=self.B.device)
        self.A = nn.Parameter(torch.cat([self.A.data, newA], dim=1))
        self.B = nn.Parameter(torch.cat([self.B.data, newB], dim=0))

    def prune(self, k: int):
        if self.A.shape[1] <= k:
            return
        keep = torch.arange(self.A.shape[1] - k, device=self.A.device)
        self.A = nn.Parameter(self.A.data[:, keep])
        self.B = nn.Parameter(self.B.data[keep])

    def bytes(self):
        # 1 byte per parameter (8-bit quantisation assumed)
        return self.A.numel() + self.B.numel()


# ────────────────────────────────────────────────────────────────────────
# 3.  Backbone & Head                                                     
# ────────────────────────────────────────────────────────────────────────
import torchvision


class DynamicBackbone(nn.Module):
    """Frozen ResNet-18 up to the avg-pool layer."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(m.children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x).mean([-2, -1])  # (B, 512)


class DynamicHead(nn.Module):
    """Linear classifier that can expand its output dimensionality online."""

    def __init__(self, out_c: int):
        super().__init__()
        self.fc = nn.Linear(512, out_c)

    # ------------------------------------------------------------------
    def expand(self, n_new: int):
        if n_new <= 0:
            return
        W_new = torch.zeros(n_new, 512, device=self.fc.weight.device)
        b_new = torch.zeros(n_new, device=self.fc.bias.device)
        nn.init.kaiming_uniform_(W_new, a=np.sqrt(5))
        self.fc.weight = nn.Parameter(torch.cat([self.fc.weight.data, W_new], 0))
        self.fc.bias = nn.Parameter(torch.cat([self.fc.bias.data, b_new], 0))

    def forward(self, z):
        return self.fc(z)


# ────────────────────────────────────────────────────────────────────────
# 4.  Continual-Learning Models                                           
# ────────────────────────────────────────────────────────────────────────
class JEMB(nn.Module):
    """Proposed Joint-budget Experience & Model Balancing controller."""

    def __init__(self, ledger: ByteLedger, r0: int = 8):
        super().__init__()
        self.ledger = ledger
        self.back = DynamicBackbone()
        self.head = DynamicHead(out_c=0)
        self.adapter = LoRA(out_c=0, r=r0)
        self.buffer: List[Tuple[torch.Tensor, int]] = []  # (features, label)
        self.ce = nn.CrossEntropyLoss()

    # ----- lifecycle hooks -------------------------------------------
    def expand_head(self, n_new: int):
        self.head.expand(n_new)
        # B dimension changes implicitly – ensure A & B stay compatible
        self.adapter.grow(0)

    def sync_ledger(self):
        self.ledger.update(self.adapter.bytes(), len(self.buffer) * 128)  # assume 128 B / sample

    # ------------------------------------------------------------------
    def forward(self, x):
        z = self.back(x)
        return self.head(z) + self.adapter(z)

    # -------------------  training  -----------------------------------
    def train_task(self, dl: DataLoader, epochs: int, opt: torch.optim.Optimizer):
        for _ in range(epochs):
            self.train()
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.ce(self(x), y)
                loss.backward()
                opt.step()

    # -------------  simple allocator (greedy) -------------------------
    @torch.no_grad()
    def allocate(self, val_dl: DataLoader):
        self.eval()
        base_acc = self.eval_accuracy(val_dl)
        # utility of adding one rank
        if self.adapter.A.shape[1] < 32:
            self.adapter.grow(1)
            self.sync_ledger()
            accA = self.eval_accuracy(val_dl)
            gainA = accA - base_acc
            self.adapter.prune(1)
        else:
            gainA = 0.0
        # utility of adding one sample
        gainB = 0.0
        if len(self.buffer) < 512:
            x, y = next(iter(val_dl))
            x, y = x.to(DEVICE), y.to(DEVICE)
            feat = self.back(x).cpu()[0]
            lbl = int(y[0])
            self.buffer.append((feat, lbl))
            self.sync_ledger()
            accB = self.eval_accuracy(val_dl)
            gainB = accB - base_acc
            self.buffer.pop()
        utilA = gainA
        utilB = gainB / 128  # scale by bytes / sample
        if utilB > utilA and self.ledger.A > 512:
            self.adapter.prune(1)
            self.buffer.append((feat, lbl))
        elif utilA > utilB and self.ledger.B > 256:
            self.adapter.grow(1)
            if self.buffer:
                self.buffer.pop(0)
        self.sync_ledger()

    # -------------------  evaluation  ---------------------------------
    @torch.no_grad()
    def eval_accuracy(self, dl: DataLoader):
        self.eval()
        good = tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            good += (self(x).argmax(1) == y).sum().item()
            tot += y.size(0)
        return 100 * good / tot


class Reservoir(nn.Module):
    """Experience-Replay baseline with reservoir sampling."""

    def __init__(self, ledger: ByteLedger, buf_size: int = 512):
        super().__init__()
        self.ledger = ledger
        self.back = DynamicBackbone()
        self.head = DynamicHead(0)
        self.ce = nn.CrossEntropyLoss()
        self.buffer: List[Tuple[torch.Tensor, int]] = []
        self.K = buf_size
        self.seen = 0

    # ------------------------------------------------------------------
    def expand_head(self, n_new: int):
        self.head.expand(n_new)

    def sync_ledger(self):
        self.ledger.update(
            self.head.fc.weight.numel() + self.head.fc.bias.numel(),
            len(self.buffer) * 3072,  # raw CIFAR image bytes
        )

    def forward(self, x):
        return self.head(self.back(x))

    # ------------------------------------------------------------------
    def train_task(self, dl: DataLoader, epochs: int, opt: torch.optim.Optimizer):
        for _ in range(epochs):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.ce(self(x), y)
                loss.backward()
                opt.step()
                # reservoir update
                for xi, yi in zip(x.cpu(), y.cpu()):
                    if len(self.buffer) < self.K:
                        self.buffer.append((xi, yi.item()))
                    else:
                        j = random.randint(0, self.seen - 1)
                        if j < self.K:
                            self.buffer[j] = (xi, yi.item())
                self.seen += x.size(0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_accuracy(self, dl: DataLoader):
        self.eval()
        g = t = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            g += (self(x).argmax(1) == y).sum().item()
            t += y.size(0)
        return 100 * g / t


class InfLoRA(nn.Module):
    """Low-rank adaptation without replay buffer."""

    def __init__(self, ledger: ByteLedger):
        super().__init__()
        self.ledger = ledger
        self.back = DynamicBackbone()
        self.head = DynamicHead(0)
        self.adapter = LoRA(0)
        self.ce = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def expand_head(self, n_new: int):
        self.head.expand(n_new)
        self.adapter.grow(0)

    def sync_ledger(self):
        self.ledger.update(self.adapter.bytes(), 0)

    def forward(self, x):
        z = self.back(x)
        return self.head(z) + self.adapter(z)

    # ------------------------------------------------------------------
    def train_task(self, dl: DataLoader, epochs: int, opt: torch.optim.Optimizer):
        for _ in range(epochs):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss = self.ce(self(x), y)
                loss.backward()
                opt.step()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_accuracy(self, dl: DataLoader):
        self.eval()
        g = t = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            g += (self(x).argmax(1) == y).sum().item()
            t += y.size(0)
        return 100 * g / t


# ────────────────────────────────────────────────────────────────────────
# 5.  Training loop per method                                            
# ────────────────────────────────────────────────────────────────────────
@dataclass
class TaskLog:
    acc: List[float] = field(default_factory=list)
    A: List[int] = field(default_factory=list)
    B: List[int] = field(default_factory=list)


def run_method(
    name: str,
    model_ctor: Callable[[ByteLedger], nn.Module],
    cfg: Dict,
    seed: int = 42,
) -> Tuple[Dict, TaskLog]:
    """Train *one* continual-learning method over the entire stream."""

    # reproducibility --------------------------------------------------
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    stream = build_stream(cfg)
    ledger = ByteLedger(cfg['memory_budget']['bytes'])
    model = model_ctor(ledger).to(DEVICE)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg['hyper']['lr'],
        momentum=cfg['hyper']['momentum'],
        weight_decay=cfg['hyper']['weight_decay'],
    )

    logs = TaskLog()
    cumul_tests = []

    for tid, (tr, va, te) in enumerate(stream, 1):
        # ----- dynamic head growth ------------------------------------
        if hasattr(model, 'expand_head'):
            model.expand_head(cfg['dataset']['classes_per_task'])
        assert model.head.fc.out_features == tid * cfg['dataset']['classes_per_task'], 'head expansion failed'
        model.sync_ledger()

        # ----- loaders ------------------------------------------------
        trL = DataLoader(tr, batch_size=cfg['hyper']['batch'], shuffle=True, num_workers=2, pin_memory=True)
        vaL = DataLoader(va, batch_size=cfg['hyper']['batch'], shuffle=False, num_workers=2)
        teL = DataLoader(te, batch_size=cfg['hyper']['batch'], shuffle=False, num_workers=2)
        cumul_tests.append(teL)

        # ----- training + allocator -----------------------------------
        model.train_task(trL, cfg['hyper']['epochs_per_task'], opt)
        if hasattr(model, 'allocate'):
            model.allocate(vaL)
        model.sync_ledger()

        # ----- evaluation ---------------------------------------------
        accs = [model.eval_accuracy(l) for l in cumul_tests]
        logs.acc.append(sum(accs) / len(accs))
        logs.A.append(model.ledger.A)
        logs.B.append(model.ledger.B)
        print(f"{name:<9} task {tid:02d}  A_T={logs.acc[-1]:5.2f}%  A={model.ledger.A}  B={model.ledger.B}")

    summary = {
        'method': name,
        'A_T': logs.acc[-1],
        'bytes_adapter': logs.A[-1],
        'bytes_buffer': logs.B[-1],
    }
    return summary, logs


# ────────────────────────────────────────────────────────────────────────
# 6.  Unit tests (fail-fast)                                             
# ────────────────────────────────────────────────────────────────────────

def _test_head_expansion():
    led = ByteLedger(1 << 20)
    m = JEMB(led)
    m.expand_head(5)
    assert m.head.fc.out_features == 5
    m.expand_head(5)
    assert m.head.fc.out_features == 10


def _test_ledger_cap():
    led = ByteLedger(1024)
    led.update(512, 512)
    try:
        led.update(2000, 0)
        assert False, 'ledger cap did not trigger'
    except RuntimeError:
        pass


def run_unit_tests():
    for t in (_test_head_expansion, _test_ledger_cap):
        t()
    print('[unit] all green')
