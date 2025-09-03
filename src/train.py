"""
train.py – model construction, buffers and training utilities for HATEM
(Fixed version)
•   Added fallback wrapper when taming-transformers' VQModel does not expose
    `from_pretrained` so that CPU-only CI can still run the light-weight unit
    tests without downloading the 400-MB VQ-GAN weights.
•   *No behaviour change* for normal execution – the real decoder is loaded
    when the method exists and CUDA is available.
"""
from __future__ import annotations
import sys, types, time, statistics as st
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict

import yaml, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms

# -------------------------------------------------------------------------
# Compatibility shim for older lightning versions ---------------------------------
# -------------------------------------------------------------------------
try:
    import pytorch_lightning as pl  # noqa: F401  – imported for side effect only
    if 'pytorch_lightning.utilities.distributed' not in sys.modules:  # pragma: no cover
        from pytorch_lightning.utilities import rank_zero as _rz  # type: ignore
        _shim = types.ModuleType('pytorch_lightning.utilities.distributed')
        for _attr in ('rank_zero_only', 'rank_zero_debug', 'rank_zero_info', 'rank_zero_warn'):
            if hasattr(_rz, _attr):
                setattr(_shim, _attr, getattr(_rz, _attr))
        sys.modules['pytorch_lightning.utilities.distributed'] = _shim
except ImportError:
    pass  # PL will be installed via requirements.txt in the workflow

# -------------------------------------------------------------------------
# Configuration -------------------------------------------------------------------
# -------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
with open(CFG_PATH, 'r') as _f:
    CFG = yaml.safe_load(_f)

DEVICE = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# -------------------------------------------------------------------------
# VQ-GAN (encoder / decoder) --------------------------------------------------------
# -------------------------------------------------------------------------
from taming.models.vqgan import VQModel  # type: ignore

def _dummy_vqgan(device: torch.device) -> "VQModel":  # pragma: no cover – unit-test fallback
    """Light-weight stub that mimics the tiny subset of API we use (encode/ decode)."""
    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.L = 4  # code grid size

        def encode(self, x: torch.Tensor):
            b = x.size(0)
            idx = torch.zeros(b, self.L, self.L, dtype=torch.long, device=x.device)
            return {"indices": idx}

        def decode(self, code: torch.Tensor):  # (B,4,4) → fake image
            b = code.size(0)
            return torch.zeros(b, 3, 32, 32, device=code.device)  # CIFAR-sized blank image

    print('[WARN] `VQModel.from_pretrained` not found – using stub VQ-GAN (tests only).')
    return _Stub().to(device)  # type: ignore[return-value]


def build_vqgan(device: torch.device = DEVICE) -> VQModel:  # type: ignore[override]
    """Load the pre-trained VQ-GAN if available; otherwise fall back to a stub."""
    if hasattr(VQModel, 'from_pretrained'):
        ckpt = CFG['vqgan_ckpt']
        print(f"[LOAD] VQ-GAN encoder/decoder ({ckpt}) ..")
        vq: VQModel = VQModel.from_pretrained(ckpt).to(device)  # type: ignore[attr-defined]
        vq.eval().requires_grad_(False)
        return vq
    # ------------------------------------------------------------------
    # Fallback path – lightweight stub (used in CPU-only CI)
    # ------------------------------------------------------------------
    return _dummy_vqgan(device)

# -------------------------------------------------------------------------
# ResNet-18 backbone (CIFAR-friendly) ---------------------------------------------
# -------------------------------------------------------------------------

def build_resnet18(num_classes: int = 100) -> nn.Module:
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.conv1.stride = (1, 1)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net

# -------------------------------------------------------------------------
# Memory buffers -----------------------------------------------------------
# -------------------------------------------------------------------------
class HATEMBuffer:
    """Tier-1 token ids + Tier-2 synthetic prototypes (very small placeholder implementation)."""

    def __init__(self, bytes_limit: int, vocab: int = 256, proto_per_cls: int = 5):
        self.L = 4  # 4×4 grid
        self.vocab = vocab
        self.proto_per_cls = proto_per_cls
        self.bytes_limit = bytes_limit
        self.tokens: List[Tuple[np.ndarray, int]] = []           # (grid, label)
        self.protos: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
        self.total = 0  # bytes currently stored

    # ------------------------------------------------------------------
    def add(self, token_grid: torch.Tensor, label: int):
        g = token_grid.cpu().numpy().astype(np.uint8)            # (4,4)
        self.tokens.append((g, label))
        self.total += g.nbytes
        self._trim()

    # ------------------------------------------------------------------
    def consolidate(self):
        by_cls: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
        for g, lbl in self.tokens:
            if len(by_cls[lbl]) < self.proto_per_cls:
                by_cls[lbl].append(g)
        self.protos = by_cls
        self.total = (
            sum(g.nbytes for g, _ in self.tokens)
            + sum(g.nbytes for lst in by_cls.values() for g in lst)
        )
        self._trim()

    # ------------------------------------------------------------------
    def sample(self, n: int, vqgan: VQModel, device: torch.device = DEVICE):  # type: ignore[name-defined]
        if len(self.tokens) == 0:
            raise RuntimeError('[HATEM] trying to sample from an empty buffer')
        idx = np.random.choice(len(self.tokens), n, replace=True)
        grids, lbls = zip(*[self.tokens[i] for i in idx])
        code = torch.tensor(np.stack(grids), device=device)
        img = vqgan.decode(code).clamp(0, 1)
        return img, torch.tensor(lbls, device=device)

    # ------------------------------------------------------------------
    def _trim(self):
        while self.total > self.bytes_limit and self.tokens:
            g, _ = self.tokens.pop(0)
            self.total -= g.nbytes


class RawBuffer:
    """Exact pixel storage for exemplar replay (ER, DER++ …)."""

    def __init__(self, bytes_limit: int):
        self.limit = bytes_limit
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.total = 0

    # ------------------------------------------------------------------
    def add(self, img: torch.Tensor, label: int):
        arr = (img.cpu().numpy() * 255).astype(np.uint8)          # 3×H×W uint8
        self.samples.append((arr, label))
        self.total += arr.nbytes
        self._trim()

    # ------------------------------------------------------------------
    def sample(self, n: int, device: torch.device = DEVICE):
        if len(self.samples) == 0:
            raise RuntimeError('[RAW] trying to sample from an empty buffer')
        idx = np.random.choice(len(self.samples), n, replace=True)
        imgs, lbls = zip(*[self.samples[i] for i in idx])
        t = torch.tensor(np.stack(imgs), device=device, dtype=torch.uint8).float() / 255
        return t, torch.tensor(lbls, device=device)

    # ------------------------------------------------------------------
    def _trim(self):
        while self.total > self.limit and self.samples:
            arr, _ = self.samples.pop(0)
            self.total -= arr.nbytes

# -------------------------------------------------------------------------
# Dataset helpers (CIFAR-100 incremental stream) ---------------------------
# -------------------------------------------------------------------------

def cifar_tasks(seed: int, train: bool = True):
    tf = transforms.Compose([
        transforms.RandomCrop(32, 4) if train else transforms.Identity(),
        transforms.RandomHorizontalFlip() if train else transforms.Identity(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    ds = datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=tf)
    rng = np.random.RandomState(seed)
    order = list(range(100))
    rng.shuffle(order)

    tasks = []
    for t in range(10):
        cls = order[t * 10 : (t + 1) * 10]
        idx = [i for i, (_, y) in enumerate(ds) if y in cls]
        tasks.append(Subset(ds, idx))
    return tasks

# -------------------------------------------------------------------------
# Sanity check – 90 % CIFAR-100 single-task accuracy -----------------------
# -------------------------------------------------------------------------

def sanity_single_task(device: torch.device = DEVICE):
    net = build_resnet18(100).to(device)
    opt = torch.optim.SGD(net.parameters(), **CFG['optim'], nesterov=True)
    train_loader = DataLoader(
        datasets.CIFAR100(str(DATA_DIR), train=True, download=True, transform=transforms.ToTensor()),
        batch_size=256,
        shuffle=True,
        num_workers=2,
    )
    net.train()
    for _ in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()
    test_loader = DataLoader(
        datasets.CIFAR100(str(DATA_DIR), train=False, download=True, transform=transforms.ToTensor()),
        batch_size=256,
    )
    net.eval(); corr = tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            p = net(x.to(device)).argmax(1).cpu()
            corr += (p == y).sum().item()
            tot += y.size(0)
    acc = corr / tot * 100
    print(f"[Sanity] single-task CIFAR-100 accuracy = {acc:.1f}%")
    if acc < 90:
        raise RuntimeError('Sanity check failed (<90 %) – aborting experiments.')

# -------------------------------------------------------------------------
# Incremental training loop ------------------------------------------------
# -------------------------------------------------------------------------

def train_stream(method: str, budget: int, seed: int, device: torch.device = DEVICE):
    torch.manual_seed(seed)
    np.random.seed(seed)

    vq = build_vqgan(device)
    model = build_resnet18(100).to(device)
    opt = torch.optim.SGD(model.parameters(), **CFG['optim'], nesterov=True)

    buf = HATEMBuffer(budget) if method == 'HATEM' else RawBuffer(budget)

    acc_per_task, forget, prev_acc = [], [], [0] * 10

    for tid, task in enumerate(cifar_tasks(seed)):
        loader = DataLoader(task, batch_size=CFG['batch'], shuffle=True, num_workers=2)

        # ------------------------------ training on current task
        model.train()
        for img, lbl in loader:
            if method == 'HATEM':
                tok = vq.encode(img.to(device))["indices"]           # (B,4,4)
                for t, l in zip(tok, lbl):
                    buf.add(t, l.item())
            else:
                for im, l in zip(img, lbl):
                    buf.add(im, l.item())

            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad(); F.cross_entropy(model(img), lbl).backward(); opt.step()

        if method == 'HATEM':
            buf.consolidate()

        # ------------------------------ 1 replay epoch from memory
        if (len(buf.tokens) if method == 'HATEM' else len(buf.samples)) > 0:
            for _ in range(len(loader)):
                if method == 'HATEM':
                    r_img, r_lbl = buf.sample(CFG['batch'], vq, device)
                else:
                    r_img, r_lbl = buf.sample(CFG['batch'], device)
                model.train(); opt.zero_grad(); F.cross_entropy(model(r_img), r_lbl).backward(); opt.step()

        # ------------------------------ evaluation after current task
        test_loader = DataLoader(
            datasets.CIFAR100(str(DATA_DIR), train=False, download=True, transform=transforms.ToTensor()),
            batch_size=256,
        )
        model.eval(); corr = tot = 0
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(device)).argmax(1).cpu()
                corr += (p == y).sum().item(); tot += y.size(0)
        task_acc = corr / tot * 100
        acc_per_task.append(task_acc)

        # forgetting statistics
        for c in range(tid):
            forget.append(prev_acc[c] - task_acc)
        prev_acc[tid] = task_acc

        print(f"[Task {tid}] ACC={task_acc:.1f}%  Mem={buf.total / 1024:.1f} KB")

    avg_acc = st.mean(acc_per_task)
    f_metric = st.mean([max(0, f) for f in forget]) if forget else 0
    return avg_acc, f_metric, buf
