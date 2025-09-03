"""
train.py – model definitions, algorithms and the generic training loop
The module is self-contained except for utility helpers that live inside this
file to avoid circular imports.  Nothing is written to disk dynamically; every
symbol is imported in the normal, static way.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import (ResNet50_Weights, resnet50)

from .preprocess import get_loaders
from .utils import set_seed, timing

################################################################################
# ─── CONFIGURATION ────────────────────────────────────────────────────────────
################################################################################

import yaml
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CFG_PATH, "r") as _f:
    cfg = yaml.safe_load(_f)

################################################################################
# ─── BACKBONE FACTORY ────────────────────────────────────────────────────────
################################################################################

class Backbone:
    """Factory that returns an ImageNet-pre-trained backbone with correct head"""

    @staticmethod
    def build(name: str, n_cls: int) -> nn.Module:
        name = name.lower()
        if name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(2048, n_cls)
            return model
        if name in {"vit_b16", "vit-b16", "vit_b/16"}:
            return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=n_cls)
        raise KeyError(f"Unknown backbone: {name}")

################################################################################
# ─── OPTIMISER HELPER ────────────────────────────────────────────────────────
################################################################################

def _make_optim(params, backbone_name: str):
    o_cfg = cfg["backbones"][backbone_name]["optim"]
    if o_cfg["type"].lower() == "sgd":
        optim = torch.optim.SGD(params,
                                lr=o_cfg["lr"],
                                momentum=o_cfg.get("momentum", 0.9),
                                weight_decay=o_cfg.get("wd", 0.0))
    elif o_cfg["type"].lower() == "adamw":
        optim = torch.optim.AdamW(params,
                                  lr=o_cfg["lr"],
                                  betas=tuple(o_cfg.get("betas", (0.9, 0.999))),
                                  weight_decay=o_cfg.get("wd", 0.0))
    else:
        raise KeyError(o_cfg["type"])
    return optim

################################################################################
# ─── ALGORITHMS / TRAINING OBJECTIVES ────────────────────────────────────────
################################################################################

class ERM:
    """Empirical Risk Minimisation – default cross-entropy"""

    def __init__(self, model: nn.Module, optimiser, device: torch.device):
        self.m, self.o, self.d = model, optimiser, device

    def update(self, x, y):
        x, y = x.to(self.d), y.to(self.d)
        logits = self.m(x)
        loss = F.cross_entropy(logits, y)
        self.o.zero_grad()
        loss.backward()
        self.o.step()
        return loss.item()

# Optional methods – import lazily so that the project still installs even if
# reviewers do not have every exotic dependency.
try:
    from wilds.algorithms import IRM as _IRM, GroupDRO as _GroupDRO
except ImportError:      # pragma: no cover – missing wilds during unit tests
    _IRM = _GroupDRO = None

class IRM(_IRM):
    pass  # subclass only so that hasattr() checks below succeed even if wilds missing

class GroupDRO(_GroupDRO):
    pass

class DiCA:
    """Stub for DiCA. Fail-fast if user actually tries to run it."""
    def __init__(self, *_, **__):
        raise RuntimeError("DiCA full implementation not included in public refactor – aborting as per fail-fast policy.")

# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_algorithm(method: str, model: nn.Module, optimiser, device):
    method = method.lower()
    if method == "erm":
        return ERM(model, optimiser, device)
    if method == "irm":
        if _IRM is None:
            raise RuntimeError("wilds is required for IRM – package not found")
        return IRM(model, optimiser, irm_lambda=1.0, device=device)
    if method == "groupdro":
        if _GroupDRO is None:
            raise RuntimeError("wilds is required for GroupDRO – package not found")
        return GroupDRO(model, optimiser, device=device)
    if method == "dica":
        return DiCA(model, optimiser, device)
    raise NotImplementedError(method)

################################################################################
# ─── TRAINER ─────────────────────────────────────────────────────────────────
################################################################################

class Trainer:
    """Handles one complete train → val → test cycle."""

    def __init__(self, dataset: str, backbone: str, method: str, seed: int):
        self.ds, self.bk, self.meth, self.seed = dataset, backbone, method, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        # data ----------------------------------------------------------------
        batch_size = cfg["backbones"][backbone]["batch"]
        self.loaders, n_cls, _ = get_loaders(dataset, batch_size, cfg["hardware"]["num_workers"])

        # model ----------------------------------------------------------------
        self.model = Backbone.build(backbone, n_cls).to(self.device)
        self.optim = _make_optim(self.model.parameters(), backbone)
        self.alg   = make_algorithm(method, self.model, self.optim, self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["hardware"]["amp"]) if torch.cuda.is_available() else None
        self.best_val = 0.0

        run_ts = int(time.time())
        self.run_id = f"{dataset}_{backbone}_{method}_seed{seed}_{run_ts}"
        self.ckpt_dir = Path("outputs/checkpoints"); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = Path("outputs/runs");       self.json_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _accuracy(self, loader):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x = x.to(self.device); y = y.to(self.device)
            preds = self.model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return correct / max(total, 1)

    def _dump_metrics(self, metrics: Dict):
        record = {
            "run_id":  self.run_id,
            "dataset": self.ds,
            "backbone": self.bk,
            "method":  self.meth,
            "seed":    self.seed,
            **metrics
        }
        with open(self.json_dir / f"{self.run_id}.json", "w") as f:
            json.dump(record, f, indent=2)

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def fit(self) -> float:
        epochs   = cfg["training"]["epochs"]
        patience = cfg["training"]["patience"]
        stall    = 0

        with timing(self.run_id):
            for ep in range(epochs):
                self.model.train()
                for batch in self.loaders["train"]:
                    # WILDS datasets sometimes return (x, y, metadata).  We only
                    # need (x, y) here.
                    x, y = batch[:2]
                    if self.scaler is None:
                        loss = self.alg.update(x, y)
                    else:
                        with torch.cuda.amp.autocast():
                            loss = self.alg.update(x, y)

                val_acc = self._accuracy(self.loaders["val"])
                print(f"{self.run_id} | ep={ep:02d} | val={val_acc:.3f}")

                if val_acc > self.best_val:
                    self.best_val = val_acc
                    stall = 0
                    torch.save(self.model.state_dict(), self.ckpt_dir / f"{self.run_id}.pt")
                else:
                    stall += 1

                if stall >= patience:
                    break

        test_acc = self._accuracy(self.loaders["test"])
        self._dump_metrics({"AccID": test_acc})
        return test_acc
