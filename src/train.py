"""
train.py – model definitions, algorithms and training loop
Structured refactor of the original single-file experiment script.
All heavy logic that is required for model construction and optimisation
is collected here so that other modules can simply import it.
"""
from __future__ import annotations
import time, random, hashlib, json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
#                         ─── Helper utilities ───
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Make every stochastic source deterministic on the current process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode()).hexdigest()[:8]


class timing:
    """Context-manager for wall-clock measurement."""

    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        print(f"[TIMER] {self.msg}: {dt:.2f}s", flush=True)


# -----------------------------------------------------------------------------
#                         ───  Back-bone factory  ───
# -----------------------------------------------------------------------------

import timm
from torchvision.models import resnet50, ResNet50_Weights


class Backbone:
    """Plug-and-play image backbone factory."""

    @staticmethod
    def build(name: str, num_classes: int) -> nn.Module:
        if name.lower() == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if name.lower() == "vit_b16":
            return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
        raise KeyError(f"Backbone {name} not supported")


# -----------------------------------------------------------------------------
#                         ───  Learning algorithms  ───
# -----------------------------------------------------------------------------

class AlgorithmBase:
    """Abstract interface every training algorithm must follow."""

    def __init__(self, model: nn.Module, optim: torch.optim.Optimizer, cfg: dict, device: str):
        self.model = model
        self.optim = optim
        self.cfg = cfg
        self.device = device

    def update(self, batch):  # pragma: no cover
        raise NotImplementedError


class ERM(AlgorithmBase):
    """Empirical-risk minimisation baseline."""

    def update(self, batch):
        # A WILDS dataset sample is (x, y, metadata).  Handle both 2- or 3-tuple cases.
        x, y = batch[0], batch[1]
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}


class SimpleIRM(AlgorithmBase):
    """Light-weight IRM implementation that works with any DataLoader tuple.

    It follows the toy IRM objective: empirical risk + λ‖∇_w R(Φ·w)‖² where Φ
    are logits and w is a learnable scalar.  This keeps the dependency list
    minimal and avoids the heavy wilds.algorithms training loop, while still
    demonstrating IRM behaviour for the demo experiment.
    """

    def __init__(self, model, optim, cfg, device, irm_lambda: float = 1.0):
        super().__init__(model, optim, cfg, device)
        self.irm_lambda = irm_lambda
        # dummy scaling parameter w as in Arjovsky et al.
        self.w = torch.tensor(1.0, requires_grad=True, device=device)
        self.w_opt = torch.optim.SGD([self.w], lr=1e-3)

    def update(self, batch):
        x, y = batch[0], batch[1]
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x) * self.w
        loss = F.cross_entropy(logits, y)
        grad_w = torch.autograd.grad(loss, [self.w], create_graph=True)[0]
        penalty = torch.square(grad_w)
        total_loss = loss + self.irm_lambda * penalty
        # update network parameters
        self.optim.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optim.step()
        # update the dummy scalar separately
        self.w_opt.step()
        self.w_opt.zero_grad()
        return {"loss": loss.item(), "irm_penalty": penalty.item()}


# Optional additional methods (IRM / GroupDRO) rely on WILDS.  We expose a
# graceful fallback so that the refactored project can be executed even if the
# user does not have the full WILDS stack compiled.


def get_algorithm(name: str, model: nn.Module, optim: torch.optim.Optimizer, cfg: dict, device: str):
    name = name.lower()
    if name == "erm":
        return ERM(model, optim, cfg, device)
    if name == "irm":
        # Use the light-weight internal IRM implementation to avoid signature
        # mismatch with wilds.algorithms.  This keeps the demo self-contained.
        return SimpleIRM(model, optim, cfg, device, irm_lambda=1.0)
    try:
        from wilds.algorithms import GroupDRO  # heavy import guarded
        if name == "groupdro":
            return GroupDRO(model, optim)
    except Exception as e:
        raise RuntimeError(f"Algorithm '{name}' requires wilds>=2.0 – {e}")
    raise KeyError(name)


# -----------------------------------------------------------------------------
#                         ───           Trainer           ───
# -----------------------------------------------------------------------------

from .evaluate import evaluate_accuracy  # local import – avoids circularity
from .preprocess import get_dataloaders


class Trainer:
    """Generic training loop.  One Trainer == one random seed run."""

    def __init__(self, cfg: dict, dataset: str, method: str, backbone: str, seed: int):
        self.cfg = cfg
        self.dataset = dataset
        self.method = method
        self.backbone_name = backbone
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(seed)

        # 1) data ----------------------------------------------------------------
        loaders, num_classes = get_dataloaders(
            dataset,
            batch_size=cfg["backbones"][backbone]["batch"],
            num_workers=cfg["hardware"]["num_workers"],
        )
        self.loaders = loaders

        # 2) model ----------------------------------------------------------------
        self.model: nn.Module = Backbone.build(backbone, num_classes).to(self.device)

        # 3) optimiser ------------------------------------------------------------
        opt_cfg = cfg["backbones"][backbone]["optim"]
        if opt_cfg["type"].lower() == "sgd":
            self.optim = torch.optim.SGD(
                self.model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg["momentum"], weight_decay=opt_cfg["weight_decay"]
            )
        else:  # AdamW default
            self.optim = torch.optim.AdamW(
                self.model.parameters(), lr=opt_cfg["lr"], betas=opt_cfg["betas"], weight_decay=opt_cfg["weight_decay"]
            )

        # 4) algorithm wrapper ----------------------------------------------------
        self.alg = get_algorithm(method, self.model, self.optim, cfg, self.device)

        # 5) AMP scaler -----------------------------------------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["hardware"].get("amp", True))

        # misc --------------------------------------------------------------------
        self.best_val = 0.0
        self.run_id = f"{dataset}_{backbone}_{method}_seed{seed}_{int(time.time())}"
        Path("models").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    #                               public API
    # -------------------------------------------------------------------------

    def fit(self) -> Dict[str, float]:
        """Train until early-stopping; return metrics collected on test set."""
        start = time.perf_counter()
        max_epochs = self.cfg["training"]["epochs"]
        patience = self.cfg["training"].get("patience", 5)
        no_improve = 0

        for epoch in range(max_epochs):
            self.model.train()
            for batch in self.loaders["train"]:
                # AMP context – gradient scaling is handled inside algorithm.update (if needed)
                with torch.cuda.amp.autocast(enabled=self.cfg["hardware"].get("amp", True)):
                    _ = self.alg.update(batch)

            # ---------------- evaluation ----------------
            val_acc = evaluate_accuracy(self.model, self.loaders["val"], self.device)
            print(f"{self.run_id}  epoch={epoch}  val-acc={val_acc:.4f}")
            if val_acc > self.best_val:
                self.best_val = val_acc
                no_improve = 0
                torch.save(self.model.state_dict(), Path("models") / f"{self.run_id}.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    break  # early stop

        elapsed_h = (time.perf_counter() - start) / 3600.0
        test_acc = evaluate_accuracy(self.model, self.loaders["test"], self.device)

        metrics = {"AccID": test_acc, "TrainHrs": elapsed_h, "Seed": self.seed}
        self._dump_metrics(metrics)
        return metrics

    # ---------------------------------------------------------------------
    #                         internal helpers
    # ---------------------------------------------------------------------

    def _dump_metrics(self, metrics: Dict[str, float]):
        Path("outputs/runs").mkdir(parents=True, exist_ok=True)
        out = {"run_id": self.run_id, "dataset": self.dataset, "method": self.method, **metrics}
        with open(Path("outputs/runs") / f"{self.run_id}.json", "w") as f:
            json.dump(out, f, indent=2)
