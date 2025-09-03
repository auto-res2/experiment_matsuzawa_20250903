"""
train.py – model architectures and training loop
"""
from types import SimpleNamespace
from typing import Tuple, Any
from pathlib import Path
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm

# -----------------------------------------------------------------------------
# Helper – backbone wrapper
# -----------------------------------------------------------------------------
class ClassifierBackbone(nn.Module):
    """Creates a timm backbone with the requested number of classes."""

    def __init__(self, cfg: Any, num_classes: int = 2):
        super().__init__()
        bname: str = cfg.backbone
        pretrained: bool = bool(getattr(cfg, "pretrained", True))
        if bname == "resnet50":
            self.net = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
        elif bname.startswith("vit"):
            drop_path = float(getattr(cfg, "patch_dropout", 0.0))
            self.net = timm.create_model(bname, pretrained=pretrained, num_classes=num_classes,
                                         drop_path_rate=drop_path)
        else:
            raise ValueError(f"Unknown backbone {bname}")

    def forward(self, x, return_features: bool = False):
        feats = self.net.forward_features(x)
        logits = self.net.head(feats)
        return (logits, feats) if return_features else logits


# -----------------------------------------------------------------------------
# Diffusion in-painting helper (loaded lazily – heavy!)
# -----------------------------------------------------------------------------
class DiffusionInpaintWrapper:
    """Memory-efficient Stable-Diffusion in-painting wrapper."""

    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from diffusers import StableDiffusionInpaintPipeline  # heavyweight import
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
            # xFormers for VRAM savings (silently ignored if unavailable)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(f"Stable-Diffusion checkpoint or diffusers lib missing: {exc}")

    @torch.no_grad()
    def inpaint(self, img: torch.Tensor, mask: torch.Tensor, prompt: str,
                num_images: int, guidance: float, steps: int) -> torch.Tensor:
        # diffusers requires PIL input
        from torchvision.transforms.functional import to_pil_image
        from torchvision import transforms as T
        pil_img = to_pil_image(img.cpu())
        pil_mask = to_pil_image(mask.float().cpu())
        gen = self.pipe(prompt=prompt, image=pil_img, mask_image=pil_mask,
                        num_inference_steps=int(steps), guidance_scale=float(guidance),
                        num_images_per_prompt=int(num_images))
        cf = gen.images[0]  # take first generated
        cf = T.ToTensor()(cf)
        return cf


# -----------------------------------------------------------------------------
# AutoCF-Diff classifier (with causal-consistency loss) & plain ERM baseline
# -----------------------------------------------------------------------------
class AutoCFDiffClassifier(nn.Module):

    def __init__(self, cfg: SimpleNamespace, num_classes: int = 2):
        super().__init__()
        self.cfg = cfg
        self.backbone = ClassifierBackbone(cfg.model, num_classes)
        if cfg.autocf.enabled:
            self.diffuser = DiffusionInpaintWrapper(cfg.autocf.diffusion_ckpt)

    def forward(self, x, y=None):
        logits, feats = self.backbone(x, return_features=True)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, feats, loss
        return logits

    # ---------------------------------------------------------------------
    # Causal-consistency auxiliary objective (on-the-fly counterfactuals)
    # ---------------------------------------------------------------------
    def causal_consistency_loss(self, x: torch.Tensor, y: torch.Tensor):
        B = x.size(0)
        device = x.device
        total_l2, total_ce_cf = 0.0, 0.0
        for i in range(B):
            xi, yi = x[i:i + 1], y[i:i + 1]
            # Placeholder square mask – replace with discovered masks in research.
            mask = torch.zeros((1, 1, 256, 256), device=device)
            cf_img = self.diffuser.inpaint(F.interpolate(xi, 256), mask, prompt="",
                                           num_images=self.cfg.autocf.k_diverse,
                                           guidance=self.cfg.autocf.guidance_scale,
                                           steps=self.cfg.autocf.ddim_steps)
            cf_img = F.interpolate(cf_img.unsqueeze(0).to(device), 224)
            with autocast(enabled=self.cfg.train.amp):
                _, feat_o, ce_o = self.forward(xi, yi)
                _, feat_cf, ce_cf = self.forward(cf_img, yi)
            l2 = F.mse_loss(feat_o, feat_cf)
            total_l2 += l2
            total_ce_cf += ce_cf
        return total_l2 / B, total_ce_cf / B


class ERMClassifier(nn.Module):
    """Standard empirical-risk minimisation baseline."""

    def __init__(self, cfg: SimpleNamespace, num_classes: int = 2):
        super().__init__()
        self.backbone = ClassifierBackbone(cfg.model, num_classes)

    def forward(self, x, y=None):
        logits = self.backbone(x)
        if y is not None:
            return logits, F.cross_entropy(logits, y)
        return logits


# -----------------------------------------------------------------------------
# Trainer (uses evaluation utilities from src.evaluate)
# -----------------------------------------------------------------------------
class Trainer:
    """Lightweight single-GPU trainer."""

    def __init__(self, cfg: SimpleNamespace, model: nn.Module,
                 loaders: Tuple[DataLoader, DataLoader, DataLoader]):
        from .evaluate import evaluate_model  # local import to avoid circularity
        self.evaluate_model = evaluate_model

        self.cfg = cfg
        self.model = model.cuda()
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
        self.scaler = GradScaler(enabled=cfg.train.amp)
        self.best_val_wga = 0.0
        self.best_state = None

    def _one_epoch(self, epoch: int):
        accumulate = int(self.cfg.train.accum_steps)
        self.model.train()
        for step, (x, y, _) in enumerate(self.train_loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast(enabled=self.cfg.train.amp):
                if self.cfg.autocf.enabled:
                    _, _, ce = self.model(x, y)
                    cc_l2, ce_cf = self.model.causal_consistency_loss(x, y)
                    loss = ce + ce_cf + self.cfg.autocf.lambda_cc * cc_l2
                else:
                    _, loss = self.model(x, y)
            self.scaler.scale(loss / accumulate).backward()
            if (step + 1) % accumulate == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            if self.cfg.train.fast_dev_run:
                break

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self):
        epochs = 1 if self.cfg.train.fast_dev_run else int(self.cfg.train.epochs)
        for ep in range(epochs):
            self._one_epoch(ep)
            val_stats = self.evaluate_model(self.model, self.val_loader)
            print({"epoch": ep, **val_stats})
            if val_stats["wga"] > self.best_val_wga:
                self.best_val_wga = val_stats["wga"]
                self.best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        # load best → final test
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.evaluate_model(self.model, self.test_loader)
