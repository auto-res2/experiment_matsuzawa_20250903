# src/train.py
# -*- coding: utf-8 -*-
"""
Training-related utilities for the DCiG project
• seed_everything – reproducibility helper
• Trainer          – generic supervised trainer (AMP-aware)
• ViTSaliency       – light-weight Grad-CAM for ViT
• DCIGEngine        – counterfactual image generator (masked LoRA fine-tune + inpainting)
All heavy libraries are imported locally so that other modules remain light-weight.
"""
from __future__ import annotations
import random, math, time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# 3rd-party libs used only here
import timm                               # vision backbones
from timm.models.vision_transformer import Attention

# diffusers / LoRA stuff (optional but required for full DCiG run)
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torchvision import transforms as tvf

# -----------------------------------------------------------------------------
#  Path & device helpers
# -----------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
#  Re-producibility helper
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch for full determinism."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#  Trainer – AMP aware, generic supervised classification
# -----------------------------------------------------------------------------

class Trainer:
    """Thin wrapper around a supervised training loop (classification)."""

    def __init__(self, model: nn.Module, optimiser, scheduler=None, *, precision: str = "fp16"):
        self.model = model.to(DEVICE)
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.precision = precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    def _step(self, batch, train: bool = True):
        x, y, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if train:
            self.optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(self.precision == "fp16")):
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        if train:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
        return loss.item(), logits.detach().argmax(1).cpu(), y.cpu()

    def fit(self, train_loader, val_loader, *, epochs: int, exp_name: str):
        history = {"train_loss": [], "val_acc": []}
        for ep in range(1, epochs + 1):
            # ---------- training ----------
            self.model.train()
            losses = []
            for batch in tqdm.tqdm(train_loader, desc=f"[{exp_name}] epoch {ep}/{epochs}"):
                l, *_ = self._step(batch, train=True)
                losses.append(l)
            mean_loss = float(sum(losses) / len(losses))
            history["train_loss"].append(mean_loss)

            # ---------- validation ----------
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    _, preds, targets = self._step(batch, train=False)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            val_acc = 100.0 * correct / max(total, 1)
            history["val_acc"].append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()
            print(f"Ep{ep:03d}: train-loss={mean_loss:.4f}, val-acc={val_acc:.2f}%")
        return history

# -----------------------------------------------------------------------------
#  ViT Grad-CAM style saliency (simplified attention rollout)
# -----------------------------------------------------------------------------

class ViTSaliency:
    """Generate approximate Grad-CAM heat-maps for Vision-Transformer models."""

    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self._grads = None
        # hook last attention block
        block = [m for m in self.model.modules() if isinstance(m, Attention)][-1]
        block.attn_drop.register_backward_hook(self._hook)

    def _hook(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]

    @torch.no_grad()
    def generate(self, x: torch.Tensor, class_idx: int):
        x = x.unsqueeze(0).to(DEVICE)
        logits = self.model(x)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
        # rollout
        attn = self._grads.mean(1).squeeze(0)  # (tokens,tokens)
        sal = attn.mean(0)[1:]                 # skip CLS
        sz = int(math.sqrt(sal.size(0)))
        sal = sal.view(1, 1, sz, sz)
        sal = torch.nn.functional.interpolate(sal, size=(224, 224), mode="bilinear", align_corners=False)
        sal = sal.clamp(min=0)
        sal /= sal.max().clamp(min=1e-6)
        return sal.squeeze().detach().cpu()     # (224,224)

# -----------------------------------------------------------------------------
#  DCiG counterfactual image generator (high-level, simplified)
# -----------------------------------------------------------------------------

class DCIGEngine:
    """Minimal implementation of DCiG – masked LoRA fine-tune + local editing."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._load_pipeline()

    # ---------------------------------------------------------------------
    #  internal helpers
    # ---------------------------------------------------------------------

    def _load_pipeline(self):
        print("[DCIG] Loading Stable-Diffusion-XL in-paint pipeline (fp16)…")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(DEVICE)
        self.pipe.enable_xformers_memory_efficient_attention()

        # attach LoRA adapters (text encoder only for brevity)
        lora_cfg = LoraConfig(r=self.cfg["lora_rank"], lora_alpha=self.cfg["lora_rank"] * 2,
                              target_modules=["to_q", "to_k", "to_v", "to_out"])
        self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, lora_cfg)

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def finetune_on_masks(self, imgs: torch.Tensor, masks: torch.Tensor, prompts: List[str]):
        """LoRA fine-tune restricted to spurious regions. Skeleton implementation."""
        optimiser = torch.optim.Adam(self.pipe.text_encoder.parameters(), lr=self.cfg["lora_lr"])
        scheduler = get_cosine_schedule_with_warmup(optimiser, 100, self.cfg["lora_steps"])
        bs = 4  # fits into 16-GB T4 in fp16
        self.pipe.train()
        for step in tqdm.trange(self.cfg["lora_steps"], desc="LoRA-finetune", leave=False):
            idx = torch.randint(0, imgs.size(0), (bs,))
            # TODO: convert tensors to PIL + feed to pipeline
            raise RuntimeError("LoRA fine-tuning not fully implemented; provide image/mask conversion.")
        self.pipe.eval()

    @torch.no_grad()
    def generate_counterfactual(self, img: torch.Tensor, mask: torch.Tensor, *,
                                prompt_keep: str, prompt_change: str):
        if img.dtype != torch.float16:
            img = img.half()
        cf = self.pipe(prompt_change, image=img, mask_image=mask, guidance_scale=7.5).images[0]
        cf = tvf.ToTensor()(cf)
        return cf
