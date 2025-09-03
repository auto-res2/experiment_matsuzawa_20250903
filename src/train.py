"""
train.py – all model architectures and training utilities
"""
from __future__ import annotations
import os
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models

# ----------------------------------------------------------------------------
#                Seed helpers
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------
#                BYOL backbone with Low-Rank Amplification
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """Minimal BYOL for ResNet-18 sized backbones (single-GPU friendly)."""

    def __init__(self, base_encoder: nn.Module):
        super().__init__()
        self.online_encoder = base_encoder
        self.target_encoder = models.resnet18()
        self._init_target()
        dim = self.online_encoder.fc.in_features
        self.online_encoder.fc = nn.Identity()
        self.target_encoder.fc = nn.Identity()
        self.online_proj = MLP(dim)
        self.target_proj = MLP(dim)
        self.predictor = MLP(256, 512, 256)

    def _init_target(self):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def _update_target(self, m: float = 0.996):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        o1, o2 = self.online_encoder(x1), self.online_encoder(x2)
        z1, z2 = self.online_proj(o1), self.online_proj(o2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        with torch.no_grad():
            t1 = self.target_proj(self.target_encoder(x1))
            t2 = self.target_proj(self.target_encoder(x2))
        loss = (F.mse_loss(p1, t2) + F.mse_loss(p2, t1)) * 0.5
        return loss


@torch.no_grad()
def amplify_latent_directions(embeddings: torch.Tensor, r: int = 8, amplify_factor: float = 4.0):
    """Low-rank amplification (DeFund style)."""
    mu = embeddings.mean(0, keepdim=True)
    X = embeddings - mu
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    S[:r] = S[:r] * amplify_factor
    return (U * S.unsqueeze(0)) @ Vt + mu

# ----------------------------------------------------------------------------
#                Statistical & optimisation helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def riesz_total_effect(Z: torch.Tensor, y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    I = torch.eye(Z.size(1), device=Z.device)
    w = torch.linalg.solve(Z.T @ Z + lam * I, Z.T @ y)
    return w


def irm_penalty(loss: torch.Tensor, predictor: nn.Module):
    grad = torch.autograd.grad(loss, predictor.parameters(), retain_graph=True, create_graph=True)
    return torch.cat([g.view(-1) for g in grad]).pow(2).mean()


class FourierLoss(nn.Module):
    """High-frequency spectral consistency loss."""

    def __init__(self, cutoff: float = 0.5):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor, x_cf: torch.Tensor):
        xf, xcf = torch.fft.fft2(x, norm="ortho"), torch.fft.fft2(x_cf, norm="ortho")
        mag, mag_cf = xf.abs(), xcf.abs()
        B, C, H, W = mag.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
        center = torch.tensor([H // 2, W // 2], device=x.device)[:, None, None]
        dist = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2).sqrt()
        mask = (dist > self.cutoff * (H / 2)).float()
        diff = (mag * mask - mag_cf * mask).abs()
        return diff.mean()

# ----------------------------------------------------------------------------
#                Counterfactual Diffusion Generator (LoRA)
# ----------------------------------------------------------------------------

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    from peft import get_peft_model, LoraConfig
except ImportError as _e:
    StableDiffusionImg2ImgPipeline, get_peft_model, LoraConfig = None, None, None  # handled later


class CDGPipeline:
    """Lightweight wrapper around Stable-Diffusion Img2Img with LoRA adapters."""

    def __init__(self, cfg: Dict[str, Any], device: str):
        if StableDiffusionImg2ImgPipeline is None:
            raise ImportError("diffusers / peft missing – install to enable CDG.")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg["base_model"], torch_dtype=torch.float16).to(device)
        peft_cfg = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_rank"] * 2,
                              target_modules=["attn2", "attn1"], lora_dropout=0.05,
                              bias="none", task_type="UNET")
        self.pipe.unet = get_peft_model(self.pipe.unet, peft_cfg)
        self.pipe.unet.train()
        self.device = device

    def fine_tune(self, train_dl: DataLoader, epochs: int, lr: float):
        opt = torch.optim.AdamW(self.pipe.unet.parameters(), lr=lr)
        for ep in range(epochs):
            for i, (img, _) in enumerate(train_dl):
                img = img.to(self.device, dtype=torch.float16)
                noise = torch.randn_like(img)
                loss = F.mse_loss(self.pipe.unet(img, 0).sample, noise)
                loss.backward()
                opt.step(); opt.zero_grad()
                if i % 10 == 0:
                    print(f"[CDG] ep{ep} it{i} loss {loss.item():.3f}")
        self.pipe.unet.eval()

    @torch.no_grad()
    def generate(self, img: torch.Tensor, latent_delta: torch.Tensor):
        latents = self.pipe.vae.encode(img.half()).latent_dist.sample() * 0.18215
        latents_cf = latents + latent_delta
        out = self.pipe.vae.decode(latents_cf / 0.18215).sample
        return out
