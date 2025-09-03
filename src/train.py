"""src/train.py
Module that defines all model‐related classes and the training loop that
is task-agnostic (i.e. does not know anything about a particular dataset
stream).  Everything here is import-able from other modules without
triggering any heavy side-effects such as downloading data or writing to
disk.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# -----------------------------------------------------------------------------
#  Model definitions
# -----------------------------------------------------------------------------

class Backbone(nn.Module):
    """ResNet-18 backbone with a replaceable classifier head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = resnet18(weights=None)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W) → (B,num_cls)
        feats = self.feature_extractor(x)
        return self.classifier(feats)


# -------------  VQ-GAN wrapper ------------------------------------------------
from vqgan_jax.modeling_flax_vqgan import VQModel  # pip install vqgan-jax


class VQGANWrapper(nn.Module):
    """Lightweight PyTorch wrapper around a pre-trained (Flax) VQ-GAN model."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.vq = VQModel.from_pretrained(ckpt_path)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.LongTensor:  # (B,3,H,W) → (B,256)
        _z, idx = self.vq.encode(x)
        return idx.view(x.size(0), -1)

    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:  # (B,256) → (B,3,H,W)
        z_grid = tokens.view(tokens.size(0), 16, 16)
        return self.vq.decode_code(z_grid)


# -------------  Tiny latent-space diffusion model ----------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        half = self.dim // 2
        emb_scale = math.log(10000.0) / (half - 1)
        device = x.device
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TinyLatentDiffusion(nn.Module):
    """6-layer transformer (~1.9 M params) operating in token space."""

    def __init__(self, code_dim: int = 8, depth: int = 6, heads: int = 4):
        super().__init__()
        self.code_dim = code_dim
        self.pos_emb = SinusoidalPosEmb(code_dim)
        self.tok_emb = nn.Embedding(512, code_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=code_dim, nhead=heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.to_logits = nn.Linear(code_dim, 512)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:  # (B,256) → (B,256,512)
        seq_len = tokens.size(1)
        pos = torch.arange(seq_len, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        x = self.transformer(x)
        return self.to_logits(x)


# -------------  Token Buffer (memory) ----------------------------------------

class TokenBuffer:
    """Fixed-capacity token grid storage obeying a byte budget."""

    def __init__(self, max_bytes: int = 512_000, grid_bytes: int = 512):
        self.max_bytes = max_bytes
        self.grid_bytes = grid_bytes
        self.storage: Dict[int, List[np.ndarray]] = {}
        self.n_bytes = 0

    # ---------------------------------------------------------------------
    def _evict_one(self):
        cls = np.random.choice(list(self.storage.keys()))
        self.storage[cls].pop(0)
        if not self.storage[cls]:
            del self.storage[cls]
        self.n_bytes -= self.grid_bytes

    # ---------------------------------------------------------------------
    def add(self, cls: int, grids: np.ndarray):
        """Add token grids for class *cls* (grids shape = (K,256))."""

        grids = grids.astype(np.uint16)
        for g in grids:
            if self.n_bytes + self.grid_bytes > self.max_bytes:
                self._evict_one()
            self.storage.setdefault(cls, []).append(g)
            self.n_bytes += self.grid_bytes

    # ---------------------------------------------------------------------
    def sample(self, n: int):
        if self.n_bytes == 0:
            return None, None
        choices, labels = [], []
        for _ in range(n):
            cls = np.random.choice(list(self.storage.keys()))
            g = np.random.choice(self.storage[cls])
            choices.append(g)
            labels.append(cls)
        return (
            torch.LongTensor(np.stack(choices)),
            torch.LongTensor(labels),
        )

    # ---------------------------------------------------------------------
    def evolve(self):
        """Placeholder for Wasserstein-Gradient-Flow evolution (omitted)."""
        pass


# -----------------------------------------------------------------------------
#  Continual Learner – task loop, replay etc.
# -----------------------------------------------------------------------------


class ContinualLearner:
    def __init__(self, cfg_exp, cfg_shared):
        self.cfg = cfg_exp
        self.shared = cfg_shared
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- models -------------------------------------
        self.backbone = Backbone(cfg_exp["num_classes"]).to(self.device)
        self.vqgan = VQGANWrapper(cfg_shared["vq_ckpt"]).to(self.device)
        self.diffuser = (
            TinyLatentDiffusion().to(self.device) if cfg_exp["use_diffusion"] else None
        )

        # ---------------- optimisation -------------------------------
        params = list(self.backbone.parameters()) + list(self.vqgan.parameters())
        if self.diffuser is not None:
            params += list(self.diffuser.parameters())
        self.opt = torch.optim.AdamW(params, **cfg_shared["optim"])
        self.scaler = GradScaler()

        # ---------------- memory -------------------------------------
        self.buffer = TokenBuffer(
            max_bytes=int(cfg_exp["memory_mb"] * 1024 * 1024)
        )

    # -----------------------------------------------------------------
    def train_task(self, task_id: int, loader_real: DataLoader, loader_val: DataLoader):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.shared["epochs"]
        )
        for _epoch in range(self.shared["epochs"]):
            for imgs, y in loader_real:
                imgs, y = imgs.to(self.device), y.to(self.device)

                # ----- latent replay --------------------------------
                if self.buffer.n_bytes > 0:
                    toks_rep, y_rep = self.buffer.sample(len(imgs))
                    toks_rep, y_rep = toks_rep.to(self.device), y_rep.to(self.device)

                    if self.diffuser is not None:
                        logits = self.diffuser(toks_rep)
                        toks_rep = torch.argmax(logits, dim=-1)

                    imgs_rep = self.vqgan.decode(toks_rep)
                    imgs = torch.cat([imgs, imgs_rep])
                    y = torch.cat([y, y_rep])

                # ----- forward / backward ---------------------------
                with autocast():
                    out = self.backbone(imgs)
                    loss = F.cross_entropy(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            scheduler.step()

        # ------------- store tokens from the finished task -----------
        all_imgs = torch.stack(
            [loader_real.dataset[i][0] for i in range(len(loader_real.dataset))]
        ).to(self.device)
        toks = self.vqgan.encode(all_imgs).cpu().numpy()
        labels = np.array(
            [loader_real.dataset[i][1] for i in range(len(loader_real.dataset))]
        )
        for c in np.unique(labels):
            idx = np.where(labels == c)[0][: self.shared["K"]]
            self.buffer.add(int(c), toks[idx])
            if self.cfg["use_wgf"]:
                self.buffer.evolve()

    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.backbone.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.backbone(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        self.backbone.train()
        return correct / total
