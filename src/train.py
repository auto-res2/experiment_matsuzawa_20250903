"""src/train.py
Model definition and training utilities for Tokenized Compressed Rehearsal (TCR).
All heavy-lifting computations live here – the rest of the project only calls the
public APIs declared below.
"""
from __future__ import annotations

import math, random, warnings, time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange
import timm

__all__ = [
    "GumbelVectorQuantizer", "TokenHead", "TCRModel", "BudgetScheduler",
    "Trainer"
]

################################################################################
# 1.  MODEL COMPONENTS
################################################################################

class GumbelVectorQuantizer(nn.Module):
    """Product-quantisation code-book with straight-through Gumbel-Softmax."""
    def __init__(self, dim: int, num_codebooks: int = 4, codebook_size: int = 256,
                 temp: float = 1.0):
        super().__init__()
        self.dim  = dim
        self.M    = num_codebooks
        self.K    = codebook_size
        self.temp = temp
        # (M, K, d_sub) where d_sub = dim // M
        self.codebooks = nn.Parameter(torch.randn(self.M, self.K, dim // num_codebooks))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # x: (B, dim)
        B, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {D} .")

        x_split = x.view(B, self.M, D // self.M)
        # similarity (B,M,K)
        logits = torch.einsum("bmd,mkd->bmk", x_split, self.codebooks)
        # sample gumbel(0,1)
        g = -torch.empty_like(logits).exponential_().log()
        y = F.softmax((logits + g) / self.temp, dim=-1)
        y_hard = F.one_hot(y.argmax(-1), self.K).type_as(y)  # straight-through trick
        quant = torch.einsum("bmk,mkd->bmd", y_hard, self.codebooks)
        return quant.reshape(B, D), y_hard.argmax(-1)  # (B, dim), (B, M)


class TokenHead(nn.Module):
    """Two-layer LN→Linear head operating on token embeddings."""
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes, bias=False)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, dim)
        return self.fc(z)


class TCRModel(nn.Module):
    """Tokenised Compressed Rehearsal – end-to-end model (backbone frozen)."""
    def __init__(self, backbone_name: str = "vit_base_patch16_224", *,
                 k_tokens: int = 8, num_classes: int = 1000,
                 codebook_temp: float = 0.5, buffer_max_bytes: int = 5*1024*1024):
        super().__init__()
        # 1. Frozen backbone ----------------------------------------------------
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # 2. Light adapter  -----------------------------------------------------
        backbone_dim = getattr(self.backbone, "num_features", 768)
        self.adapter = nn.Sequential(
            nn.Conv1d(backbone_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )  # → (B,128,1)

        # 3. Vector quantiser & classification head -----------------------------
        self.quantizer = GumbelVectorQuantizer(dim=128, num_codebooks=k_tokens,
                                               codebook_size=256, temp=codebook_temp)
        self.head = TokenHead(dim=128, num_classes=num_classes)

        # 4. Replay memory ------------------------------------------------------
        self.register_buffer("replay_tokens", torch.empty(0, k_tokens, dtype=torch.uint8))
        self.register_buffer("replay_labels", torch.empty(0, dtype=torch.long))
        self.buffer_max_bytes = buffer_max_bytes
        self.k_tokens = k_tokens

    # -------------------------------------------------------------------------
    # Forward paths
    # -------------------------------------------------------------------------
    def _encode_img(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) → (B,128)
        feat = self.backbone.forward_features(x)            # (B,seq,feat)
        feat = feat.transpose(1, 2)                         # (B,feat,seq)
        adapted = self.adapter(feat).squeeze(-1)            # (B,128)
        return adapted

    def forward_current(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z   = self._encode_img(x)
        z_q, toks = self.quantizer(z)                       # (B,128), (B,k_tokens)
        logits = self.head(z_q)
        loss = F.cross_entropy(logits, y)
        return loss, toks

    def forward_tokens(self, z_tokens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward using stored discrete tokens (no backbone pass)."""
        if z_tokens.dtype != torch.long:
            z_tokens = z_tokens.long()
        B = z_tokens.size(0)
        # one-hot (B,k_tokens,256)
        toks_onehot = F.one_hot(z_tokens, num_classes=256).float()
        codebooks = self.quantizer.codebooks         # (k_tokens,256,d_sub)
        embeds = torch.einsum("bkd,kdf->bkf", toks_onehot, codebooks)  # (B,k_tokens,d_sub)
        embeds = embeds.reshape(B, -1)               # (B,128)
        logits = self.head(embeds)
        return F.cross_entropy(logits, y)

    # -------------------------------------------------------------------------
    # Replay buffer  (very small & simple reservoir sampling)
    # -------------------------------------------------------------------------
    def maybe_store(self, token_indices: torch.Tensor, labels: torch.Tensor):
        assert token_indices.ndim == 2
        bytes_per_sample = token_indices.size(1)            # uint8 = 1 Byte
        for tok, lbl in zip(token_indices.cpu(), labels.cpu()):
            if (self.replay_tokens.numel() + bytes_per_sample) < self.buffer_max_bytes:
                self.replay_tokens = torch.cat([
                    self.replay_tokens, tok.unsqueeze(0).to(torch.uint8)
                ])
                self.replay_labels = torch.cat([
                    self.replay_labels, lbl.unsqueeze(0)
                ])
            else:
                # reservoir replacement ------------------------------------------------
                j = random.randint(0, len(self.replay_tokens) - 1)
                self.replay_tokens[j] = tok
                self.replay_labels[j] = lbl

    def sample_tokens(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.replay_tokens) == 0:
            raise RuntimeError("Replay buffer is empty – cannot sample.")
        idx = torch.randint(0, len(self.replay_tokens), (batch_size,), device=self.replay_tokens.device)
        return self.replay_tokens[idx].long(), self.replay_labels[idx]

################################################################################
# 2.  SCHEDULER (Compute-budget aware)
################################################################################

class BudgetScheduler:
    """Simple proportional split of steps between current data & replay."""
    def __init__(self, total_steps_per_task: int = 200, current_ratio: float = 0.5):
        assert 0. < current_ratio < 1., "current_ratio must be between 0 and 1"
        self.total = total_steps_per_task
        self.cur_steps = int(total_steps_per_task * current_ratio)
        self.replay_steps_cnt = self.total - self.cur_steps

    def replay_steps(self) -> int:
        return self.replay_steps_cnt

################################################################################
# 3.  TRAINER
################################################################################

class Trainer:
    """Light-weight helper encapsulating the optimisation procedure for one model."""
    def __init__(self, model: TCRModel, *, lr: float = 3e-4, weight_decay: float = 0.02):
        self.model  = model.cuda().train()
        self.opt    = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999),
                                        weight_decay=weight_decay)
        self.scaler = GradScaler()
        self.scheduler = BudgetScheduler()

    # ---------------------------------------------------------------------
    # current data passes --------------------------------------------------
    # ---------------------------------------------------------------------
    def _train_on_loader(self, loader: torch.utils.data.DataLoader, *, cur: bool):
        passes = self.scheduler.cur_steps if cur else self.scheduler.replay_steps()
        if passes == 0: return
        for _ in range(passes):
            for x, y in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with autocast():
                    if cur:
                        loss, toks = self.model.forward_current(x, y)
                    else:
                        loss = self.model.forward_tokens(x, y)  # here x already tokens
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if cur:
                    # only store when training on *current* samples
                    self.model.maybe_store(toks, y)

    # ---------------------------------------------------------------------
    # Public API -----------------------------------------------------------
    # ---------------------------------------------------------------------
    def train_task(self, loader: torch.utils.data.DataLoader):
        # 1) Train on current task samples ---------------------------------
        self._train_on_loader(loader, cur=True)
        # 2) Re-play stored tokens -----------------------------------------
        if len(self.model.replay_tokens) > 0:
            tokens, labels = self.model.sample_tokens(batch_size=64)
            token_loader = [(tokens.cuda(), labels.cuda())]  # synthetic loader
            self._train_on_loader(token_loader, cur=False)

    def eval_loader(self, loader: torch.utils.data.DataLoader) -> float:
        """Return accuracy (%) on *loader*."""
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                z = self.model._encode_img(x)
                z_q, _ = self.model.quantizer(z)
                logits = self.model.head(z_q)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        self.model.train()
        return 100.0 * correct / total
