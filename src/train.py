"""src/train.py
Model architectures and training-time helpers.
Contains:
• ProductQuantiser – discrete latent encoder.
• TokenHead         – classifier that consumes tokens.
• TCR               – Tokenised Compressed Rehearsal model used in all
  experiments (back-bone frozen – only adapter, PQ and head are trained).
"""
from __future__ import annotations

import math
import random
from typing import Tuple, List

import einops  # noqa: F401 – needed by timm backbones that rely on einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

__all__: List[str] = [
    "ProductQuantiser",
    "TokenHead",
    "TCR"
]


class ProductQuantiser(nn.Module):
    """Multi-codebook product quantiser (vector-quantisation with straight
    through Gumbel-Softmax) used to compress backbone features into a handful
    of 8-bit tokens.

    Args
    ----
    dim : int
        Feature dimensionality to be quantised.
    M : int, default=4
        Number of codebooks (sub-vector splits).
    K : int, default=256
        Codewords per codebook (token values 0…K-1).
    tau : float, default=0.5
        Gumbel-Softmax temperature.
    """

    def __init__(self, dim: int, M: int = 4, K: int = 256, tau: float = 0.5):
        super().__init__()
        assert dim % M == 0, "dim must be divisible by M"
        self.M, self.K, self.dsub, self.tau = M, K, dim // M, tau
        # Codebook:  M × K × dsub  (one independent K×dsub table per sub-vector)
        self.codebook = nn.Parameter(torch.randn(M, K, self.dsub))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing the quantised feature and the discrete tokens.

        Returns
        -------
        z_q : torch.Tensor
            Quantised feature, same shape as *x*.
        tokens : torch.Tensor[uint8]
            B × M matrix of token IDs (stored in the replay buffer).
        """
        B, D = x.shape
        x = x.view(B, self.M, self.dsub)                                    # B × M × dsub
        # Similarity between input sub-vectors and codewords --------------
        logits = (x.unsqueeze(2) * self.codebook).sum(-1)                   # B × M × K

        # Straight-through Gumbel-Softmax ---------------------------------
        g = -torch.empty_like(logits).exponential_().log()                  # Gumbel noise
        y = F.softmax((logits + g) / self.tau, dim=-1)                      # soft sample
        y_hard = F.one_hot(y.argmax(-1), self.K).type_as(y)                 # hard sample (ST)

        # Reconstruct quantised feature per codebook then concat ----------
        #   y_hard : B × M × K
        #   codebk : M × K × dsub
        # → einsum gives             B × M × dsub
        z_q = torch.einsum("bmk,mkd->bmd", y_hard, self.codebook).reshape(B, -1)  # B × D
        return z_q, y_hard.argmax(-1).to(torch.uint8)


class TokenHead(nn.Module):
    """2-layer classifier that consumes the quantised tokens."""

    def __init__(self, dim: int, ncls: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ncls, bias=False)
        )

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:  # logits
        return self.fc(z)


class TCR(nn.Module):
    """Full Tokenised-Compressed-Rehearsal model.

    The *backbone* (any image model from timm) is kept frozen to minimise the
    computational load; a light adapter ↓128 dims → product-quantiser produces
    M×K discrete tokens that are stored in a replay buffer.
    """

    def __init__(
        self,
        backbone: str,
        ncls: int,
        k: int = 4,
        tau: float = 0.5,
        residual_bytes: int = 32,
    ) -> None:
        super().__init__()
        # 1) frozen backbone ------------------------------------------------
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        # 2) tiny trainable adapter + PQ -----------------------------------
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.quant = ProductQuantiser(128, M=k, K=256, tau=tau)
        # 3) token-to-logit head ------------------------------------------
        self.head = TokenHead(128, ncls)
        # 4) hierarchical replay buffer (token-level only for brevity) -----
        self.k = k  # tokens per sample
        self.residual_bytes = residual_bytes
        self.max_bytes = 5 * 1024 * 1024                # 5 MiB allowed memory

        # -----------------------------------------------------------------
        # IMPORTANT: Buffers live PERMANENTLY on **CPU** memory so that
        #            GPU VRAM stays within the tight 5 MiB budget enforced
        #            by the Jetson-Nano scenario.  We therefore store them
        #            as *plain* attributes (NOT as registered buffers)
        #            which prevents automatic `.to()`/`.cuda()` migration.
        # -----------------------------------------------------------------
        self.tok_buf: torch.Tensor = torch.empty(0, k, dtype=torch.uint8, device="cpu")
        self.lbl_buf: torch.Tensor = torch.empty(0, dtype=torch.long, device="cpu")

    # ------------- forward paths -----------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.backbone(x))

    def forward_current(self, x: torch.Tensor, y: torch.Tensor):
        z = self.encode(x)
        z_q, tok = self.quant(z)
        logits = self.head(z_q)
        loss = F.cross_entropy(logits, y)
        return loss, tok

    def forward_tokens(self, tok: torch.Tensor, y: torch.Tensor):
        B = tok.size(0)
        onehot = F.one_hot(tok, 256).float()            # B × M × 256
        code = self.quant.codebook                      # M × 256 × dsub
        z = torch.einsum("bmk,mkd->bmd", onehot, code).reshape(B, -1)  # B × 128
        logits = self.head(z)
        return F.cross_entropy(logits, y)

    # ------------- replay-buffer management ------------------------------
    def _bytes(self, n: int) -> int:
        return n * self.k  # token = 1 byte (uint8)

    def store(self, tok: torch.Tensor, lbl: torch.Tensor) -> None:
        """Reservoir-sampling insertion under a fixed byte budget.
        The incoming *tok* & *lbl* tensors are on **CPU** already.  The
        internal buffers are therefore kept on CPU, eliminating expensive
        device transfers when concatenating.
        """
        # Ensure CPU (safety-net – inexpensive when already on host)
        tok = tok.cpu()
        lbl = lbl.cpu()

        for t, l in zip(tok, lbl):
            if self._bytes(len(self.tok_buf) + 1) < self.max_bytes:
                # still room → append
                self.tok_buf = torch.cat((self.tok_buf, t.unsqueeze(0)), dim=0)
                self.lbl_buf = torch.cat((self.lbl_buf, l.unsqueeze(0)), dim=0)
            else:
                # reservoir replacement ----------------------------------
                j = random.randrange(len(self.tok_buf))
                self.tok_buf[j] = t
                self.lbl_buf[j] = l

    def sample(self, bs: int):
        assert len(self.tok_buf) > 0, "Buffer empty – cannot sample"
        idx = torch.randint(0, len(self.tok_buf), (bs,), device="cpu")
        return self.tok_buf[idx].long(), self.lbl_buf[idx]
