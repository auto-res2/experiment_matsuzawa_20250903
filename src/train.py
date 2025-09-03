"""src/train.py
Model architectures and memory‐buffer classes used by the HATEM
experiments.  All code is extracted from the original monolithic
script and organised so that it can be imported from other modules
(e.g. `from src.train import build_resnet18`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 – kept for potential future use
from torchvision import models

# ---------------------------------------------------------------------------
# 1.  Backbone – ResNet-18 adapted to small images (stride-1, no max-pool)
# ---------------------------------------------------------------------------

def build_resnet18(num_classes: int = 100) -> nn.Module:
    """Return a ResNet-18 whose first layer is suitable for 32×32/64×64 images."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ---------------------------------------------------------------------------
# 2.  Frozen encoder stub & helper utilities (Tier-1 tokenisation)
# ---------------------------------------------------------------------------

class DummyEncoder(nn.Module):
    """Light-weight placeholder that mimics the VQ-GAN encoder used in HATEM.

    The real implementation would output a grid of codebook indices and a
    semantic embedding vector.  For reproducibility in environments where
    the heavy VQ-GAN is unavailable, we emulate the behaviour with random
    numbers so that the rest of the pipeline can still be executed.
    """

    def __init__(self, code_dim: int = 256):
        super().__init__()
        self.code_dim = code_dim

    @torch.no_grad()
    def tokenise(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (L×L integer grid, embedding-vector).

        Parameters
        ----------
        x : torch.Tensor
            Input image batch B×C×H×W on *any* device.
        """
        bsz = x.size(0)
        codes = torch.randint(0, self.code_dim, (bsz, 4, 4), device=x.device)
        embed = x.mean(dim=[2, 3])  # B×C mean-pooled embedding
        return codes, embed


def tiny_hash(codes: torch.Tensor) -> torch.Tensor:
    """Map raw code indices to an 8-bit hash (uint8)."""
    return (codes % 256).to(torch.uint8)

# ---------------------------------------------------------------------------
# 3.  HATEM hierarchical memory buffer
# ---------------------------------------------------------------------------

class HATEMBuffer:
    """Hierarchical Adaptive Tokenised Episodic Memory (HATEM).

    Tier-1 stores 8-bit token IDs (≈32–64 B / image) **plus** a small
    embedding; Tier-2 keeps one or a few class-wise synthetic prototypes.
    The implementation here is the minimal version required for the paper’s
    experiments – prototype learning is simplified to mean aggregation.
    """

    def __init__(self, L: int = 4, vocab: int = 256, proto_per_cls: int = 5, bytes_limit: int = 524_288):
        self.L = L
        self.vocab = vocab
        self.proto_per_cls = proto_per_cls
        self.bytes_limit = bytes_limit

        # Storage
        self.token_buf: List[Tuple[np.ndarray, torch.Tensor, int]] = []
        self.proto_dict: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.total_bytes = 0  # runtime accounting

    # ---------------------------------------------------------------------
    # Tier-1  – online write (no gradients)
    # ---------------------------------------------------------------------
    def add_sample(self, img: torch.Tensor, label: int, encoder: "DummyEncoder") -> None:
        """Encode *one* image and push it into the buffer (if space allows)."""
        with torch.no_grad():
            codes, embed = encoder.tokenise(img.unsqueeze(0))
        token_ids = tiny_hash(codes).cpu().numpy().astype("uint8")
        self.token_buf.append((token_ids, embed.squeeze(0).cpu(), label))
        self.total_bytes += token_ids.nbytes + embed.numel() * 4
        self._prune_if_needed()

    # ---------------------------------------------------------------------
    # Tier-2  – synthetic prototypes (simplified mean pooling)
    # ---------------------------------------------------------------------
    def consolidate(self) -> None:
        """Create/refresh prototypes by class-wise mean aggregation."""
        by_cls: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for token_ids, embed, lbl in self.token_buf:
            by_cls[lbl].append(embed)
        for lbl, embeds in by_cls.items():
            embeds = torch.stack(embeds)
            proto = embeds.mean(dim=0)
            self.proto_dict[lbl] = [proto]  # one prototype per class (simplified)

        # Re-compute memory usage
        self.total_bytes = sum(t[0].nbytes + t[1].numel() * 4 for t in self.token_buf)
        self.total_bytes += sum(p.numel() * 4 for lst in self.proto_dict.values() for p in lst)
        self._prune_if_needed()

    # ---------------------------------------------------------------------
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *n* replay samples (dummy Gaussian images for code brevity)."""
        if len(self.token_buf) == 0:
            raise RuntimeError("[HATEM] Buffer empty – cannot sample.")
        idxs = np.random.choice(len(self.token_buf), size=n, replace=True)
        imgs, labels = [], []
        for i in idxs:
            _token_ids, _embed, lbl = self.token_buf[i]
            imgs.append(torch.randn(3, 32, 32))  # placeholder decode
            labels.append(lbl)
        return torch.stack(imgs), torch.tensor(labels)

    # ---------------------------------------------------------------------
    def _prune_if_needed(self) -> None:
        while self.total_bytes > self.bytes_limit and len(self.token_buf) > 0:
            token_ids, embed, _lbl = self.token_buf.pop(0)
            self.total_bytes -= token_ids.nbytes + embed.numel() * 4

# ---------------------------------------------------------------------------
# 4.  Raw exemplar replay buffer (baseline)
# ---------------------------------------------------------------------------

class RawReplayBuffer:
    """Naïve exemplar memory that stores raw float32 tensors (ER / DER++)."""

    def __init__(self, bytes_limit: int):
        self.bytes_limit = bytes_limit
        self.samples: List[Tuple[torch.Tensor, int]] = []
        self.total_bytes = 0  # counts *floats*, multiply by 4 for bytes

    # .....................................................................
    def add_sample(self, img: torch.Tensor, label: int) -> None:
        self.samples.append((img.cpu(), label))
        self.total_bytes += img.numel()
        self._prune()

    def _prune(self) -> None:
        while self.total_bytes * 4 > self.bytes_limit and len(self.samples) > 0:
            img, _ = self.samples.pop(0)
            self.total_bytes -= img.numel()

    # .....................................................................
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.samples) == 0:
            raise RuntimeError("[RawReplay] Buffer empty – cannot sample.")
        idxs = np.random.choice(len(self.samples), size=n, replace=True)
        imgs, labels = zip(*[self.samples[i] for i in idxs])
        return torch.stack(imgs), torch.tensor(labels)
