"""src/train.py – model definitions and training routines for the GCDI project"""
from __future__ import annotations
import json, random, tarfile, time, math, hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from .preprocess import (
    prepare_waterbirds_datasets,
    sha256sum,
    download,
)

# Matplotlib is only needed for saving training-time sanity plots.  Do **not**
# switch to an interactive backend – we always run head-less on the server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Heavy external dependencies (fail fast if unavailable)
from diffusers import StableDiffusionImg2ImgPipeline
from peft import get_peft_model, LoraConfig, TaskType

# ---------------------------------------------------------------------------
#                               LOSSES & UTILS
# ---------------------------------------------------------------------------
class HighPassMaskCache:
    """Caches a radial high-pass mask to avoid recomputation.
    The key also includes the target device so the same mask can be reused for
    CPU/GPU tensors without additional .to() calls."""

    _cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    @classmethod
    def get(cls, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device)
        if key not in cls._cache:
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            center = torch.tensor([(h - 1) / 2, (w - 1) / 2])[:, None, None]
            d = ((y - center[0]) ** 2 + (x - center[1]) ** 2).sqrt()
            mask = (d > 0.5 * (h / 2)).float().to(device)
            cls._cache[key] = mask
        return cls._cache[key]


class FourierFeatureLoss(nn.Module):
    """L1 distance between the high-frequency magnitude spectra of two feature maps."""

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):  # B×C×H×W
        F1, F2 = torch.fft.fft2(f1, norm="ortho"), torch.fft.fft2(f2, norm="ortho")
        m = HighPassMaskCache.get(f1.shape[-2], f1.shape[-1], f1.device)
        return ((F1.abs() - F2.abs()).abs() * m).mean()


# ---------------------------------------------------------------------------
#                               BYOL BACKBONE
# ---------------------------------------------------------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 2048, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """Minimal BYOL implementation with separate online & target encoders."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone_o = backbone
        self.backbone_t = models.resnet18(weights=None)
        for p in self.backbone_t.parameters():
            p.requires_grad = False

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone_t.fc = nn.Identity()

        self.proj_o, self.proj_t = ProjectionMLP(feat_dim), ProjectionMLP(feat_dim)
        self.pred = ProjectionMLP(256, 512, 256)
        self._ema(0)  # initial hard copy

    @torch.no_grad()
    def _ema(self, m: float = 0.996):
        for po, pt in zip(self.backbone_o.parameters(), self.backbone_t.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1 - m)

    def forward(self, v1, v2):
        o1, o2 = self.backbone_o(v1), self.backbone_o(v2)
        z1, z2 = self.proj_o(o1), self.proj_o(o2)
        p1, p2 = self.pred(z1), self.pred(z2)
        with torch.no_grad():
            t1, t2 = self.proj_t(self.backbone_t(v1)), self.proj_t(self.backbone_t(v2))
        loss = (F.mse_loss(p1, t2) + F.mse_loss(p2, t1)) / 2
        return loss


# ---------------------------------------------------------------------------
#                    LOW-RANK AMPLIFICATION & RIESZ ESTIMATOR
# ---------------------------------------------------------------------------
@torch.no_grad()
def low_rank_amplify(X: torch.Tensor, r: int, factor: float):
    mu = X.mean(0, keepdim=True)
    Y = X - mu
    U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
    S[:r] = S[:r] * factor
    return (U * S.unsqueeze(0)) @ Vt + mu, Vt  # amplified embeddings, PCs


@torch.no_grad()
def riesz(Z: torch.Tensor, y: torch.Tensor, lam: float = 1e-3):
    I = torch.eye(Z.shape[1], device=Z.device)
    w = torch.linalg.solve(Z.T @ Z + lam * I, Z.T @ y)
    return w  # (K,)


# ---------------------------------------------------------------------------
#                  COUNTERFACTUAL DIFFUSION GENERATOR (CDG)
# ---------------------------------------------------------------------------
class CDG:
    """Light-weight wrapper around SD-Nano-1.1 with LoRA fine-tuning."""

    def __init__(self, cfg: Dict[str, Any], device: str):
        self.cfg = cfg
        print("[CDG] loading diffusion backbone – this takes ~400 MB …")
        dtype = torch.bfloat16 if cfg["env"]["bf16"] else torch.float16
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg["cdg"]["model_name"], torch_dtype=dtype, safety_checker=None
        ).to(device)

        lora_cfg = LoraConfig(
            r=cfg["cdg"]["lora_rank"],
            lora_alpha=2 * cfg["cdg"]["lora_rank"],
            target_modules=["attn1", "attn2"],
            bias="none",
            task_type=TaskType.UNET_2D_CONDITIONAL,
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)
        self.pipe.unet.enable_gradient_checkpointing()
        self.opt = torch.optim.AdamW(self.pipe.unet.parameters(), lr=cfg["cdg"]["lr"])
        self.device = device

    def fine_tune(self, dl: DataLoader):
        base_loss: float | None = None
        for ep in range(self.cfg["cdg"]["epochs"]):
            for it, (img, _) in enumerate(dl):
                img = (img * 0.5 + 0.5).to(self.device, dtype=self.pipe.dtype)
                noise = torch.randn_like(img)
                with torch.autocast(self.device, self.pipe.dtype):
                    out = self.pipe.unet(img, 0).sample  # type: ignore[arg-type]
                    loss = F.mse_loss(out, noise)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                if base_loss is None:
                    base_loss = loss.item()
                # Divergence check
                if torch.isnan(loss) or loss.item() > 5 * base_loss:
                    raise RuntimeError("[CDG] LoRA diverged – aborting")
            print(f"[CDG] epoch {ep + 1}/{self.cfg['cdg']['epochs']}  loss={loss.item():.4f}")
        self.pipe.unet.eval()

    @torch.no_grad()
    def generate(self, x: torch.Tensor, latent_delta: torch.Tensor):
        """Generate counterfactuals: x in (-1,1); latent_delta broadcastable."""
        self.pipe.to(self.device)
        x = (x * 0.5 + 0.5).to(self.pipe.dtype)
        lat = self.pipe.vae.encode(x).latent_dist.sample() * 0.18215
        lat_cf = lat + latent_delta.to(lat.dtype)
        img = self.pipe.vae.decode(lat_cf / 0.18215).sample
        return (img * 2 - 1).clamp(-1, 1)


# ---------------------------------------------------------------------------
#                               MEMORY GUARD
# ---------------------------------------------------------------------------
_DEF_MAX_GB = 15.2

def assert_memory(stage: str, max_gb: float = _DEF_MAX_GB):
    if not torch.cuda.is_available():
        return
    gb = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[MEM] peak after {stage}: {gb:5.2f} GB")
    if gb > max_gb:
        raise RuntimeError(f"VRAM budget exceeded during {stage}")


# ---------------------------------------------------------------------------
#                          FULL WATERBIRDS PIPELINE
# ---------------------------------------------------------------------------

def run_waterbirds_experiment(seed: int, cfg: Dict[str, Any]):
    """Full end-to-end pipeline.  Returns a dict with raw outputs for further
    evaluation & aggregation."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = cfg["env"]["device"]

    # ---------------- DATA ----------------
    ds_train_full, ds_val, ds_test, tf_ssl = prepare_waterbirds_datasets(cfg)

    # ---------------- STAGE 1 – BYOL ----------------
    backbone = models.resnet18(weights=None)
    byol = BYOL(backbone).to(device)
    opt_ssl = torch.optim.AdamW(byol.parameters(), lr=cfg["ssl"]["lr"])

    dl_ssl = DataLoader(
        ds_train_full,
        batch_size=cfg["ssl"]["batch"],
        shuffle=True,
        num_workers=cfg["env"]["num_workers"],
        drop_last=True,
        persistent_workers=True,
    )

    for ep in range(cfg["ssl"]["epochs"]):
        for xb, _ in dl_ssl:
            x1 = xb.to(device)
            x2 = xb[torch.randperm(xb.size(0))].to(device)
            with torch.autocast(device, torch.bfloat16 if cfg["env"]["bf16"] else torch.float16):
                loss = byol(x1, x2)
            loss.backward()
            opt_ssl.step()
            opt_ssl.zero_grad()
            byol._ema()
        if (ep + 1) % 20 == 0:
            print(f"[BYOL] epoch {ep + 1}/{cfg['ssl']['epochs']}  loss={loss.item():.4f}")
    assert_memory("BYOL")

    # Cache embeddings for causal analysis
    all_emb, all_y = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(ds_train_full, batch_size=512):
            feat = backbone(xb.to(device)).cpu()
            all_emb.append(feat)
            all_y.append(yb)
    emb = torch.cat(all_emb)
    labels = torch.cat(all_y).float()

    # ---------------- STAGE 2 – RIESZ ----------------
    emb_amp, PCs = low_rank_amplify(emb, cfg["lra"]["rank"], cfg["lra"]["amp_factor"])
    n_lab = max(1, int(len(ds_train_full) * cfg["riesz"]["label_frac"]))
    idx_lab = random.sample(range(len(ds_train_full)), n_lab)
    Z = emb_amp[idx_lab] @ PCs[: cfg["lra"]["rank"], :].T  # N×K
    w_te = riesz(Z, labels[idx_lab], cfg["riesz"]["ridge"])
    spur_idx = torch.where(w_te.abs() > w_te.abs().mean())[0].tolist()
    if not spur_idx:
        raise RuntimeError("[FAIL] No spurious directions discovered – aborting")
    print("[INFO] Spurious directions:", spur_idx)

    # ---------------- STAGE 3 – CDG ----------------
    cdg = CDG(cfg, device)
    subset, _ = random_split(
        ds_train_full,
        [5000, len(ds_train_full) - 5000],
        generator=torch.Generator().manual_seed(seed),
    )
    dl_cdg = DataLoader(subset, batch_size=cfg["cdg"]["batch"], shuffle=True, num_workers=2)
    cdg.fine_tune(dl_cdg)
    assert_memory("CDG")

    # ---------------- STAGE 4 – INVARIANT LEARNER ----------------
    def make_env_ds(base_ds: Dataset):
        class _Env(Dataset):
            def __len__(self):
                return len(base_ds) * (1 + len(spur_idx))

            def __getitem__(self, idx: int):
                base_i = idx % len(base_ds)
                env_id = idx // len(base_ds)
                x, y = base_ds[base_i]
                if env_id == 0:
                    return x, y, 0
                dir_vec = torch.from_numpy(PCs[spur_idx[env_id - 1]].copy()).to(device)
                delta = dir_vec.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                x_cf = cdg.generate(x.unsqueeze(0).to(device), delta)[0].cpu()
                return x_cf, y, env_id

        return _Env()

    env_ds = make_env_ds(ds_train_full)
    dl_env = DataLoader(env_ds, batch_size=cfg["invariant"]["batch"], shuffle=True, num_workers=4)

    net = models.resnet18(num_classes=2).to(device)
    opt_inv = torch.optim.AdamW(net.parameters(), lr=cfg["invariant"]["lr"])
    fourier = FourierFeatureLoss()
    criterion = nn.CrossEntropyLoss()

    def conv_stem(x):  # feature extractor before layer1
        x = net.conv1(x)
        x = net.bn1(x)
        return net.relu(x)

    for ep in range(cfg["invariant"]["epochs"]):
        for xb, yb, env in dl_env:
            xb, yb, env = xb.to(device), yb.to(device), env.to(device)
            with torch.autocast(device, torch.bfloat16 if cfg["env"]["bf16"] else torch.float16):
                logits = net(xb)
                loss_ce = criterion(logits, yb)
                # IRM penalty
                irm_pen = 0.0
                for eid in env.unique():
                    mask = env == eid
                    l_e = criterion(logits[mask], yb[mask])
                    g = torch.autograd.grad(l_e, net.fc.parameters(), create_graph=True)[0]
                    irm_pen += (g ** 2).mean()
                # Fourier consistency between original and counterfactual batches
                orig_mask, cf_mask = env == 0, env > 0
                if orig_mask.any() and cf_mask.any():
                    f_o = conv_stem(xb[orig_mask])
                    f_cf = conv_stem(xb[cf_mask])
                    loss_four = fourier(f_o, f_cf)
                else:
                    loss_four = torch.tensor(0.0, device=device)
                loss = (
                    loss_ce
                    + cfg["invariant"]["irm_lambda"] * irm_pen
                    + cfg["invariant"]["fourier_lambda"] * loss_four
                )
            opt_inv.zero_grad()
            loss.backward()
            opt_inv.step()
        if (ep + 1) % 25 == 0:
            print(
                f"[INV] epoch {ep + 1}/{cfg['invariant']['epochs']}  loss={loss.item():.4f}"
            )
    assert_memory("Invariant-Learner")

    # Return artefacts required for evaluation
    artefacts = {
        "model": net,
        "cdg": cdg,
        "pcs": PCs,
        "spur_idx": spur_idx,
        "ds_test": ds_test,
        "seed": seed,
    }
    return artefacts
