"""src/train.py – model definition, LoRA helpers, generator wrapper, ring-buffer and the Trainer that performs task-wise training.
The code is refactored from the original monolithic script. No algorithmic changes were
introduced, only structural improvements and small safety guards.
"""
from __future__ import annotations

import math
import pathlib
import random
import time
from collections import deque
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from torch.cuda.amp import GradScaler, autocast

# -----------------------------------------------------------------------------
# LoRA helpers
# -----------------------------------------------------------------------------

class LoRAInject:
    """Inject low-rank adapters (LoRA) into attention blocks of a model.

    Only modules that expose `to_q` / `to_v` Linear layers are patched. The
    patched forward keeps the original path *frozen* and adds the LoRA path.
    """

    def __init__(self, rank: int = 8, alpha: int = 16):
        self.rank = rank
        self.alpha = alpha
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    # .........................................................................
    # public helpers
    # .........................................................................

    def inject(self, model: nn.Module) -> None:
        for _name, module in model.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_v"):
                self._patch_linear(module, "to_q")
                self._patch_linear(module, "to_v")

    def parameters(self):
        seen = set()
        for h in self.handles:
            # grab *only* LoRA params we created (avoid duplicates)
            mod = h._hooked_module  # pyright: ignore [reportPrivateUsage]
            for p in mod.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # .........................................................................
    # internal implementation details
    # .........................................................................

    def _patch_linear(self, parent: nn.Module, attr: str) -> None:
        orig: nn.Linear = getattr(parent, attr)
        dim_out, dim_in = orig.weight.shape

        lora_a = nn.Parameter(torch.zeros(self.rank, dim_in))
        lora_b = nn.Parameter(torch.zeros(dim_out, self.rank))
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        nn.init.zeros_(lora_b)
        scale = self.alpha / self.rank

        def fwd(x, *, _orig=orig, _A=lora_a, _B=lora_b):
            # (B, L, C) – handle arbitrary leading dims by flattening
            out = _orig(x)
            delta = (_B @ (_A @ x.transpose(-2, -1))).transpose(-2, -1)
            return out + delta * scale

        handle = parent.register_forward_hook(lambda _mod, inp, _out: fwd(inp[0]))
        self.handles.append(handle)


# -----------------------------------------------------------------------------
# Frozen diffusion generator + LoRA adapters
# -----------------------------------------------------------------------------

class FrozenGenerator:
    """Wrapper around diffusers pipeline with optional LoRA injection."""

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        from diffusers import StableDiffusionPipeline  # local import (heavy)

        repo = cfg["common"]["frozen_generator"]["hf_repo"]
        self.pipe = StableDiffusionPipeline.from_pretrained(
            repo, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.enable_model_cpu_offload()
        self.device = device
        self.lora: LoRAInject | None = None
        self.cfg = cfg

    # .................................................................
    def attach_adapter(self, *, rank: int = 8, alpha: int = 16) -> LoRAInject:
        self.lora = LoRAInject(rank, alpha)
        self.lora.inject(self.pipe.unet)
        return self.lora

    # .................................................................
    @torch.no_grad()
    def sample(self, prompts: Sequence[str]):
        steps = self.cfg["common"]["frozen_generator"]["num_inference_steps"]
        guidance = self.cfg["common"]["frozen_generator"]["guidance_scale"]
        imgs = self.pipe(list(prompts), num_inference_steps=steps, guidance_scale=guidance).images
        return imgs


# -----------------------------------------------------------------------------
# Rehearsal buffer (experience replay)
# -----------------------------------------------------------------------------

class RingBuffer:
    """Fixed-size ring buffer that stores (x, y) examples on *CPU* memory."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._buf: deque = deque(maxlen=max_items)

    def __len__(self):
        return len(self._buf)

    def add(self, x: torch.Tensor, y: int):
        # store cpu tensors to save GPU memory
        self._buf.append((x.cpu(), int(y)))

    def sample(self, k: int):
        k = min(k, len(self._buf))
        xs, ys = zip(*random.sample(self._buf, k))
        return torch.stack(xs), torch.tensor(ys)


# -----------------------------------------------------------------------------
# Classifier backbones with Orthogonal-Subspace heads
# -----------------------------------------------------------------------------

class OrthoSubspaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, proj_dim: int = 256, lam: float = 5e-3):
        super().__init__()
        self.P1 = nn.Linear(in_dim, proj_dim, bias=False)
        self.P2 = nn.Linear(in_dim, proj_dim, bias=False)
        self.fc = nn.Linear(proj_dim * 2, num_classes, bias=False)
        self.lam = lam

    def forward(self, x):
        z1 = self.P1(x)
        z2 = self.P2(x)
        return self.fc(torch.cat([z1, z2], dim=-1))

    def ortho_penalty(self):
        I = torch.eye(self.P1.weight.shape[0], device=self.P1.weight.device)
        p1 = (self.P1.weight @ self.P1.weight.T) - I
        p2 = (self.P2.weight @ self.P2.weight.T) - I
        return (p1.pow(2).mean() + p2.pow(2).mean()) * self.lam


# .............................................................................
# backbone helpers
# .............................................................................

def resnet18(num_classes: int) -> nn.Module:
    net = torchvision.models.resnet18(weights=None)
    in_dim = net.fc.in_features
    net.fc = nn.Identity()
    head = OrthoSubspaceHead(in_dim, num_classes)
    return nn.Sequential(net, head)


def vit_small(num_classes: int) -> nn.Module:
    net = timm.create_model("vit_small_patch16_224", pretrained=True)
    in_dim = net.head.in_features
    net.reset_classifier(0)
    head = OrthoSubspaceHead(in_dim, num_classes)
    return nn.Sequential(net, head)


# -----------------------------------------------------------------------------
# Trainer (task stream training)
# -----------------------------------------------------------------------------

class Trainer:
    """Task-incremental continual-learning trainer."""

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.gen = FrozenGenerator(cfg, device)  # frozen diffusion generator

    # .........................................................................
    def _build_model(self, num_classes: int):
        name = self.cfg["experiment_1"]["classifier"]
        if name == "resnet18":
            model = resnet18(num_classes)
        elif name.startswith("vit"):
            model = vit_small(num_classes)
        else:
            raise ValueError(f"Unknown backbone {name}")
        return model.to(self.device)

    # .........................................................................
    def train_stream(self, tasks: List[torch.utils.data.Dataset], testset,
                     *, method: str, seed: int, out_dir: pathlib.Path) -> None:
        import numpy as np  # local import to keep global namespace clean
        from src.evaluate import evaluate_model, lineplot

        torch.manual_seed(seed)
        np.random.seed(seed)

        num_classes = len(set(getattr(testset, "targets", []))) or 100
        model = self._build_model(num_classes)

        # single optimiser for backbone + (optional) LoRA adapter params
        optim = torch.optim.AdamW(model.parameters(), lr=self.cfg["common"]["optimizer"]["lr_backbone"])
        scaler = GradScaler()

        buf: RingBuffer | None = None
        if "er" in method:
            # roughly 25 images take ~100 KB fp32; adjust heuristic here
            cap_kb = self.cfg["experiment_1"].get("memory_cap_kb", 512)
            buf = RingBuffer(max_items=int(cap_kb * 2.5))

        aa: List[float] = []  # accuracy after every task

        # =====================================================================
        for task_id, task_ds in enumerate(tasks):
            loader = torch.utils.data.DataLoader(
                task_ds,
                batch_size=self.cfg["common"]["batch_size"],
                shuffle=True,
                num_workers=self.cfg["common"].get("num_workers", 4),
            )

            for _epoch in range(self.cfg["experiment_1"]["epochs"]):
                for x, y in loader:
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                    # rehearsal
                    if buf and len(buf):
                        rx, ry = buf.sample(len(x))
                        rx, ry = rx.to(self.device), ry.to(self.device)
                        x = torch.cat([x, rx])
                        y = torch.cat([y, ry])

                    with autocast(enabled=(self.device == "cuda")):
                        out = model(x)
                        loss = F.cross_entropy(out, y)

                    optim.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()

            # after the task: add samples to rehearsal buffer
            if buf:
                for x, y in torch.utils.data.DataLoader(task_ds, batch_size=1, shuffle=True):
                    buf.add(x.squeeze(), int(y))

            # evaluation -------------------------------------------------------
            acc = evaluate_model(model, testset, self.device)
            aa.append(acc)
            print(f"[Task {task_id:02d}] accuracy = {acc:.2f}%")

        # =====================================================================
        out_dir.mkdir(parents=True, exist_ok=True)
        lineplot(list(range(len(aa))), aa, title=f"AA_{method}",
                 xlabel="task", ylabel="accuracy", fname=out_dir / f"accuracy_{method}.pdf")

        # save raw scores for later statistical aggregation
        (out_dir / "scores.json").write_text(json.dumps({"AA": aa[-1], "curve": aa}))
