"""src/train.py – model definitions and training logic for DiCA/ERM experiments
Fixed:  • correct Grad-CAM import path  • avoid heavy `diffusers` import at
module load-time  • graceful degradation when optional packages are absent
"""
from __future__ import annotations

import itertools, math, random, time
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import timm

# -----------------------------------------------------------------------------
# Optional external libraries --------------------------------------------------
# -----------------------------------------------------------------------------

try:
    # correct library name on PyPI: "pytorch-grad-cam"
    from pytorch_grad_cam import GradCAM  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    GradCAM = None  # type: ignore

# -----------------------------------------------------------------------------
# Local imports ----------------------------------------------------------------
# -----------------------------------------------------------------------------
from .preprocess import get_dataset

# --------------------------------------------------
# Small utility helpers (kept local to avoid extra file)
# --------------------------------------------------

@contextmanager
def timing(msg: str):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print(f"[TIMER] {msg}: {toc - tic:.2f}s")

# --------------------------------------------------
# Model factory
# --------------------------------------------------

class BackboneFactory:
    """Return backbone with final layer adapted to num_classes."""

    @staticmethod
    def get(name: str, num_classes: int):
        if name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if name == "vit_b16":
            return timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=num_classes
            )
        raise KeyError(f"Unknown backbone: {name}")

# --------------------------------------------------
# Diffusion-based counterfactual image editor (optional)
# --------------------------------------------------

class DiffusionEditor:
    """Wrapper around Stable-Diffusion + LoRA fine-tuning for counterfactuals.
    Heavy dependencies (diffusers, peft) and multi-GB weights are imported and
    loaded lazily so that running plain ERM experiments does **not** force the
    download.  If the required packages or checkpoint are unavailable, we raise
    a clear error prompting the user to switch to `method="erm"`.
    """

    def __init__(self, cfg: Dict[str, Any], device: str = "cuda"):
        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler  # type: ignore
            from peft import LoraConfig, get_peft_model  # type: ignore
            import torchvision.transforms as T  # local import to avoid T dependency if unused
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DiffusionEditor requires the optional packages `diffusers` and "
                "`peft`. Install them or run with method='erm'."
            ) from e

        self.T = T  # store for reuse in `sample_cf`
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg = cfg

        ckpt = cfg["models"]["stable_diffusion"]["checkpoint"]
        with timing("Load Stable Diffusion v1.5 (this can take a while)"):
            self.pipe = StableDiffusionPipeline.from_pretrained(
                ckpt, torch_dtype=torch.float16
            )
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)

        # LoRA adapter ---------------------------------------------------------
        lora_cfg = LoraConfig(r=cfg["models"]["stable_diffusion"]["lora"]["rank"])
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)
        self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, lora_cfg)
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            self.pipe.enable_xformers_memory_efficient_attention()

    # ---------------------------------------------------------------------
    # Counterfactual sampling
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample_cf(self, images: torch.Tensor, masks: torch.Tensor, k: int = 3) -> torch.Tensor:  # noqa: D401,E501
        """Generate *k* counterfactuals for each image in *images*.
        Args:
            images: (B,C,H,W) float tensor in range [0,1]
            masks : (B,1,H,W) float binary mask indicating region to edit
        Returns:
            Tensor of shape (B,k,C,H,W)
        """

        images = images.to(self.device)
        masks = masks.to(self.device)
        cfs = []
        for _ in range(k):
            out_imgs = self.pipe.image_editor(
                images, masks.float(), guidance_scale=7.5, strength=0.8
            ).images
            cf_batch = torch.stack(
                [self.T.ToTensor()(im.resize((224, 224))) for im in out_imgs]
            )
            cfs.append(cf_batch.to(images.dtype).to(self.device))
        return torch.stack(cfs, dim=1)  # (B,k,C,H,W)

# --------------------------------------------------
# Training loop encapsulation
# --------------------------------------------------

class Trainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        dataset_name: str,
        method: str,
        backbone: str,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.method = method.lower()
        self.backbone_name = backbone
        self.device = device if torch.cuda.is_available() else "cpu"

        # ---------- Dataset ----------
        self.train_ds, self.val_ds, self.test_ds = get_dataset(dataset_name, cfg)
        num_classes = (
            len(self.train_ds.dataset.classes)
            if hasattr(self.train_ds, "dataset")
            else len(self.train_ds.classes)
        )

        # ---------- Model & helper modules ----------
        self.model = BackboneFactory.get(backbone, num_classes).to(self.device)

        # DiCA components are optional and initialised only when the method is
        # requested **and** the required libraries are available.
        self.editor = None
        self.cam = None
        if self.method == "dica":
            if GradCAM is None:
                raise ImportError(
                    "Method 'dica' requires `pytorch-grad-cam`. Install the package "
                    "or run with `method='erm'`."
                )
            # Attempt to build DiffusionEditor – may fail if diffusers weights
            # are not reachable. We surface a user-friendly error.
            try:
                self.editor = DiffusionEditor(cfg, self.device)
            except Exception as e:  # pragma: no cover – fail-fast
                raise RuntimeError(
                    "Failed to initialise DiffusionEditor. See the original error "
                    "below and consider switching to `method='erm'` if diffusion "
                    "resources are unavailable."
                ) from e
            self.cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])

        # ---------- Optimisation ----------
        self._build_optim()
        self.scaler = GradScaler() if cfg["hardware"]["amp"] else None

    # --------------------------------------------------
    # private helpers
    # --------------------------------------------------

    def _build_optim(self) -> None:
        opt_cfg = self.cfg["training"]["optim"][self.backbone_name]
        if opt_cfg["type"].upper() == "SGD":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg["lr"],
                momentum=opt_cfg["momentum"],
                weight_decay=opt_cfg["weight_decay"],
            )
        else:
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg["lr"],
                betas=opt_cfg["betas"],
                weight_decay=opt_cfg["weight_decay"],
            )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.cfg["training"]["epochs"]
        )

    # --------------------------------------------------
    # counterfactual helper (DiCA only)
    # --------------------------------------------------

    def _counterfactual_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cam is None or self.editor is None:  # pragma: no cover
            raise RuntimeError("Counterfactual batch requested but DiCA components are missing.")
        # grad-cam saliency mask
        mask = self.cam(x, eigen_smooth=True)[0]  # (B,1,H,W)
        topk = torch.quantile(mask.flatten(1), 1 - self.cfg["dica"]["top_p"], dim=1).view(
            -1, 1, 1, 1
        )
        mask_bin = (mask >= topk).float()
        x_cf = self.editor.sample_cf(x, mask_bin, k=self.cfg["dica"]["k"])
        return x_cf, mask_bin

    # --------------------------------------------------
    # public API
    # --------------------------------------------------

    def train(self) -> None:
        bs = self.cfg["training"]["batch_size"][self.backbone_name]
        tr_loader = DataLoader(
            self.train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=self.cfg["hardware"]["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(self.val_ds, batch_size=64, shuffle=False)

        best_val = 0.0
        patience = 0

        for epoch in range(self.cfg["training"]["epochs"]):
            self.model.train()
            for x, y in tr_loader:
                x, y = x.to(self.device), y.to(self.device)
                with autocast(enabled=self.cfg["hardware"]["amp"]):
                    logits = self.model(x)
                    loss_cls = F.cross_entropy(logits, y)

                    if self.method == "dica":
                        x_cf, _ = self._counterfactual_batch(x)
                        logits_cf = self.model(x_cf.view(-1, *x.shape[1:]))
                        logits_rep = (
                            logits.unsqueeze(1)
                            .expand(-1, self.cfg["dica"]["k"], -1)
                            .contiguous()
                            .view_as(logits_cf)
                        )
                        loss_cons = F.mse_loss(logits_cf, logits_rep)
                        loss = loss_cls + self.cfg["dica"]["lambda_consistency"] * loss_cons
                    else:
                        loss = loss_cls

                self.opt.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.opt.step()

            self.sched.step()
            # ------------- validation -------------
            acc = self.evaluate(val_loader)
            print(f"Epoch {epoch}: val-acc = {acc:.3f}")
            if acc > best_val:
                best_val = acc
                patience = 0
                Path("models").mkdir(parents=True, exist_ok=True)
                ckpt_name = f"models/{self.dataset_name}_{self.method}_{self.backbone_name}.pt"
                torch.save(self.model.state_dict(), ckpt_name)
            else:
                patience += 1
                if patience >= self.cfg["training"]["early_stop_patience"]:
                    print("Early stopping triggered.")
                    break

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return correct / (total + 1e-8)
