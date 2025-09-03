"""src/train.py
Model building and training utilities for PCCM.
"""
from __future__ import annotations

import shutil
import requests
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.cuda import amp
from torchvision import transforms as T
import timm
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import mutual_info_score
import pandas as pd
from tqdm import tqdm
from PIL import Image

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
AMP_ENABLED: bool = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# project-wide directories -----------------------------------------------------
DATA_ROOT = Path("data")
CACHE_ROOT = Path("cache")
CKPT_ROOT = Path("checkpoints")

for p in (DATA_ROOT, CACHE_ROOT, CKPT_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Make experiment deterministic on the given seed."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
#  Model definition
# -----------------------------------------------------------------------------

class FeatExtractor(nn.Module):
    """Wrap timm models to return both logits and penultimate features."""

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes)
        # NOTE: timm always builds the classifier last – grab its in_features
        self.feat_dim = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        if return_feat:
            features = self.model.forward_features(x)  # timm API
            logits = self.model.get_classifier()(features)
            return logits, features
        return self.model(x)


def _download_state_dict(url: str, target_path: Path) -> bool:
    """Download a file with streaming to avoid RAM overflow.
    Returns True on success, False if the download failed for any reason. The
    training code will fall back to randomly initialised weights when the
    download is unavailable (e.g. CI environment without external internet).
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(target_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return True
    except (requests.RequestException, IOError) as e:
        print(f"[train] WARNING: Failed to download weights from {url}. "
              f"Proceeding with random initialisation. ({e})")
        # Make sure half-downloaded files do not pollute cache
        if target_path.exists():
            try:
                target_path.unlink()
            except IOError:
                pass
        return False


def _try_load_moco_weights(model: FeatExtractor, weight_path: Path) -> None:
    """Attempt to load a saved MoCo-v3 checkpoint. If loading fails, the model
    remains randomly initialised and a warning is printed instead of raising an
    exception so that training can continue in restricted environments."""

    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        # remove prefix of MoCo keys
        cleaned = {
            k.replace("module.encoder.", ""): v
            for k, v in ckpt["state_dict"].items()
            if "encoder" in k
        }
        msg = model.model.load_state_dict(cleaned, strict=False)
        print("[train] Loaded MoCo-v3 weights:", msg)
    except Exception as e:  # noqa: BLE001 – broad on purpose, just warn
        print(f"[train] WARNING: Failed to load cached MoCo-v3 checkpoint from {weight_path}. "
              f"Proceeding with random initialisation. ({e})")


def build_model(arch: str, num_classes: int) -> FeatExtractor:
    """Build a backbone (ResNet-50 / ViT-B16) and optionally load MoCo-v3
    self-supervised weights when they are available locally. The function never
    raises when the weights cannot be obtained so that unit-tests and CI runs
    without external network still succeed."""

    if arch not in {"resnet50", "vit_base_patch16_224"}:
        raise ValueError(f"Unsupported architecture {arch}")

    model = FeatExtractor(arch, num_classes)

    # ------------------------------------------------------------------
    # Load MoCo-v3 weights if we already have them or can download them
    # ------------------------------------------------------------------
    moco_urls = {
        "resnet50": "https://dl.fbaipublicfiles.com/moco-v3/r50-300ep/r50-300ep.pth.tar",
        "vit_base_patch16_224": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
    }
    weight_url = moco_urls[arch]
    weight_path = CACHE_ROOT / Path(weight_url).name

    if not weight_path.exists():
        _download_state_dict(weight_url, weight_path)

    if weight_path.exists():
        _try_load_moco_weights(model, weight_path)
    else:
        print("[train] MoCo-v3 weights unavailable – using randomly initialised model.")

    return model.to(DEVICE)

# -----------------------------------------------------------------------------
# PCCM Trainer
# -----------------------------------------------------------------------------

PROMPT_BANK: List[str] = [
    "a photo of red color",
    "a photo of green color",
    "a photo of blue color",
    "sunny weather",
    "cloudy sky",
    "snowy ground",
    "water background",
    "wooden texture",
    "metallic surface",
    "cartoon style",
    "painting style",
]

# image transforms used only inside the trainer for counterfactuals
_val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# constants needed for (un)normalisation
_MEAN_T = torch.as_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD_T = torch.as_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


class PCCMTrainer:
    """Prompted Counterfactual Confounding Mitigation trainer."""

    def __init__(
        self,
        ds_name: str,
        arch: str,
        num_classes: int,
        lambda1: float,
        lambda2: float,
        K: int,
        seed: int,
    ) -> None:
        # reproducibility
        set_seed(seed)
        self.seed = seed

        # hyper-params
        self.ds_name = ds_name
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.K = K
        self.step_idx = 0

        # cache directory for this run
        self.cache_dir = CACHE_ROOT / f"{ds_name}_{arch}_seed{seed}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # model & optimiser --------------------------------------------------
        self.model = build_model(arch, num_classes)
        lr = 3e-4 if "resnet" in arch else 1e-4
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
        self.scaler = amp.GradScaler(enabled=AMP_ENABLED)
        self.criterion = nn.CrossEntropyLoss()

        # CLIP & diffusion pipelines ----------------------------------------
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.inp_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float16 if AMP_ENABLED else torch.float32,
            safety_checker=None,
        ).to(DEVICE)
        self.inp_pipe.enable_attention_slicing()

        self.best_val_wgacc = -1.0

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _unnormalize(img: torch.Tensor) -> torch.Tensor:
        """Map tensor in Normalised space back to [0,1] range for visual models."""
        return img * _STD_T.to(img.device) + _MEAN_T.to(img.device)

    # ---------------------------------------------------------------------
    # 1) Attribute mining
    # ---------------------------------------------------------------------
    def _mine_spurious_attrs(self, dl: torch.utils.data.DataLoader) -> torch.Tensor:
        print("[PCCM] Mining spurious attributes …")
        all_sims, all_labels = [], []
        for x, y, _ in tqdm(dl, leave=False):
            with torch.no_grad():
                x = x.to(DEVICE)
                # De-normalise before feeding to CLIP
                x_vis = self._unnormalize(x).clamp(0, 1)
                inputs = self.clip_proc(text=PROMPT_BANK, images=x_vis, return_tensors="pt", padding=True).to(DEVICE)
                outs = self.clip_model(**inputs)
                sims = outs.logits_per_image  # [B, |A|]
            all_sims.append(sims.float().cpu())
            all_labels.append(y)
        S = torch.cat(all_sims)  # (N, |A|)  – on CPU
        Y = torch.cat(all_labels).cpu()  # (N,)

        # Convert labels once to NumPy for repeated reuse
        Y_np = Y.numpy()
        C_scores: List[float] = []
        for j in range(S.shape[1]):
            sj = S[:, j]
            sj_np = sj.numpy()
            # Pearson correlation (absolute value)
            corr = abs(np.corrcoef(sj_np, Y_np)[0, 1])
            # Mutual information – need discrete inputs. Use pandas qcut on sj.
            mi = mutual_info_score(Y_np, pd.qcut(sj_np, q=10, duplicates="drop"))
            C_scores.append(corr - mi)
        C_scores = np.asarray(C_scores)
        sel = np.where(C_scores > np.percentile(C_scores, 90))[0]  # top-10 % as spurious
        return torch.as_tensor(sel, dtype=torch.long)

    # ---------------------------------------------------------------------
    # 2) Counterfactual synthesis (coarse full-image in-paint for speed)
    # ---------------------------------------------------------------------
    def _gen_counterfactual_batch(self, x: torch.Tensor, attr_idx: torch.Tensor) -> torch.Tensor:
        # Convert indices tensor to Python ints and craft negative prompt list
        neg_prompt = [PROMPT_BANK[int(i)].replace("a photo of ", "remove ") for i in attr_idx.tolist()]
        cf_imgs = []
        # Denormalise once for PIL conversion
        x_vis = self._unnormalize(x).clamp(0, 1)
        with torch.autocast("cuda", enabled=AMP_ENABLED):
            for img in x_vis:
                pil = T.ToPILImage()(img.cpu())
                mask = Image.new("L", pil.size, color=255)  # white mask ⇒ full image
                out = self.inp_pipe(prompt="", negative_prompt=neg_prompt, image=pil, mask_image=mask).images[0]
                cf_imgs.append(_val_tf(out))
        return torch.stack(cf_imgs).to(DEVICE)

    # ---------------------------------------------------------------------
    # single optimisation step
    # ---------------------------------------------------------------------
    def _train_step(self, x: torch.Tensor, y: torch.Tensor, x_cf: torch.Tensor):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        with amp.autocast(enabled=AMP_ENABLED):
            logit_o, feat_o = self.model(x, return_feat=True)
            logit_c, feat_c = self.model(x_cf, return_feat=True)
            ce = self.criterion(logit_o, y) + self.criterion(logit_c, y)
            inv_f = (feat_o - feat_c).pow(2).mean()
            inv_p = (logit_o - logit_c).abs().mean()
            loss = ce + self.lambda1 * inv_f + self.lambda2 * inv_p
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item(), ce.item(), inv_f.item(), inv_p.item()

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    def fit(self, train_dl: torch.utils.data.DataLoader, val_dl: torch.utils.data.DataLoader, epochs: int):
        print(f"==========  PCCM training ({self.ds_name})  ==========")
        from .evaluate import evaluate_model  # local import to avoid circular

        best_model_w = None
        patience, patience_cnt = 3, 0

        # initial mining before first epoch
        spurious_attr_idx = self._mine_spurious_attrs(train_dl)

        for epoch in range(epochs):
            losses = []
            for x, y, _ in tqdm(train_dl, desc=f"Epoch {epoch}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                x_cf = self._gen_counterfactual_batch(x, spurious_attr_idx)
                losses.append(self._train_step(x, y, x_cf))

            val_acc, val_wg = evaluate_model(self.model, val_dl)
            print(f"Epoch {epoch}  val Acc={val_acc:.2f}  val WGAcc={val_wg:.2f}")

            if val_wg > self.best_val_wgacc + 1e-3:
                self.best_val_wgacc = val_wg
                best_model_w = self.model.state_dict()
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                break

            if (epoch + 1) % self.K == 0:
                spurious_attr_idx = self._mine_spurious_attrs(train_dl)

        if best_model_w is None:
            raise RuntimeError("Training failed to improve WGAcc even once!")
        self.model.load_state_dict(best_model_w)

    # ---------------------------------------------------------------------
    # external evaluation helper used by main scripts
    # ---------------------------------------------------------------------
    def test(self, test_dl: torch.utils.data.DataLoader) -> Tuple[float, float]:
        from .evaluate import evaluate_model  # avoid circular

        return evaluate_model(self.model, test_dl)
