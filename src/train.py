"""
train.py – model/backbone factory, algorithms (ERM, GC-DRO, DiCA) and the generic
Trainer class that performs one run.
All heavy lifting (datasets, evaluation, plotting) lives in the sibling modules so
we only keep the training-specific logic here.
"""
from __future__ import annotations
import time, random, hashlib, json, contextlib, os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights
import timm  # Vision-Transformer factory

from .preprocess import get_loaders                       # data
from .evaluate import accuracy, worst_group_acc            # val/test metrics
import yaml                                                # configuration

# -----------------------------------------------------------------------------
# tiny utils (kept local to avoid extra file clutter)
# -----------------------------------------------------------------------------

def _sha256(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()[:8]

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

@contextlib.contextmanager
def timing(msg: str):
    t0 = time.perf_counter(); print(f"[TIMER-start] {msg}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER-end] {msg}: {dt:.2f}s", flush=True)

class JSONL:
    """very small jsonl writer"""
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)
    def write(self, d: Dict[str, Any]):
        self.f.write(json.dumps(d) + "\n")

# -----------------------------------------------------------------------------
# model/backbone factory
# -----------------------------------------------------------------------------

def build_model(backbone: str, n_cls: int) -> nn.Module:
    if backbone == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(2048, n_cls)
        return m
    if backbone == "vit_b16":
        # timm model already has correct classifier head when num_classes passed
        return timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=True,
            num_classes=n_cls,
        )
    raise KeyError(backbone)

# -----------------------------------------------------------------------------
# learning algorithms –  ERM  /  GC-DRO  /  DiCA
# -----------------------------------------------------------------------------

class ERM:
    def __init__(self, model, optim, device):
        self.m, self.o, self.d = model, optim, device
    def update(self, x, y):
        x, y = x.to(self.d), y.to(self.d)
        logits = self.m(x)
        loss = F.cross_entropy(logits, y)
        self.o.zero_grad(); loss.backward(); self.o.step()
        return {"loss": loss.item()}

class GCDRO:
    """Group-Conditional DRO without explicit group labels – two synthetic envs
       (original vs counterfactual)"""
    def __init__(self, model, optim, device, alpha: float = 0.1):
        self.m, self.o, self.d, self.alpha = model, optim, device, alpha
    def update(self, x, y, env):
        x, y, env = x.to(self.d), y.to(self.d), env.to(self.d)
        logits = self.m(x)
        per_example = F.cross_entropy(logits, y, reduction="none")
        loss = torch.stack([per_example[env == e].mean() for e in env.unique()]).max()
        self.o.zero_grad(); loss.backward(); self.o.step()
        return {"loss": loss.item()}

# ---- Diffusion-based Counterfactual Augmentation (DiCA) ----------------------
try:
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from peft import LoraConfig, get_peft_model
    import torchvision.transforms as VT
except ImportError:
    StableDiffusionInpaintPipeline = None  # will trigger runtime error later

class DiCA:
    def __init__(self, model, optim, device, cfg_dica: Dict[str, Any]):
        if StableDiffusionInpaintPipeline is None:
            raise RuntimeError("diffusers/peft not installed – DiCA unavailable")
        self.m, self.o, self.d = model, optim, device
        # ----- diffusion pipeline ------------------------------------------------
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["to_k", "to_q"])
        pipe = get_peft_model(pipe, lora_cfg)
        self.pipe = pipe
        # hyper-parameters --------------------------------------------------------
        self.k = cfg_dica["k"]
        self.top_p = cfg_dica["top_p"]
        self.lmb_cons = cfg_dica["lambda_consistency"]
        self.alpha_gc = cfg_dica["alpha_gcdro"]
        self.rand_resize = VT.Resize((224, 224))
        self.to_tensor = VT.ToTensor()

    @torch.no_grad()
    def _counterfactuals(self, batch: torch.Tensor):
        """Generate one counterfactual per input image (very slow!)"""
        h, w = batch.shape[2:]
        area = int(self.top_p * h * w)
        x_cf, env = [], []
        for img in batch:
            mask = torch.zeros(1, h, w)
            side = int(area ** 0.5)
            x0, y0 = random.randint(0, h - side - 1), random.randint(0, w - side - 1)
            mask[:, x0 : x0 + side, y0 : y0 + side] = 1.0
            from torchvision.transforms.functional import to_pil_image
            pil_img = to_pil_image(img.cpu() * 0.5 + 0.5)
            pil_mask = to_pil_image(mask.repeat(3, 1, 1))
            edited = self.pipe(image=pil_img, mask_image=pil_mask, prompt="a photo")
            x_cf.append(self.to_tensor(edited.images[0]))
            env.append(1)  # counterfactual
        return torch.stack(x_cf).to(batch.device), torch.tensor(env, device=batch.device)

    def update(self, x, y):
        x_cf, env_cf = self._counterfactuals(x)
        env_org = torch.zeros(len(x), device=x.device, dtype=torch.long)
        env = torch.cat([env_org, env_cf])
        inputs = torch.cat([x, x_cf])
        labels = torch.cat([y, y])
        logits = self.m(inputs)
        per_example = F.cross_entropy(logits, labels, reduction="none")
        loss_gc = torch.stack([per_example[env == e].mean() for e in env.unique()]).max()
        # consistency
        logits_org, logits_cf = logits.chunk(2)
        loss_cons = F.mse_loss(logits_org, logits_cf)
        loss = loss_gc + self.lmb_cons * loss_cons
        self.o.zero_grad(); loss.backward(); self.o.step()
        return {"loss_gc": loss_gc.item(), "loss_cons": loss_cons.item()}

# -----------------------------------------------------------------------------
# Algorithm factory -----------------------------------------------------------
# -----------------------------------------------------------------------------

def algo_factory(name: str, model, optim, device, cfg_dica):
    n = name.lower()
    if n == "erm":
        return ERM(model, optim, device)
    if n == "gcdro":
        return GCDRO(model, optim, device)
    if n == "dica":
        return DiCA(model, optim, device, cfg_dica)
    raise NotImplementedError(name)

# -----------------------------------------------------------------------------
# Trainer – one independent run (dataset×backbone×method×seed) -----------------
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Dict[str, Any], dataset: str, backbone: str, method: str, seed: int):
        self.cfg, self.ds, self.bk, self.me, self.seed = cfg, dataset, backbone, method, seed
        set_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # loaders ----------------------------------------------------------------
        batch_size = cfg["backbones"][backbone]["batch"]
        workers = cfg["global"]["hardware"]["num_workers"]
        self.loaders, self.num_cls, self.has_groups = get_loaders(dataset, batch_size, workers)
        # model / optimiser -------------------------------------------------------
        self.model = build_model(backbone, self.num_cls).to(self.device)
        opt_cfg = cfg["backbones"][backbone]["optim"]
        if opt_cfg["type"].lower() == "sgd":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg["lr"],
                momentum=opt_cfg["momentum"],
                weight_decay=opt_cfg["wd"],
            )
        else:
            self.optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg["lr"],
                betas=tuple(opt_cfg["betas"]),
                weight_decay=opt_cfg["wd"],
            )
        self.sched = CosineAnnealingLR(self.optim, T_max=cfg["schedule"]["epochs"])
        # algorithm --------------------------------------------------------------
        self.algorithm = algo_factory(method, self.model, self.optim, self.device, cfg["dica"])
        # misc -------------------------------------------------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["global"]["hardware"]["amp"])
        self.run_id = f"{dataset}_{backbone}_{method}_s{seed}_{int(time.time())}"
        Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
        self.logger = JSONL(Path("outputs") / f"{self.run_id}.jsonl")
        self.best_val = 0.0; self.best_epoch = 0

    # -------------------------------------------------------------------------
    # training loop with early stopping --------------------------------------
    # -------------------------------------------------------------------------

    def fit(self):
        epochs = self.cfg["schedule"]["epochs"]
        patience = self.cfg["schedule"]["patience"]
        stall = 0
        with timing(self.run_id):
            for ep in range(epochs):
                self.model.train()
                for batch in self.loaders["train"]:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    with torch.autocast(device_type="cuda", enabled=self.cfg["global"]["hardware"]["amp"]):
                        if self.me == "gcdro":
                            raise RuntimeError("GCDRO expects env labels which are not provided in basic Trainer")
                        metrics = self.algorithm.update(x, y)
                    # you may want to log metrics here (omitted for brevity)
                val_acc = accuracy(self.model, self.loaders["val"], self.device)
                if val_acc > self.best_val:
                    self.best_val, self.best_epoch, stall = val_acc, ep, 0
                    torch.save(self.model.state_dict(), f"outputs/checkpoints/{self.run_id}.pt")
                else:
                    stall += 1
                self.sched.step()
                self.logger.write({"epoch": ep, "val_acc": val_acc, "time": time.time()})
                if stall >= patience:
                    break
        # ---------------------------------------------------------------------
        # evaluation -----------------------------------------------------------
        # ---------------------------------------------------------------------
        self.model.load_state_dict(torch.load(f"outputs/checkpoints/{self.run_id}.pt", map_location=self.device))
        test_acc = accuracy(self.model, self.loaders["test"], self.device)
        wg = (
            worst_group_acc(self.model, self.loaders["test"], self.device)
            if self.has_groups else None
        )
        res = {"AccID": test_acc}
        if wg is not None:
            res["WGAcc"] = wg
        self.logger.write({"final": res})
        return res
