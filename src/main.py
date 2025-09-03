# src/main.py
# -*- coding: utf-8 -*-
"""Main orchestration script – must be executed via `python -m src.main`."""
from __future__ import annotations
import json, time
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import timm

from .preprocess import WaterbirdsDataset, MiniImageNet9, get_transforms
from .train import Trainer, seed_everything
from .evaluate import save_line_plot

# -----------------------------------------------------------------------------
#  Paths & configuration
# -----------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE} (torch {torch.__version__})")

SEED_LIST = {"exp1": [0, 1, 2], "exp2": [3, 4, 5], "exp3": [42]}

# -----------------------------------------------------------------------------
#  Experiment 1 – Bias-conflict accuracy (Waterbirds)
# -----------------------------------------------------------------------------

def experiment1():
    print("\n================ Experiment-1 : Bias-Conflict Accuracy ==================")
    cfg_ds = CONFIG["datasets"]["waterbirds"]
    img_size = int(cfg_ds["img_size"])
    tf_train = get_transforms(img_size, train=True)
    tf_val   = get_transforms(img_size, train=False)

    ds_train = WaterbirdsDataset("train", tf_train)
    ds_val   = WaterbirdsDataset("val", tf_val)
    dl_train = DataLoader(ds_train, batch_size=CONFIG["common"]["batch_size"]["exp1"],
                           shuffle=True, num_workers=CONFIG["common"]["num_workers"], pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=256, shuffle=False,
                           num_workers=CONFIG["common"]["num_workers"], pin_memory=True)

    for seed in SEED_LIST["exp1"]:
        seed_everything(seed)
        print(f"--- Seed {seed} ---")
        model = timm.create_model(CONFIG["models"]["vit_s16"]["timm_id"],
                                  pretrained=True, num_classes=2)
        optimiser = optim.AdamW(model.parameters(), lr=CONFIG["optim"]["vit"]["lr"],
                                weight_decay=CONFIG["optim"]["vit"]["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CONFIG["optim"]["vit"]["epochs"])
        trainer = Trainer(model, optimiser, scheduler, precision=CONFIG["common"]["precision"])
        hist = trainer.fit(dl_train, dl_val, epochs=5, exp_name=f"ERM_seed{seed}")

        fig_name = save_line_plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"],
                                  "epoch", "train loss", f"ERM (seed {seed}) – Waterbirds",
                                  f"training_loss_erm_seed{seed}.pdf")
        print("Experiment description: Waterbirds ERM baseline")
        print("Numerical results:\n", json.dumps(hist, indent=2))
        print("Figures:", fig_name)

# -----------------------------------------------------------------------------
#  Experiment 2 – Domain & corruption robustness (mini-ImageNet-9)
# -----------------------------------------------------------------------------

def experiment2():
    print("\n================ Experiment-2 : Domain & Corruption =====================")
    cfg_ds = CONFIG["datasets"]["mini_imagenet9"]
    img_size = int(cfg_ds["img_size"])
    tf_train = get_transforms(img_size, train=True)
    tf_val   = get_transforms(img_size, train=False)

    ds_train = MiniImageNet9("train", tf_train)
    ds_val   = MiniImageNet9("val", tf_val)
    dl_train = DataLoader(ds_train, batch_size=CONFIG["common"]["batch_size"]["exp2"],
                           shuffle=True, num_workers=CONFIG["common"]["num_workers"], pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=256, shuffle=False,
                           num_workers=CONFIG["common"]["num_workers"], pin_memory=True)

    seed_everything(SEED_LIST["exp2"][0])
    model = timm.create_model(CONFIG["models"]["resnet18"]["timm_id"], pretrained=True, num_classes=9)
    optimiser = optim.SGD(model.parameters(), lr=CONFIG["optim"]["resnet"]["lr"],
                          momentum=CONFIG["optim"]["resnet"]["momentum"],
                          weight_decay=CONFIG["optim"]["resnet"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CONFIG["optim"]["resnet"]["epochs"])
    trainer = Trainer(model, optimiser, scheduler, precision=CONFIG["common"]["precision"])
    hist = trainer.fit(dl_train, dl_val, epochs=5, exp_name="ResNet18-ERM-miniIN9")

    fig_name = save_line_plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"],
                              "epoch", "train loss", "ResNet-18 ERM mini-IN-9",
                              "training_loss_resnet18_erm.pdf")
    print("Experiment description: mini-ImageNet-9 ERM baseline – corruption eval TBD")
    print("Numerical results:\n", json.dumps(hist, indent=2))
    print("Figures:", fig_name)

# -----------------------------------------------------------------------------
#  Experiment 3 – Causal fidelity demo (LPIPS)
# -----------------------------------------------------------------------------

def experiment3():
    print("\n================ Experiment-3 : Causal Fidelity =========================")
    from lpips import LPIPS
    lpips_metric = LPIPS(net="alex").to(DEVICE)
    dummy_img = torch.rand(1, 3, 224, 224).to(DEVICE)
    d = lpips_metric(dummy_img, dummy_img)
    fig_name = save_line_plot(["self"], [d.item()], "condition", "LPIPS", "Causal fidelity demo",
                              "fidelity_demo.pdf")
    print("LPIPS self-similarity (sanity): %.4f" % d.item())
    print("Figures:", fig_name)

# -----------------------------------------------------------------------------
#  Main entry point
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    experiment1()
    experiment2()
    experiment3()
    print(f"All experiments finished in {(time.time() - t0) / 60:.1f} min.")

if __name__ == "__main__":
    main()
