"""
main.py – orchestrates the full Waterbirds experiment
USAGE:  python -m src.main   (must be executed from project root)
"""
from __future__ import annotations
import json, random, pathlib
from typing import Any, Dict, List

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from .preprocess import download_url, DATA_ROOT, WaterbirdsDataset
from .train import (BYOL, amplify_latent_directions, riesz_total_effect, CDGPipeline,
                    FourierLoss, irm_penalty, set_seed)
from .evaluate import accuracy, delta_prob, save_bar_figure

CONFIG_PATH = pathlib.Path("config/config.yaml")
CFG: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
SEED_LIST: List[int] = [0, 1, 2]

def run_single(seed: int):
    device = CFG["global"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # -------------------------   data prep   -------------------------
    img_size = CFG["datasets"]["waterbirds"]["img_size"]
    t_train_ssl = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(0.2), transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # download & extract Waterbirds (fail-fast)
    wb_cfg = CFG["datasets"]["waterbirds"]
    tar_path = DATA_ROOT / "waterbirds.tar.gz"
    download_url(wb_cfg["url"], tar_path, wb_cfg["compressed_md5"])
    if not (DATA_ROOT / "waterbird_complete95_forest2water2").exists():
        import tarfile
        with tarfile.open(tar_path) as t:
            t.extractall(DATA_ROOT)

    ds_train = WaterbirdsDataset(str(DATA_ROOT), "train", transform=t_train_ssl)
    ds_val = WaterbirdsDataset(str(DATA_ROOT), "val", transform=t_eval)
    ds_test = WaterbirdsDataset(str(DATA_ROOT), "test", transform=t_eval)

    # --------------------   self-supervised BYOL   -------------------
    ssl_cfg = CFG["models"]["resnet18_ssl"]
    byol = BYOL(models.resnet18()).to(device)
    opt_ssl = torch.optim.AdamW(byol.parameters(), lr=ssl_cfg["lr"])
    dl_ssl = DataLoader(ds_train, batch_size=ssl_cfg["batch_size"], shuffle=True,
                        num_workers=CFG["global"]["num_workers"], drop_last=True)
    for ep in range(ssl_cfg["epochs"]):
        for x, _ in dl_ssl:
            x1 = x.to(device)
            x2 = x1[torch.randperm(x1.size(0))]
            loss = byol(x1, x2)
            loss.backward(); opt_ssl.step(); opt_ssl.zero_grad(); byol._update_target()
        if ep % 10 == 0:
            print(f"[BYOL] epoch {ep} loss {loss.item():.3f}")

    embeds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds_train, batch_size=256):
            embeds.append(byol.online_encoder(x.to(device)).cpu())
    embeds = torch.cat(embeds)
    amplified = amplify_latent_directions(embeds, r=8)
    X = embeds - embeds.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    dirs = Vt[:8]

    # ---------------   causal effect (2% labels)   -------------------
    k = int(0.02 * len(ds_train))
    idx_lab = random.sample(range(len(ds_train)), k)
    Z = amplified[idx_lab] @ dirs.T
    y_lab = torch.tensor([ds_train[i][1] for i in idx_lab], dtype=torch.float32)
    w_te = riesz_total_effect(Z, y_lab)
    S_idx = torch.where(w_te.abs() > w_te.abs().mean())[0].tolist()
    if not S_idx:
        raise RuntimeError("No spurious directions detected – abort.")
    print("[INFO] Spurious directions:", S_idx)

    # ----------------------   CDG fine-tune   ------------------------
    cdg_cfg = CFG["cdg"]
    cdg = CDGPipeline(cdg_cfg, device)
    subset_5k, _ = random_split(ds_train, [min(5000, len(ds_train)), len(ds_train) - min(5000, len(ds_train))])
    dl_cdg = DataLoader(subset_5k, batch_size=cdg_cfg["train_batch"], shuffle=True, num_workers=2)
    cdg.fine_tune(dl_cdg, cdg_cfg["train_epochs"], cdg_cfg["lr"])

    # -------------------   build environments   ----------------------
    class EnvDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, dirs, s_idx):
            self.base_ds, self.dirs, self.s_idx = base_ds, dirs, s_idx
        def __len__(self):
            return len(self.base_ds) * (1 + len(self.s_idx))
        def __getitem__(self, idx):
            base_idx = idx % len(self.base_ds); env = idx // len(self.base_ds)
            x, y = self.base_ds[base_idx]
            if env == 0:
                return x, y, 0
            dir_vec = self.dirs[self.s_idx[env - 1]].to(device)
            latent_delta = dir_vec.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            with torch.no_grad():
                x_cf = cdg.generate(x.unsqueeze(0).to(device), latent_delta)[0].cpu()
            return x_cf, y, env

    env_ds = EnvDataset(ds_train, dirs, S_idx)
    clf_cfg = CFG["models"]["classifier_resnet18"]
    dl_env = DataLoader(env_ds, batch_size=clf_cfg["batch_size"], shuffle=True, num_workers=4)

    # ------------------   invariant classifier   ---------------------
    net = models.resnet18(num_classes=2).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=clf_cfg["lr"])
    criterion = torch.nn.CrossEntropyLoss(); four_loss = FourierLoss()

    for ep in range(clf_cfg["epochs"]):
        for x, y, e in dl_env:
            x, y, e = x.to(device), y.to(device), e.to(device)
            logits = net(x); loss_ce = criterion(logits, y)
            irm_sum = 0.0
            for env in e.unique():
                idx = e == env
                irm_sum += irm_penalty(criterion(logits[idx], y[idx]), net)
            loss_f = four_loss(x[e == 0], x[e > 0])
            loss = loss_ce + clf_cfg["irm_lambda"] * irm_sum + clf_cfg["fourier_lambda"] * loss_f
            loss.backward(); opt.step(); opt.zero_grad()
        if ep % 10 == 0:
            print(f"[INV] epoch {ep} loss {loss.item():.3f}")

    # ----------------------   evaluation   ---------------------------
    acc = accuracy(net, ds_test, device)
    sample_ds, _ = random_split(ds_test, [min(2000, len(ds_test)), len(ds_test) - min(2000, len(ds_test))])
    dprob = delta_prob(net, sample_ds, dirs, S_idx, cdg, device)
    fig_path = save_bar_figure(acc, "Test Acc", f"accuracy_seed{seed}.pdf")

    result = {"seed": seed, "dataset": "Waterbirds", "test_accuracy": acc, "delta_prob": dprob, "figure": fig_path}
    print("===== EXPERIMENT – WATERBIRDS =====")
    print(json.dumps(result, indent=2))
    print("Figure saved to", fig_path)


def main():
    print("============= GCDI (refactored) =============")
    for s in SEED_LIST:
        run_single(s)

if __name__ == "__main__":
    main()
