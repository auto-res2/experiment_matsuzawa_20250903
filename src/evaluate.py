"""
evaluate.py – evaluation utilities & plotting
"""
from __future__ import annotations
import os, json
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IMG_DIR = os.path.join(".research", "iteration1", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def accuracy(net: torch.nn.Module, dataset, device: str) -> float:
    net.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=256):
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total


def delta_prob(net, sample_ds, dirs, spurious_idx: List[int], cdg, device: str) -> float:
    """Reliance measure ∆Prob for first spurious direction."""
    dp = []
    dir_vec = dirs[spurious_idx[0]].to(device)
    latent_delta = dir_vec.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    loader = DataLoader(sample_ds, batch_size=64)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p0 = F.softmax(net(x), 1)
            x_cf = cdg.generate(x, latent_delta).to(device)
            p1 = F.softmax(net(x_cf), 1)
            dp.extend((p0 - p1).abs().max(1)[0].cpu().tolist())
    return sum(dp) / len(dp)


def save_bar_figure(val: float, title: str, fname: str):
    plt.figure(figsize=(4, 4))
    sns.barplot(x=[title], y=[val])
    plt.ylim(0, 100)
    plt.text(0, val + 0.5, f"{val:.1f}%", ha="center")
    plt.title(title)
    fpath = os.path.join(IMG_DIR, fname)
    plt.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close()
    return fpath
