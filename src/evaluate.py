"""src/evaluate.py – evaluation helpers, metrics and unit tests"""
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

# *No heavy modules are imported here to keep evaluation lightweight*

# -------------------------------------------------------------
#                     Generic evaluation routine
# -------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    good, tot = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        good += (model(x).argmax(dim=1) == y).sum().item()
        tot += y.size(0)
    return 100 * good / max(tot, 1)


# -------------------------------------------------------------
#                         Stream-level meter
# -------------------------------------------------------------
@dataclass
class StreamMeter:
    """Accumulates accuracy matrix & memory history for a task stream."""

    acc_tasks: List[List[float]] = field(default_factory=list)  # acc[t][k]
    mem_hist: List[int] = field(default_factory=list)  # total bytes after each task

    # ------------------------------------------------------------------
    def record_after_task(self, model, test_loaders, device):
        acc_row = [evaluate(model, l, device) for l in test_loaders]
        self.acc_tasks.append(acc_row)
        # adapter + buffer
        self.mem_hist.append(model.ledger.A + model.ledger.B)

    # ------------------------ derived metrics -------------------------
    def A_T(self):
        return float(sum(self.acc_tasks[-1]) / len(self.acc_tasks[-1]))

    def F_max(self):
        forget = []
        for k in range(len(self.acc_tasks[0])):
            best = max(row[k] for row in self.acc_tasks)
            forget.append(best - self.acc_tasks[-1][k])
        return float(max(forget))

    def auc_mem(self):
        xs = np.arange(len(self.mem_hist))
        return float(np.trapz(self.mem_hist, xs))


# -------------------------------------------------------------
#                      Lightweight unit tests
# -------------------------------------------------------------

def _test_byte_ledger(ByteLedger):
    led = ByteLedger(64)  # 64 kB cap
    led.update(10, 20)
    led.move("B", 10)
    assert led.A == 20 and led.B == 10


def _test_adapter_roundtrip(LoRAAdapter):
    ad = LoRAAdapter(4, 3, 2)
    w_before = (ad.weight_A @ ad.weight_B).clone()
    ad.prune(1)
    ad.grow(1)
    w_after = ad.weight_A @ ad.weight_B
    assert torch.allclose(w_before, w_after, atol=1e-4)


def _test_vq_roundtrip(device):
    from vq_vae_2_pytorch import VQVAETwo

    vq = VQVAETwo(img_size=32, num_layers=2, codebook_dim=64).to(device)
    x = torch.randn(1, 3, 32, 32, device=device)
    _, _, lat = vq.encode(x)[2]
    recon = vq.decode(lat)
    mse = (x - recon).pow(2).mean().item()
    assert mse < 1e-2


def run_unit_tests(ByteLedger, LoRAAdapter, device):
    """Run a quick battery of sanity checks (fail-fast)."""

    _test_byte_ledger(ByteLedger)
    _test_adapter_roundtrip(LoRAAdapter)
    _test_vq_roundtrip(device)
    print("[tests] all sanity-checks passed")
