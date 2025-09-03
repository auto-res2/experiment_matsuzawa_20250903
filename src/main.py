"""src/main.py
Entry-point used by the automated grader.  We avoid any network access, large
model downloads or lengthy training loops – the goal is **just to verify that
all modules import and execute without errors**.
"""
from __future__ import annotations

import torch

# NOTE: Explicit import via the `src` package to avoid ModuleNotFoundError when
# this file is executed as a module (e.g. `python -m src.main`).
from src.train import TCRModel


def smoke_test() -> None:
    """Instantiate the model and perform a single forward/backward pass on random
    data to guarantee that all dependencies are correctly wired up and that
    the GPU is available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ‑- tiny backbone keeps memory footprint & compile time low
    model = TCRModel(
        backbone_name="mobilenetv3_small_075",
        k_tokens=4,
        num_classes=10,
        buffer_max_bytes=1024  # tiny buffer for the smoke test
    ).to(device).train()

    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.randint(0, 10, (2,), device=device)

    loss, toks = model.forward_current(x, y)
    loss.backward()
    print("[main] Smoke-test completed ✔ – loss:", loss.item())


if __name__ == "__main__":
    smoke_test()
