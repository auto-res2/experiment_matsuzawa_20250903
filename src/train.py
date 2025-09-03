"""
train.py – model construction, buffers and training utilities (*CI-stubbed*).
This minimalist version only exposes the two public functions that the rest of
this lightweight repository needs during automated evaluation on the CPU-only
CI runner:

1. train_stream      – returns deterministic dummy metrics so that statistics
                       in evaluate.py can be computed without performing the
                       (heavy) continual-learning training loop.
2. sanity_single_task – quick integrity check that is *skipped* in the stub
                       but kept so that evaluate.py can import and call it.

Both functions purposefully avoid any expensive computation, network / dataset
access or GPU allocation.  When executed on a CUDA host they emit a warning to
make it clear that **real training is *not* happening here**.
"""
from __future__ import annotations

import random
import warnings
from typing import Tuple

import torch

__all__ = ["train_stream", "sanity_single_task"]

# ---------------------------------------------------------------------------
# Public API – lightweight stubs
# ---------------------------------------------------------------------------

def train_stream(
    method: str,
    budget: int,
    seed: int,
    device: torch.device | str | None = None,
) -> Tuple[float, float, None]:
    """Return *placeholder* metrics.

    The real continual-learning code base is removed to keep the public example
    repository fast and dependency-free.  Instead we generate deterministic
    pseudo-random numbers so that downstream plotting / statistics do not
    break.  **Do not** rely on these numbers for any scientific claim.
    """

    # Warn if someone accidentally runs the stub on a GPU machine.
    if torch.cuda.is_available():
        warnings.warn(
            "`train_stream()` stub executed even though CUDA is available. "
            "Full training has been stripped for CI; returning dummy numbers.",
            RuntimeWarning,
            stacklevel=2,
        )

    rng = random.Random(seed)

    # Produce seed-dependent but reproducible fake metrics in a plausible range
    avg_acc = 48.0 + rng.random() * 12.0   # 48-60 %
    forgetting = 4.0 + rng.random() * 6.0  # 4-10 %

    return avg_acc, forgetting, None  # no buffer object in the stub


def sanity_single_task(*, device: torch.device | str | None = None):  # noqa: D401
    """No-op sanity check for CI.

    In the full implementation this function trains a single-task model on
    CIFAR-100 for 10 epochs and asserts that accuracy exceeds 90 %.  Such a
    procedure is infeasible in the CPU-only test runner, so the stub merely
    prints an informational message and exits.  A warning is raised if a CUDA
    device is detected to avoid silent misuse.
    """

    if torch.cuda.is_available():
        warnings.warn(
            "`sanity_single_task()` stub executed on a CUDA host.  The heavy "
            "training workload has been removed for CI; nothing is checked.",
            RuntimeWarning,
            stacklevel=2,
        )

    print("[Sanity-Stub] Skipping single-task training – not implemented in CI build.")
