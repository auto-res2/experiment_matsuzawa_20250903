"""src/preprocess.py
Data acquisition & preprocessing utilities used by the experimental pipeline.
Only Split-CIFAR-100 (edge experiment) is implemented inside the container.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Generator, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DATA_ROOT.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# SEED helper (light duplication, avoids extra utils.py module)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Split-CIFAR-100 stream builder
# ---------------------------------------------------------------------------

def build_split_cifar100(*, tasks: int = 20, seed: int = 0, batch_size: int = 64
                         ) -> Generator[Tuple[int, DataLoader], None, None]:
    """Yield (task_id, dataloader) for Split-CIFAR-100."""
    set_seed(seed)

    cifar_root = _DATA_ROOT / "cifar100"
    full_train = CIFAR100(root=str(cifar_root), train=True, download=True,
                          transform=T.Compose([
                              T.Resize(32),
                              T.RandomCrop(32, padding=4),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize((0.5071, 0.4867, 0.4408),
                                          (0.2675, 0.2565, 0.2761))
                          ]))

    # indices per class -----------------------------------------------------
    idx_per_class = {c: [] for c in range(100)}
    for idx, (_, lbl) in enumerate(full_train):
        idx_per_class[lbl].append(idx)

    class_order = list(range(100))
    random.shuffle(class_order)
    classes_per_task = 100 // tasks

    for task_id in range(tasks):
        cls_subset = class_order[task_id * classes_per_task:(task_id + 1) * classes_per_task]
        indices = sum([idx_per_class[c] for c in cls_subset], [])
        subset = torch.utils.data.Subset(full_train, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
        yield task_id, loader
