"""
preprocess.py – data download & preprocessing utilities
Only Waterbirds is implemented for brevity – extend identically for other datasets.
"""
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Optional – HuggingFace datasets
HF_AVAILABLE = False
try:
    from datasets import load_dataset  # noqa: F401
    HF_AVAILABLE = True
except ImportError:
    print("[Warning] huggingface-datasets not installed – Waterbirds download will fail.")

# -----------------------------------------------------------------------------
# Transforms (ImageNet statistics)
# -----------------------------------------------------------------------------
ImageTransformTrain = transforms.Compose([
    transforms.Resize(288),
    transforms.RandomResizedCrop(224),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ImageTransformEval = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -----------------------------------------------------------------------------
# Waterbirds dataset wrapper
# -----------------------------------------------------------------------------
class WaterbirdsDataset(Dataset):
    """Loads Waterbirds splits from the HuggingFace hub. Fails clearly if unavailable."""

    def __init__(self, split: str, root: str = "data", transform=None):
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface-datasets missing – install to use Waterbirds.")
        ds = load_dataset("grodino/waterbirds", split=split, cache_dir=root)

        # The official dataset uses the singular key "label" – fall back to that
        # if the pluralised variant is absent. This keeps backward-compatibility
        # with any older cached versions that might still expose "labels".
        self.labels = ds["labels"] if "labels" in ds.column_names else ds["label"]
        self.images = ds["image"]
        # group label (bird × background) for worst-group accuracy
        self.groups = list(zip(ds["y"], ds["place"]))
        self.transform = transform or ImageTransformEval

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].convert("RGB")
        y = int(self.labels[idx])
        g = self.groups[idx]
        if self.transform:
            img = self.transform(img)
        return img, y, g


# -----------------------------------------------------------------------------
# Convenience loaders
# -----------------------------------------------------------------------------

def get_waterbirds_splits(root: str = "data",
                          transform_train=None,
                          transform_eval=None) -> Tuple[Dataset, Dataset, Dataset]:
    t_tr = transform_train or ImageTransformTrain
    t_ev = transform_eval or ImageTransformEval
    train = WaterbirdsDataset("train", root=root, transform=t_tr)
    val = WaterbirdsDataset("validation", root=root, transform=t_ev)
    test = WaterbirdsDataset("test", root=root, transform=t_ev)
    return train, val, test
