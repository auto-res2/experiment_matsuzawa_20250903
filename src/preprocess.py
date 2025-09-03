"""src/preprocess.py
Dataset utilities: downloading, extraction, and continual-learning task
splits used in the experiments.
"""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
from torch.utils.data import Subset, Dataset  # noqa: F401 – Subset used
from torchvision import datasets, transforms

import yaml

# ---------------------------------------------------------------------------
# 1.  Configuration – load YAML once
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with CONFIG_PATH.open("r") as _f:
    CONFIG = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# 2.  Data root
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# 3.  Download helpers (generic – may be unused for CIFAR)
# ---------------------------------------------------------------------------

def _download(url: str, target: Path) -> None:
    print(f"[DOWNLOAD] {url} -> {target}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total) if total else 0
                        sys.stdout.write("\r[{:<50s}] {:>6.2f}%".format("■" * done, downloaded * 100 / total if total else 0))
                        sys.stdout.flush()
        sys.stdout.write("\n")
    except Exception as e:
        print("[ERROR] Download failed:", e)
        if target.exists():
            target.unlink(missing_ok=True)
        raise


def _maybe_download_and_extract(key: str) -> Path:
    entry = CONFIG["datasets"][key]
    url = entry["url"]
    fname = url.split("/")[-1]
    download_path = DATA_ROOT / fname
    extract_dir = DATA_ROOT / key
    if extract_dir.exists():
        return extract_dir
    if not download_path.exists():
        _download(url, download_path)
    print(f"[EXTRACT] {download_path} -> {extract_dir}")
    extract_dir.mkdir(exist_ok=True)
    if fname.endswith((".tar.gz", ".tgz")):
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(extract_dir)
    elif fname.endswith(".zip"):
        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive type: {fname}")
    return extract_dir

# ---------------------------------------------------------------------------
# 4.  Continual-learning task generators (CIFAR-100 only for brevity)
# ---------------------------------------------------------------------------

def get_cifar100_tasks(seed: int = 0, tasks: int = 10, classes_per_task: int = 10, train: bool = True):
    """Return list[torch.utils.data.Subset] – each subset is one task."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = datasets.CIFAR100(root=str(DATA_ROOT), train=train, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    class_order = list(range(100))
    rng.shuffle(class_order)

    task_datasets: List[Subset] = []
    for t in range(tasks):
        cls = class_order[t * classes_per_task : (t + 1) * classes_per_task]
        indices = [i for i, (_, y) in enumerate(dataset) if y in cls]
        task_datasets.append(Subset(dataset, indices))
    return task_datasets
