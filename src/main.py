from __future__ import annotations
"""src/main.py
Entry-point: `python -m src.main`
Sets up the runtime environment, loads the YAML config and orchestrates the
whole experimental workflow via the sub-modules defined in this package.
"""
import json, random, time
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch
import torch.optim as optim

# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Missing config/config.yaml – please provide it before running.")

with open(CONFIG_PATH) as f:
    CFG = SimpleNamespace(**yaml.safe_load(f))

# -----------------------------------------------------------------------------
# Runtime directories
# -----------------------------------------------------------------------------
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / ".research" / "iteration3" / "images"  # updated path per instructions
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Seeding – reproducibility first
# -----------------------------------------------------------------------------
SEED_LIST = CFG.meta["seeds"]
random.seed(SEED_LIST[0]); torch.manual_seed(SEED_LIST[0]); torch.cuda.manual_seed_all(SEED_LIST[0])  # noqa: E702

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ENV] using device: {DEVICE}")

# -----------------------------------------------------------------------------
# Deferred heavy imports – only after the device is set up
# -----------------------------------------------------------------------------
from . import preprocess as pp  # noqa: E402 – deferred import
from . import train as trn      # noqa: E402
from . import evaluate as ev    # noqa: E402

# -----------------------------------------------------------------------------
# Data acquisition – only what we truly need for the executed experiment(s).
# For the current set-up we rely solely on CIFAR-100; attempting to download
# additional datasets (MiniImageNet, RotMNIST, SpeechCommands) caused a hard
# failure due to remote access restrictions (HTTP 403).  We therefore limit the
# downloads to the indispensable dataset to avoid unnecessary external
# dependencies while still conforming to the *fail-fast* policy: if CIFAR-100
# cannot be downloaded the program will terminate immediately with a clear
# error message.
# -----------------------------------------------------------------------------
pp.download_and_prepare_cifar100(CFG.__dict__)

################################################################################
#                        ───  EXPERIMENT 1  –  BENCHMARK ───                  #
################################################################################

def _num_classes(ds):
    """Infer the number of classes represented in a dataset. Supports both
    SplitDataset and torch.utils.data.Subset."""
    base = getattr(ds, "dataset", ds)  # unwrap Subset if needed
    cls_ids = getattr(base, "cls_ids", None)
    if cls_ids is not None:
        return len(cls_ids)
    # Fallback – iterate once (acceptable because datasets are tiny after split)
    seen = set()
    for _, y in ds:
        seen.add(int(y))
    return len(seen)

def run_experiment_1(seed: int) -> None:
    print("\n================ EXPERIMENT 1 – Standard C-L Benchmarks ================")
    t0 = time.perf_counter()

    tasks = pp.build_cifar_splits(CFG.__dict__, seed=seed)

    model = trn.JEMBModel(CFG.__dict__).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=CFG.opt["lr"], momentum=0.9, weight_decay=5e-4)

    acc_list = []
    for task_id, (train_ds, val_ds, test_ds) in enumerate(tasks, 1):
        n_cls = _num_classes(train_ds)
        print(f"\n[Task {task_id}] classes={n_cls}  samples={len(train_ds)}")
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=CFG.opt["batch"], shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=CFG.opt["batch"], shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=CFG.opt["batch"], shuffle=False
        )

        trn.train_one_task(model, train_loader, val_loader, optimizer, epochs=CFG.opt["epochs"])
        model.allocate_memory()
        acc = ev.accuracy(model, test_loader)
        acc_list.append(acc)
        print(f"[RESULT] Task-{task_id}   accuracy={acc:.2f}")

    # final aggregated metrics & figure
    A_T = sum(acc_list) / len(acc_list)
    fig_path = ev.plot_task_accuracies(acc_list)

    print(f"\nExperiment-1 CIFAR100 seed={seed}   Final Average Accuracy A_T = {A_T:.2f}")
    print(json.dumps({"Figure": fig_path.name, "A_T": A_T, "acc_per_task": acc_list}, indent=2))
    print(f"Wall-clock: {(time.perf_counter() - t0) / 60:.1f} min")

################################################################################
#                                    MAIN                                      #
################################################################################

def main() -> None:  # noqa: D401 – obvious entry point
    for sd in SEED_LIST:
        run_experiment_1(sd)
    # Placeholders for additional experiments --------------------------------
    # run_experiment_2(); run_experiment_3()


if __name__ == "__main__":
    main()
