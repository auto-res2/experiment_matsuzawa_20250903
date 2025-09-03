from __future__ import annotations
"""
main.py – top-level entry point  (run with  python -m src.main )
"""
import importlib
import subprocess
import sys
from pathlib import Path
import yaml

# ---------------------------------------------------------------------
# 0. Reproducibility guard & on-the-fly wheel installer
# ---------------------------------------------------------------------
# Strict reproducibility would require a fixed Python minor version.
# However, the execution environment may differ, so we emit a warning
# instead of aborting when the version is not exactly 3.10.*.
REQ_PY = (3, 10)
if sys.version_info[:2] != REQ_PY:
    print(
        f"[warn] Expected Python {REQ_PY[0]}.{REQ_PY[1]}.* for strict reproducibility, "
        f"but running on {sys.version_info.major}.{sys.version_info.minor}. Proceeding anyway."
    )

# Lightweight dependency check – install only if a module is *missing*.
# This prevents unnecessary re-installation of huge wheels that are
# typically already present in the execution image (e.g. torch / CUDA).
_PINNED = {
    "torch": "torch==2.1.2+cu122",
    "torchvision": "torchvision==0.16.2+cu122",
    "tqdm": "tqdm==4.66.2",
    "numpy": "numpy==1.26.4",
    "fvcore": "fvcore==0.1.5.post20221221",
    "torchmetrics": "torchmetrics==1.3.2",
    "matplotlib": "matplotlib==3.8.4",
    "seaborn": "seaborn==0.13.2",
    "bitsandbytes": "bitsandbytes==0.43.1",
    "PyYAML": "PyYAML==6.0.1",
}

for module, wheel in _PINNED.items():
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"[setup] installing {wheel} …", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                wheel,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu122",
            ]
        )

# ---------------------------------------------------------------------
# 1.  Load YAML configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "config.yaml"
with CFG_PATH.open() as fh:
    cfg = yaml.safe_load(fh)

# Inject default data path if not provided in the YAML -----------------
cfg["dataset"].setdefault("data_root", str(ROOT / "data"))

# ---------------------------------------------------------------------
# 2.  Run experiment (delegated to evaluate.py)
# ---------------------------------------------------------------------
from .evaluate import run_experiment


def main():
    run_experiment(cfg)


if __name__ == "__main__":
    main()
