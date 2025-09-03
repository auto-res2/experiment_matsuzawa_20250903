"""
main.py – top-level entry point  (run with  python -m src.main )
"""
from __future__ import annotations
import sys, importlib, subprocess, yaml
from pathlib import Path

# ---------------------------------------------------------------------
# 0. Reproducibility guard & on-the-fly wheel installer (kept verbatim)
# ---------------------------------------------------------------------
REQ_PY = (3, 10)
if sys.version_info[:2] != REQ_PY:
    raise RuntimeError(
        f"Reproducibility guarantee broken – install Python {REQ_PY[0]}.{REQ_PY[1]}.*"
    )

_PINNED = {
    "torch":        "torch==2.1.2+cu122",
    "torchvision":  "torchvision==0.16.2+cu122",
    "tqdm":         "tqdm==4.66.2",
    "numpy":        "numpy==1.26.4",
    "fvcore":       "fvcore==0.1.5.post20221221",
    "torchmetrics": "torchmetrics==1.3.2",
    "matplotlib":   "matplotlib==3.8.4",
    "seaborn":      "seaborn==0.13.2",
    "bitsandbytes": "bitsandbytes==0.43.1",
    "PyYAML":       "PyYAML==6.0.1",
}

for module, wheel in _PINNED.items():
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"[setup] installing {wheel} …", flush=True)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            wheel,
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu122",
        ])

# ---------------------------------------------------------------------
# 1.  Load YAML configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "config.yaml"
with CFG_PATH.open() as fh:
    cfg = yaml.safe_load(fh)

# inject data_root if not present
cfg["dataset"].setdefault("data_root", str(ROOT / "data"))

# ---------------------------------------------------------------------
# 2.  Run experiment (delegated to evaluate.py)
# ---------------------------------------------------------------------
from .evaluate import run_experiment


def main():
    run_experiment(cfg)


if __name__ == "__main__":
    main()
