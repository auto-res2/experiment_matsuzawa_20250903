"""
main.py – orchestrates the full experimental workflow.
Run exactly with:  python -m src.main
"""
from __future__ import annotations
import itertools, os, json
from pathlib import Path
import yaml

from .train import Trainer, timing
from .evaluate import bar_chart

# -----------------------------------------------------------------------------
# load configuration -----------------------------------------------------------
# -----------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)

print("================  EXPERIMENT DESCRIPTION  =================")
print("Structured refactor – running benchmark grid as specified in config.yaml")
print("===========================================================")

FAST_DEMO = os.environ.get("FAST_DEMO", "0") == "1"

if FAST_DEMO:
    grid = [("waterbirds", "resnet50", "erm")]
else:
    grid = list(itertools.product(
        ["waterbirds", "celeba", "imagenet9", "ninco"],
        ["resnet50", "vit_b16"],
        CFG["methods"],
    ))

all_results = {}
for ds, bk, meth in grid:
    for seed in CFG["global"]["seeds"]:
        label = f"{ds}_{bk}_{meth}_s{seed}"
        with timing(label):
            res = Trainer(CFG, ds, bk, meth, seed).fit()
            all_results[label] = res

# -----------------------------------------------------------------------------
# aggregate + example figure ---------------------------------------------------
# -----------------------------------------------------------------------------
keys = [k for k in all_results if "waterbirds_resnet50" in k]
fig_path = bar_chart({k: all_results[k]["AccID"] for k in keys}, "Waterbirds AccID (ResNet-50)", "accuracy_waterbirds")

print("================  NUMERICAL RESULTS  ======================")
for k, v in all_results.items():
    print(k, v)
print("================  FIGURE PATHS  ===========================")
print(fig_path)
