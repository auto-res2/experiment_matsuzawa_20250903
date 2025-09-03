"""
main.py – orchestrates the complete experimental workflow
Run with:  python -m src.main
"""
from __future__ import annotations
import itertools, pprint

from .train import Trainer
from .evaluate import bar_chart
import yaml
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(_CFG_PATH, "r") as _f:
    cfg = yaml.safe_load(_f)

print("================  EXPERIMENT DESCRIPTION  =================", flush=True)
print("Demonstration – trains Waterbirds & CelebA with ERM and IRM (3 seeds each).\n"
      "Full 168-run grid can be activated by setting RUN_FULL_GRID=True.")
print("===========================================================", flush=True)

RUN_FULL_GRID = False   # flip for the full camera-ready sweep

###########################################################################
# ─── EXPERIMENT GRID ──────────────────────────────────────────────────────
###########################################################################

def _run(dataset: str, backbone: str, method: str):
    accs = []
    for seed in cfg["seeds"]:
        trainer = Trainer(dataset, backbone, method, seed)
        accs.append(trainer.fit())
    return sum(accs) / len(accs)

if RUN_FULL_GRID:
    GRID = itertools.product(
        ["waterbirds", "celeba", "imagenet9", "ninco"],
        ["resnet50", "vit_b16"],
        cfg["methods"]
    )
else:
    GRID = [(d, "resnet50", m) for d in ["waterbirds", "celeba"] for m in ["erm", "irm"]]

results = {}
for ds, bk, meth in GRID:
    key = f"{ds}_{meth}"
    print(f"\n>>> Launching {key} ({bk})", flush=True)
    results[key] = _run(ds, bk, meth)

###########################################################################
# ─── VISUALISATION & OUTPUT ───────────────────────────────────────────────
###########################################################################

fig_path = bar_chart(results, "AccID (demo)", "training_accuracy_demo")

print("\n================  EXPERIMENTAL NUMERICAL DATA  ============")
print(pprint.pformat(results, compact=True))
print("================  FIGURE FILENAMES  =======================")
print(fig_path)
