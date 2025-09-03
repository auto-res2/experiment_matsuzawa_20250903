"""src/main.py – orchestrates the full GCDI Waterbirds experiment"""
from __future__ import annotations
import json, time
from pathlib import Path

import yaml

from .train import run_waterbirds_experiment
from .evaluate import evaluate_waterbirds, aggregate_results, overview_plot

CONFIG_PATH = Path("config/config.yaml")
with CONFIG_PATH.open() as f:
    CFG = yaml.safe_load(f)

print("============  GCDI – Generative Causal Diffusion Interventions  ============")
print("Experiment configuration (excerpt):")
print(json.dumps({k: v for k, v in CFG.items() if k != "datasets"}, indent=2))

all_results = []
start = time.perf_counter()
for seed in CFG["seeds"]:
    artefacts = run_waterbirds_experiment(seed, CFG)
    res = evaluate_waterbirds(artefacts, CFG)
    all_results.append(res)

agg = aggregate_results(all_results)
print("\n=== AGGREGATED RESULTS – WATERBIRDS ===")
print(json.dumps(agg, indent=2))
fig_name = overview_plot(agg)
print("Figures:", fig_name)
print(f"Total wall-clock: {(time.perf_counter() - start) / 3600:.2f} h")
