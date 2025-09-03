"""src/main.py – orchestrates Experiment-1 using the refactored modules"""
from __future__ import annotations
import sys, subprocess, importlib, json, time
from pathlib import Path
from typing import Dict

import yaml

# ----------------------------------------------------------------------
# 1.  Load configuration                                                 
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # /project/src
CFG_PATH = ROOT.parent / 'config' / 'config.yaml'
CFG: Dict = yaml.safe_load(CFG_PATH.read_text())

# ----------------------------------------------------------------------
# 2.  Environment guard & wheel bootstrap (unchanged logic)              
# ----------------------------------------------------------------------
REQ_MAJOR, REQ_MINOR = 3, 10
if sys.version_info[:2] != (REQ_MAJOR, REQ_MINOR):
    raise RuntimeError(f"Python {REQ_MAJOR}.{REQ_MINOR}.x required, found {sys.version}")

for mod, wheel in CFG['environment']['pinned_wheels'].items():
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"[setup] installing pinned wheel {wheel}")
        subprocess.check_call([
            sys.executable,
            '-m',
            'pip',
            'install',
            wheel,
            '--extra-index-url',
            'https://download.pytorch.org/whl/cu122',
        ])

# now heavy imports are safe -------------------------------------------
import torch  # noqa: E402  pylint: disable=wrong-import-position

from .train import (
    JEMB,
    Reservoir,
    InfLoRA,
    run_method,
    run_unit_tests,
)
from .evaluate import plot_curves

RUNS_DIR = ROOT.parent / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTION = (
    """
Experiment-1  –  Correct-Task End-to-End Benchmark\n"
    "Dataset : Split-CIFAR100 (20 × 5 classes)\n"
    "Budget  : 1 MB (joint adapter + buffer)    Seed : 42\n"
    "Models  : JEMB, Reservoir, InfLoRA\n"
    "Metrics : per-task average accuracy, bytes_adapter, bytes_buffer\n"""
)


def main():
    print('\n' + DESCRIPTION + '\n')
    run_unit_tests()
    start = time.time()

    logs = {}
    results = []

    for name, ctor in [
        ('JEMB', lambda led: JEMB(led)),
        ('Reservoir', lambda led: Reservoir(led)),
        ('InfLoRA', lambda led: InfLoRA(led)),
    ]:
        res, lg = run_method(name, ctor, CFG, seed=42)
        results.append(res)
        logs[name] = lg

    # -------------------- sanity gates --------------------------------
    if results[0]['A_T'] < 30:
        raise RuntimeError('ci_final_acc() gate failed – JEMB accuracy below 30 %')
    if max(logs['JEMB'].B) == 0 or len(set(logs['JEMB'].B)) < 2:
        raise RuntimeError('controller_allocates_buffer() failed – buffer never used')
    if not 45 <= results[1]['A_T'] <= 65:
        raise RuntimeError('reservoir_in_range() failed – baseline sanity')

    # -------------------- persist & plots ------------------------------
    out = {
        'description': DESCRIPTION,
        'per_task': {k: {'acc': v.acc, 'bytesA': v.A, 'bytesB': v.B} for k, v in logs.items()},
        'summary': results,
        'wall_clock_s': round(time.time() - start, 2),
    }
    (RUNS_DIR / 'run_ci.json').write_text(json.dumps(out, indent=2))
    print('\n[results]\n', json.dumps(results, indent=2))
    print('[saved] runs/run_ci.json')

    plot_curves(logs)


if __name__ == '__main__':
    main()
