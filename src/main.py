from __future__ import annotations

"""
main.py – reproducible entry-point that orchestrates the experiment.
It loads the YAML config, runs unit-tests, trains all methods and finally
creates the accuracy figure.  Execute with

    python -m src.main
"""
import sys, subprocess, importlib, os, time, json
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# 0.  ENVIRONMENT GUARD --------------------------------------------------------
# Allow any Python 3 version >= 3.10 (e.g. 3.11 which is used by the CI runner)
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    raise RuntimeError(
        f"Python >=3.10 required – found {sys.version.split()[0]}"  # fail-fast, clear message
    )

# optional: auto-install pinned CUDA wheels for CI -----------------------------
WHEELS = {
    'torch': 'torch==2.1.2+cu122',
    'torchvision': 'torchvision==0.16.2+cu122',
    'numpy': 'numpy==1.26.4',
    'tqdm': 'tqdm==4.66.2',
    'fvcore': 'fvcore==0.1.5.post20221221',
    'torchmetrics': 'torchmetrics==1.3.2',
    'matplotlib': 'matplotlib==3.8.4',
    'seaborn': 'seaborn==0.13.2',
    'pyyaml': 'pyyaml==6.0.1'
}
for mod, wheel in WHEELS.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([
            sys.executable,
            '-m',
            'pip',
            'install',
            wheel,
            '--extra-index-url',
            'https://download.pytorch.org/whl/cu122'
        ])

# -----------------------------------------------------------------------------
# 1.  PATHS --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / 'runs'
RUN_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 2.  LOAD CONFIG --------------------------------------------------------------
CFG_PATH = ROOT / 'config' / 'config.yaml'
with open(CFG_PATH, 'r') as fh:
    CFG = yaml.safe_load(fh)

# allow quick-CI override via env var -----------------------------------------
if os.getenv('QUICK_CI') == '1':
    CFG['quick_ci']['enabled'] = True

# -----------------------------------------------------------------------------
# 3.  IMPORT INTERNAL MODULES --------------------------------------------------
from .train import (
    JEMB,
    Reservoir,
    InfLoRA,
    run_unit_tests,
    run_method,
)
from .evaluate import plot_accuracy

# -----------------------------------------------------------------------------
DESC = f"""
Experiment-1  –  Bug-free End-to-End Benchmark (CI Gate)
Dataset : {CFG['dataset']['name']}  (tasks = {CFG['dataset']['tasks']})
Budget  : {CFG['memory_budget']['bytes'] // (1 << 20)} MB (adapter + buffer)
Seed    : 42{'  QUICK-CI' if CFG['quick_ci']['enabled'] else ''}
Models  : JEMB-dynamic, Reservoir, InfLoRA
Metrics : per-task A_T curve, bytes_A, bytes_B
"""

# -----------------------------------------------------------------------------
# 4.  MAIN ---------------------------------------------------------------------

def main():
    print(DESC)
    run_unit_tests()  # fast sanity checks
    start = time.time()

    METHODS = {
        'JEMB': lambda led, mp: JEMB(led, mp),
        'Reservoir': lambda led, mp: Reservoir(led, mp),
        'InfLoRA': lambda led, mp: InfLoRA(led, mp),
    }

    all_logs = {}
    summary = []
    for name, ctor in METHODS.items():
        s, log = run_method(name, ctor, CFG)
        all_logs[name] = log
        summary.append(s)

    # — CI gates ---------------------------------------------------------
    if summary[0]['A_T'] < 30 or summary[0]['A_T'] < summary[2]['A_T']:
        raise RuntimeError('finalAcc gate failed')

    if max(all_logs['JEMB'].B) == 0 or len(set(all_logs['JEMB'].B)) < 4:
        raise RuntimeError('controller_allocates_buffer gate failed')

    if not 45 <= summary[1]['A_T'] <= 65 and not CFG['quick_ci']['enabled']:
        raise RuntimeError('reservoir_sanity gate failed')

    out = {
        'description': DESC,
        'summary': summary,
        'per_task': {k: vars(v) for k, v in all_logs.items()},
        'wall_clock_s': round(time.time() - start, 2),
    }
    (RUN_DIR / 'run_ci.json').write_text(json.dumps(out, indent=2))
    print('\n[results]\n', json.dumps(summary, indent=2))
    print('[saved] runs/run_ci.json')

    plot_accuracy(all_logs)


if __name__ == '__main__':
    main()
