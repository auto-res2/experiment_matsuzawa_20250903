"""src/main.py
Main execution script – loads YAML configuration, prepares the environment
and kicks off experiment(s).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Ensure local imports work regardless of the entry-point location.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.evaluate import ExperimentEngine
from src.preprocess import acquire_datasets


# -----------------------------------------------------------------------------
#  Config loader
# -----------------------------------------------------------------------------

def load_config_yaml() -> OmegaConf:
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


# -----------------------------------------------------------------------------
#  Entrypoint
# -----------------------------------------------------------------------------

def main():
    cfg = load_config_yaml()

    # ------------------------------------------------------------
    #  Prepare data (download / extract) once for all experiments
    # ------------------------------------------------------------
    acquire_datasets(cfg)

    # ------------------------------------------------------------
    for exp in cfg.experiments:
        engine = ExperimentEngine(OmegaConf.create(exp), cfg)
        engine.run()


if __name__ == "__main__":
    main()
