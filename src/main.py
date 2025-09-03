"""src/main.py – lightweight entry-point for automated evaluation.
It delegates the actual work to `evaluate.experiment1()` so that the CI can
simply run `python -m src.main`.
"""
from __future__ import annotations

def _run():
    from .evaluate import experiment1
    experiment1()


if __name__ == "__main__":
    _run()
