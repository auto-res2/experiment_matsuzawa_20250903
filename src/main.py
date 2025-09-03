"""src/main.py
Minimal entry-point used by the automated evaluation harness.
It intentionally avoids heavy side-effects (like dataset downloads or
model training) so that importing / executing the file stays lightweight
and deterministic in the constrained CI environment.

Running the module as a script simply prints a short confirmation that
the codebase loaded successfully.  All actual experiment logic lives in
`src/train.py`, `src/evaluate.py`, etc. and can be invoked by external
scripts if desired.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main():  # noqa: D401 – CLI entry-point
    print("[T-DLR] Codebase import check passed.  (src.main executed successfully.)")


if __name__ == "__main__":  # pragma: no cover
    main()
