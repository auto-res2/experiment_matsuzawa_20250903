"""src/preprocess.py
A **placeholder** script.  In a full research pipeline this would download and
pre-process raw data into a format suitable for `train.py`.

For the purpose of CI testing (where internet access is disabled) we simply
provide a no-op implementation so that importing/running the script does not
raise any errors.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy pre-processing script.")
    parser.add_argument("--output", default="data/", help="Where to write outputs.")
    _ = parser.parse_args()
    print("[preprocess] Nothing to do – this is a stub.")


if __name__ == "__main__":
    main()
