"""
main.py – project entry-point that orchestrates the experimental workflow
"""
import time, yaml
from pathlib import Path

from src.evaluate import experiment1

# -------------------------------------------------------------------------
# Load configuration
# -------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CFG = yaml.safe_load(f)

# ensure plot directory exists
Path('.research/iteration2/images').mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    start = time.time()
    experiment1()
    print(f"\nAll done in {(time.time() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
