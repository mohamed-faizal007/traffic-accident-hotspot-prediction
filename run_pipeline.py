"""Run the pipeline stages in order. Never evaluates the 2025 test set.

    python run_pipeline.py                 # everything up to and including validation
    python run_pipeline.py --from grid     # restart from a stage (grid, monthly, dbscan, features, train, compare)

After it finishes: run the tests, then `python src/freeze_config.py`, then (once only)
`python src/evaluate_test.py`.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("preprocess", ["src/preprocessing.py"]),
    ("grid", ["src/spatial_grid.py"]),
    ("monthly", ["src/temporal_dataset.py"]),
    ("dbscan", ["src/hotspot_detection.py", "--reuse"]),
    ("features", ["src/feature_engineering.py"]),
    ("train", ["src/train_validate.py"]),
    ("compare", ["src/validation_comparison.py"]),
]


def main():
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STAGES[0][0]
    names = [n for n, _ in STAGES]
    if start not in names:
        sys.exit(f"unknown stage {start!r}; choose from {names}")
    for name, cmd in STAGES[names.index(start):]:
        print(f"\n>>> stage: {name}")
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
