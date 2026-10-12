import hashlib
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"
REGRESSION_FORECAST_PATH = OUTPUTS_DIR / "regression_forecast_2026-01.csv"
REGRESSION_FORECAST_META_PATH = OUTPUTS_DIR / "regression_forecast_2026-01.meta.json"

# Never written by any dashboard export; asserted by the tests.
FROZEN_FILES = [
    "results/frozen_config.json", "results/metrics.json", "results/test_evaluated.lock",
    "models/final_model.joblib", "models/calibrator.joblib",
    "data/processed/ml_features.parquet", "data/processed/grid_monthly_collisions.csv",
    "results/regression/regression_frozen_config.json", "results/regression/regression_test_evaluated.lock",
    "models/regression/final_model.joblib",
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes() -> dict:
    return {rel: sha256(PROJECT_ROOT / rel) for rel in FROZEN_FILES}


def size_kb(path) -> str:
    return f"{Path(path).stat().st_size / 1024:,.0f} KB"
