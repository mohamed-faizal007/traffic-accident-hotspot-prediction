"""Read-only data access for the React dashboard's API.

Mirrors exactly what app.py (the Streamlit dashboard) reads: results/metrics.json,
results/feature_importance.csv, outputs/forecast_2026-01.csv and
outputs/test_predictions_2025.parquet. This module never writes, trains, or loads a model.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METRICS_PATH = BASE_DIR / "results" / "metrics.json"
FROZEN_PATH = BASE_DIR / "results" / "frozen_config.json"
IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.csv"
FORECAST_PATH = BASE_DIR / "outputs" / "forecast_2026-01.csv"
TEST_PRED_PATH = BASE_DIR / "outputs" / "test_predictions_2025.parquet"

TIER_ORDER = ["Critical", "High", "Medium", "Low"]
MODEL_NAMES = {
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "persistence": "Baseline: last month >= 2 (persistence)",
    "trailing_12_month_count": "Baseline: trailing 12-month count",
}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=8)
def _load_json_cached(path_str: str, mtime: float):
    path = Path(path_str)
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=8)
def _load_csv_cached(path_str: str, mtime: float):
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else None


@lru_cache(maxsize=8)
def _load_parquet_cached(path_str: str, mtime: float):
    path = Path(path_str)
    return pd.read_parquet(path) if path.exists() else None


def load_metrics():
    return _load_json_cached(str(METRICS_PATH), _mtime(METRICS_PATH))


def load_frozen():
    return _load_json_cached(str(FROZEN_PATH), _mtime(FROZEN_PATH))


def load_feature_importance():
    return _load_csv_cached(str(IMPORTANCE_PATH), _mtime(IMPORTANCE_PATH))


def load_forecast():
    return _load_csv_cached(str(FORECAST_PATH), _mtime(FORECAST_PATH))


def load_test_predictions():
    return _load_parquet_cached(str(TEST_PRED_PATH), _mtime(TEST_PRED_PATH))


def model_family_name(metrics: dict) -> str:
    key = metrics["selection"]["model_family"]
    return MODEL_NAMES.get(key, key)


def headline(metrics: dict):
    """(split label, raw-score block, baselines block) for the split we can show."""
    test, val = metrics.get("test"), metrics["validation"]
    if test:
        return "2025 test (evaluated once, frozen)", test["score_raw"], test["baselines"]
    return "2024 validation (test not yet run)", val["chosen_model_raw_score"], val["baselines"]
