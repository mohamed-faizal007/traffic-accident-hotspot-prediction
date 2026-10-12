"""Read-only data access for the React dashboard's API.

Reads only files tracked in git under results/ and outputs/ (never data/ or models/). This module never
writes, trains, or loads a model.
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

REGRESSION_METRICS_PATH = BASE_DIR / "results" / "regression" / "regression_metrics.json"
QUANTILE_TIERS_PATH = BASE_DIR / "results" / "regression" / "quantile_tiers_2025.parquet"
NEIGHBOR_CLF_PATH = BASE_DIR / "results" / "neighbor_features_ablation.json"
NEIGHBOR_REG_PATH = BASE_DIR / "results" / "regression" / "neighbor_features_ablation.json"
DBSCAN_SELECTION_PATH = BASE_DIR / "results" / "dbscan_selection.json"
DBSCAN_SWEEP_PATH = BASE_DIR / "results" / "dbscan_sweep.csv"
REGRESSION_FORECAST_PATH = BASE_DIR / "outputs" / "regression_forecast_2026-01.csv"
REGRESSION_FORECAST_META_PATH = BASE_DIR / "outputs" / "regression_forecast_2026-01.meta.json"
FORECAST_CELLS_PATH = BASE_DIR / "outputs" / "dashboard" / "forecast_cells_2026-01.csv"
FORECAST_CELLS_META_PATH = BASE_DIR / "outputs" / "dashboard" / "forecast_cells_2026-01.meta.json"
DBSCAN_CLUSTERS_PATH = BASE_DIR / "outputs" / "dashboard" / "dbscan_clusters.csv"
TEMPORAL_PATH = BASE_DIR / "outputs" / "dashboard" / "temporal_patterns.json"
MORANS_PATH = BASE_DIR / "outputs" / "dashboard" / "morans_i.json"
FOLIUM_MAP_PATH = BASE_DIR / "outputs" / "report_compliance" / "hotspot_map.html"

TIER_ORDER = ["Critical", "High", "Medium", "Low"]
MODEL_NAMES = {
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "persistence": "Baseline: last month >= 2 (persistence)",
    "trailing_12_month_count": "Baseline: trailing 12-month count",
}


class DataFileError(Exception):
    """A saved output exists but cannot be parsed or has an unexpected structure."""


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=8)
def _load_json_cached(path_str: str, mtime: float):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise DataFileError(f"{path.name} is not valid JSON ({exc.__class__.__name__}). Re-generate it with the pipeline.") from exc


@lru_cache(maxsize=8)
def _load_csv_cached(path_str: str, mtime: float):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # corrupt/truncated file: report it, do not crash
        raise DataFileError(f"{path.name} could not be read as CSV ({exc.__class__.__name__}). Re-generate it with the pipeline.") from exc


@lru_cache(maxsize=8)
def _load_parquet_cached(path_str: str, mtime: float):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # corrupt/truncated file: report it, do not crash
        raise DataFileError(f"{path.name} could not be read as PARQUET ({exc.__class__.__name__}). Re-generate it with the pipeline.") from exc


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


def _json(path: Path):
    return _load_json_cached(str(path), _mtime(path))


def _csv(path: Path):
    return _load_csv_cached(str(path), _mtime(path))


def load_regression_metrics():
    return _json(REGRESSION_METRICS_PATH)


def load_quantile_tiers():
    return _load_parquet_cached(str(QUANTILE_TIERS_PATH), _mtime(QUANTILE_TIERS_PATH))


def load_neighbor_clf():
    return _json(NEIGHBOR_CLF_PATH)


def load_neighbor_reg():
    return _json(NEIGHBOR_REG_PATH)


def load_dbscan_selection():
    return _json(DBSCAN_SELECTION_PATH)


def load_dbscan_sweep():
    return _csv(DBSCAN_SWEEP_PATH)


def load_regression_forecast():
    return _csv(REGRESSION_FORECAST_PATH)


def load_regression_forecast_meta():
    return _json(REGRESSION_FORECAST_META_PATH)


def load_forecast_cells():
    return _csv(FORECAST_CELLS_PATH)


def load_forecast_cells_meta():
    return _json(FORECAST_CELLS_META_PATH)


def load_dbscan_clusters():
    return _csv(DBSCAN_CLUSTERS_PATH)


def load_temporal():
    return _json(TEMPORAL_PATH)


def load_morans():
    return _json(MORANS_PATH)


# A cell is tagged by the same rule as the Folium map (src/report_compliance/build_folium_map.py):
# share of its past collisions on single carriageways / in adverse weather above one half.
SHARE_TAG_THRESHOLD = 0.5


def road_tag(share) -> str:
    return "single" if share > SHARE_TAG_THRESHOLD else "other"


def weather_tag(share) -> str:
    return "adverse" if share > SHARE_TAG_THRESHOLD else "fine"


@lru_cache(maxsize=2)
def _action_frame_cached(mtimes: tuple):
    fc, cells, reg = load_forecast(), load_forecast_cells(), load_regression_forecast()
    if fc is None or cells is None:
        return None
    df = fc.merge(cells, on="grid_id", how="inner", validate="one_to_one")
    if reg is not None:
        df = df.merge(reg[["grid_id", "predicted_count_xgboost", "predicted_count_baseline"]], on="grid_id", how="left")
    else:
        df["predicted_count_xgboost"] = float("nan")
        df["predicted_count_baseline"] = float("nan")
    df["road_tag"] = df["historical_single_carriageway_share"].map(road_tag)
    df["weather_tag"] = df["historical_adverse_weather_share"].map(weather_tag)
    return df.sort_values("rank").reset_index(drop=True)


def action_frame():
    """Classifier forecast + per-cell descriptive drivers (+ regression forecast when available)."""
    return _action_frame_cached(tuple(_mtime(p) for p in (FORECAST_PATH, FORECAST_CELLS_PATH, REGRESSION_FORECAST_PATH)))


def clean(obj):
    """Replace NaN/inf (which json can't encode) with None, recursively; numpy scalars become Python."""
    import math

    if hasattr(obj, "item") and not isinstance(obj, (list, dict)):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def model_family_name(metrics: dict) -> str:
    key = metrics["selection"]["model_family"]
    return MODEL_NAMES.get(key, key)


def headline(metrics: dict):
    """(split label, raw-score block, baselines block) for the split we can show."""
    test, val = metrics.get("test"), metrics["validation"]
    if test:
        return "2025 test (evaluated once, frozen)", test["score_raw"], test["baselines"]
    return "2024 validation (test not yet run)", val["chosen_model_raw_score"], val["baselines"]
