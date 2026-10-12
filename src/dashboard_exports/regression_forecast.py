"""Jan-2026 count forecast from the FROZEN regression model: inference only.

Reads only the 'unlabeled' rows (features through Dec-2025, no outcome exists for the target month)
and the 'train' rows (needed to fit the pre-registered historical-average baseline). The 2025 test
rows are excluded at read time, so no 2025 label is ever loaded. The model file is hashed before and
after loading and the script aborts if the hash differs from the frozen config's hash or from itself.

Writes only outputs/regression_forecast_2026-01.csv (+ a .meta.json). Nothing here is validated:
there are no outcomes for Jan 2026 yet.
"""

import datetime
import hashlib
import json

import joblib
import pandas as pd

try:
    from ._common import PROJECT_ROOT, REGRESSION_FORECAST_META_PATH, REGRESSION_FORECAST_PATH, sha256
except ImportError:  # run as a script
    from _common import PROJECT_ROOT, REGRESSION_FORECAST_META_PATH, REGRESSION_FORECAST_PATH, sha256

from report_compliance.regression_baseline import fit_historical_average_baseline, predict_historical_average_baseline
from report_compliance.regression_evaluation import to_raw_scale

MODEL_PATH = PROJECT_ROOT / "models" / "regression" / "final_model.joblib"
FROZEN_PATH = PROJECT_ROOT / "results" / "regression" / "regression_frozen_config.json"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "regression_features.parquet"
STATUS = "unvalidated forecast (no outcomes yet)"


def main():
    frozen = json.loads(FROZEN_PATH.read_text())
    body = {k: v for k, v in frozen.items() if k != "config_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != frozen["config_sha256"]:
        raise SystemExit("regression_frozen_config.json was modified after freezing.")
    expected = frozen["hashes_sha256"]["models/regression/final_model.joblib"]

    model_sha_before = sha256(MODEL_PATH)
    if model_sha_before != expected:
        raise SystemExit("regression model does not match the hash in the frozen config.")
    model = joblib.load(MODEL_PATH)

    cols = ["grid_id", "grid_latitude", "grid_longitude", "year_month", "target_month", "split",
            "next_month_collisions"] + frozen["features"]
    df = pd.read_parquet(FEATURES_PATH, columns=sorted(set(cols)), filters=[("split", "in", ["train", "unlabeled"])])
    unlabeled = df[df["split"] == "unlabeled"].reset_index(drop=True)
    train = df[df["split"] == "train"].reset_index(drop=True)
    assert unlabeled["next_month_collisions"].isna().all(), "unlabeled rows must have no outcome"
    assert unlabeled["target_month"].nunique() == 1

    pred = to_raw_scale(model.predict(unlabeled[frozen["features"]]))
    baseline = predict_historical_average_baseline(fit_historical_average_baseline(train), unlabeled)

    model_sha_after = sha256(MODEL_PATH)
    if model_sha_after != model_sha_before:
        raise SystemExit("model file changed while loading it")

    out = unlabeled[["grid_id", "grid_latitude", "grid_longitude", "target_month"]].copy()
    out["target_month"] = out["target_month"].dt.strftime("%Y-%m-%d")
    out["predicted_count_xgboost"] = pred
    out["predicted_count_baseline"] = baseline
    out["rank"] = out["predicted_count_xgboost"].rank(method="first", ascending=False).astype(int)
    out = out.sort_values("rank")
    out.to_csv(REGRESSION_FORECAST_PATH, index=False, float_format="%.5f")

    meta = {
        "status": STATUS,
        "target_month": out["target_month"].iloc[0],
        "n_cells": int(len(out)),
        "model": frozen["model"],
        "target_transform": frozen["target_transform"],
        "regression_config_sha256": frozen["config_sha256"],
        "model_sha256_before_load": model_sha_before,
        "model_sha256_after_load": model_sha_after,
        "model_hash_matches_frozen_config": model_sha_before == expected,
        "inputs": "unlabeled rows (features through 2025-12) + train rows (baseline fit only); 2025 test rows not read",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    REGRESSION_FORECAST_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"{len(out):,} cells -> {REGRESSION_FORECAST_PATH}")
    print(f"model sha256 before == after == frozen: {model_sha_before == model_sha_after == expected}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
