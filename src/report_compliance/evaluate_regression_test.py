"""Evaluate the FROZEN regression model on the 2025 test targets. Runs
EXACTLY ONCE. Mirrors src/evaluate_test.py's discipline (verify the frozen
config's integrity and file hashes before scoring, write a lock file
immediately after, refuse to run a second time) but is entirely separate:

  lock file   results/regression/regression_test_evaluated.lock
              (NOT results/test_evaluated.lock -- the classifier's lock)
  config      results/regression/regression_frozen_config.json
              (NOT results/frozen_config.json -- the classifier's config)
  model       models/regression/final_model.joblib
              (NOT models/final_model.joblib -- the classifier's model)

Never reads or writes any file under the classifier's own
results/frozen_config.json, results/metrics.json, results/test_evaluated.lock,
models/final_model.joblib, models/calibrator.joblib, or
data/processed/ml_features.parquet.

DO NOT RUN without explicit go-ahead: this is the one-shot 2025 evaluation.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_compliance.register_regression_selection_rule import (  # noqa: E402
    REGRESSION_RESULTS_DIR, REGRESSION_FROZEN_CONFIG_PATH, REGRESSION_LOCK_PATH,
)
from report_compliance.train_validate_regression import REGRESSION_METRICS_PATH, REGRESSION_MODELS_DIR  # noqa: E402
from report_compliance.regression_features import (  # noqa: E402
    OUTPUT_PATH as REGRESSION_FEATURES_PATH, RAW_TARGET_COLUMN,
)
from report_compliance.regression_evaluation import to_raw_scale, evaluate_regression  # noqa: E402
from report_compliance.regression_baseline import fit_historical_average_baseline, predict_historical_average_baseline  # noqa: E402
from freeze_regression_config import sha256  # noqa: E402


def verify_frozen(frozen):
    if frozen.get("status") != "frozen":
        sys.exit("Regression config is not frozen; run freeze_regression_config.py first.")
    body = {k: v for k, v in frozen.items() if k != "config_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != frozen["config_sha256"]:
        sys.exit("regression_frozen_config.json was modified after freezing.")
    files = {
        "models/regression/final_model.joblib": REGRESSION_MODELS_DIR / "final_model.joblib",
        "data/processed/regression_features.parquet": REGRESSION_FEATURES_PATH,
    }
    for rel, path in files.items():
        if sha256(path) != frozen["hashes_sha256"][rel]:
            sys.exit(f"{rel} changed since freezing.")


def main():
    if REGRESSION_LOCK_PATH.exists():
        sys.exit("The regression track's 2025 test set has already been evaluated. Re-running is not allowed.")
    frozen = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())
    verify_frozen(frozen)

    features = frozen["features"]
    model = joblib.load(REGRESSION_MODELS_DIR / "final_model.joblib")

    df = pd.read_parquet(REGRESSION_FEATURES_PATH)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    y_raw = test[RAW_TARGET_COLUMN].to_numpy()
    pred_log = model.predict(test[features])
    pred_raw = to_raw_scale(pred_log)
    test_metrics = evaluate_regression(y_raw, pred_raw)

    # Baseline comparison on the SAME 2025 test set, fit on train only (no
    # hyperparameters to tune, so this is not a second "attempt" at the test
    # set -- it's the same pre-registered baseline the validation comparison
    # already used, just scored once here alongside the frozen model.
    baseline_fit = fit_historical_average_baseline(train)
    baseline_pred = predict_historical_average_baseline(baseline_fit, test)
    baseline_metrics = evaluate_regression(y_raw, baseline_pred)

    evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    test_block = {
        "evaluated_at_utc": evaluated_at,
        "frozen_config_sha256": frozen["config_sha256"],
        "model": frozen["model"]["family"],
        "xgboost": test_metrics,
        "historical_average_baseline": baseline_metrics,
        "note": "evaluated once; nothing was tuned or changed after seeing these numbers",
    }

    metrics = json.loads(REGRESSION_METRICS_PATH.read_text())
    metrics["test"] = test_block
    REGRESSION_METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    REGRESSION_LOCK_PATH.write_text(evaluated_at)

    out = test[["grid_id", "grid_latitude", "grid_longitude", "target_month"]].copy()
    out[RAW_TARGET_COLUMN] = y_raw
    out["predicted_count_xgboost"] = pred_raw
    out["predicted_count_baseline"] = baseline_pred
    out.to_parquet(REGRESSION_RESULTS_DIR / "test_predictions_2025.parquet", index=False)

    print(f"TEST 2025 xgboost:  MAE {test_metrics['mae']:.4f}  RMSE {test_metrics['rmse']:.4f}  R2 {test_metrics['r2']:.4f}")
    print(f"TEST 2025 baseline: MAE {baseline_metrics['mae']:.4f}  RMSE {baseline_metrics['rmse']:.4f}  R2 {baseline_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
