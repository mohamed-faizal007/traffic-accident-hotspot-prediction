"""Freeze the regression track's model, features and selection outcome
BEFORE the 2025 test set is evaluated -- mirrors src/freeze_config.py's
discipline exactly, but is a wholly separate script writing to wholly
separate paths:

  results/regression/regression_frozen_config.json   (NOT results/frozen_config.json)
  results/regression/regression_test_evaluated.lock   (NOT results/test_evaluated.lock, does not exist until evaluate_regression_test.py runs)

Refuses to run if the tests fail, if the selection rule differs from the one
registered before tuning, or if the regression test set has already been
evaluated. evaluate_regression_test.py verifies these hashes before scoring
2025. Never reads or writes anything under the classifier's own
results/frozen_config.json, results/metrics.json, results/test_evaluated.lock,
models/final_model.joblib, or models/calibrator.joblib.
"""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, PROCESSED_DATA_DIR, MODELS_DIR  # noqa: E402
from report_compliance.register_regression_selection_rule import (  # noqa: E402
    REGRESSION_FROZEN_CONFIG_PATH, REGRESSION_LOCK_PATH,
)
from report_compliance.train_validate_regression import REGRESSION_METRICS_PATH  # noqa: E402
from report_compliance.regression_features import OUTPUT_PATH as REGRESSION_FEATURES_PATH  # noqa: E402

REGRESSION_MODEL_PATH = MODELS_DIR / "regression" / "final_model.joblib"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()


def main():
    if REGRESSION_LOCK_PATH.exists():
        sys.exit("Regression test set already evaluated; the frozen config must not change.")

    tests = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=PROJECT_ROOT)
    if tests.returncode != 0:
        sys.exit("Tests failed; not freezing.")

    registered = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())
    if registered.get("status") == "frozen":
        sys.exit("regression_frozen_config.json is already frozen.")
    rule = registered["selection_rule"]
    rule_hash = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
    if rule_hash != registered["selection_rule_sha256"]:
        sys.exit("Selection rule differs from the one registered before tuning.")

    metrics = json.loads(REGRESSION_METRICS_PATH.read_text())
    if metrics.get("test") is not None:
        sys.exit("regression_metrics.json already contains test results.")
    sel = metrics["selection"]

    # re-build the feature-table hash fresh so it matches whatever is on disk right now
    if not REGRESSION_FEATURES_PATH.exists():
        sys.exit(f"{REGRESSION_FEATURES_PATH} not found; run regression_features.py before freezing.")

    frozen = {
        "status": "frozen",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "rule_registered_at_utc": registered["written_at_utc"],
        "selection_rule": rule,
        "selection_rule_sha256": rule_hash,
        "selection_outcome": sel["decision"],
        "model": {
            "family": sel["model_family"],
            "params": sel["model_params"],
            "training": "targets <= 2023-12, full training set (no class sampling; this is a regression, not a classifier)",
            "file": "models/regression/final_model.joblib",
        },
        "features": sel["features"],
        "target_transform": sel["target_transform"],
        "xgboost_retuning_note": sel.get("xgboost_retuning_note"),
        "split": {"train_targets": "<= 2023-12", "validation_targets": "2024", "test_targets": "2025"},
        "hashes_sha256": {
            "models/regression/final_model.joblib": sha256(REGRESSION_MODEL_PATH),
            "data/processed/regression_features.parquet": sha256(REGRESSION_FEATURES_PATH),
            "metrics.validation": hashlib.sha256(
                json.dumps(metrics["validation"], sort_keys=True, default=float).encode()).hexdigest(),
        },
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_files": git("status", "--porcelain").splitlines(),
        "note": (
            "This is the REGRESSION TRACK's frozen config, entirely separate from the "
            "classifier's results/frozen_config.json / results/test_evaluated.lock. "
            "The one-shot 2025 evaluation for this track writes "
            "results/regression/regression_test_evaluated.lock, never results/test_evaluated.lock."
        ),
    }
    frozen["config_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    REGRESSION_FROZEN_CONFIG_PATH.write_text(json.dumps(frozen, indent=2))
    print(f"Frozen regression config saved: {REGRESSION_FROZEN_CONFIG_PATH}\nconfig sha256: {frozen['config_sha256']}")


if __name__ == "__main__":
    main()
