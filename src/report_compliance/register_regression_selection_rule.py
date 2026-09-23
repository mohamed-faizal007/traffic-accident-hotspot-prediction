"""Pre-register the regression track's model-selection rule BEFORE any
validation tuning is run -- same discipline as the classifier's
results/frozen_config.json["selection_rule"] (written before train_validate.py
tunes anything, then checked for tampering by freeze_config.py).

Writes results/regression/regression_frozen_config.json with status
"registered" (NOT "frozen" -- that only happens in
freeze_regression_config.py, right before the one-shot 2025 test
evaluation). This file's path and lock file are entirely separate from the
classifier's results/frozen_config.json / results/test_evaluated.lock.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR  # noqa: E402

REGRESSION_RESULTS_DIR = RESULTS_DIR / "regression"
REGRESSION_FROZEN_CONFIG_PATH = REGRESSION_RESULTS_DIR / "regression_frozen_config.json"
REGRESSION_LOCK_PATH = REGRESSION_RESULTS_DIR / "regression_test_evaluated.lock"

SELECTION_RULE = {
    "candidates": ["historical_average_baseline", "random_forest", "xgboost"],
    "simplicity_order_simplest_first": ["historical_average_baseline", "random_forest", "xgboost"],
    "metric": "validation MAE on the RAW COUNT scale (log1p model predictions inverted via expm1, clipped at 0, before computing MAE)",
    "tuning_data": "train on target months <= 2023-12; hyperparameters for random_forest and xgboost chosen by validation (2024 targets) MAE only; the baseline has no hyperparameters",
    "step_1": "tune random_forest and xgboost each on validation MAE; take each family's best configuration",
    "step_2": "rank the three candidates (baseline, tuned random_forest, tuned xgboost) by validation MAE, lower is better",
    "step_3": "if the top two are within bootstrap noise (95% grid-bootstrap CI of their validation MAE difference includes 0, 300 resamples), choose the simpler of the two by simplicity_order; otherwise choose the lower-MAE model",
    "test_set_policy": "2025 is evaluated exactly once, after regression_frozen_config.json is completed with the frozen model and feature list; never re-run",
}


def main():
    if REGRESSION_LOCK_PATH.exists():
        sys.exit("The regression track's 2025 test set has already been evaluated; the rule must not change.")
    REGRESSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if REGRESSION_FROZEN_CONFIG_PATH.exists():
        existing = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())
        if existing.get("status") == "frozen":
            sys.exit("regression_frozen_config.json is already frozen; refusing to overwrite the registered rule.")

    rule_sha256 = hashlib.sha256(json.dumps(SELECTION_RULE, sort_keys=True).encode()).hexdigest()
    registered = {
        "status": "registered",
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": SELECTION_RULE,
        "selection_rule_sha256": rule_sha256,
        "note": "status becomes 'frozen' only in freeze_regression_config.py, run after validation and BEFORE the one-shot test evaluation",
    }
    REGRESSION_FROZEN_CONFIG_PATH.write_text(json.dumps(registered, indent=2))
    print(f"Registered regression selection rule -> {REGRESSION_FROZEN_CONFIG_PATH}")
    print(f"selection_rule_sha256: {rule_sha256}")


if __name__ == "__main__":
    main()
