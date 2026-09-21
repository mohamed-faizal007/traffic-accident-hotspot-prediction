"""Freeze everything chosen on 2021-2024 data BEFORE the 2025 test set is evaluated.

Run after train_validate.py, validation_comparison.py and the tests. Refuses to run if the
tests fail, if the selection rule differs from the one saved before tuning, or if the test
set has already been evaluated. evaluate_test.py verifies these hashes before scoring 2025.
"""

import datetime
import hashlib
import json
import subprocess
import sys

from config import FROZEN_CONFIG_PATH, METRICS_PATH, MODELS_DIR, FEATURES_PATH, PROJECT_ROOT, RESULTS_DIR

LOCK_PATH = RESULTS_DIR / "test_evaluated.lock"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()


def main():
    if LOCK_PATH.exists():
        sys.exit("Test set already evaluated; the frozen config must not change.")
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=PROJECT_ROOT)
    if tests.returncode != 0:
        sys.exit("Tests failed; not freezing.")

    saved = json.loads(FROZEN_CONFIG_PATH.read_text())
    rule = saved["selection_rule"]
    rule_hash = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
    if rule_hash != saved["selection_rule_sha256"]:
        sys.exit("Selection rule differs from the one saved before tuning.")

    metrics = json.loads(METRICS_PATH.read_text())
    if metrics.get("test") is not None:
        sys.exit("metrics.json already contains test results.")
    sel = metrics["selection"]

    frozen = {
        "status": "frozen",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "rule_written_at_utc": saved["written_at_utc"],
        "selection_rule": rule,
        "selection_rule_sha256": rule_hash,
        "selection_outcome": sel["decision"],
        "model": {
            "family": sel["model_family"],
            "params": sel["model_params"],
            "training": f"targets <= 2023-12, all positives + negatives sampled 1:{sel['negatives_per_positive']} (seed 42)",
            "file": "models/final_model.joblib",
        },
        "features": sel["features"],
        "cluster_features_used": sel["use_cluster_features"],
        "calibration": {"kind": sel["calibration"]["chosen"], "fit_on": "all 2024 target months",
                        "file": "models/calibrator.joblib"},
        "decision_threshold": {"raw_score": sel["raw_f1_threshold"],
                               "calibrated_probability": sel["calibrated_threshold"],
                               "chosen_by": "max F1 on 2024 validation targets"},
        "risk_tiers": {"type": "rank-based within each month (share of grids)", "fractions": sel["tier_top_fractions"]},
        "split": {"train_targets": "<= 2023-12", "validation_targets": "2024", "test_targets": "2025"},
        "hashes_sha256": {
            "models/final_model.joblib": sha256(MODELS_DIR / "final_model.joblib"),
            "models/calibrator.joblib": sha256(MODELS_DIR / "calibrator.joblib"),
            "data/processed/ml_features.parquet": sha256(FEATURES_PATH),
            "metrics.validation": hashlib.sha256(
                json.dumps(metrics["validation"], sort_keys=True, default=float).encode()).hexdigest(),
        },
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_files": git("status", "--porcelain").splitlines(),
    }
    frozen["config_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    FROZEN_CONFIG_PATH.write_text(json.dumps(frozen, indent=2))
    print(f"Frozen config saved: {FROZEN_CONFIG_PATH}\nconfig sha256: {frozen['config_sha256']}")


if __name__ == "__main__":
    main()
