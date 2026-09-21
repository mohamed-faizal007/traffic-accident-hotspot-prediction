"""Evaluate the FROZEN model on the 2025 test targets. Runs exactly once.

Also writes the Jan-2026 forecast (features up to Dec 2025, no label) and the dashboard
prediction files. The lock file stops any second evaluation; nothing here changes the model,
threshold, calibrator or tiers, which are all read from results/frozen_config.json.
"""

import datetime
import hashlib
import json
import sys

import joblib
import pandas as pd

from config import (FROZEN_CONFIG_PATH, METRICS_PATH, MODELS_DIR, FEATURES_PATH, RESULTS_DIR,
                    OUTPUTS_DIR, TEST_YEAR)
from evaluation import assign_tiers, evaluate_scores
from freeze_config import LOCK_PATH, sha256
from train_validate import baseline_report, coverage_report


def load_calibrator(path):
    """The calibrator was pickled while train_validate.py ran as a script, so its class is recorded
    as __main__.<Class>. Make those names resolvable; the pickled file itself is unchanged."""
    import __main__
    import train_validate
    for name in ("PlattCalibrator", "IsotonicCalibrator"):
        setattr(__main__, name, getattr(train_validate, name))
    return joblib.load(path)


def verify_frozen(frozen):
    if frozen.get("status") != "frozen":
        sys.exit("Config is not frozen; run freeze_config.py first.")
    body = {k: v for k, v in frozen.items() if k != "config_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != frozen["config_sha256"]:
        sys.exit("frozen_config.json was modified after freezing.")
    files = {
        "models/final_model.joblib": MODELS_DIR / "final_model.joblib",
        "models/calibrator.joblib": MODELS_DIR / "calibrator.joblib",
        "data/processed/ml_features.parquet": FEATURES_PATH,
    }
    for rel, path in files.items():
        if sha256(path) != frozen["hashes_sha256"][rel]:
            sys.exit(f"{rel} changed since freezing.")


def plot_pr(y, raw, test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, s in (("frozen model", raw), ("trailing-12-month count", test["base_trailing_12"].to_numpy()),
                     ("persistence (last month)", test["base_last_month_count"].to_numpy())):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=label)
    ax.axhline(y.mean(), color="k", ls="--", lw=1, label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall, 2025 test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    if LOCK_PATH.exists():
        sys.exit("The 2025 test set has already been evaluated. Re-running is not allowed.")
    frozen = json.loads(FROZEN_CONFIG_PATH.read_text())
    verify_frozen(frozen)

    features = frozen["features"]
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    calibrator = load_calibrator(MODELS_DIR / "calibrator.joblib")
    thr = frozen["decision_threshold"]
    fractions = frozen["risk_tiers"]["fractions"]

    df = pd.read_parquet(FEATURES_PATH)
    test = df[df["split"] == "test"].reset_index(drop=True)
    future = df[df["split"] == "unlabeled"].reset_index(drop=True)
    y = test["hotspot"].to_numpy().astype(int)
    months = test["target_month"].to_numpy()

    raw = model.predict_proba(test[features])[:, 1]
    prob = calibrator.predict(raw)

    raw_res = evaluate_scores(y, raw, months, threshold=thr["raw_score"])
    cal_res = evaluate_scores(y, prob, months, probability=True, threshold=thr["calibrated_probability"])
    coverage = coverage_report(test, TEST_YEAR)
    hot_all = coverage["hotspot_grid_months_all_grids"]
    conf = raw_res["confusion"]
    system_wide = {
        "recall_on_modelled_grids_at_threshold": conf["recall"],
        "recall_system_wide_at_threshold": conf["tp"] / hot_all,
        "top_k_per_month": {
            k: {"recall_on_modelled_grids": v["recall"],
                "recall_system_wide": v["recall"] * coverage["in_modelled_grids"] / hot_all}
            for k, v in raw_res["at_top_k_per_month"].items()
        },
    }
    test_metrics = {
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "frozen_config_sha256": frozen["config_sha256"],
        "model": frozen["model"]["family"],
        "score_raw": raw_res,
        "score_calibrated": cal_res,
        "baselines": baseline_report(test),
        "coverage": coverage,
        "system_wide": system_wide,
        "note": "evaluated once; nothing was tuned or changed after seeing these numbers",
    }

    metrics = json.loads(METRICS_PATH.read_text())
    metrics["test"] = test_metrics
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    LOCK_PATH.write_text(test_metrics["evaluated_at_utc"])

    # dashboard outputs: same frozen model, calibrator and tiers, nothing re-tuned
    OUTPUTS_DIR.mkdir(exist_ok=True)
    test_out = test[["grid_id", "grid_latitude", "grid_longitude", "target_month", "hotspot", "base_trailing_12"]].copy()
    test_out["score"], test_out["probability"] = raw, prob
    test_out["risk_tier"] = assign_tiers(raw, months, fractions)
    test_out.to_parquet(OUTPUTS_DIR / "test_predictions_2025.parquet", index=False)

    fut_raw = model.predict_proba(future[features])[:, 1]
    fut = future[["grid_id", "grid_latitude", "grid_longitude", "target_month"]].copy()
    fut["score"], fut["probability"] = fut_raw, calibrator.predict(fut_raw)
    fut["risk_tier"] = assign_tiers(fut_raw, fut["target_month"].to_numpy(), fractions)
    fut["rank"] = fut["score"].rank(ascending=False, method="first").astype(int)
    fut.sort_values("rank").to_csv(OUTPUTS_DIR / "forecast_2026-01.csv", index=False)
    plot_pr(y, raw, test, RESULTS_DIR / "figures" / "test_pr_curve.png")

    print(f"TEST 2025: base rate {raw_res['base_rate']:.4f}  AP {raw_res['average_precision']:.4f}  "
          f"AUC {raw_res['roc_auc']:.4f}  Brier(cal) {cal_res['brier']:.4f}")
    for k, r in test_metrics["baselines"].items():
        print(f"  {k}: AP {r['average_precision']:.4f}")
    print(f"Forecast rows: {len(fut):,} ({fut['target_month'].iloc[0]:%Y-%m})")


if __name__ == "__main__":
    main()
