"""Train on 2021-2023 targets; tune, select, calibrate and set thresholds on 2024 targets.

The 2025 test set is NEVER touched here (see evaluate_test.py, run once at the end).
The model-selection rule is read from results/frozen_config.json, where it was saved
BEFORE any of this tuning was run.
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, make_scorer, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURES_PATH, RESULTS_DIR, METRICS_PATH, FROZEN_CONFIG_PATH, INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, HOTSPOT_MIN_COLLISIONS, TIER_TOP_FRACTIONS, VAL_YEAR,
)
from evaluation import evaluate_scores, paired_bootstrap_ap_diff, precision_recall_at_k
from features import FEATURE_COLUMNS, build_feature_table
from freeze_config import LOCK_PATH
from splits import assign_split

NEGATIVES_PER_POSITIVE = 4

# RF grid widened beyond the previous edge (depth 10 / leaf 25 were the edge values before).
RF_GRID = [
    dict(n_estimators=150, max_depth=d, min_samples_leaf=leaf)
    for d in (4, 6, 8, 10, 14) for leaf in (25, 50, 100, 200)
]
LR_GRID = [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)]
HGB_GRID = [
    dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl)
    for lr in (0.03, 0.1) for leaves in (8, 15, 31) for msl in (50, 200)
]


# ------------------------------------------------------------------ data
def load_split_frames(df=None):
    """Train and validation rows only. Test rows are never returned by this module."""
    if df is None:
        df = pd.read_parquet(FEATURES_PATH)
    return {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "val")}


def sample_training(train, seed=RANDOM_STATE, ratio=NEGATIVES_PER_POSITIVE):
    pos = train[train["hotspot"] == 1]
    negatives = train[train["hotspot"] == 0]
    neg = negatives.sample(n=min(len(pos) * ratio, len(negatives)), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


# ------------------------------------------------------------------ models
def fit_rf(train_s, features, params, seed=RANDOM_STATE):
    model = RandomForestClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=-1)
    return model.fit(train_s[features], train_s["hotspot"].astype(int))


def fit_lr(train_s, features, params, seed=RANDOM_STATE):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced",
                                                               random_state=seed, **params))
    return model.fit(train_s[features], train_s["hotspot"].astype(int))


def fit_hgb(train_s, features, params, seed=RANDOM_STATE):
    model = HistGradientBoostingClassifier(
        max_iter=400, l2_regularization=1.0, early_stopping=True, n_iter_no_change=20,
        validation_fraction=0.1, random_state=seed, **params)
    return model.fit(train_s[features], train_s["hotspot"].astype(int))


FAMILIES = {
    "logistic_regression": (fit_lr, LR_GRID),
    "hist_gradient_boosting": (fit_hgb, HGB_GRID),
    "random_forest": (fit_rf, RF_GRID),
}


def score(model, frame, features):
    return model.predict_proba(frame[features])[:, 1]


def tune_family(name, train_s, val, features):
    fit, grid = FAMILIES[name]
    y = val["hotspot"].to_numpy().astype(int)
    rows, best = [], None
    for params in grid:
        model = fit(train_s, features, params)
        s = score(model, val, features)
        ap = float(average_precision_score(y, s))
        rows.append({"family": name, **params, "val_average_precision": ap})
        if best is None or ap > best["ap"]:
            best = {"ap": ap, "params": params, "model": model, "scores": s}
    print(f"{name}: best val AP {best['ap']:.4f} with {best['params']}")
    return best, rows


def apply_selection_rule(rule, results, val, y):
    """Implements the rule saved in frozen_config.json (see its `selection_rule`)."""
    ranked = sorted(results, key=lambda f: results[f]["ap"], reverse=True)
    first, second = ranked[0], ranked[1]
    diff = paired_bootstrap_ap_diff(y, results[first]["scores"], results[second]["scores"], val["grid_id"].to_numpy())
    within_noise = diff["ci95"][0] <= 0 <= diff["ci95"][1]
    order = rule["simplicity_order_simplest_first"]
    if within_noise:
        chosen = min((first, second), key=order.index)
        reason = "top two within bootstrap noise; chose the simpler"
    else:
        chosen, reason = first, "top model clearly ahead of the runner-up"
    return chosen, {"ranking_by_val_ap": [(f, results[f]["ap"]) for f in ranked],
                    "top_two": [first, second], "top_two_ap_diff": diff,
                    "within_noise": bool(within_noise), "chosen": chosen, "reason": reason}


# ------------------------------------------------------------- calibration
class PlattCalibrator:
    """Sigmoid calibration on the logit of the raw score (monotone)."""

    def fit(self, s, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(s), y)
        return self

    @staticmethod
    def _x(s):
        s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def predict(self, s):
        return self.lr.predict_proba(self._x(s))[:, 1]


class IsotonicCalibrator:
    def fit(self, s, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return self

    def predict(self, s):
        return self.iso.predict(s)


def make_calibrator(kind):
    return IsotonicCalibrator() if kind == "isotonic" else PlattCalibrator()


def half_masks(val):
    months = pd.to_datetime(val["target_month"]).dt.month
    return (months <= 6).to_numpy(), (months > 6).to_numpy()


def choose_calibrator(val, raw):
    """Fit on the first half of 2024 targets, judge on the second half (honest Brier)."""
    h1, h2 = half_masks(val)
    y = val["hotspot"].to_numpy().astype(int)
    report = {"raw_brier_h2": float(brier_score_loss(y[h2], raw[h2])), "candidates": {}}
    for kind in ("isotonic", "platt"):
        p2 = make_calibrator(kind).fit(raw[h1], y[h1]).predict(raw[h2])
        report["candidates"][kind] = {
            "brier_h2": float(brier_score_loss(y[h2], p2)),
            "mean_pred_h2": float(p2.mean()),
            "ap_h2": float(average_precision_score(y[h2], p2)),
        }
    report["base_rate_h2"] = float(y[h2].mean())
    report["base_rate_brier_h2"] = float(brier_score_loss(y[h2], np.full(h2.sum(), y[h2].mean())))
    report["chosen"] = min(report["candidates"], key=lambda k: report["candidates"][k]["brier_h2"])
    final = make_calibrator(report["chosen"]).fit(raw, y)   # frozen calibrator: all of 2024
    return final, report


def reliability_table(y, p, bins=10):
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()


# --------------------------------------------------------------- threshold
def best_f1_threshold(y, s):
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def threshold_transfer_check(val, raw):
    """Choose the threshold on H1 of 2024, apply it to H2: is the choice stable?"""
    h1, h2 = half_masks(val)
    y = val["hotspot"].to_numpy().astype(int)
    t, f1_h1 = best_f1_threshold(y[h1], raw[h1])
    pred = raw[h2] >= t
    tp = int((pred & (y[h2] == 1)).sum())
    p = tp / max(pred.sum(), 1)
    r = tp / max(y[h2].sum(), 1)
    return {"threshold_from_h1": t, "f1_h1": f1_h1, "f1_h2": 2 * p * r / max(p + r, 1e-12),
            "flagged_fraction_h2": float(pred.mean())}


# --------------------------------------------------------------- baselines
def baseline_report(frame):
    y = frame["hotspot"].to_numpy().astype(int)
    months = frame["target_month"].to_numpy()
    out = {}
    persist = frame["base_last_month_count"].to_numpy()
    res = evaluate_scores(y, persist, months)
    binary = (persist >= HOTSPOT_MIN_COLLISIONS).astype(int)
    tp = int(((binary == 1) & (y == 1)).sum())
    precision, recall = tp / max(binary.sum(), 1), tp / max(y.sum(), 1)
    res["binary_rule"] = {
        "rule": f"hotspot if last month had >= {HOTSPOT_MIN_COLLISIONS} collisions",
        "flagged_fraction": float(binary.mean()), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    out["persistence"] = res
    res = evaluate_scores(y, frame["base_trailing_12"].to_numpy(), months)
    res["note"] = "score = collisions in the last 12 months (month t inclusive); top-k ranking"
    out["trailing_12_month_count"] = res
    return out


def coverage_report(frame, year):
    """Share of ALL hotspot grid-months of `year` (any grid) that lie in the modelled active grids."""
    c = pd.read_csv(INTERIM_DATA_DIR / "collisions_with_grid.csv", usecols=["grid_id", "datetime"])
    c["m"] = pd.to_datetime(c["datetime"]).dt.to_period("M")
    c = c[c["m"].dt.year == year]
    gm = c.groupby(["grid_id", "m"]).size()
    hot = gm[gm >= HOTSPOT_MIN_COLLISIONS]
    in_active = int(hot.index.get_level_values(0).isin(set(frame["grid_id"].unique())).sum())
    return {"hotspot_grid_months_all_grids": int(len(hot)), "in_modelled_grids": in_active,
            "coverage": in_active / len(hot),
            "note": "system-wide recall is capped by this coverage; grids first active after 2023 are not modelled"}


def run_alignment_ablation(monthly, features, rf_params):
    """Same RF on the ORIGINAL alignment (delay=1) vs the new one (delay=0), validation only."""
    out = {}
    for delay in (1, 0):
        df = build_feature_table(monthly, delay=delay)
        df["split"] = assign_split(df)
        frames = load_split_frames(df=df)
        s = score(fit_rf(sample_training(frames["train"]), features, rf_params), frames["val"], features)
        y = frames["val"]["hotspot"].to_numpy().astype(int)
        at_k = precision_recall_at_k(y, s, frames["val"]["target_month"].to_numpy())
        out[f"delay_{delay}"] = {
            "alignment": "features use months <= t (new)" if delay == 0 else "features use months <= t-1 (original)",
            "val_average_precision": float(average_precision_score(y, s)),
            "val_top_1pct_precision": at_k["top_1pct"]["precision"],
        }
        print(f"alignment delay={delay}: val AP={out[f'delay_{delay}']['val_average_precision']:.4f}")
    return out


def plot_reliability(raw, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(raw.mean_predicted, raw.observed_rate, "o-", label="raw score")
    ax.plot(cal.mean_predicted, cal.observed_rate, "s-", label="calibrated (fit on 2024, in-sample)")
    hi = max(raw.mean_predicted.max(), raw.observed_rate.max())
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect")
    ax.set_xlabel("mean predicted")
    ax.set_ylabel("observed hotspot rate")
    ax.set_title("Reliability (validation 2024, deciles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "reliability_val.png", dpi=130)


def feature_importance(model, val, features):
    """Permutation importance (drop in AP) on a validation sample; works for any model family."""
    sample = val.sample(n=min(80_000, len(val)), random_state=RANDOM_STATE)
    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    result = permutation_importance(model, sample[features], sample["hotspot"].astype(int), scoring=scorer,
                                    n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"feature": features, "importance": result.importances_mean, "std": result.importances_std})
            .sort_values("importance", ascending=False))


# -------------------------------------------------------------------- main
def main():
    if LOCK_PATH.exists():
        sys.exit("The 2025 test set has been evaluated; results and models are frozen. Not overwriting.")
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    frozen = json.loads(FROZEN_CONFIG_PATH.read_text())
    rule = frozen["selection_rule"]
    features = FEATURE_COLUMNS

    frames = load_split_frames()
    train, val = frames["train"], frames["val"]
    y_val = val["hotspot"].to_numpy().astype(int)
    months_val = val["target_month"].to_numpy()
    print(f"train rows {len(train):,} (pos {int(train.hotspot.sum()):,}); val rows {len(val):,} (pos {int(y_val.sum()):,})")
    train_s = sample_training(train)

    results, tuning_rows = {}, []
    for name in rule["candidates"]:
        results[name], rows = tune_family(name, train_s, val, features)
        tuning_rows += rows
    pd.DataFrame(tuning_rows).to_csv(RESULTS_DIR / "tuning_results.csv", index=False)

    chosen_name, decision = apply_selection_rule(rule, results, val, y_val)
    print(f"SELECTED: {chosen_name} ({decision['reason']})")
    chosen = results[chosen_name]
    model, raw_val = chosen["model"], chosen["scores"]

    calibrator, cal_report = choose_calibrator(val, raw_val)
    cal_val = calibrator.predict(raw_val)
    raw_threshold, _ = best_f1_threshold(y_val, raw_val)
    cal_threshold = float(calibrator.predict(np.array([raw_threshold]))[0])
    threshold_check = threshold_transfer_check(val, raw_val)

    rel_raw, rel_cal = reliability_table(y_val, raw_val), reliability_table(y_val, cal_val)
    pd.concat([rel_raw.assign(kind="raw"), rel_cal.assign(kind="calibrated")]).to_csv(
        RESULTS_DIR / "val_reliability.csv", index=False)
    plot_reliability(rel_raw, rel_cal)

    chosen_raw = evaluate_scores(y_val, raw_val, months_val, threshold=raw_threshold)
    chosen_raw["raw_score_note"] = "raw score: NOT a probability (1:4 sampling / class weights)"
    chosen_cal = evaluate_scores(y_val, cal_val, months_val, probability=True, threshold=cal_threshold)

    scores_out = val[["grid_id", "target_month", "hotspot", "base_trailing_12"]].copy()
    for name, r in results.items():
        scores_out[name] = r["scores"]
    scores_out.to_parquet(RESULTS_DIR / "val_scores.parquet", index=False)

    monthly = pd.read_csv(PROCESSED_DATA_DIR / "grid_monthly_collisions.csv", parse_dates=["year_month"])
    validation = {
        "n_train": int(len(train)), "n_val": int(len(val)),
        "chosen_model_raw_score": chosen_raw,
        "chosen_model_calibrated": chosen_cal,
        "all_candidates": {n: evaluate_scores(y_val, r["scores"], months_val) for n, r in results.items()},
        "baselines": baseline_report(val),
        "alignment_ablation": run_alignment_ablation(monthly, features, results["random_forest"]["params"]),
        "coverage": coverage_report(val, VAL_YEAR),
    }
    selection = {
        "model_family": chosen_name,
        "model_params": chosen["params"],
        "candidate_best_params": {n: r["params"] for n, r in results.items()},
        "decision": decision,
        "features": features,
        "use_cluster_features": False,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "calibration": cal_report,
        "raw_f1_threshold": raw_threshold,
        "calibrated_threshold": cal_threshold,
        "threshold_transfer_check_h1_to_h2": threshold_check,
        "tier_top_fractions": TIER_TOP_FRACTIONS,
        "selected_on": f"target months of {VAL_YEAR} only",
    }

    joblib.dump(model, MODELS_DIR / "final_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")
    feature_importance(model, val, features).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    metrics = {
        "protocol": "train targets 2021-2023 | validation targets 2024 | test targets 2025 (evaluated once, frozen)",
        "selection": selection, "validation": validation, "test": None,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print_summary(validation, selection)


def print_summary(v, sel):
    m, base = v["chosen_model_raw_score"], v["baselines"]
    print("\n===== VALIDATION (2024 targets) =====")
    print(f"base rate {m['base_rate']:.4f}")
    rows = [(f"CHOSEN {sel['model_family']}", m)] + [(k, b) for k, b in base.items()]
    for k, r in rows:
        print(f"{k:34s} AP {r['average_precision']:.4f}  AUC {r['roc_auc']:.4f}  top1% P {r['at_top_k_per_month']['top_1pct']['precision']:.4f}")
    print("calibration:", json.dumps(sel["calibration"]["candidates"]), "chosen:", sel["calibration"]["chosen"])


if __name__ == "__main__":
    main()
