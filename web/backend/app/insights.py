"""Read-only endpoints for the action list, patterns, regression track, tiers and ablations.

Every number is read from tracked saved outputs (results/ and outputs/); nothing is computed by a model
here and nothing is hard-coded. The scripts that produce the outputs/ exports are in src/dashboard_exports/.
"""

import math
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from . import data

router = APIRouter(prefix="/api")

FORECAST_STATUS = "unvalidated forecast (no outcomes yet)"
DRIVERS_NOTE = "descriptive, not a model attribution"
REGENERATE = {
    "cells": "Run src/dashboard_exports/forecast_cells.py.",
    "regression_forecast": "Run src/dashboard_exports/regression_forecast.py.",
    "temporal": "Run src/dashboard_exports/temporal_patterns.py.",
    "clusters": "Run src/dashboard_exports/dbscan_clusters.py.",
}
SHARE_COLUMNS = [
    "historical_rush_hour_share", "historical_weekend_share", "historical_night_share",
    "historical_adverse_weather_share", "historical_wet_road_share",
    "historical_single_carriageway_share", "historical_high_speed_share",
]


def _need(obj, name: str, hint: str = ""):
    if obj is None:
        raise HTTPException(status_code=503, detail=f"{name} not found. {hint}".strip())
    return obj


def _csv_list(value: Optional[str]) -> Optional[list]:
    return None if value is None else [v for v in value.split(",") if v]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _drivers(row) -> list:
    """Descriptive facts about the cell, with percentiles computed from the data in the export."""
    return [
        {"key": "collisions_last_12_months", "label": "Collisions in the last 12 months",
         "value": row["collisions_last_12_months"], "percentile": row["pct_last_12_months"]},
        {"key": "historical_avg_collisions", "label": "Long-run average collisions per month",
         "value": row["historical_avg_collisions"], "percentile": row["pct_historical_avg"]},
        {"key": "months_since_last_collision", "label": "Months since the last collision",
         "value": row["months_since_last_collision"], "percentile": None},
        {"key": "collisions_lag_0", "label": "Collisions in the latest month",
         "value": row["collisions_lag_0"], "percentile": None},
        {"key": "historical_max_collisions", "label": "Most collisions in any single month",
         "value": row["historical_max_collisions"], "percentile": None},
    ]


def _summary(row) -> str:
    n12 = int(row["collisions_last_12_months"])
    return (
        f"{_plural(n12, 'collision')} in the last 12 months (more than {math.floor(row['pct_last_12_months'])}% of modelled cells); "
        f"long-run average {row['historical_avg_collisions']:.2f} per month (more than {math.floor(row['pct_historical_avg'])}% of modelled cells); "
        f"{_plural(int(row['months_since_last_collision']), 'month')} since the last collision; "
        f"busiest month had {int(row['historical_max_collisions'])}."
    )


def _cell_record(row, history_start: Optional[str]) -> dict:
    history = [int(x) for x in str(row["history_12m"]).split(";")]
    months = None
    if history_start:
        start = pd.Period(history_start, "M")
        months = [str(start + i) for i in range(len(history))]
    return {
        "rank": int(row["rank"]),
        "grid_id": row["grid_id"],
        "grid_latitude": row["grid_latitude"],
        "grid_longitude": row["grid_longitude"],
        "risk_tier": row["risk_tier"],
        "quantile_tier": row["quantile_tier"],
        "score": row["score"],
        "probability": row["probability"],
        "predicted_count_xgboost": row["predicted_count_xgboost"],
        "predicted_count_baseline": row["predicted_count_baseline"],
        "road": row["road_tag"],
        "weather": row["weather_tag"],
        "shares": {c.removeprefix("historical_").removesuffix("_share"): row[c] for c in SHARE_COLUMNS},
        "drivers": _drivers(row),
        "summary": _summary(row),
        "history_12m": history,
        "history_months": months,
    }


def filter_frame(df: pd.DataFrame, tiers=None, quantile_tiers=None, road=None, weather=None,
                 min_adverse_share=None, min_single_share=None) -> pd.DataFrame:
    if tiers is not None:
        df = df[df["risk_tier"].isin(tiers)]
    if quantile_tiers is not None:
        df = df[df["quantile_tier"].isin([q.upper() for q in quantile_tiers])]
    if road:
        df = df[df["road_tag"] == road]
    if weather:
        df = df[df["weather_tag"] == weather]
    if min_adverse_share is not None:
        df = df[df["historical_adverse_weather_share"] >= min_adverse_share]
    if min_single_share is not None:
        df = df[df["historical_single_carriageway_share"] >= min_single_share]
    return df


def share_percentiles(df: pd.DataFrame) -> dict:
    """Data-derived cut points, so the UI can offer presets that actually select a sensible number of cells."""
    return {
        name: {f"p{int(q * 100)}": float(df[col].quantile(q)) for q in (0.5, 0.75, 0.9)}
        for name, col in (("adverse_weather", "historical_adverse_weather_share"),
                          ("single_carriageway", "historical_single_carriageway_share"))
    }


def _frame():
    return _need(data.action_frame(), "The forecast or per-cell drivers file", REGENERATE["cells"])


@router.get("/action-list")
def action_list(
    n: int = Query(20, ge=1, le=500),
    offset: int = Query(0, ge=0),
    tiers: Optional[str] = Query(None, description="Comma-separated classifier risk tiers"),
    quantile_tiers: Optional[str] = Query(None, description="Comma-separated LOW,MEDIUM,HIGH"),
    road: Optional[str] = Query(None, pattern="^(single|other)$"),
    weather: Optional[str] = Query(None, pattern="^(adverse|fine)$"),
    sort: str = Query("classifier", pattern="^(classifier|regression)$"),
    min_adverse_share: Optional[float] = Query(None, ge=0, le=1, description="minimum share of the cell's past collisions in adverse weather"),
    min_single_share: Optional[float] = Query(None, ge=0, le=1, description="minimum share on single carriageways"),
):
    df = _frame()
    meta = data.load_forecast_cells_meta() or {}
    shown = filter_frame(df, _csv_list(tiers), _csv_list(quantile_tiers), road, weather, min_adverse_share, min_single_share)
    if sort == "regression":
        shown = shown.sort_values("predicted_count_xgboost", ascending=False, na_position="last")
    rows = shown.iloc[offset:offset + n]
    return data.clean({
        "target_month": df["target_month"].iloc[0][:7] if len(df) else None,
        "status": FORECAST_STATUS,
        "drivers_note": DRIVERS_NOTE,
        "sort": sort,
        "regression_available": bool(df["predicted_count_xgboost"].notna().any()),
        "filters": {"tiers": _csv_list(tiers), "quantile_tiers": _csv_list(quantile_tiers), "road": road, "weather": weather,
                    "min_adverse_share": min_adverse_share, "min_single_share": min_single_share},
        "share_percentiles": share_percentiles(df),
        "total": int(len(shown)),
        "total_unfiltered": int(len(df)),
        "offset": offset,
        "n": n,
        "road_weather_rule": f"single carriageway / adverse weather when that share of the cell's past collisions is above {data.SHARE_TAG_THRESHOLD}",
        "rows": [_cell_record(r, meta.get("history_start")) for _, r in rows.iterrows()],
    })


@router.get("/cell/{grid_id}")
def cell(grid_id: str):
    df = _frame()
    match = df[df["grid_id"] == grid_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Grid cell {grid_id} is not among the modelled cells.")
    meta = data.load_forecast_cells_meta() or {}
    return data.clean({
        "target_month": match["target_month"].iloc[0][:7],
        "status": FORECAST_STATUS,
        "drivers_note": DRIVERS_NOTE,
        **_cell_record(match.iloc[0], meta.get("history_start")),
    })


@router.get("/temporal-patterns")
def temporal_patterns():
    t = _need(data.load_temporal(), "outputs/dashboard/temporal_patterns.json", REGENERATE["temporal"])
    wk = t["weekend"]
    share = wk["weekend_share_of_days"]
    per_day_ratio = (wk["weekend"] / share) / (wk["weekday"] / (1 - share)) if wk["weekday"] else None
    return data.clean({**t, "weekend": {**wk, "per_day_ratio_weekend_vs_weekday": per_day_ratio}})


@router.get("/dbscan/summary")
def dbscan_summary():
    sel = _need(data.load_dbscan_selection(), "results/dbscan_selection.json")
    sweep = _need(data.load_dbscan_sweep(), "results/dbscan_sweep.csv")
    return data.clean({
        "fit_period": sel["fit_period"],
        "criterion": sel["criterion"],
        "chosen": {"eps_km": sel["eps_km"], "min_samples": sel["min_samples"]},
        "stats": sel["stats"],
        "sweep": sweep.to_dict(orient="records"),
        "morans_i": data.load_morans(),
        "ablation_note": "Cluster features were tested as predictors and did not improve prediction (see the ablations).",
    })


@router.get("/dbscan/clusters")
def dbscan_clusters(min_size: int = Query(0, ge=0), limit: int = Query(3000, ge=1, le=5000)):
    df = _need(data.load_dbscan_clusters(), "outputs/dashboard/dbscan_clusters.csv", REGENERATE["clusters"])
    sel = data.load_dbscan_selection() or {}
    view = df[df["size"] >= min_size]
    return data.clean({
        "fit_period": sel.get("fit_period"),
        "total": int(len(df)),
        "matching": int(len(view)),
        "returned": int(min(len(view), limit)),
        "clusters": view.head(limit).to_dict(orient="records"),
    })


def _gap(block: dict) -> dict:
    x, b = block["xgboost"]["mae"], block["historical_average_baseline"]["mae"]
    return {"xgboost_mae": x, "baseline_mae": b, "xgboost_minus_baseline": x - b, "relative_difference": (x - b) / b}


@router.get("/regression/results")
def regression_results():
    m = _need(data.load_regression_metrics(), "results/regression/regression_metrics.json")
    val, test, sel = m["validation"], m.get("test"), m["selection"]
    cand = val["all_candidates"]
    out = {
        "protocol": m["protocol"],
        "model": sel["model_family"],
        "params": sel["model_params"],
        "target_transform": sel["target_transform"],
        "features": sel["features"],
        "selection_decision": sel["decision"],
        "validation": {
            "split_label": "2024 validation (used for tuning and model selection)",
            "n": val["n_val"],
            "models": [{"model": r["model"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]} for r in val["comparison_table"]],
            "mean_actual": cand["xgboost"]["mean_actual"],
            "mae_gap": _gap(cand),
        },
        "test": None,
    }
    if test:
        out["test"] = {
            "split_label": "2025 test (evaluated once, frozen)",
            "evaluated_at_utc": test["evaluated_at_utc"],
            "n": test["xgboost"]["n"],
            "models": [{"model": k, **{x: test[k][x] for x in ("mae", "rmse", "r2")}}
                       for k in ("xgboost", "historical_average_baseline")],
            "mean_actual": test["xgboost"]["mean_actual"],
            "mae_gap": _gap(test),
            "note": test["note"],
        }
    return data.clean(out)


@router.get("/regression/forecast")
def regression_forecast(n: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0)):
    df = _need(data.load_regression_forecast(), "outputs/regression_forecast_2026-01.csv", REGENERATE["regression_forecast"])
    meta = data.load_regression_forecast_meta() or {}
    model = meta.get("model")
    rows = df.sort_values("rank").iloc[offset:offset + n]
    return data.clean({
        "status": meta.get("status", FORECAST_STATUS),
        "target_month": df["target_month"].iloc[0][:7] if len(df) else None,
        "total": int(len(df)),
        "model": model.get("family") if isinstance(model, dict) else model,
        "model_hash_verified": meta.get("model_hash_matches_frozen_config"),
        "generated_at_utc": meta.get("generated_at_utc"),
        "rows": rows.to_dict(orient="records"),
    })


@router.get("/quantile-tiers")
def quantile_tiers():
    q = _need(data.load_quantile_tiers(), "results/regression/quantile_tiers_2025.parquet")
    metrics = data.load_metrics() or {}
    order_q, order_r = ["LOW", "MEDIUM", "HIGH"], data.TIER_ORDER

    def block(col, order):
        g = q.groupby(col)["hotspot"].agg(["size", "sum"]).reindex(order)
        return [{"tier": t, "cell_months": int(r["size"]), "hotspots": int(r["sum"]), "hotspot_rate": float(r["sum"] / r["size"])}
                for t, r in g.iterrows()]

    cross = pd.crosstab(q["risk_tier"], q["quantile_tier"]).reindex(index=order_r, columns=order_q, fill_value=0)
    out = {
        "split_label": "2025 test predictions (frozen classifier scores)",
        "quantile_rule": "within-month terciles of the frozen classifier's score (lowest third LOW, middle third MEDIUM, top third HIGH)",
        "rank_rule_top_fractions": (metrics.get("selection") or {}).get("tier_top_fractions"),
        "overall_hotspot_rate": float(q["hotspot"].mean()),
        "quantile_tiers": block("quantile_tier", order_q),
        "rank_tiers": block("risk_tier", order_r),
        "crosstab": {"rows": order_r, "columns": order_q, "counts": cross.to_numpy().tolist()},
        "forecast": None,
    }
    fc = data.action_frame()
    if fc is not None:
        out["forecast"] = {
            "target_month": fc["target_month"].iloc[0][:7],
            "status": FORECAST_STATUS,
            "quantile_tiers": [{"tier": t, "cells": int((fc["quantile_tier"] == t).sum())} for t in order_q],
            "rank_tiers": [{"tier": t, "cells": int((fc["risk_tier"] == t).sum())} for t in order_r],
        }
    return data.clean(out)


@router.get("/ablations")
def ablations():
    metrics = _need(data.load_metrics(), "results/metrics.json", "Run the pipeline first.")
    val = metrics["validation"]
    items = []

    ca = val.get("cluster_ablation")
    if ca:
        ap = ca["summary"]["ap"]
        ci = ((val.get("bootstrap") or {}).get("comparisons") or {}).get("random_forest_with_cluster - random_forest", {}).get("ap", {}).get("ci95")
        items.append({
            "id": "dbscan_clusters", "track": "classifier", "title": "DBSCAN cluster features",
            "split": "2024 validation", "metric": "average precision", "higher_is_better": True,
            "without": ap["without_cluster_mean"], "with": ap["with_cluster_mean"], "difference": ap["diff_mean"],
            "ci95": ci, "ci_note": "single-seed grid bootstrap" if ci else None,
            "seeds": len(ca["seeds"]), "seeds_with_better": ap["seeds_where_cluster_better"],
            "helps": ca["cluster_features_help"], "used_in_final_model": ca["used_in_final_model"],
            "decision_rule": ca["decision_rule"],
        })

    nc = data.load_neighbor_clf()
    if nc:
        ap = nc["summary"]["ap"]
        comp = next(iter(nc["bootstrap"]["comparisons"].values()))["ap"]
        items.append({
            "id": "neighbor_classifier", "track": "classifier", "title": "Neighbouring-cell features",
            "split": "2024 validation", "metric": "average precision", "higher_is_better": True,
            "without": ap["without_neighbor_mean"], "with": ap["with_neighbor_mean"], "difference": ap["diff_mean"],
            "ci95": comp["ci95"], "ci_note": "single-seed grid bootstrap",
            "seeds": len(nc["seeds"]), "seeds_with_better": ap["seeds_where_neighbor_better"],
            "helps": nc["neighbor_features_help"], "used_in_final_model": False,
            "decision_rule": nc["decision_rule"],
        })

    nr = data.load_neighbor_reg()
    if nr:
        mae = nr["summary"]["mae"]
        b = nr["bootstrap_mae_diff_with_minus_without"]
        items.append({
            "id": "neighbor_regression", "track": "regression", "title": "Neighbouring-cell features",
            "split": "2024 validation", "metric": "MAE", "higher_is_better": False,
            "without": mae["without_neighbor_mean"], "with": mae["with_neighbor_mean"], "difference": mae["diff_mean"],
            "ci95": b["ci95"], "ci_note": "grid bootstrap",
            "seeds": len(nr["seeds"]), "seeds_with_better": mae["seeds_where_neighbor_better"],
            "helps": nr["neighbor_features_help"], "used_in_final_model": False,
            "decision_rule": nr["decision_rule"],
        })
    return data.clean({"items": items})


@router.get("/map/folium")
def folium_map():
    if not data.FOLIUM_MAP_PATH.exists():
        raise HTTPException(status_code=503, detail="outputs/report_compliance/hotspot_map.html not found. Run src/report_compliance/build_folium_map.py.")
    return FileResponse(data.FOLIUM_MAP_PATH, media_type="text/html")
