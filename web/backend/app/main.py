"""Read-only FastAPI backend for the React dashboard.

Serves the same numbers as the Streamlit dashboard (app.py) from the same files
(results/metrics.json, results/feature_importance.csv, outputs/forecast_2026-01.csv,
outputs/test_predictions_2025.parquet). No write endpoints, no model loading, no
training code is imported here.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import data
from .insights import filter_frame, router as insights_router

app = FastAPI(title="Hotspot Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


app.include_router(insights_router)


@app.exception_handler(data.DataFileError)
async def _data_file_error(_: Request, exc: data.DataFileError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(KeyError)
@app.exception_handler(TypeError)
@app.exception_handler(IndexError)
@app.exception_handler(AttributeError)
async def _unexpected_structure(_: Request, exc: Exception):
    # A saved output parsed but lacks the fields the API expects (e.g. an empty or hand-edited file).
    return JSONResponse(
        status_code=503,
        content={"detail": f"A saved output has an unexpected structure (missing {exc!s}). Re-generate it with the pipeline."},
    )


def _require_metrics() -> dict:
    metrics = data.load_metrics()
    if metrics is None:
        raise HTTPException(status_code=503, detail="results/metrics.json not found. Run the pipeline first.")
    return metrics


_clean = data.clean


@app.get("/api/summary")
def summary():
    metrics = _require_metrics()
    frozen = data.load_frozen()
    test, val, sel = metrics.get("test"), metrics["validation"], metrics["selection"]
    label, m, base = data.headline(metrics)
    fractions = sel["tier_top_fractions"]
    cov = (test or val)["coverage"]

    return _clean({
        "model_family": data.model_family_name(metrics),
        "task": "will a 500 m grid cell have >= 2 collisions next month?",
        "frozen": {
            "status": frozen.get("status") if frozen else None,
            "config_sha256": frozen.get("config_sha256") if frozen else None,
            "frozen_at_utc": frozen.get("frozen_at_utc") if frozen else None,
        } if frozen else None,
        "test_evaluated": bool(test),
        "split_label": label,
        "base_rate": m["base_rate"],
        "average_precision": m["average_precision"],
        "ap_lift_over_base_rate": m["ap_lift_over_base_rate"],
        "top_1pct_precision": m["at_top_k_per_month"]["top_1pct"]["precision"],
        "top_1pct_lift": m["at_top_k_per_month"]["top_1pct"]["lift"],
        "baseline_trailing12_top_1pct_precision": base["trailing_12_month_count"]["at_top_k_per_month"]["top_1pct"]["precision"],
        "baseline_trailing12_average_precision": base["trailing_12_month_count"]["average_precision"],
        "ap_gain_over_trailing12": m["average_precision"] - base["trailing_12_month_count"]["average_precision"],
        "coverage": cov["coverage"],
        "tier_top_fractions": fractions,
        "calibration_method": sel["calibration"]["chosen"],
    })


@app.get("/api/predictions")
def predictions(
    tiers: Optional[str] = Query(None, description="Comma-separated risk tiers, e.g. Critical,High"),
    grid_id: Optional[str] = Query(None, description="Substring filter on grid_id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
):
    fc = data.load_forecast()
    if fc is None:
        raise HTTPException(status_code=503, detail="The forecast is written when the 2025 evaluation step runs.")

    shown = fc
    if tiers:
        tier_list = [t for t in tiers.split(",") if t]
        shown = shown[shown["risk_tier"].isin(tier_list)]
    if grid_id:
        shown = shown[shown["grid_id"].astype(str).str.contains(grid_id, regex=False)]

    total = len(shown)
    start = (page - 1) * page_size
    page_rows = shown.iloc[start:start + page_size]
    cols = ["rank", "grid_id", "risk_tier", "score", "probability", "grid_latitude", "grid_longitude"]

    return _clean({
        "target_month": fc["target_month"].iloc[0][:7] if len(fc) else None,
        "total": total,
        "total_unfiltered": len(fc),
        "page": page,
        "page_size": page_size,
        "rows": page_rows[cols].to_dict(orient="records"),
    })


DEFAULT_MAP_TIERS = ["Critical", "High", "Medium"]


@app.get("/api/risk-map")
def risk_map(
    source: str = Query("forecast", pattern="^(forecast|test)$"),
    month: Optional[str] = Query(None, description="YYYY-MM; defaults to the latest available"),
    tiers: Optional[str] = Query(None, description="Comma-separated risk tiers; omitted = Critical,High,Medium; empty = none"),
    road: Optional[str] = Query(None, pattern="^(single|other)$", description="forecast source only"),
    weather: Optional[str] = Query(None, pattern="^(adverse|fine)$", description="forecast source only"),
    min_adverse_share: Optional[float] = Query(None, ge=0, le=1, description="forecast source only"),
    min_single_share: Optional[float] = Query(None, ge=0, le=1, description="forecast source only"),
    limit: int = Query(5000, ge=1, le=30000),
):
    if source == "forecast":
        raw = data.load_forecast()
        if raw is None:
            raise HTTPException(status_code=503, detail="The forecast is written when the 2025 evaluation step runs.")
        df = raw.assign(month=raw["target_month"].str[:7])
        has_hotspot = False
    else:
        raw = data.load_test_predictions()
        if raw is None:
            raise HTTPException(status_code=503, detail="Test predictions are written when the 2025 evaluation step runs.")
        df = raw.assign(month=raw["target_month"].astype(str).str.slice(0, 7))
        has_hotspot = "hotspot" in df.columns

    months = sorted(df["month"].unique().tolist())
    target_month = month if month in months else (months[-1] if months else None)

    view = df[df["month"] == target_month] if target_month else df.iloc[0:0]
    tier_list = DEFAULT_MAP_TIERS if tiers is None else [t for t in tiers.split(",") if t]
    view = view[view["risk_tier"].isin(tier_list)]

    filters_applied = False
    has_share_filter = any(v is not None for v in (road, weather, min_adverse_share, min_single_share))
    if has_share_filter and source == "forecast":
        frame = data.action_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="Per-cell road/weather shares not found. Run src/dashboard_exports/forecast_cells.py.")
        keep = filter_frame(frame, road=road, weather=weather, min_adverse_share=min_adverse_share, min_single_share=min_single_share)["grid_id"]
        view = view[view["grid_id"].isin(set(keep))]
        filters_applied = True

    total_matching = len(view)
    truncated = total_matching > limit
    if truncated:
        view = view.nlargest(limit, "score")

    cols = ["grid_id", "grid_latitude", "grid_longitude", "risk_tier", "score", "probability"]
    if has_hotspot:
        cols.append("hotspot")

    return _clean({
        "source": source,
        "months": months,
        "month": target_month,
        "tiers": tier_list,
        "limit": limit,
        "total_matching": total_matching,
        "truncated": truncated,
        "road_weather_filters_applied": filters_applied,
        "count": len(view),
        "cells": view[cols].to_dict(orient="records"),
    })


@app.get("/api/model-performance")
def model_performance():
    metrics = _require_metrics()
    test, val, sel = metrics.get("test"), metrics["validation"], metrics["selection"]
    label, m, base = data.headline(metrics)
    family = data.model_family_name(metrics)

    comparison_rows = [{"method": family + " (frozen)", **_row_from_result(m)}]
    for key, result in base.items():
        comparison_rows.append({"method": data.MODEL_NAMES.get(key, key), **_row_from_result(result)})

    system_wide = None
    if test:
        sw, cov = test["system_wide"], test["coverage"]
        system_wide = {
            "frozen_threshold": {
                "recall_on_modelled_grids": sw["recall_on_modelled_grids_at_threshold"],
                "recall_system_wide": sw["recall_system_wide_at_threshold"],
            },
            "top_k_per_month": sw["top_k_per_month"],
            "coverage": {
                "in_modelled_grids": cov["in_modelled_grids"],
                "hotspot_grid_months_all_grids": cov["hotspot_grid_months_all_grids"],
                "coverage": cov["coverage"],
            },
        }

    cal = (test or val)["score_calibrated" if test else "chosen_model_calibrated"]
    conf = m["confusion"]

    candidates = [
        {"model": data.MODEL_NAMES.get(k, k), "val_average_precision": v["average_precision"],
         "val_roc_auc": v["roc_auc"], "val_top_1pct_precision": v["at_top_k_per_month"]["top_1pct"]["precision"]}
        for k, v in val["all_candidates"].items()
    ]
    d = sel["decision"]

    bootstrap = None
    if "bootstrap" in val:
        bootstrap = [
            {"comparison": k, "ap_point_diff": v["ap"]["point_diff"], "ap_ci95_low": v["ap"]["ci95"][0],
             "ap_ci95_high": v["ap"]["ci95"][1]}
            for k, v in val["bootstrap"]["comparisons"].items()
        ]

    cluster_ablation = None
    if "cluster_ablation" in val:
        cluster_ablation = val["cluster_ablation"]["summary"]["ap"]

    alignment_ablation = None
    if "alignment_ablation" in val:
        a = val["alignment_ablation"]
        alignment_ablation = {
            "original_alignment_ap": a["delay_1"]["val_average_precision"],
            "months_leq_t_ap": a["delay_0"]["val_average_precision"],
        }

    return _clean({
        "split_label": label,
        "base_rate": m["base_rate"],
        "n": m["n"],
        "positives": m["positives"],
        "comparison_rows": comparison_rows,
        "system_wide": system_wide,
        "confusion": conf,
        "calibration": {
            "brier": cal["brier"],
            "brier_base_rate_reference": cal["brier_base_rate_reference"],
        },
        "per_month_average_precision": {k: v["average_precision"] for k, v in m["per_month"].items()},
        "validation_candidates": candidates,
        "selection_decision": {
            "chosen": data.MODEL_NAMES.get(d["chosen"], d["chosen"]),
            "reason": d["reason"],
            "top_two_ap_point_diff": d["top_two_ap_diff"]["point_diff"],
            "top_two_ap_ci95": d["top_two_ap_diff"]["ci95"],
        },
        "bootstrap_comparisons": bootstrap,
        "cluster_ablation": cluster_ablation,
        "alignment_ablation": alignment_ablation,
        "test_evaluated": bool(test),
        "test_evaluated_at_utc": test.get("evaluated_at_utc") if test else None,
    })


def _row_from_result(result: dict) -> dict:
    k1, k5 = result["at_top_k_per_month"]["top_1pct"], result["at_top_k_per_month"]["top_5pct"]
    return {
        "average_precision": result["average_precision"],
        "ap_lift_over_base_rate": result["ap_lift_over_base_rate"],
        "roc_auc": result["roc_auc"],
        "precision_top_1pct": k1["precision"],
        "recall_top_1pct": k1["recall"],
        "precision_top_5pct": k5["precision"],
        "recall_top_5pct": k5["recall"],
    }


@app.get("/api/feature-importance")
def feature_importance():
    imp = data.load_feature_importance()
    if imp is None:
        raise HTTPException(status_code=503, detail="results/feature_importance.csv not found.")
    if imp.empty or not {"feature", "importance", "std"}.issubset(imp.columns):
        raise HTTPException(status_code=503, detail="results/feature_importance.csv is empty or lacks the expected columns.")
    return _clean({"rows": imp.to_dict(orient="records")})


@app.get("/api/health")
def health():
    return {"status": "ok"}
