"""Streamlit dashboard. Every number shown is read from results/metrics.json or from the saved
prediction files; nothing is hard-coded. Run:  streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.ui import (
    confusion_matrix_html,
    hero,
    inject_css,
    section_title,
    style_tier_column,
    themed_bar_chart,
    tier_badge,
    tier_legend,
)

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "results" / "metrics.json"
FROZEN_PATH = BASE_DIR / "results" / "frozen_config.json"
IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.csv"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FORECAST_PATH = BASE_DIR / "outputs" / "forecast_2026-01.csv"
TEST_PRED_PATH = BASE_DIR / "outputs" / "test_predictions_2025.parquet"

TIER_ORDER = ["Critical", "High", "Medium", "Low"]
TIER_COLOURS = {"Critical": [208, 59, 59], "High": [236, 131, 90], "Medium": [250, 178, 25], "Low": [12, 163, 12]}
MODEL_NAMES = {"random_forest": "Random forest", "logistic_regression": "Logistic regression",
               "hist_gradient_boosting": "Histogram gradient boosting",
               "persistence": "Baseline: last month >= 2 (persistence)",
               "trailing_12_month_count": "Baseline: trailing 12-month count"}

st.set_page_config(page_title="Traffic Accident Hotspot Prediction", page_icon="🚦", layout="wide")
inject_css(BASE_DIR)


# ------------------------------------------------------------------ loading
@st.cache_data
def load_json(path, mtime):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


@st.cache_data
def load_csv(path, mtime):
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_data
def load_parquet(path, mtime):
    return pd.read_parquet(path) if Path(path).exists() else None


def mtime(path):
    return Path(path).stat().st_mtime if Path(path).exists() else 0


metrics = load_json(METRICS_PATH, mtime(METRICS_PATH))
frozen = load_json(FROZEN_PATH, mtime(FROZEN_PATH))
if metrics is None:
    st.error("results/metrics.json not found. Run the pipeline first (see README).")
    st.stop()

test, val, sel = metrics.get("test"), metrics["validation"], metrics["selection"]
family = MODEL_NAMES.get(sel["model_family"], sel["model_family"])
fractions = sel["tier_top_fractions"]
split_label = "2025 test" if test else "2024 validation"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def headline():
    """(label, raw-score block, baselines block) for the split we can show."""
    if test:
        return "2025 test (evaluated once, frozen)", test["score_raw"], test["baselines"]
    return "2024 validation (test not yet run)", val["chosen_model_raw_score"], val["baselines"]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🚦 Hotspot Prediction")
    page = st.radio("Page", ["Home", "Hotspot Predictions", "Risk Map", "Model Performance", "Feature Importance"])
    st.markdown("---")
    st.markdown(f"**Model:** {family}")
    st.markdown(f"**Task:** will a 500 m grid cell have ≥ 2 collisions next month?")
    if frozen and frozen.get("status") == "frozen":
        st.markdown(
            f'<span class="sidebar-badge">🔒 frozen {frozen["config_sha256"][:10]} '
            f'&middot; {frozen["frozen_at_utc"][:10]}</span>',
            unsafe_allow_html=True,
        )
    st.caption("Numbers on this dashboard are read from results/metrics.json.")
    if not test:
        st.warning("The 2025 test set has not been evaluated yet; showing 2024 validation results.")


# --------------------------------------------------------------------- home
if page == "Home":
    hero(
        "🚦 Spatio-temporal traffic accident hotspot prediction",
        "UK road-safety collisions 2021-2025, aggregated to 500 m British National Grid cells and months — "
        "ranking the risk that each modelled cell has at least two collisions next month.",
    )
    st.write(
        "DBSCAN is used to define and analyse hotspot structure; the cluster features tested in an ablation "
        "did **not** improve prediction and are not used in the final model."
    )
    label, m, base = headline()
    st.subheader(label)
    c = st.columns(4)
    c[0].metric("Base rate (hotspot cell-months)", pct(m["base_rate"], 2))
    c[1].metric("Average precision", f"{m['average_precision']:.3f}",
                f"{m['ap_lift_over_base_rate']:.1f}x base rate", delta_color="off")
    top1 = m["at_top_k_per_month"]["top_1pct"]
    c[2].metric("Precision in top 1% each month", pct(top1["precision"]), f"{top1['lift']:.1f}x base rate", delta_color="off")
    tb = base["trailing_12_month_count"]["at_top_k_per_month"]["top_1pct"]["precision"]
    c[3].metric("Same, trailing-12-month baseline", pct(tb))
    ap_gain = m["average_precision"] - base["trailing_12_month_count"]["average_precision"]
    st.info(
        f"Average precision is {m['average_precision']:.3f} versus {base['trailing_12_month_count']['average_precision']:.3f} "
        f"for simply ranking cells by their collisions in the last 12 months ({ap_gain:+.3f}). "
        "See Model Performance for confidence intervals and the full comparison."
    )
    cov = (test or val)["coverage"]
    st.warning(
        f"**Coverage ceiling.** Only cells with ≥ 3 collisions in 2021-2023 are modelled. They contain "
        f"{pct(cov['coverage'])} of all hotspot cell-months, so system-wide recall cannot exceed that."
    )
    section_title("How to read the outputs", "📖")
    st.markdown(
        f"- **Risk tier** is rank-based within each month: {tier_badge('Critical')} = top {pct(fractions['Critical'], 0)} of cells, "
        f"{tier_badge('High')} = next {pct(fractions['High'] - fractions['Critical'], 0)}, "
        f"{tier_badge('Medium')} = next {pct(fractions['Medium'] - fractions['High'], 0)}, otherwise {tier_badge('Low')}.\n"
        "- **Score** is the raw model output; it is *not* a probability.\n"
        f"- **Estimated probability** is the score passed through a {sel['calibration']['chosen']} calibrator fitted on 2024 data.",
        unsafe_allow_html=True,
    )
    section_title("Limitations", "⚠")
    st.markdown(
        "- The model is trained on targets up to 2023-12 only; 2024 is used for calibration and the threshold, so recent data are not in the fit.\n"
        "- Grids first active after 2023 are not modelled.\n"
        "- Neighbouring cells are spatially correlated; metrics assume independent cells and understate uncertainty.\n"
        "- Confidence intervals resample grids only (not years); one test year is a single draw.\n"
        "- The label is a count rule (≥ 2 collisions), not a causal risk measure; reporting is police-recorded injury collisions only."
    )

# ------------------------------------------------------------- predictions
elif page == "Hotspot Predictions":
    st.title("🔮 Hotspot forecast")
    fc = load_csv(FORECAST_PATH, mtime(FORECAST_PATH))
    if fc is None:
        st.info("The forecast is written when the 2025 evaluation step runs (src/evaluate_test.py).")
    else:
        st.caption(f"Forecast for {fc['target_month'].iloc[0][:7]} from data up to the previous month. "
                   "Tiers are rank-based; probabilities are calibrated on 2024 data.")
        tiers = st.multiselect("Risk tiers", TIER_ORDER, default=["Critical", "High"])
        query = st.text_input("Filter by grid id")
        shown = fc[fc["risk_tier"].isin(tiers)]
        if query:
            shown = shown[shown["grid_id"].astype(str).str.contains(query)]
        st.write(f"{len(shown):,} of {len(fc):,} modelled cells")
        table = shown[["rank", "grid_id", "risk_tier", "score", "probability", "grid_latitude", "grid_longitude"]].rename(
            columns={"score": "score (not a probability)", "probability": "estimated probability"})
        st.dataframe(style_tier_column(table.head(1000)), use_container_width=True, hide_index=True)
        st.download_button("Download filtered forecast (CSV)", shown.to_csv(index=False), "forecast_filtered.csv")

# ---------------------------------------------------------------------- map
elif page == "Risk Map":
    st.title("🗺 Risk map")
    source = st.radio("Data", ["Forecast (next month)", "2025 retrospective (test predictions)"], horizontal=True)
    if source.startswith("Forecast"):
        data = load_csv(FORECAST_PATH, mtime(FORECAST_PATH))
        if data is not None:
            data = data.assign(month=data["target_month"].str[:7], actual=None)
    else:
        data = load_parquet(TEST_PRED_PATH, mtime(TEST_PRED_PATH))
        if data is not None:
            data = data.assign(month=pd.to_datetime(data["target_month"]).dt.strftime("%Y-%m"))
    if data is None:
        st.info("Prediction files are written when the 2025 evaluation step runs (src/evaluate_test.py).")
    else:
        month = st.selectbox("Target month", sorted(data["month"].unique()))
        tiers = st.multiselect("Risk tiers", TIER_ORDER, default=["Critical", "High", "Medium"])
        view = data[(data["month"] == month) & data["risk_tier"].isin(tiers)].copy()
        if len(view) > 30000:
            view = view.nlargest(30000, "score")
            st.warning("Showing the 30,000 highest-scoring cells.")
        view["colour"] = view["risk_tier"].map(TIER_COLOURS)
        layers = [pdk.Layer("ScatterplotLayer", data=view, get_position="[grid_longitude, grid_latitude]",
                            get_fill_color="colour", get_radius=250, pickable=True, opacity=0.75)]
        if "hotspot" in view and view["hotspot"].notna().any():
            hits = view[view["hotspot"] == 1]
            layers.append(pdk.Layer("ScatterplotLayer", data=hits, get_position="[grid_longitude, grid_latitude]",
                                    get_radius=420, stroked=True, filled=False, get_line_color=[255, 255, 255],
                                    line_width_min_pixels=2))
            st.caption(f"White rings mark cells that actually had ≥ 2 collisions ({len(hits):,} shown).")
        if len(view):
            state = pdk.ViewState(latitude=float(view["grid_latitude"].mean()), longitude=float(view["grid_longitude"].mean()), zoom=5.5)
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=state,
                                     tooltip={"text": "{grid_id}\n{risk_tier}\nscore {score}\nprob {probability}"}))
        st.markdown(tier_legend(TIER_ORDER), unsafe_allow_html=True)
        st.caption("Tiers are rank-based within the month.")

# ------------------------------------------------------------------ metrics
elif page == "Model Performance":
    st.title("📊 Model performance")
    label, m, base = headline()
    st.subheader(label)
    rows = []
    for name, r in [(family + " (frozen)", m)] + [(MODEL_NAMES[k], v) for k, v in base.items()]:
        k1, k5 = r["at_top_k_per_month"]["top_1pct"], r["at_top_k_per_month"]["top_5pct"]
        rows.append({"Method": name, "Avg precision": r["average_precision"], "Lift over base": r["ap_lift_over_base_rate"],
                     "ROC-AUC (ref.)": r["roc_auc"], "Precision @ top 1%": k1["precision"], "Recall @ top 1%": k1["recall"],
                     "Precision @ top 5%": k5["precision"], "Recall @ top 5%": k5["recall"]})
    st.dataframe(pd.DataFrame(rows).style.format(precision=3), use_container_width=True, hide_index=True)
    st.caption(f"Base rate {pct(m['base_rate'], 2)} on {m['n']:,} cell-months ({m['positives']:,} hotspots). "
               "Top-k is taken within each month. Recall is on modelled cells.")

    if test:
        sw = test["system_wide"]
        cov = test["coverage"]
        section_title("Recall on modelled cells vs system-wide")
        st.html(pd.DataFrame([
            {"Operating point": "Frozen F1 threshold", "Recall (modelled cells)": sw["recall_on_modelled_grids_at_threshold"],
             "Recall (system-wide)": sw["recall_system_wide_at_threshold"]},
            *[{"Operating point": f"Top {k.split('_')[1].replace('pct', '%')} per month",
               "Recall (modelled cells)": v["recall_on_modelled_grids"], "Recall (system-wide)": v["recall_system_wide"]}
              for k, v in sw["top_k_per_month"].items()],
        ]).style.format({"Recall (modelled cells)": "{:.3f}", "Recall (system-wide)": "{:.3f}"}).hide(axis="index").to_html())
        st.caption(f"{cov['in_modelled_grids']:,} of {cov['hotspot_grid_months_all_grids']:,} hotspot cell-months "
                   f"({pct(cov['coverage'])}) lie in modelled cells.")

    cal = (test or val)["score_calibrated" if test else "chosen_model_calibrated"]
    conf = m["confusion"]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Confusion matrix at the frozen threshold** (score ≥ {conf['threshold']:.3f}, tuned on 2024)")
        st.markdown(confusion_matrix_html(conf["tn"], conf["fp"], conf["fn"], conf["tp"]), unsafe_allow_html=True)
        st.caption(f"Precision {conf['precision']:.3f}, recall {conf['recall']:.3f}, F1 {conf['f1']:.3f}; flags {pct(conf['flagged_fraction'])} of cell-months.")
    with right:
        st.markdown("**Calibration (Brier score, lower is better)**")
        st.write(f"Calibrated probabilities: **{cal['brier']:.4f}** vs constant base-rate forecast {cal['brier_base_rate_reference']:.4f}. "
                 f"Raw scores are not probabilities (mean score is far above the base rate).")
        rel = FIGURES_DIR / "reliability_val.png"
        if rel.exists():
            st.image(str(rel), caption="Validation reliability (calibrator fitted on the same 2024 data, in-sample)")

    section_title("Per-month average precision")
    per_month = m["per_month"]
    st.altair_chart(
        themed_bar_chart(pd.Series({k: v["average_precision"] for k, v in per_month.items()}, name="average precision"),
                          "average precision", "month"),
        use_container_width=True,
    )

    st.subheader("Validation (2024) and model selection")
    cand = pd.DataFrame([{"Model": MODEL_NAMES[k], "Val AP": v["average_precision"], "Val ROC-AUC": v["roc_auc"],
                          "Val top-1% precision": v["at_top_k_per_month"]["top_1pct"]["precision"]}
                         for k, v in val["all_candidates"].items()])
    st.dataframe(cand.style.format(precision=3), use_container_width=True, hide_index=True)
    d = sel["decision"]
    st.write(f"Selection rule (saved before tuning): highest validation AP; if the top two are within bootstrap noise, take the simpler. "
             f"Outcome: **{MODEL_NAMES[d['chosen']]}** ({d['reason']}). Top-two AP difference "
             f"{d['top_two_ap_diff']['point_diff']:+.4f}, 95% CI [{d['top_two_ap_diff']['ci95'][0]:+.4f}, {d['top_two_ap_diff']['ci95'][1]:+.4f}].")
    if "bootstrap" in val:
        rows = [{"Comparison": k, "AP difference": v["ap"]["point_diff"], "95% CI low": v["ap"]["ci95"][0], "95% CI high": v["ap"]["ci95"][1]}
                for k, v in val["bootstrap"]["comparisons"].items()]
        st.markdown("**Validation AP differences (grid bootstrap)**")
        st.dataframe(pd.DataFrame(rows).style.format(precision=4), use_container_width=True, hide_index=True)
    if "cluster_ablation" in val:
        s = val["cluster_ablation"]["summary"]["ap"]
        st.markdown("**DBSCAN cluster-feature ablation (validation)**")
        st.write(f"Average precision without cluster features {s['without_cluster_mean']:.4f}, with {s['with_cluster_mean']:.4f} "
                 f"(difference {s['diff_mean']:+.4f}, mean over seeds). Cluster features are not used in the final model.")
    if "alignment_ablation" in val:
        a = val["alignment_ablation"]
        st.write(f"Feature timing ablation (validation AP): original alignment {a['delay_1']['val_average_precision']:.4f} "
                 f"vs months ≤ t {a['delay_0']['val_average_precision']:.4f}.")
    pr = FIGURES_DIR / "test_pr_curve.png"
    if test and pr.exists():
        st.image(str(pr), caption="Precision-recall curve, 2025 test")
    if test:
        st.caption(f"Test evaluated once at {test['evaluated_at_utc']}; frozen config {test['frozen_config_sha256'][:10]}.")

# --------------------------------------------------------------- importance
elif page == "Feature Importance":
    st.title("🧩 Feature importance")
    imp = load_csv(IMPORTANCE_PATH, mtime(IMPORTANCE_PATH))
    if imp is None:
        st.info("results/feature_importance.csv not found.")
    else:
        st.write("Permutation importance on a 2024 validation sample: the drop in average precision when a feature is shuffled. "
                 "Correlated features share credit, so small values do not mean a feature is useless.")
        st.altair_chart(
            themed_bar_chart(imp.set_index("feature")["importance"], "importance", "feature"),
            use_container_width=True,
        )
        st.dataframe(imp.style.format(precision=4), use_container_width=True, hide_index=True)
