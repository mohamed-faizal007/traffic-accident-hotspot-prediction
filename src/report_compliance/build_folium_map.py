"""Static exported Folium map (DA1 proposal item: "Folium for interactive
mapping"), a literal demonstration of the proposed tool, shown ALONGSIDE
(not replacing) the React dashboard.

Combines, read-only:
  * the FROZEN classifier's 2025 predictions (via
    results/regression/quantile_tiers_2025.parquet, which already carries
    the classifier's risk_tier/score/probability alongside the new
    quantile_tier -- itself built read-only from
    outputs/test_predictions_2025.parquet in Phase 1)
  * the new regression track's 2025 predictions
    (results/regression/test_predictions_2025.parquet)
  * the new grid-month road/weather aggregate shares
    (data/processed/regression_features.parquet, test split only), used to
    derive the road-type / weather filter tags

Does not read, write, or import from any frozen classifier CODE path
(src/train_validate.py, src/freeze_config.py, src/evaluate_test.py); it only
reads their already-written OUTPUT files.

Output: outputs/report_compliance/hotspot_map.html
"""

import sys
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import TagFilterButton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

MAP_OUTPUT_PATH = OUTPUTS_DIR / "report_compliance" / "hotspot_map.html"

# Marker caps per risk tier, same convention as legacy/hotspot_map.py (Low
# tier is never plotted; it is by far the largest and least informative
# group, and would make a "literal demonstration" map unusably heavy).
MAX_CRITICAL_GRIDS = 1000
MAX_HIGH_GRIDS = 1000
MAX_MEDIUM_GRIDS = 500

TIER_COLOR = {"Critical": "#8b0000", "High": "#e6550d", "Medium": "#fdae6b"}
TIER_SEVERITY = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def load_merged_2025_predictions():
    classifier = pd.read_parquet(RESULTS_DIR / "regression" / "quantile_tiers_2025.parquet")
    regression = pd.read_parquet(RESULTS_DIR / "regression" / "test_predictions_2025.parquet")
    features = pd.read_parquet(PROCESSED_DATA_DIR / "regression_features.parquet")
    features = features[features["split"] == "test"][
        ["grid_id", "target_month", "historical_single_carriageway_share", "historical_adverse_weather_share"]
    ]

    merged = classifier.merge(
        regression[["grid_id", "target_month", "predicted_count_xgboost", "predicted_count_baseline", "next_month_collisions"]],
        on=["grid_id", "target_month"], how="inner",
    ).merge(features, on=["grid_id", "target_month"], how="left")
    return merged


def road_label(share):
    if share is None or pd.isna(share) or share < 0:
        return "Unknown"
    return "Single carriageway" if share > 0.5 else "Other road type"


def weather_label(share):
    if share is None or pd.isna(share) or share < 0:
        return "Unknown"
    return "Adverse-weather prone" if share > 0.5 else "Predominantly fine weather"


def pick_peak_month_per_grid(merged):
    """One row per grid: the 2025 target month where the classifier's
    probability was highest -- the grid's "peak risk" snapshot for the year."""
    merged = merged[merged["risk_tier"].isin(["Critical", "High", "Medium"])].copy()
    idx = merged.groupby("grid_id")["probability"].idxmax()
    return merged.loc[idx].reset_index(drop=True)


def cap_by_tier(df):
    parts = []
    for tier, cap in [("Critical", MAX_CRITICAL_GRIDS), ("High", MAX_HIGH_GRIDS), ("Medium", MAX_MEDIUM_GRIDS)]:
        sub = df[df["risk_tier"] == tier].nlargest(cap, "probability")
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def build_map(df):
    center = [df["grid_latitude"].mean(), df["grid_longitude"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap")

    for _, row in df.iterrows():
        road = road_label(row["historical_single_carriageway_share"])
        weather = weather_label(row["historical_adverse_weather_share"])
        tags = [f"Risk: {row['risk_tier']}", f"Quantile: {row['quantile_tier']}", f"Road: {road}", f"Weather: {weather}"]

        popup = folium.Popup(
            f"<b>Grid {row['grid_id']}</b><br>"
            f"Target month: {pd.Timestamp(row['target_month']):%Y-%m}<br>"
            f"Classifier risk tier: <b>{row['risk_tier']}</b> (probability {row['probability']:.3f})<br>"
            f"Quantile tier (DA1-compliance demo): <b>{row['quantile_tier']}</b><br>"
            f"Regression predicted count (XGBoost): {row['predicted_count_xgboost']:.2f}<br>"
            f"Historical-average baseline predicted count: {row['predicted_count_baseline']:.2f}<br>"
            f"Actual collisions that month: {row['next_month_collisions']:.0f}<br>"
            f"Road type: {road}<br>"
            f"Weather: {weather}",
            max_width=320,
        )

        folium.CircleMarker(
            location=[row["grid_latitude"], row["grid_longitude"]],
            radius=5,
            color=TIER_COLOR[row["risk_tier"]],
            fill=True,
            fill_color=TIER_COLOR[row["risk_tier"]],
            fill_opacity=0.75,
            weight=1,
            popup=popup,
            tags=tags,
        ).add_to(m)

    all_tags = sorted({t for tags in df.apply(
        lambda r: [f"Risk: {r['risk_tier']}", f"Quantile: {r['quantile_tier']}",
                   f"Road: {road_label(r['historical_single_carriageway_share'])}",
                   f"Weather: {weather_label(r['historical_adverse_weather_share'])}"],
        axis=1,
    ) for t in tags})
    TagFilterButton(all_tags, title="Filter: risk / quantile tier / road type / weather").add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
                padding: 10px 14px; border: 1px solid #999; border-radius: 4px; font-size: 13px;">
      <b>Classifier risk tier</b><br>
      <span style="color:#8b0000;">&#9679;</span> Critical&nbsp;&nbsp;
      <span style="color:#e6550d;">&#9679;</span> High&nbsp;&nbsp;
      <span style="color:#fdae6b;">&#9679;</span> Medium<br>
      <small>Low tier grids are not plotted (too numerous to show usefully).<br>
      Popups show the new regression track's predicted count and quantile tier,<br>
      plus road-type / weather-based filter tags (DA1 proposal demonstration).</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def main():
    merged = load_merged_2025_predictions()
    peak = pick_peak_month_per_grid(merged)
    capped = cap_by_tier(peak)
    print(f"Grids plotted: {len(capped):,} "
          f"(Critical {(capped.risk_tier == 'Critical').sum():,}, "
          f"High {(capped.risk_tier == 'High').sum():,}, "
          f"Medium {(capped.risk_tier == 'Medium').sum():,}; Low excluded)")

    m = build_map(capped)
    MAP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(MAP_OUTPUT_PATH))
    print(f"Saved: {MAP_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
