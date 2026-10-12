"""Recurring temporal patterns of collisions, descriptive (all five years of the cleaned data), for the dashboard.

Read-only over data/interim/collisions_clean.csv (and collisions_with_hotspots.csv for the clustered-vs-
other hour profile, which covers 2021-2023 only, the DBSCAN fit period). Reuses the time-of-day buckets and
the STATS19 day coding (1 = Sunday) from src/report_compliance/temporal_eda.py. Writes only
outputs/dashboard/temporal_patterns.json.
"""

import datetime
import json

import pandas as pd

try:
    from ._common import DASHBOARD_DIR, PROJECT_ROOT
except ImportError:  # run as a script
    from _common import DASHBOARD_DIR, PROJECT_ROOT

from report_compliance.temporal_eda import DAY_NAMES, TIME_OF_DAY_BINS, TIME_OF_DAY_LABELS, add_temporal_features

CLEAN = PROJECT_ROOT / "data" / "interim" / "collisions_clean.csv"
WITH_CLUSTERS = PROJECT_ROOT / "data" / "interim" / "collisions_with_hotspots.csv"
OUT = DASHBOARD_DIR / "temporal_patterns.json"
WEEK_ORDER = [2, 3, 4, 5, 6, 7, 1]  # Monday first


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    df = add_temporal_features(pd.read_csv(CLEAN, usecols=["collision_year", "date", "time", "day_of_week"]))
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df = df.dropna(subset=["hour"]).astype({"hour": int})

    matrix = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0).reindex(index=WEEK_ORDER, columns=range(24), fill_value=0)
    monthly = df.groupby(["collision_year", "month"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    tod = df.groupby("time_of_day", observed=True).size().reindex(TIME_OF_DAY_LABELS)

    wk = df.groupby("is_weekend").size()
    n_weekend_days = 2 / 7  # share of calendar days that are weekend days, for the per-day normalisation
    pts = pd.read_csv(WITH_CLUSTERS, usecols=["time", "hotspot_id"])
    pts["hour"] = pd.to_datetime(pts["time"], format="%H:%M", errors="coerce").dt.hour
    pts = pts.dropna(subset=["hour"])
    pts["clustered"] = pts["hotspot_id"] >= 0
    prof = pts.groupby(["clustered", "hour"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)

    out = {
        "scope": "descriptive pattern of all collisions in the cleaned data, all years; not used by the frozen classifier",
        "n_collisions": int(len(df)),
        "years": sorted(int(y) for y in df["collision_year"].unique()),
        "hours": list(range(24)),
        "weekday_order": [DAY_NAMES[d] for d in WEEK_ORDER],
        "hour_by_weekday": matrix.to_numpy().tolist(),
        "by_hour": df.groupby("hour").size().reindex(range(24), fill_value=0).tolist(),
        "by_weekday": [int((df["day_of_week"] == d).sum()) for d in WEEK_ORDER],
        "monthly_by_year": {str(y): row.tolist() for y, row in monthly.iterrows()},
        "monthly_total": monthly.sum().tolist(),
        "weekend": {"weekend": int(wk.get(True, 0)), "weekday": int(wk.get(False, 0)),
                    "weekend_share_of_days": n_weekend_days},
        "time_of_day": {"buckets": TIME_OF_DAY_LABELS, "hour_bins_right_inclusive": TIME_OF_DAY_BINS,
                        "collisions": [int(x) for x in tod.tolist()]},
        "clustered_vs_other_hours": {
            "period": "2021-2023 (DBSCAN fit period)",
            "clustered": prof.loc[True].tolist(), "other": prof.loc[False].tolist(),
        },
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(out))
    print(f"{out['n_collisions']:,} collisions -> {OUT}")


if __name__ == "__main__":
    main()
