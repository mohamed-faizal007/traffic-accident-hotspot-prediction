"""Collision-level temporal EDA additions (DA1 proposal item: "explicit
hour/day-of-week/weekend/time-of-day features").

Pure, read-only analysis of data/interim/collisions_clean.csv. Produces
descriptive plots/tables only -- these columns are NOT fed into the frozen
classifier's feature table (data/processed/ml_features.parquet, untouched).
The grid-month AGGREGATE versions of these same signals (rush-hour share,
weekend share, etc.) that the new regression track actually trains on are
built separately in a later phase, with their own leakage tests.

day_of_week follows the STATS19 convention already used by the raw data:
1=Sunday ... 7=Saturday (verified against known calendar dates).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLEAN_COLLISIONS_PATH, OUTPUTS_DIR  # noqa: E402

DAY_NAMES = {
    1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
    5: "Thursday", 6: "Friday", 7: "Saturday",
}
WEEKEND_DAYS = {1, 7}  # Sunday, Saturday

TIME_OF_DAY_BINS = [-1, 5, 9, 15, 18, 23]
TIME_OF_DAY_LABELS = ["night", "morning_rush", "midday", "evening_rush", "evening"]


def add_temporal_features(df):
    """Add hour, day_name, is_weekend, time_of_day columns from `time` and
    `day_of_week`. Does not modify the input frame in place."""
    out = df.copy()
    hour = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.hour
    out["hour"] = hour
    out["day_name"] = out["day_of_week"].map(DAY_NAMES)
    out["is_weekend"] = out["day_of_week"].isin(WEEKEND_DAYS)
    out["time_of_day"] = pd.cut(hour, bins=TIME_OF_DAY_BINS, labels=TIME_OF_DAY_LABELS, right=True)
    return out


def summarize(df):
    """Descriptive tables used for the report/demo: collision counts by hour,
    by day of week, weekend vs weekday, and by time-of-day bucket."""
    return {
        "by_hour": df.groupby("hour").size().rename("collisions").reset_index(),
        "by_day_of_week": (
            df.groupby(["day_of_week", "day_name"]).size().rename("collisions").reset_index()
            .sort_values("day_of_week")
        ),
        "weekend_vs_weekday": df.groupby("is_weekend").size().rename("collisions").reset_index(),
        "by_time_of_day": (
            df.groupby("time_of_day", observed=True).size().rename("collisions")
            .reindex(TIME_OF_DAY_LABELS).rename_axis("time_of_day").reset_index()
        ),
    }


def plot_summary(tables, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tables["by_hour"]["hour"], tables["by_hour"]["collisions"])
    ax.set_xlabel("hour of day")
    ax.set_ylabel("collisions")
    ax.set_title("Collisions by hour of day")
    fig.tight_layout()
    fig.savefig(out_dir / "collisions_by_hour.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    dow = tables["by_day_of_week"]
    ax.bar(dow["day_name"], dow["collisions"])
    ax.set_ylabel("collisions")
    ax.set_title("Collisions by day of week")
    fig.tight_layout()
    fig.savefig(out_dir / "collisions_by_day_of_week.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    wk = tables["weekend_vs_weekday"].copy()
    wk["label"] = wk["is_weekend"].map({True: "weekend", False: "weekday"})
    ax.bar(wk["label"], wk["collisions"])
    ax.set_ylabel("collisions")
    ax.set_title("Weekend vs weekday collisions")
    fig.tight_layout()
    fig.savefig(out_dir / "collisions_weekend_vs_weekday.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    tod = tables["by_time_of_day"]
    ax.bar(tod["time_of_day"].astype(str), tod["collisions"])
    ax.set_ylabel("collisions")
    ax.set_title("Collisions by time-of-day bucket")
    fig.tight_layout()
    fig.savefig(out_dir / "collisions_by_time_of_day.png", dpi=130)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_COLLISIONS_PATH, low_memory=False)
    df = add_temporal_features(df)
    tables = summarize(df)

    out_dir = OUTPUTS_DIR / "figures" / "report_compliance"
    plot_summary(tables, out_dir)

    csv_dir = OUTPUTS_DIR / "report_compliance"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(csv_dir / f"temporal_eda_{name}.csv", index=False)

    print(f"Collisions: {len(df):,}")
    print(tables["weekend_vs_weekday"])
    print(f"Saved plots to {out_dir} and tables to {csv_dir}")


if __name__ == "__main__":
    main()
