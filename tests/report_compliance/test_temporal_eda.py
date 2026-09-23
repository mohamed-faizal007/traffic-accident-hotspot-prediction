import pandas as pd

from report_compliance.temporal_eda import add_temporal_features, summarize, DAY_NAMES, WEEKEND_DAYS


def make_df():
    return pd.DataFrame(
        {
            "time": ["00:30", "05:59", "06:00", "09:59", "10:00", "15:59", "16:00", "18:59", "19:00", "23:59"],
            "day_of_week": [1, 2, 3, 4, 5, 6, 7, 1, 2, 7],
        }
    )


def test_hour_extracted_from_time_string():
    df = add_temporal_features(make_df())
    assert list(df["hour"]) == [0, 5, 6, 9, 10, 15, 16, 18, 19, 23]


def test_day_name_matches_stats19_convention():
    df = add_temporal_features(make_df())
    # 1=Sunday .. 7=Saturday, verified against real calendar dates in the plan
    assert df["day_name"].iloc[0] == "Sunday"
    assert df["day_name"].iloc[6] == "Saturday"
    assert df["day_name"].iloc[2] == "Tuesday"


def test_weekend_flag_matches_sunday_and_saturday_only():
    df = add_temporal_features(make_df())
    expected_weekend = df["day_of_week"].isin(WEEKEND_DAYS)
    pd.testing.assert_series_equal(df["is_weekend"], expected_weekend, check_names=False)
    assert set(df.loc[df["is_weekend"], "day_of_week"]) <= {1, 7}


def test_time_of_day_bucket_boundaries():
    df = add_temporal_features(make_df())
    expected = ["night", "night", "morning_rush", "morning_rush", "midday",
                "midday", "evening_rush", "evening_rush", "evening", "evening"]
    assert list(df["time_of_day"].astype(str)) == expected


def test_summarize_counts_add_up():
    df = add_temporal_features(make_df())
    tables = summarize(df)
    assert tables["by_hour"]["collisions"].sum() == len(df)
    assert tables["by_day_of_week"]["collisions"].sum() == len(df)
    assert tables["weekend_vs_weekday"]["collisions"].sum() == len(df)
    assert tables["by_time_of_day"]["collisions"].sum() == len(df)
