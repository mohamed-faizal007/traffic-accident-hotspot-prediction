import pandas as pd
import joblib

from pathlib import Path

from config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features_advanced.csv"
)

MODEL_PATH = Path(
    "models/advanced_random_forest_model.pkl"
)

OUTPUT_DIR = Path("outputs")

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_hotspot_predictions.csv"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

FINAL_THRESHOLD = 0.90


FEATURE_COLUMNS = [

    "grid_latitude",
    "grid_longitude",

    "year",
    "month",

    "month_sin",
    "month_cos",

    "collisions_lag_1",
    "collisions_lag_2",
    "collisions_lag_3",

    "casualties_lag_1",

    "collisions_rolling_3",
    "collisions_rolling_6",
    "collisions_rolling_sum_3",

    "historical_total_collisions",
    "historical_avg_collisions",
    "historical_max_collisions",
    "historical_collision_frequency",

    "historical_hotspot_frequency",

    "collisions_last_3_months",
    "collisions_last_6_months",
    "collisions_last_12_months",

    "months_since_last_collision"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def load_data():

    print_section(
        "STEP 1: LOADING ADVANCED MACHINE LEARNING DATASET"
    )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"Unique grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"\nDataset loaded from:\n{DATA_PATH}"
    )

    return df


# ============================================================
# STEP 2: SELECT PREDICTION PERIOD
# ============================================================

def select_prediction_data(df):

    print_section(
        "STEP 2: SELECTING PREDICTION DATA"
    )

    prediction_df = df[
        df["year"] == 2025
    ].copy()

    print(
        "Prediction period: 2025"
    )

    print(
        f"\nTotal prediction records: "
        f"{len(prediction_df):,}"
    )

    print(
        f"Unique spatial grids: "
        f"{prediction_df['grid_id'].nunique():,}"
    )

    print(
        f"Unique months: "
        f"{prediction_df['year_month'].nunique()}"
    )

    return prediction_df


# ============================================================
# STEP 3: LOAD MODEL
# ============================================================

def load_model():

    print_section(
        "STEP 3: LOADING FINAL MACHINE LEARNING MODEL"
    )

    print(
        "Final model: "
        "Advanced Random Forest"
    )

    print(
        f"Probability threshold: "
        f"{FINAL_THRESHOLD}"
    )

    print(
        "\nLoading trained model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "\nModel loaded successfully."
    )

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    return model


# ============================================================
# STEP 4: GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    df,
    model
):

    print_section(
        "STEP 4: GENERATING HOTSPOT PREDICTIONS"
    )

    print(
        f"Features used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    X = df[
        FEATURE_COLUMNS
    ]

    print(
        "\nGenerating prediction probabilities..."
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    print(
        "Generating hotspot predictions..."
    )

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    df[
        "prediction_probability"
    ] = probabilities

    df[
        "predicted_hotspot"
    ] = predictions

    print(
        "\nPredictions generated successfully."
    )

    return df


# ============================================================
# STEP 5: CREATE RISK LEVELS
# ============================================================

def create_risk_levels(df):

    print_section(
        "STEP 5: CREATING RISK LEVELS"
    )

    print(
        "Risk classification:"
    )

    print(
        "Low Risk      : Probability < 0.50"
    )

    print(
        "Medium Risk   : 0.50 - 0.70"
    )

    print(
        "High Risk     : 0.70 - 0.90"
    )

    print(
        "Critical Risk : >= 0.90"
    )


    def assign_risk(probability):

        if probability >= 0.90:

            return "Critical"

        elif probability >= 0.70:

            return "High"

        elif probability >= 0.50:

            return "Medium"

        else:

            return "Low"


    df[
        "risk_level"
    ] = df[
        "prediction_probability"
    ].apply(
        assign_risk
    )

    print(
        "\nRisk levels created successfully."
    )

    return df


# ============================================================
# STEP 6: ANALYZE PREDICTIONS
# ============================================================

def analyze_predictions(df):

    print_section(
        "STEP 6: PREDICTION ANALYSIS"
    )

    total_predictions = len(df)

    predicted_hotspots = (
        df[
            "predicted_hotspot"
        ].sum()
    )

    predicted_non_hotspots = (
        total_predictions
        - predicted_hotspots
    )

    hotspot_percentage = (
        predicted_hotspots
        / total_predictions
        * 100
    )


    print(
        f"Total predictions: "
        f"{total_predictions:,}"
    )

    print(
        f"Predicted hotspots: "
        f"{predicted_hotspots:,}"
    )

    print(
        f"Predicted non-hotspots: "
        f"{predicted_non_hotspots:,}"
    )

    print(
        f"Hotspot percentage: "
        f"{hotspot_percentage:.2f}%"
    )


    print_section(
        "RISK LEVEL DISTRIBUTION"
    )

    risk_distribution = (
        df[
            "risk_level"
        ]
        .value_counts()
    )

    print(
        risk_distribution
    )


    print_section(
        "TOP 10 HIGHEST RISK LOCATIONS"
    )

    top_predictions = (
        df[
            [
                "grid_id",
                "year_month",
                "grid_latitude",
                "grid_longitude",
                "prediction_probability",
                "risk_level"
            ]
        ]
        .sort_values(
            by="prediction_probability",
            ascending=False
        )
        .head(10)
    )

    print(
        top_predictions.to_string(
            index=False
        )
    )

    return df


# ============================================================
# STEP 7: COMPARE WITH ACTUAL HOTSPOTS
# ============================================================

def compare_with_actual(df):

    print_section(
        "STEP 7: ACTUAL VS PREDICTED HOTSPOTS"
    )

    if "hotspot" not in df.columns:

        print(
            "Actual hotspot column not available."
        )

        return


    actual_hotspots = (
        df["hotspot"].sum()
    )

    predicted_hotspots = (
        df[
            "predicted_hotspot"
        ].sum()
    )


    true_positives = (
        (
            df["hotspot"] == 1
        )
        &
        (
            df[
                "predicted_hotspot"
            ] == 1
        )
    ).sum()


    false_positives = (
        (
            df["hotspot"] == 0
        )
        &
        (
            df[
                "predicted_hotspot"
            ] == 1
        )
    ).sum()


    false_negatives = (
        (
            df["hotspot"] == 1
        )
        &
        (
            df[
                "predicted_hotspot"
            ] == 0
        )
    ).sum()


    print(
        f"Actual hotspots: "
        f"{actual_hotspots:,}"
    )

    print(
        f"Predicted hotspots: "
        f"{predicted_hotspots:,}"
    )

    print(
        f"\nTrue Positives: "
        f"{true_positives:,}"
    )

    print(
        f"False Positives: "
        f"{false_positives:,}"
    )

    print(
        f"False Negatives: "
        f"{false_negatives:,}"
    )


# ============================================================
# STEP 8: SAVE FINAL PREDICTIONS
# ============================================================

def save_predictions(df):

    print_section(
        "STEP 8: SAVING FINAL PREDICTIONS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_columns = [

        "grid_id",

        "year_month",

        "year",

        "month",

        "grid_latitude",

        "grid_longitude",

        "collision_count",

        "total_casualties",

        "prediction_probability",

        "predicted_hotspot",

        "risk_level",

        "hotspot"
    ]


    available_columns = [

        column

        for column in output_columns

        if column in df.columns

    ]


    output_df = df[
        available_columns
    ].copy()


    output_df.to_csv(

        OUTPUT_PATH,

        index=False

    )


    print(
        "Final prediction dataset "
        "saved successfully."
    )

    print(
        f"\nOutput path:\n{OUTPUT_PATH}"
    )

    print(
        f"\nTotal saved records: "
        f"{len(output_df):,}"
    )

    print(
        f"Total saved columns: "
        f"{len(output_df.columns)}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")

    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "FINAL HOTSPOT PREDICTION PIPELINE"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    df = load_data()


    # --------------------------------------------------------
    # STEP 2: SELECT PREDICTION DATA
    # --------------------------------------------------------

    prediction_df = (
        select_prediction_data(df)
    )


    # --------------------------------------------------------
    # STEP 3: LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # STEP 4: GENERATE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = (
        generate_predictions(

            prediction_df,

            model

        )
    )


    # --------------------------------------------------------
    # STEP 5: CREATE RISK LEVELS
    # --------------------------------------------------------

    prediction_df = (
        create_risk_levels(
            prediction_df
        )
    )


    # --------------------------------------------------------
    # STEP 6: ANALYZE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = (
        analyze_predictions(
            prediction_df
        )
    )


    # --------------------------------------------------------
    # STEP 7: ACTUAL VS PREDICTED
    # --------------------------------------------------------

    compare_with_actual(
        prediction_df
    )


    # --------------------------------------------------------
    # STEP 8: SAVE PREDICTIONS
    # --------------------------------------------------------

    save_predictions(
        prediction_df
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "FINAL HOTSPOT PREDICTION COMPLETE"
    )

    print(
        "Final model used:"
    )

    print(
        "Advanced Random Forest"
    )

    print(
        f"\nProbability threshold: "
        f"{FINAL_THRESHOLD}"
    )

    print(
        "\nOutput generated:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nThe prediction dataset contains:"
    )

    print(
        "- Spatial grid location"
    )

    print(
        "- Prediction month"
    )

    print(
        "- Accident risk probability"
    )

    print(
        "- Predicted hotspot"
    )

    print(
        "- Risk level"
    )


if __name__ == "__main__":

    main()