import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

ML_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features.csv"
)


MODEL_DIR = Path("models")


RESULTS_DIR = Path("outputs")


LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_baseline.pkl"
)


RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR
    / "random_forest_model.pkl"
)


RESULTS_PATH = (
    RESULTS_DIR
    / "threshold_optimization_results.csv"
)


FEATURE_COLUMNS = [

    "grid_latitude",

    "grid_longitude",

    "year",

    "month",

    "collisions_lag_1",

    "collisions_lag_2",

    "collisions_lag_3",

    "casualties_lag_1",

    "collisions_rolling_3",

    "collisions_rolling_6",

    "collisions_rolling_sum_3",

    "month_sin",

    "month_cos"

]


TARGET_COLUMN = "hotspot"


THRESHOLDS = [

    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90

]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print formatted section heading."""

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)


# ============================================================
# STEP 1: LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load ML dataset and select
    complete unseen 2025 test data.
    """

    print_section(
        "STEP 1: LOADING COMPLETE TEST DATA"
    )

    df = pd.read_csv(
        ML_DATA_PATH,
        low_memory=False
    )


    test_df = df[
        df["year"] == 2025
    ].copy()


    X_test = test_df[
        FEATURE_COLUMNS
    ]


    y_test = test_df[
        TARGET_COLUMN
    ]


    print(
        "Evaluation period: 2025"
    )


    print(
        f"\nTotal test records: "
        f"{len(test_df):,}"
    )


    print(
        "\nActual hotspot distribution:"
    )


    print(
        y_test.value_counts()
    )


    print(
        "\nActual hotspot percentage:"
    )


    print(
        (
            y_test
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


    return X_test, y_test


# ============================================================
# STEP 2: LOAD MODELS
# ============================================================

def load_models():
    """Load trained machine learning models."""

    print_section(
        "STEP 2: LOADING TRAINED MODELS"
    )


    print(
        "Loading Logistic Regression model..."
    )


    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )


    print(
        "Logistic Regression loaded successfully."
    )


    print(
        "\nLoading Random Forest model..."
    )


    random_forest_model = joblib.load(
        RANDOM_FOREST_MODEL_PATH
    )


    print(
        "Random Forest loaded successfully."
    )


    return {

        "Logistic Regression":
            logistic_model,

        "Random Forest":
            random_forest_model

    }


# ============================================================
# STEP 3: TEST THRESHOLDS
# ============================================================

def optimize_thresholds(
    model_name,
    model,
    X_test,
    y_test
):
    """
    Test multiple probability thresholds
    and calculate classification metrics.
    """

    print_section(
        f"STEP 3: THRESHOLD OPTIMIZATION - "
        f"{model_name.upper()}"
    )


    print(
        "Generating prediction probabilities..."
    )


    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    results = []


    print(
        "\nTesting thresholds...\n"
    )


    for threshold in THRESHOLDS:


        predictions = (

            probabilities >= threshold

        ).astype(int)


        precision = precision_score(

            y_test,

            predictions,

            zero_division=0

        )


        recall = recall_score(

            y_test,

            predictions,

            zero_division=0

        )


        f1 = f1_score(

            y_test,

            predictions,

            zero_division=0

        )


        matrix = confusion_matrix(

            y_test,

            predictions

        )


        tn, fp, fn, tp = (

            matrix.ravel()

        )


        predicted_hotspots = (
            predictions.sum()
        )


        results.append(

            {

                "model": model_name,

                "threshold": threshold,

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "true_positives": tp,

                "false_positives": fp,

                "false_negatives": fn,

                "true_negatives": tn,

                "predicted_hotspots":
                    predicted_hotspots

            }

        )


        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Predicted Hotspots: "
            f"{predicted_hotspots:,}"
        )


    results_df = pd.DataFrame(
        results
    )


    return results_df


# ============================================================
# STEP 4: DISPLAY BEST THRESHOLD
# ============================================================

def display_best_threshold(
    results_df,
    model_name
):
    """
    Identify threshold with
    highest F1 score.
    """

    print_section(
        f"BEST THRESHOLD - "
        f"{model_name.upper()}"
    )


    best_result = (

        results_df

        .loc[
            results_df[
                "f1_score"
            ].idxmax()
        ]

    )


    print(
        f"Best Threshold: "
        f"{best_result['threshold']:.2f}"
    )


    print(
        f"\nPrecision: "
        f"{best_result['precision']:.4f}"
    )


    print(
        f"Recall: "
        f"{best_result['recall']:.4f}"
    )


    print(
        f"F1 Score: "
        f"{best_result['f1_score']:.4f}"
    )


    print(
        f"\nTrue Positives: "
        f"{int(best_result['true_positives']):,}"
    )


    print(
        f"False Positives: "
        f"{int(best_result['false_positives']):,}"
    )


    print(
        f"False Negatives: "
        f"{int(best_result['false_negatives']):,}"
    )


    print(
        f"Predicted Hotspots: "
        f"{int(best_result['predicted_hotspots']):,}"
    )


    return best_result


# ============================================================
# STEP 5: COMPARE MODELS
# ============================================================

def compare_models(
    logistic_results,
    random_forest_results
):
    """
    Compare best threshold results
    between both models.
    """

    print_section(
        "STEP 5: FINAL MODEL COMPARISON"
    )


    logistic_best = (

        logistic_results

        .loc[
            logistic_results[
                "f1_score"
            ].idxmax()
        ]

    )


    random_forest_best = (

        random_forest_results

        .loc[
            random_forest_results[
                "f1_score"
            ].idxmax()
        ]

    )


    comparison_df = pd.DataFrame(

        [

            {

                "model":
                    "Logistic Regression",

                "best_threshold":
                    logistic_best[
                        "threshold"
                    ],

                "precision":
                    logistic_best[
                        "precision"
                    ],

                "recall":
                    logistic_best[
                        "recall"
                    ],

                "f1_score":
                    logistic_best[
                        "f1_score"
                    ]

            },

            {

                "model":
                    "Random Forest",

                "best_threshold":
                    random_forest_best[
                        "threshold"
                    ],

                "precision":
                    random_forest_best[
                        "precision"
                    ],

                "recall":
                    random_forest_best[
                        "recall"
                    ],

                "f1_score":
                    random_forest_best[
                        "f1_score"
                    ]

            }

        ]

    )


    print(
        comparison_df
        .to_string(
            index=False
        )
    )


    return comparison_df


# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================

def save_results(
    logistic_results,
    random_forest_results
):
    """
    Save threshold optimization
    results to CSV.
    """

    print_section(
        "STEP 6: SAVING THRESHOLD RESULTS"
    )


    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    all_results = pd.concat(

        [

            logistic_results,

            random_forest_results

        ],

        ignore_index=True

    )


    all_results.to_csv(

        RESULTS_PATH,

        index=False

    )


    print(
        "Threshold optimization results "
        "saved successfully:"
    )


    print(
        RESULTS_PATH
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
        "MODEL THRESHOLD OPTIMIZATION"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1: LOAD TEST DATA
    # --------------------------------------------------------

    X_test, y_test = (

        load_test_data()

    )


    # --------------------------------------------------------
    # STEP 2: LOAD MODELS
    # --------------------------------------------------------

    models = load_models()


    # --------------------------------------------------------
    # STEP 3: LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_results = (

        optimize_thresholds(

            "Logistic Regression",

            models[
                "Logistic Regression"
            ],

            X_test,

            y_test

        )

    )


    # --------------------------------------------------------
    # STEP 4: RANDOM FOREST
    # --------------------------------------------------------

    random_forest_results = (

        optimize_thresholds(

            "Random Forest",

            models[
                "Random Forest"
            ],

            X_test,

            y_test

        )

    )


    # --------------------------------------------------------
    # STEP 5: DISPLAY BEST THRESHOLDS
    # --------------------------------------------------------

    logistic_best = (

        display_best_threshold(

            logistic_results,

            "Logistic Regression"

        )

    )


    random_forest_best = (

        display_best_threshold(

            random_forest_results,

            "Random Forest"

        )

    )


    # --------------------------------------------------------
    # STEP 6: COMPARE MODELS
    # --------------------------------------------------------

    comparison_df = (

        compare_models(

            logistic_results,

            random_forest_results

        )

    )


    # --------------------------------------------------------
    # STEP 7: SAVE RESULTS
    # --------------------------------------------------------

    save_results(

        logistic_results,

        random_forest_results

    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "THRESHOLD OPTIMIZATION COMPLETE"
    )


    print(
        "Both models were evaluated "
        "using multiple probability thresholds."
    )


    print(
        "\nSelection criterion:"
    )


    print(
        "Highest F1 Score"
    )


    print(
        "\nBest Logistic Regression "
        "Threshold:"
    )


    print(
        f"{logistic_best['threshold']:.2f}"
    )


    print(
        "\nBest Random Forest "
        "Threshold:"
    )


    print(
        f"{random_forest_best['threshold']:.2f}"
    )


if __name__ == "__main__":

    main()