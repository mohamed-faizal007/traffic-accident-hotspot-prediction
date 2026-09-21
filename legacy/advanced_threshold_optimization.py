import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

ML_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features_advanced.csv"
)

MODEL_PATH = Path(
    "models/advanced_random_forest_model.pkl"
)

OUTPUT_DIR = Path(
    "outputs"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "advanced_threshold_optimization_results.csv"
)


TARGET_COLUMN = "hotspot"


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


THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95
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
    Load advanced ML dataset
    and select complete unseen 2025 data.
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

    print(
        "\nEvaluation period: 2025"
    )

    print(
        f"\nTotal test records: "
        f"{len(test_df):,}"
    )

    print(
        "\nActual hotspot distribution:"
    )

    print(
        test_df[
            TARGET_COLUMN
        ].value_counts()
    )

    print(
        "\nActual hotspot percentage:"
    )

    print(
        (
            test_df[
                TARGET_COLUMN
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    return X_test, y_test


# ============================================================
# STEP 2: LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained Advanced
    Random Forest model.
    """

    print_section(
        "STEP 2: LOADING ADVANCED RANDOM FOREST MODEL"
    )

    print(
        "Loading model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "\nAdvanced Random Forest "
        "loaded successfully."
    )

    print(
        f"\nModel path:\n{MODEL_PATH}"
    )

    return model


# ============================================================
# STEP 3: GENERATE PROBABILITIES
# ============================================================

def generate_probabilities(
    model,
    X_test
):
    """
    Generate hotspot
    prediction probabilities.
    """

    print_section(
        "STEP 3: GENERATING PREDICTION PROBABILITIES"
    )

    print(
        "Generating probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Probability generation "
        "completed successfully."
    )

    return probabilities


# ============================================================
# STEP 4: CALCULATE ROC-AUC
# ============================================================

def calculate_roc_auc(
    y_test,
    probabilities
):
    """
    Calculate ROC-AUC score
    independent of threshold.
    """

    print_section(
        "STEP 4: ROC-AUC ANALYSIS"
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"ROC-AUC Score: "
        f"{roc_auc:.4f}"
    )

    return roc_auc


# ============================================================
# STEP 5: TEST THRESHOLDS
# ============================================================

def optimize_threshold(
    y_test,
    probabilities
):
    """
    Test multiple probability thresholds
    and calculate classification metrics.
    """

    print_section(
        "STEP 5: THRESHOLD OPTIMIZATION"
    )

    print(
        "Testing probability thresholds...\n"
    )

    results = []

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

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Predicted Hotspots: "
            f"{predicted_hotspots:,}"
        )

        results.append({

            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "true_negatives":
                tn,

            "predicted_hotspots":
                predicted_hotspots
        })

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# STEP 6: FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    results_df
):
    """
    Select threshold with
    highest F1 score.
    """

    print_section(
        "STEP 6: BEST THRESHOLD SELECTION"
    )

    best_result = (
        results_df.loc[
            results_df[
                "f1_score"
            ].idxmax()
        ]
    )

    print(
        "Selection criterion:"
    )

    print(
        "Highest F1 Score"
    )

    print(
        f"\nBest Threshold: "
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
# STEP 7: SAVE RESULTS
# ============================================================

def save_results(
    results_df
):
    """
    Save threshold optimization
    results to CSV.
    """

    print_section(
        "STEP 7: SAVING THRESHOLD RESULTS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        "Threshold optimization "
        "results saved successfully:"
    )

    print(
        f"{RESULTS_PATH}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    best_result,
    roc_auc
):
    """
    Print final model
    threshold optimization summary.
    """

    print_section(
        "ADVANCED THRESHOLD OPTIMIZATION COMPLETE"
    )

    print(
        "Advanced Random Forest was "
        "evaluated using multiple "
        "probability thresholds."
    )

    print(
        "\nEvaluation strategy:"
    )

    print(
        "Complete unseen 2025 dataset"
    )

    print(
        "\nSelection criterion:"
    )

    print(
        "Highest F1 Score"
    )

    print(
        "\nAdvanced Model ROC-AUC:"
    )

    print(
        f"{roc_auc:.4f}"
    )

    print(
        "\nBest Threshold:"
    )

    print(
        f"{best_result['threshold']:.2f}"
    )

    print(
        "\nFinal Metrics:"
    )

    print(
        f"Precision: "
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
        "ADVANCED MODEL THRESHOLD OPTIMIZATION"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1: LOAD COMPLETE TEST DATA
    # --------------------------------------------------------

    X_test, y_test = (
        load_test_data()
    )


    # --------------------------------------------------------
    # STEP 2: LOAD ADVANCED MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # STEP 3: GENERATE PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        generate_probabilities(
            model,
            X_test
        )
    )


    # --------------------------------------------------------
    # STEP 4: CALCULATE ROC-AUC
    # --------------------------------------------------------

    roc_auc = (
        calculate_roc_auc(
            y_test,
            probabilities
        )
    )


    # --------------------------------------------------------
    # STEP 5: OPTIMIZE THRESHOLD
    # --------------------------------------------------------

    results_df = (
        optimize_threshold(
            y_test,
            probabilities
        )
    )


    # --------------------------------------------------------
    # STEP 6: FIND BEST THRESHOLD
    # --------------------------------------------------------

    best_result = (
        find_best_threshold(
            results_df
        )
    )


    # --------------------------------------------------------
    # STEP 7: SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        results_df
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        best_result,
        roc_auc
    )


if __name__ == "__main__":

    main()