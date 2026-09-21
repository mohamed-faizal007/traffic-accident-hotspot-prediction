import pandas as pd
import numpy as np
import joblib

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features_advanced.csv"
)

MODEL_PATH = (
    Path("models")
    / "advanced_random_forest_model.pkl"
)

OUTPUT_DIR = Path("outputs") / "visualizations"

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


TARGET_COLUMN = "hotspot"


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):

    print("\n")

    print("=" * 60)

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
        f"\nDataset loaded from:\n{DATA_PATH}"
    )

    return df


# ============================================================
# STEP 2: PREPARE TEST DATA
# ============================================================

def prepare_test_data(df):

    print_section(
        "STEP 2: PREPARING UNSEEN TEST DATA"
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
        f"\nTotal test records: {len(test_df):,}"
    )

    print(
        f"\nUnique spatial grids: "
        f"{test_df['grid_id'].nunique():,}"
    )

    print(
        "\nActual hotspot distribution:"
    )

    print(
        y_test.value_counts()
    )

    return X_test, y_test


# ============================================================
# STEP 3: LOAD FINAL MODEL
# ============================================================

def load_model():

    print_section(
        "STEP 3: LOADING FINAL MODEL"
    )

    print(
        "Model: Advanced Random Forest"
    )

    print(
        f"Threshold: {FINAL_THRESHOLD}"
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
        f"\nModel path:\n{MODEL_PATH}"
    )

    return model


# ============================================================
# STEP 4: GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test
):

    print_section(
        "STEP 4: GENERATING MODEL PREDICTIONS"
    )

    print(
        "Generating prediction probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Generating threshold predictions..."
    )

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    print(
        "\nPredictions generated successfully."
    )

    print(
        f"\nPrediction threshold: "
        f"{FINAL_THRESHOLD}"
    )

    print(
        f"\nPredicted hotspots: "
        f"{predictions.sum():,}"
    )

    return probabilities, predictions


# ============================================================
# STEP 5: CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    predictions
):

    print_section(
        "STEP 5: CREATING CONFUSION MATRIX"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Hotspot",
            "Hotspot"
        ]
    )

    display.plot(
        ax=ax
    )

    plt.title(
        "Advanced Random Forest\nConfusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nConfusion matrix saved:\n"
        f"{output_path}"
    )


# ============================================================
# STEP 6: ROC CURVE
# ============================================================

def create_roc_curve(
    y_test,
    probabilities
):

    print_section(
        "STEP 6: CREATING ROC CURVE"
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            probabilities
        )
    )

    roc_auc_score = auc(
        false_positive_rate,
        true_positive_rate
    )

    print(
        f"ROC-AUC Score: "
        f"{roc_auc_score:.4f}"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=(
            f"Advanced Random Forest "
            f"(AUC = {roc_auc_score:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nROC curve saved:\n"
        f"{output_path}"
    )


# ============================================================
# STEP 7: PRECISION RECALL CURVE
# ============================================================

def create_precision_recall_curve(
    y_test,
    probabilities
):

    print_section(
        "STEP 7: CREATING PRECISION-RECALL CURVE"
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    average_precision = (
        average_precision_score(
            y_test,
            probabilities
        )
    )

    print(
        f"Average Precision Score: "
        f"{average_precision:.4f}"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"Advanced Random Forest "
            f"(AP = {average_precision:.4f})"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "precision_recall_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nPrecision-Recall curve saved:\n"
        f"{output_path}"
    )


# ============================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(
    model
):

    print_section(
        "STEP 8: CREATING FEATURE IMPORTANCE CHART"
    )

    importances = (
        model.feature_importances_
    )

    feature_importance = pd.DataFrame({

        "feature": FEATURE_COLUMNS,

        "importance": importances

    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=True
        )
    )

    print(
        "\nFeature Importance:"
    )

    print(
        feature_importance
        .sort_values(
            by="importance",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(10, 10)
    )

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Features"
    )

    plt.title(
        "Advanced Random Forest\nFeature Importance"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFeature importance chart saved:\n"
        f"{output_path}"
    )

    feature_importance = (
        feature_importance
        .sort_values(
            by="importance",
            ascending=False
        )
    )

    csv_path = (
        OUTPUT_DIR
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nFeature importance data saved:\n"
        f"{csv_path}"
    )


# ============================================================
# STEP 9: MODEL COMPARISON CHART
# ============================================================

def create_model_comparison():

    print_section(
        "STEP 9: CREATING MODEL COMPARISON CHART"
    )

    comparison_data = pd.DataFrame({

        "Model": [

            "Logistic Regression",

            "Random Forest",

            "Advanced Random Forest"

        ],

        "ROC-AUC": [

            0.7769,

            0.7886,

            0.8508

        ],

        "F1 Score": [

            0.2187,

            0.2298,

            0.2621

        ]

    })

    print(
        "\nModel Comparison:"
    )

    print(
        comparison_data
        .to_string(
            index=False
        )
    )

    x_positions = np.arange(
        len(
            comparison_data
        )
    )

    bar_width = 0.35


    # --------------------------------------------------------
    # ROC-AUC CHART
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        comparison_data["Model"],
        comparison_data["ROC-AUC"]
    )

    plt.ylabel(
        "ROC-AUC Score"
    )

    plt.title(
        "Model Comparison - ROC-AUC"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    roc_output_path = (
        OUTPUT_DIR
        / "model_comparison_roc_auc.png"
    )

    plt.savefig(
        roc_output_path,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # F1 SCORE CHART
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        comparison_data["Model"],
        comparison_data["F1 Score"]
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Model Comparison - F1 Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    f1_output_path = (
        OUTPUT_DIR
        / "model_comparison_f1.png"
    )

    plt.savefig(
        f1_output_path,
        dpi=300
    )

    plt.close()


    comparison_csv = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    comparison_data.to_csv(
        comparison_csv,
        index=False
    )

    print(
        "\nROC-AUC comparison saved:"
    )

    print(
        roc_output_path
    )

    print(
        "\nF1 Score comparison saved:"
    )

    print(
        f1_output_path
    )

    print(
        "\nComparison data saved:"
    )

    print(
        comparison_csv
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
        "FINAL MODEL EVALUATION VISUALIZATIONS"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    df = load_data()


    # --------------------------------------------------------
    # STEP 2: PREPARE TEST DATA
    # --------------------------------------------------------

    X_test, y_test = (
        prepare_test_data(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 3: LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # STEP 4: PREDICTIONS
    # --------------------------------------------------------

    probabilities, predictions = (
        generate_predictions(
            model,
            X_test
        )
    )


    # --------------------------------------------------------
    # STEP 5: CONFUSION MATRIX
    # --------------------------------------------------------

    create_confusion_matrix(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # STEP 6: ROC CURVE
    # --------------------------------------------------------

    create_roc_curve(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # STEP 7: PRECISION RECALL CURVE
    # --------------------------------------------------------

    create_precision_recall_curve(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # STEP 8: FEATURE IMPORTANCE
    # --------------------------------------------------------

    create_feature_importance(
        model
    )


    # --------------------------------------------------------
    # STEP 9: MODEL COMPARISON
    # --------------------------------------------------------

    create_model_comparison()


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "VISUALIZATION PIPELINE COMPLETE"
    )

    print(
        "All final model evaluation "
        "visualizations created successfully."
    )

    print(
        "\nGenerated files:"
    )

    print(
        "- confusion_matrix.png"
    )

    print(
        "- roc_curve.png"
    )

    print(
        "- precision_recall_curve.png"
    )

    print(
        "- feature_importance.png"
    )

    print(
        "- feature_importance.csv"
    )

    print(
        "- model_comparison_roc_auc.png"
    )

    print(
        "- model_comparison_f1.png"
    )

    print(
        "- model_comparison.csv"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()