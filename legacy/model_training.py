import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_baseline.pkl"
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


# ------------------------------------------------------------
# SAMPLING CONFIGURATION
# ------------------------------------------------------------

NEGATIVE_TO_POSITIVE_RATIO = 4

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# STEP 1: LOAD ML DATASET
# ============================================================

def load_ml_data():
    """Load machine learning dataset."""

    print_section(
        "STEP 1: LOADING MACHINE LEARNING DATASET"
    )

    df = pd.read_csv(
        ML_DATA_PATH,
        low_memory=False
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"Dataset loaded from:\n{ML_DATA_PATH}"
    )

    return df


# ============================================================
# STEP 2: TIME-BASED TRAIN TEST SPLIT
# ============================================================

def create_time_split(df):
    """
    Create train and test datasets
    using a chronological split.
    """

    print_section(
        "STEP 2: CREATING TIME-BASED TRAIN TEST SPLIT"
    )

    train_df = df[
        df["year"] < 2025
    ].copy()

    test_df = df[
        df["year"] == 2025
    ].copy()

    print(
        "\nTraining period: 2021 - 2024"
    )

    print(
        "Testing period: 2025"
    )

    print(
        f"\nOriginal training records: "
        f"{len(train_df):,}"
    )

    print(
        f"Complete testing records: "
        f"{len(test_df):,}"
    )

    return train_df, test_df


# ============================================================
# STEP 3: CREATE CONTROLLED TRAINING DATASET
# ============================================================

def create_controlled_training_dataset(train_df):
    """
    Keep all hotspot records and sample
    non-hotspot records using a controlled ratio.

    This reduces computational requirements
    while retaining all positive examples.
    """

    print_section(
        "STEP 3: CREATING CONTROLLED TRAINING DATASET"
    )

    hotspot_df = train_df[
        train_df[TARGET_COLUMN] == 1
    ].copy()

    non_hotspot_df = train_df[
        train_df[TARGET_COLUMN] == 0
    ].copy()

    hotspot_count = len(hotspot_df)

    non_hotspot_count = len(non_hotspot_df)

    print(
        f"Total hotspot records: "
        f"{hotspot_count:,}"
    )

    print(
        f"Total non-hotspot records: "
        f"{non_hotspot_count:,}"
    )

    print(
        f"\nSampling ratio:"
    )

    print(
        f"1 hotspot : "
        f"{NEGATIVE_TO_POSITIVE_RATIO} non-hotspots"
    )

    required_non_hotspots = (
        hotspot_count
        * NEGATIVE_TO_POSITIVE_RATIO
    )

    if required_non_hotspots > non_hotspot_count:

        required_non_hotspots = (
            non_hotspot_count
        )

        print(
            "\nWarning:"
        )

        print(
            "Not enough non-hotspot records "
            "for requested sampling ratio."
        )

        print(
            "Using all available non-hotspot records."
        )

    print(
        f"\nSampling non-hotspot records: "
        f"{required_non_hotspots:,}"
    )

    sampled_non_hotspot_df = (
        non_hotspot_df.sample(
            n=required_non_hotspots,
            random_state=RANDOM_STATE
        )
    )

    controlled_train_df = pd.concat(
        [
            hotspot_df,
            sampled_non_hotspot_df
        ],
        ignore_index=True
    )

    controlled_train_df = (
        controlled_train_df.sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    print(
        f"\nFinal controlled training records: "
        f"{len(controlled_train_df):,}"
    )

    print(
        "\nControlled training distribution:"
    )

    print(
        controlled_train_df[
            TARGET_COLUMN
        ].value_counts()
    )

    print(
        "\nControlled training percentage:"
    )

    print(
        (
            controlled_train_df[
                TARGET_COLUMN
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    return controlled_train_df


# ============================================================
# STEP 4: PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df,
    test_df
):
    """Prepare feature matrices and target vectors."""

    print_section(
        "STEP 4: PREPARING FEATURES"
    )

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    print(
        "\nFeatures used:"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"- {feature}"
        )

    print(
        f"\nTotal features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"\nTraining feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Testing feature shape: "
        f"{X_test.shape}"
    )

    print(
        "\nTraining hotspot distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTesting hotspot distribution "
        "(COMPLETE 2025 DATA):"
    )

    print(
        y_test.value_counts()
    )

    print(
        "\nTesting hotspot percentage:"
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

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# STEP 5: TRAIN BASELINE MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):
    """
    Train Logistic Regression
    baseline model using feature scaling.
    """

    print_section(
        "STEP 5: TRAINING BASELINE MODEL"
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "Feature scaling: StandardScaler"
    )

    print(
        "Training data: Controlled sampled dataset"
    )

    print(
        "Solver: lbfgs"
    )

    print(
        "Maximum iterations: 500"
    )

    print(
        "\nTraining model..."
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                    class_weight=None
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nModel training completed successfully."
    )

    return model


# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate trained model on the
    complete unseen 2025 dataset.
    """

    print_section(
        "STEP 6: MODEL EVALUATION"
    )

    print(
        "Evaluation dataset:"
    )

    print(
        "Complete unseen 2025 dataset"
    )

    print(
        f"Total test records: "
        f"{len(X_test):,}"
    )

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    print(
        "Generating prediction probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print_section(
        "MODEL PERFORMANCE"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"ROC-AUC Score: {roc_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# STEP 7: SAVE MODEL
# ============================================================

def save_model(model):
    """Save trained model pipeline to disk."""

    print_section(
        "STEP 7: SAVING TRAINED MODEL"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model pipeline saved successfully:\n"
        f"{MODEL_PATH}"
    )

    print(
        "\nSaved pipeline contains:"
    )

    print(
        "- StandardScaler"
    )

    print(
        "- Logistic Regression model"
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
        "BASELINE MACHINE LEARNING MODEL"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    df = load_ml_data()


    # --------------------------------------------------------
    # STEP 2: TIME SPLIT
    # --------------------------------------------------------

    train_df, test_df = (
        create_time_split(df)
    )


    # --------------------------------------------------------
    # STEP 3: CONTROLLED TRAINING SAMPLE
    # --------------------------------------------------------

    controlled_train_df = (
        create_controlled_training_dataset(
            train_df
        )
    )


    # --------------------------------------------------------
    # FREE MEMORY
    # --------------------------------------------------------

    del df
    del train_df


    # --------------------------------------------------------
    # STEP 4: PREPARE FEATURES
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_features(
        controlled_train_df,
        test_df
    )


    # --------------------------------------------------------
    # FREE UNUSED DATA
    # --------------------------------------------------------

    del controlled_train_df


    # --------------------------------------------------------
    # STEP 5: TRAIN MODEL
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # STEP 6: EVALUATE MODEL
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )


    # --------------------------------------------------------
    # STEP 7: SAVE MODEL
    # --------------------------------------------------------

    save_model(
        model
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "Baseline machine learning "
        "model completed successfully."
    )

    print(
        "\nTraining strategy:"
    )

    print(
        "All hotspot records were retained."
    )

    print(
        f"Non-hotspot records were sampled "
        f"at a 1:{NEGATIVE_TO_POSITIVE_RATIO} ratio."
    )

    print(
        "\nEvaluation strategy:"
    )

    print(
        "The complete unseen 2025 dataset "
        "was used for final evaluation."
    )

    print(
        "\nFinal Model Metrics:"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC Score: "
        f"{metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":

    main()