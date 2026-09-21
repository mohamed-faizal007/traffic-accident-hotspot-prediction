import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
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
    / "random_forest_model.pkl"
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


RANDOM_STATE = 42


NON_HOTSPOT_RATIO = 4


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
    Create training and testing datasets
    using chronological separation.
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

def create_controlled_training_dataset(
    train_df
):
    """
    Keep all hotspot examples.

    Randomly sample non-hotspot examples
    using the configured ratio.
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


    print(
        f"Total hotspot records: "
        f"{len(hotspot_df):,}"
    )

    print(
        f"Total non-hotspot records: "
        f"{len(non_hotspot_df):,}"
    )


    non_hotspot_sample_size = (
        len(hotspot_df)
        * NON_HOTSPOT_RATIO
    )


    print(
        "\nSampling ratio:"
    )

    print(
        f"1 hotspot : "
        f"{NON_HOTSPOT_RATIO} non-hotspots"
    )


    print(
        f"\nSampling non-hotspot records: "
        f"{non_hotspot_sample_size:,}"
    )


    sampled_non_hotspot_df = (
        non_hotspot_df
        .sample(
            n=non_hotspot_sample_size,
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
        controlled_train_df
        .sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(
            drop=True
        )
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
        ]
        .value_counts()
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
    """
    Prepare training and testing
    feature matrices.
    """

    print_section(
        "STEP 4: PREPARING FEATURES"
    )


    X_train = train_df[
        FEATURE_COLUMNS
    ]


    y_train = train_df[
        TARGET_COLUMN
    ]


    X_test = test_df[
        FEATURE_COLUMNS
    ]


    y_test = test_df[
        TARGET_COLUMN
    ]


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
# STEP 5: TRAIN RANDOM FOREST MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):
    """
    Train Random Forest model.
    """

    print_section(
        "STEP 5: TRAINING RANDOM FOREST MODEL"
    )


    print(
        "Model: Random Forest Classifier"
    )


    print(
        "Number of trees: 100"
    )


    print(
        "Maximum depth: 15"
    )


    print(
        "Minimum samples per leaf: 5"
    )


    print(
        "Class weight: Balanced"
    )


    print(
        "Parallel processing: Enabled"
    )


    print(
        "\nTraining model..."
    )


    model = RandomForestClassifier(

        n_estimators=100,

        max_depth=15,

        min_samples_leaf=5,

        class_weight="balanced",

        n_jobs=-1,

        random_state=RANDOM_STATE

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
    Evaluate model on complete
    unseen 2025 dataset.
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


    matrix = confusion_matrix(

        y_test,

        predictions

    )


    print_section(
        "MODEL PERFORMANCE"
    )


    print(
        f"Precision: "
        f"{precision:.4f}"
    )


    print(
        f"Recall: "
        f"{recall:.4f}"
    )


    print(
        f"F1 Score: "
        f"{f1:.4f}"
    )


    print(
        f"ROC-AUC Score: "
        f"{roc_auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        matrix
    )


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
# STEP 7: FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(
    model
):
    """
    Display Random Forest
    feature importance.
    """

    print_section(
        "STEP 7: FEATURE IMPORTANCE ANALYSIS"
    )


    importance_df = pd.DataFrame(

        {

            "feature":
                FEATURE_COLUMNS,

            "importance":
                model.feature_importances_

        }

    )


    importance_df = (

        importance_df

        .sort_values(
            "importance",
            ascending=False
        )

        .reset_index(
            drop=True
        )

    )


    print(
        "Feature Importance:\n"
    )


    print(
        importance_df
        .to_string(
            index=False
        )
    )


    return importance_df


# ============================================================
# STEP 8: SAVE MODEL
# ============================================================

def save_model(
    model
):
    """
    Save trained Random Forest
    model to disk.
    """

    print_section(
        "STEP 8: SAVING TRAINED MODEL"
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
        f"Model saved successfully:\n"
        f"{MODEL_PATH}"
    )


    print(
        "\nSaved model contains:"
    )


    print(
        "- Random Forest Classifier"
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
        "RANDOM FOREST MACHINE LEARNING MODEL"
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

        create_time_split(
            df
        )

    )


    # --------------------------------------------------------
    # STEP 3: CONTROLLED TRAINING DATASET
    # --------------------------------------------------------

    controlled_train_df = (

        create_controlled_training_dataset(
            train_df
        )

    )


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
    # STEP 7: FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = (

        analyze_feature_importance(
            model
        )

    )


    # --------------------------------------------------------
    # STEP 8: SAVE MODEL
    # --------------------------------------------------------

    save_model(
        model
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "RANDOM FOREST TRAINING COMPLETE"
    )


    print(
        "Random Forest model "
        "completed successfully."
    )


    print(
        "\nTraining strategy:"
    )


    print(
        "All hotspot records were retained."
    )


    print(
        f"Non-hotspot records were sampled "
        f"at a 1:{NON_HOTSPOT_RATIO} ratio."
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