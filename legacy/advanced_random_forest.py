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
    / "ml_features_advanced.csv"
)


MODEL_DIR = Path("models")


MODEL_PATH = (
    MODEL_DIR
    / "advanced_random_forest_model.pkl"
)


TARGET_COLUMN = "hotspot"


RANDOM_STATE = 42


NEGATIVE_SAMPLING_RATIO = 4


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # SPATIAL FEATURES
    # --------------------------------------------------------

    "grid_latitude",
    "grid_longitude",


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    "year",
    "month",

    "month_sin",
    "month_cos",


    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    "collisions_lag_1",
    "collisions_lag_2",
    "collisions_lag_3",

    "casualties_lag_1",


    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    "collisions_rolling_3",
    "collisions_rolling_6",
    "collisions_rolling_sum_3",


    # --------------------------------------------------------
    # ADVANCED HISTORICAL FEATURES
    # --------------------------------------------------------

    "historical_total_collisions",

    "historical_avg_collisions",

    "historical_max_collisions",

    "historical_collision_frequency",

    "historical_hotspot_frequency",


    # --------------------------------------------------------
    # RECENT ACTIVITY FEATURES
    # --------------------------------------------------------

    "collisions_last_3_months",

    "collisions_last_6_months",

    "collisions_last_12_months",


    # --------------------------------------------------------
    # RECENCY FEATURE
    # --------------------------------------------------------

    "months_since_last_collision"

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
# STEP 1: LOAD DATA
# ============================================================

def load_ml_data():

    """Load advanced machine learning dataset."""

    print_section(
        "STEP 1: LOADING ADVANCED MACHINE LEARNING DATASET"
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
        f"Unique grids: "
        f"{df['grid_id'].nunique():,}"
    )


    print(
        f"\nDataset loaded from:\n"
        f"{ML_DATA_PATH}"
    )


    return df


# ============================================================
# STEP 2: TIME-BASED TRAIN TEST SPLIT
# ============================================================

def create_time_split(df):

    """
    Create chronological training
    and testing datasets.
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

def create_controlled_training_data(train_df):

    """
    Keep all hotspot records and
    randomly sample non-hotspot records.
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


    print(
        "\nSampling ratio:"
    )


    print(
        f"1 hotspot : "
        f"{NEGATIVE_SAMPLING_RATIO} non-hotspots"
    )


    sample_size = (

        len(hotspot_df)

        * NEGATIVE_SAMPLING_RATIO

    )


    print(
        f"\nSampling non-hotspot records: "
        f"{sample_size:,}"
    )


    sampled_non_hotspot_df = (

        non_hotspot_df.sample(

            n=sample_size,

            random_state=RANDOM_STATE

        )

    )


    controlled_train_df = (

        pd.concat(

            [

                hotspot_df,

                sampled_non_hotspot_df

            ],

            ignore_index=True

        )

    )


    controlled_train_df = (

        controlled_train_df.sample(

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

    """
    Prepare training and testing
    feature matrices.
    """


    print_section(
        "STEP 4: PREPARING ADVANCED FEATURES"
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
# STEP 5: TRAIN ADVANCED RANDOM FOREST
# ============================================================

def train_model(

    X_train,

    y_train

):

    """
    Train Advanced Random Forest model.
    """


    print_section(
        "STEP 5: TRAINING ADVANCED RANDOM FOREST MODEL"
    )


    print(
        "Model: Random Forest Classifier"
    )


    print(
        "Number of trees: 150"
    )


    print(
        "Maximum depth: 18"
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

        n_estimators=150,

        max_depth=18,

        min_samples_leaf=5,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1

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
    Evaluate model using complete
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


    cm = confusion_matrix(

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
        cm
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

def analyze_feature_importance(model):

    """
    Display feature importance
    from Random Forest.
    """


    print_section(
        "STEP 7: FEATURE IMPORTANCE ANALYSIS"
    )


    importance_df = pd.DataFrame({

        "feature": FEATURE_COLUMNS,

        "importance": model.feature_importances_

    })


    importance_df = (

        importance_df

        .sort_values(

            by="importance",

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
        importance_df.to_string(
            index=False
        )
    )


    return importance_df


# ============================================================
# STEP 8: SAVE MODEL
# ============================================================

def save_model(model):

    """
    Save trained Advanced
    Random Forest model.
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
        "- Advanced Random Forest Classifier"
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
        "ADVANCED RANDOM FOREST MACHINE LEARNING MODEL"
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
    # STEP 3: CONTROLLED TRAINING DATA
    # --------------------------------------------------------

    controlled_train_df = (

        create_controlled_training_data(

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

    importance_df = (

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


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_section(
        "ADVANCED RANDOM FOREST TRAINING COMPLETE"
    )


    print(
        "Advanced Random Forest model "
        "completed successfully."
    )


    print(
        "\nTraining strategy:"
    )


    print(
        "All hotspot records were retained."
    )


    print(
        "Non-hotspot records were sampled "
        "at a 1:4 ratio."
    )


    print(
        "\nEvaluation strategy:"
    )


    print(
        "The complete unseen 2025 dataset "
        "was used for final evaluation."
    )


    print(
        "\nAdvanced features added:"
    )


    advanced_features = [

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


    for feature in advanced_features:

        print(
            f"- {feature}"
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


    print(
        "\nPrevious Best Random Forest F1 Score:"
    )


    print(
        "0.2298"
    )


    print(
        "\nAdvanced Model F1 Score:"
    )


    print(
        f"{metrics['f1_score']:.4f}"
    )


if __name__ == "__main__":

    main()