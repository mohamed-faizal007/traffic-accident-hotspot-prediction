import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")

COMPARISON_PATH = (
    OUTPUT_DIR
    / "final_model_comparison.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# MODEL RESULTS
# ============================================================

def create_model_comparison():
    """
    Create final comparison between
    all trained machine learning models.
    """

    print_section(
        "STEP 1: CREATING FINAL MODEL COMPARISON"
    )

    model_results = {

        "model": [

            "Logistic Regression",

            "Random Forest",

            "Advanced Random Forest"
        ],

        "roc_auc": [

            0.7769,

            0.7886,

            0.8508
        ],

        "best_threshold": [

            0.60,

            0.90,

            0.90
        ],

        "precision": [

            0.2031,

            0.2337,

            0.2667
        ],

        "recall": [

            0.2368,

            0.2261,

            0.2577
        ],

        "f1_score": [

            0.2187,

            0.2298,

            0.2621
        ]
    }

    comparison_df = pd.DataFrame(
        model_results
    )

    print(
        "\nFinal Model Comparison:\n"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    return comparison_df


# ============================================================
# SELECT FINAL MODEL
# ============================================================

def select_final_model(comparison_df):
    """
    Select the final model based on
    highest F1 score.
    """

    print_section(
        "STEP 2: SELECTING FINAL MODEL"
    )

    final_model = (
        comparison_df.loc[
            comparison_df[
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
        "\nFINAL SELECTED MODEL"
    )

    print(
        f"\nModel: "
        f"{final_model['model']}"
    )

    print(
        f"ROC-AUC: "
        f"{final_model['roc_auc']:.4f}"
    )

    print(
        f"Best Threshold: "
        f"{final_model['best_threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{final_model['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{final_model['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{final_model['f1_score']:.4f}"
    )

    return final_model


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

def calculate_improvement(
    comparison_df,
    final_model
):
    """
    Calculate improvement of the
    final model over baseline.
    """

    print_section(
        "STEP 3: MODEL IMPROVEMENT ANALYSIS"
    )

    baseline = comparison_df[
        comparison_df["model"]
        == "Logistic Regression"
    ].iloc[0]

    f1_improvement = (
        (
            final_model["f1_score"]
            - baseline["f1_score"]
        )
        / baseline["f1_score"]
    ) * 100

    roc_auc_improvement = (
        (
            final_model["roc_auc"]
            - baseline["roc_auc"]
        )
        / baseline["roc_auc"]
    ) * 100

    print(
        "Baseline Model:"
    )

    print(
        f"{baseline['model']}"
    )

    print(
        "\nFinal Model:"
    )

    print(
        f"{final_model['model']}"
    )

    print(
        "\nF1 Score Improvement:"
    )

    print(
        f"{f1_improvement:.2f}%"
    )

    print(
        "\nROC-AUC Improvement:"
    )

    print(
        f"{roc_auc_improvement:.2f}%"
    )

    return (
        f1_improvement,
        roc_auc_improvement
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(comparison_df):
    """
    Save final model comparison.
    """

    print_section(
        "STEP 4: SAVING MODEL COMPARISON"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print(
        "Model comparison saved successfully:"
    )

    print(
        f"{COMPARISON_PATH}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    final_model,
    f1_improvement,
    roc_auc_improvement
):
    """
    Print final project
    model selection summary.
    """

    print_section(
        "FINAL MODEL SELECTION COMPLETE"
    )

    print(
        "Final selected model:"
    )

    print(
        f"{final_model['model']}"
    )

    print(
        "\nFinal Configuration:"
    )

    print(
        f"Probability Threshold: "
        f"{final_model['best_threshold']:.2f}"
    )

    print(
        "\nFinal Performance:"
    )

    print(
        f"ROC-AUC: "
        f"{final_model['roc_auc']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_model['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{final_model['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{final_model['f1_score']:.4f}"
    )

    print(
        "\nImprovement over baseline:"
    )

    print(
        f"F1 Score Improvement: "
        f"{f1_improvement:.2f}%"
    )

    print(
        f"ROC-AUC Improvement: "
        f"{roc_auc_improvement:.2f}%"
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
        "FINAL MACHINE LEARNING MODEL SELECTION"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1: CREATE COMPARISON
    # --------------------------------------------------------

    comparison_df = (
        create_model_comparison()
    )


    # --------------------------------------------------------
    # STEP 2: SELECT FINAL MODEL
    # --------------------------------------------------------

    final_model = (
        select_final_model(
            comparison_df
        )
    )


    # --------------------------------------------------------
    # STEP 3: IMPROVEMENT ANALYSIS
    # --------------------------------------------------------

    (
        f1_improvement,
        roc_auc_improvement
    ) = calculate_improvement(
        comparison_df,
        final_model
    )


    # --------------------------------------------------------
    # STEP 4: SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        comparison_df
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        final_model,
        f1_improvement,
        roc_auc_improvement
    )


if __name__ == "__main__":

    main()