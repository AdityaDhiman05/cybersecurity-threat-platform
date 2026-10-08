"""
================================================================================
ANOMALY DETECTION — SHAP EXPLAINABILITY
================================================================================

Project:
    Cybersecurity Threat Platform

Purpose:
    Explain why the Isolation Forest anomaly detector considers a network
    event normal or anomalous.

Model:
    IsolationForest

Dataset:
    UNSW-NB15

Outputs:
    outputs/xai/anomaly_feature_importance.csv
    outputs/xai/anomaly_shap_values.csv
    outputs/xai/anomaly_shap_summary.png
    outputs/xai/anomaly_shap_bar.png

Important Interpretation:
    Isolation Forest learns the normal traffic profile.

    For this model:

        Positive SHAP value
            -> pushes the model toward a MORE NORMAL decision

        Negative SHAP value
            -> pushes the model toward a MORE ANOMALOUS decision

    Therefore, SHAP is explaining deviation from learned normal behavior,
    not predicting a specific attack category.

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly_detector.pkl"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly_preprocessor.pkl"
)

XAI_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "xai"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

BACKGROUND_SAMPLE_SIZE = 100
EXPLANATION_SAMPLE_SIZE = 10

TOP_FEATURE_COUNT = 20

# Features used by the anomaly detection model.
#
# These are the same 15 numerical + 3 categorical features used during
# anomaly_detector.py training.
NUMERICAL_FEATURES = [
    "sbytes",
    "smean",
    "ct_dst_src_ltm",
    "ct_srv_dst",
    "ct_srv_src",
    "sload",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "dbytes",
    "dmean",
    "ct_flw_http_mthd",
    "trans_depth",
    "sttl",
    "dur",
    "ct_src_ltm",
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state",
]

MODEL_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_separator(title: str) -> None:
    """
    Print a formatted section separator.
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def create_output_directory() -> None:
    """
    Create the XAI output directory if it does not already exist.
    """

    XAI_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_anomaly_model():
    """
    Load the trained Isolation Forest anomaly detector and its preprocessor.

    Returns
    -------
    tuple
        anomaly_model, anomaly_preprocessor
    """

    print_separator(
        "LOADING ANOMALY DETECTION MODEL FOR SHAP EXPLAINABILITY"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model not found:\n{MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly preprocessor not found:\n{PREPROCESSOR_PATH}"
        )

    anomaly_model = joblib.load(MODEL_PATH)

    print("Anomaly model loaded successfully.")
    print(f"Model type: {type(anomaly_model).__name__}")

    anomaly_preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Anomaly preprocessor loaded successfully.")
    print(
        f"Preprocessor type: "
        f"{type(anomaly_preprocessor).__name__}"
    )

    return anomaly_model, anomaly_preprocessor


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_testing_dataset() -> pd.DataFrame:
    """
    Load the official UNSW-NB15 testing dataset.
    """

    print_separator(
        "LOADING UNSW-NB15 TESTING DATASET"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n{DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {dataframe.shape}")

    return dataframe


# =============================================================================
# PREPARE MODEL INPUT
# =============================================================================

def prepare_model_input(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract exactly the features used by the anomaly detector.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "The following required model features are missing:\n"
            + "\n".join(missing_features)
        )

    return dataframe[MODEL_FEATURES].copy()


# =============================================================================
# TRANSFORM DATA
# =============================================================================

def transform_testing_data(
    dataframe: pd.DataFrame,
    preprocessor,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Transform testing data using the already-trained anomaly preprocessor.

    Returns
    -------
    tuple
        raw feature dataframe,
        transformed dense numpy array
    """

    print_separator(
        "TRANSFORMING TESTING DATA"
    )

    model_input = prepare_model_input(dataframe)

    transformed_data = preprocessor.transform(model_input)

    # The ColumnTransformer may return a sparse matrix.
    # SHAP TreeExplainer works more reliably with dense input for this model.
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    transformed_data = np.asarray(
        transformed_data,
        dtype=np.float64,
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    print(
        f"Transformed dataset shape: "
        f"{transformed_data.shape}"
    )

    print(
        f"Number of feature names   : "
        f"{len(feature_names)}"
    )

    if transformed_data.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between transformed feature count and "
            "feature-name count."
        )

    return model_input, transformed_data


# =============================================================================
# SELECT EXPLANATION SAMPLES
# =============================================================================

def select_explanation_samples(
    model,
    transformed_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select representative samples for SHAP explanation.

    We intentionally select:
        - highly anomalous events
        - normal events
        - a mixture of both

    Isolation Forest decision_function:
        lower value  -> more anomalous
        higher value -> more normal

    Returns
    -------
    tuple
        background data,
        explanation data
    """

    print_separator(
        "SELECTING SHAP BACKGROUND AND EXPLANATION SAMPLES"
    )

    total_samples = len(transformed_data)

    if total_samples < (
        BACKGROUND_SAMPLE_SIZE
        + EXPLANATION_SAMPLE_SIZE
    ):
        raise ValueError(
            "Dataset is too small for the configured "
            "background and explanation sample sizes."
        )

    # -------------------------------------------------------------------------
    # Background
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_STATE)

    background_indices = rng.choice(
        total_samples,
        size=BACKGROUND_SAMPLE_SIZE,
        replace=False,
    )

    background_data = transformed_data[
        background_indices
    ]

    # -------------------------------------------------------------------------
    # Model decision scores
    # -------------------------------------------------------------------------

    decision_scores = model.decision_function(
        transformed_data
    )

    # Lower score = more anomalous.
    anomaly_order = np.argsort(
        decision_scores
    )

    most_anomalous_indices = anomaly_order[
        :EXPLANATION_SAMPLE_SIZE // 2
    ]

    most_normal_indices = anomaly_order[
        -(
            EXPLANATION_SAMPLE_SIZE
            - EXPLANATION_SAMPLE_SIZE // 2
        ):
    ]

    explanation_indices = np.concatenate(
        [
            most_anomalous_indices,
            most_normal_indices,
        ]
    )

    explanation_data = transformed_data[
        explanation_indices
    ]

    print(
        f"Background samples : "
        f"{len(background_data)}"
    )

    print(
        f"Explanation samples: "
        f"{len(explanation_data)}"
    )

    print(
        "\nExplanation sample composition:"
    )

    print(
        f"  Most anomalous events: "
        f"{len(most_anomalous_indices)}"
    )

    print(
        f"  Most normal events   : "
        f"{len(most_normal_indices)}"
    )

    return background_data, explanation_data


# =============================================================================
# CALCULATE SHAP VALUES
# =============================================================================

def calculate_shap_values(
    model,
    background_data: np.ndarray,
    explanation_data: np.ndarray,
):
    """
    Calculate SHAP values using TreeExplainer.

    For Isolation Forest, SHAP explains the model's decision behavior.

    Positive SHAP:
        pushes toward a more normal decision.

    Negative SHAP:
        pushes toward a more anomalous decision.
    """

    print_separator(
        "CALCULATING SHAP VALUES"
    )

    print(
        f"Background samples : "
        f"{len(background_data)}"
    )

    print(
        f"Explanation samples: "
        f"{len(explanation_data)}"
    )

    print(
        f"Feature count       : "
        f"{explanation_data.shape[1]}"
    )

    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(
        model,
        data=background_data,
    )

    print(
        "TreeExplainer created successfully."
    )

    print("\nCalculating SHAP values...")

    shap_result = explainer(
        explanation_data
    )

    # SHAP versions may return either:
    #
    #     shap.Explanation
    #
    # or a numpy array.
    #
    # We normalize the result here.

    if hasattr(shap_result, "values"):
        shap_values = shap_result.values
    else:
        shap_values = shap_result

    shap_values = np.asarray(
        shap_values
    )

    # Some SHAP/model combinations can return an extra output dimension.
    # Isolation Forest should normally produce a 2-D matrix.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    print(
        f"\nSHAP value matrix shape: "
        f"{shap_values.shape}"
    )

    return explainer, shap_values


# =============================================================================
# GLOBAL FEATURE IMPORTANCE
# =============================================================================

def calculate_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance for every feature.
    """

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": mean_absolute_shap,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance.insert(
        0,
        "rank",
        np.arange(
            1,
            len(feature_importance) + 1,
        ),
    )

    return feature_importance


# =============================================================================
# SAVE GLOBAL IMPORTANCE
# =============================================================================

def save_feature_importance(
    feature_importance: pd.DataFrame,
) -> None:
    """
    Save global SHAP feature importance to CSV.
    """

    output_path = (
        XAI_OUTPUT_DIRECTORY
        / "anomaly_feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nAnomaly feature importance saved:"
    )

    print(output_path)


# =============================================================================
# SAVE SHAP VALUES
# =============================================================================

def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Save SHAP values for all explanation samples.
    """

    output_path = (
        XAI_OUTPUT_DIRECTORY
        / "anomaly_shap_values.csv"
    )

    shap_dataframe = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nAnomaly SHAP values saved:"
    )

    print(output_path)


# =============================================================================
# CREATE SHAP BAR PLOT
# =============================================================================

def create_bar_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Create a global SHAP feature-importance bar plot.
    """

    output_path = (
        XAI_OUTPUT_DIRECTORY
        / "anomaly_shap_bar.png"
    )

    plt.figure(
        figsize=(14, 10)
    )

    shap.summary_plot(
        shap_values,
        features=np.zeros_like(shap_values),
        feature_names=feature_names,
        plot_type="bar",
        max_display=TOP_FEATURE_COUNT,
        show=False,
    )

    plt.title(
        "SHAP Feature Importance — Anomaly Detection"
    )

    plt.xlabel(
        "mean(|SHAP value|) "
        "(average impact on anomaly model output)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSHAP bar plot saved:"
    )

    print(output_path)


# =============================================================================
# CREATE SHAP SUMMARY PLOT
# =============================================================================

def create_summary_plot(
    shap_values: np.ndarray,
    explanation_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Create a SHAP summary / beeswarm plot.

    Interpretation:

        SHAP < 0
            More anomalous

        SHAP > 0
            More normal
    """

    output_path = (
        XAI_OUTPUT_DIRECTORY
        / "anomaly_shap_summary.png"
    )

    plt.figure(
        figsize=(14, 10)
    )

    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_names,
        max_display=TOP_FEATURE_COUNT,
        show=False,
    )

    plt.title(
        "SHAP Summary — Anomaly Detection"
    )

    plt.xlabel(
        "SHAP value "
        "(negative = more anomalous, "
        "positive = more normal)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSHAP summary plot saved:"
    )

    print(output_path)


# =============================================================================
# EVENT-LEVEL EXPLANATION
# =============================================================================

def print_event_explanations(
    model,
    original_data: pd.DataFrame,
    explanation_data: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Print local explanations for representative anomalous and normal events.

    This makes the XAI output useful for the project demonstration.
    """

    print_separator(
        "EVENT-LEVEL ANOMALY EXPLANATIONS"
    )

    decision_scores = model.decision_function(
        explanation_data
    )

    predictions = model.predict(
        explanation_data
    )

    # Isolation Forest:
    #
    #     -1 = anomaly
    #      1 = normal
    #

    for sample_number in range(
        len(explanation_data)
    ):

        print(
            f"\nEVENT EXPLANATION "
            f"{sample_number + 1}"
        )

        print("-" * 80)

        if predictions[sample_number] == -1:
            detection_status = "ANOMALY"
        else:
            detection_status = "NORMAL"

        print(
            f"Detection status : "
            f"{detection_status}"
        )

        print(
            f"Decision score   : "
            f"{decision_scores[sample_number]:.6f}"
        )

        print(
            "\nTop SHAP contributors:"
        )

        sample_shap_values = (
            shap_values[sample_number]
        )

        ranked_indices = np.argsort(
            np.abs(sample_shap_values)
        )[::-1]

        displayed_count = min(
            10,
            len(ranked_indices),
        )

        for rank in range(
            displayed_count
        ):

            feature_index = (
                ranked_indices[rank]
            )

            feature_name = (
                feature_names[feature_index]
            )

            shap_value = (
                sample_shap_values[
                    feature_index
                ]
            )

            if shap_value < 0:
                interpretation = (
                    "↓ More anomalous"
                )
            else:
                interpretation = (
                    "↑ More normal"
                )

            print(
                f"{rank + 1:02d}. "
                f"{feature_name:<45} "
                f"{shap_value:+.6f}  "
                f"{interpretation}"
            )


# =============================================================================
# PRINT TOP FEATURES
# =============================================================================

def print_top_features(
    feature_importance: pd.DataFrame,
) -> None:
    """
    Print the top global SHAP features.
    """

    print_separator(
        "TOP GLOBAL SHAP FEATURES"
    )

    print(
        feature_importance.head(
            TOP_FEATURE_COUNT
        ).to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main execution pipeline for anomaly detection XAI.
    """

    print_separator(
        "ANOMALY DETECTION — SHAP EXPLAINABILITY"
    )

    print(
        "Project: Cybersecurity Threat Platform"
    )

    print(
        "Model: Isolation Forest"
    )

    print(
        "Purpose: Explain anomalous network behavior"
    )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    create_output_directory()

    # -------------------------------------------------------------------------
    # Load model and preprocessor
    # -------------------------------------------------------------------------

    anomaly_model, anomaly_preprocessor = (
        load_anomaly_model()
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    testing_dataframe = (
        load_testing_dataset()
    )

    # -------------------------------------------------------------------------
    # Transform data
    # -------------------------------------------------------------------------

    (
        original_model_input,
        transformed_data,
    ) = transform_testing_data(
        testing_dataframe,
        anomaly_preprocessor,
    )

    # -------------------------------------------------------------------------
    # Get feature names
    # -------------------------------------------------------------------------

    feature_names = list(
        anomaly_preprocessor.get_feature_names_out()
    )

    # -------------------------------------------------------------------------
    # Select SHAP samples
    # -------------------------------------------------------------------------

    (
        background_data,
        explanation_data,
    ) = select_explanation_samples(
        anomaly_model,
        transformed_data,
    )

    # -------------------------------------------------------------------------
    # Calculate SHAP values
    # -------------------------------------------------------------------------

    (
        explainer,
        shap_values,
    ) = calculate_shap_values(
        anomaly_model,
        background_data,
        explanation_data,
    )

    # -------------------------------------------------------------------------
    # Validate SHAP dimensions
    # -------------------------------------------------------------------------

    if shap_values.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "SHAP feature count does not match "
            "preprocessor feature count."
        )

    # -------------------------------------------------------------------------
    # Global feature importance
    # -------------------------------------------------------------------------

    feature_importance = (
        calculate_global_feature_importance(
            shap_values,
            feature_names,
        )
    )

    # -------------------------------------------------------------------------
    # Print global importance
    # -------------------------------------------------------------------------

    print_top_features(
        feature_importance
    )

    # -------------------------------------------------------------------------
    # Save feature importance
    # -------------------------------------------------------------------------

    save_feature_importance(
        feature_importance
    )

    # -------------------------------------------------------------------------
    # Save raw SHAP values
    # -------------------------------------------------------------------------

    save_shap_values(
        shap_values,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # Create bar plot
    # -------------------------------------------------------------------------

    create_bar_plot(
        shap_values,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # Create summary plot
    # -------------------------------------------------------------------------

    create_summary_plot(
        shap_values,
        explanation_data,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # Event-level explanations
    # -------------------------------------------------------------------------

    print_event_explanations(
        anomaly_model,
        original_model_input,
        explanation_data,
        shap_values,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    print_separator(
        "ANOMALY SHAP ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {XAI_OUTPUT_DIRECTORY / 'anomaly_feature_importance.csv'}"
    )

    print(
        f"2. {XAI_OUTPUT_DIRECTORY / 'anomaly_shap_values.csv'}"
    )

    print(
        f"3. {XAI_OUTPUT_DIRECTORY / 'anomaly_shap_summary.png'}"
    )

    print(
        f"4. {XAI_OUTPUT_DIRECTORY / 'anomaly_shap_bar.png'}"
    )

    print(
        "\nThese outputs will be used for the project's"
    )

    print(
        "Explainable AI / Anomaly Detection section."
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()