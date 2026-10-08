"""
===============================================================================
CYBERSECURITY THREAT PLATFORM
Explainable AI / SHAP Analysis Module
===============================================================================

Purpose
-------
This module provides explainability for the binary Random Forest attack
detection model.

The objective is to answer:

    "Why did the model classify this network event as normal or attack?"

The module uses SHAP (SHapley Additive exPlanations) with the trained
Random Forest classifier.

Explainability Workflow
-----------------------

    UNSW-NB15 Test Event
            |
            v
    Binary Preprocessor
            |
            v
    Processed Features
            |
            v
    Random Forest
            |
            v
       SHAP Explainer
            |
            v
    Feature Contributions
            |
            v
    Human-readable Explanation


Outputs
-------
The module generates:

    outputs/xai/
        binary_shap_summary.png
        binary_shap_bar.png
        binary_shap_values.csv
        binary_feature_importance.csv

Important
---------
The SHAP explanation is generated for the binary attack detector first.

Known attack-category explanation will be implemented separately after the
binary explanation pipeline has been verified.

===============================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "models"

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "xai"

BINARY_MODEL_PATH = MODEL_DIRECTORY / "binary_classifier.pkl"

BINARY_PREPROCESSOR_PATH = MODEL_DIRECTORY / "binary_preprocessor.pkl"

TESTING_DATASET_PATH = DATA_DIRECTORY / "UNSW_NB15_testing-set.csv"


# =============================================================================
# XAI CONFIGURATION
# =============================================================================

# Number of normal records used as SHAP background data.
BACKGROUND_SAMPLE_SIZE = 100

# Number of test records to explain.
EXPLANATION_SAMPLE_SIZE = 10

# Random state for reproducibility.
RANDOM_STATE = 42


# =============================================================================
# DATASET COLUMNS
# =============================================================================

DROP_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_binary_model_and_preprocessor():
    """
    Load the trained binary Random Forest model and its preprocessor.

    Returns
    -------
    tuple
        (binary_model, binary_preprocessor)
    """

    print("=" * 80)
    print("LOADING BINARY MODEL FOR SHAP EXPLAINABILITY")
    print("=" * 80)

    if not BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Binary model not found:\n{BINARY_MODEL_PATH}"
        )

    if not BINARY_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Binary preprocessor not found:\n{BINARY_PREPROCESSOR_PATH}"
        )

    binary_model = joblib.load(BINARY_MODEL_PATH)

    binary_preprocessor = joblib.load(BINARY_PREPROCESSOR_PATH)

    print("\nBinary model loaded successfully.")

    print(
        f"Model type: "
        f"{type(binary_model).__name__}"
    )

    print("\nBinary preprocessor loaded successfully.")

    print(
        f"Preprocessor type: "
        f"{type(binary_preprocessor).__name__}"
    )

    return binary_model, binary_preprocessor


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_testing_dataset() -> pd.DataFrame:
    """
    Load the official UNSW-NB15 testing dataset.

    Returns
    -------
    pandas.DataFrame
        Testing dataset.
    """

    print("\n" + "=" * 80)
    print("LOADING UNSW-NB15 TESTING DATASET")
    print("=" * 80)

    if not TESTING_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n{TESTING_DATASET_PATH}"
        )

    dataframe = pd.read_csv(TESTING_DATASET_PATH)

    print(f"\nDataset shape: {dataframe.shape}")

    return dataframe


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features and target labels.

    Parameters
    ----------
    dataframe:
        UNSW-NB15 dataframe.

    Returns
    -------
    tuple
        X and y.
    """

    missing_columns = [
        column
        for column in DROP_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from dataset: "
            f"{missing_columns}"
        )

    X = dataframe.drop(
        columns=DROP_COLUMNS
    )

    y = dataframe["label"].copy()

    return X, y


# =============================================================================
# FEATURE NAME EXTRACTION
# =============================================================================

def get_transformed_feature_names(
    preprocessor,
) -> list[str]:
    """
    Extract feature names generated by the fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor:
        Fitted sklearn ColumnTransformer.

    Returns
    -------
    list[str]
        Names of transformed model features.
    """

    try:
        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

        return feature_names

    except Exception as exc:
        warnings.warn(
            "Could not retrieve transformed feature names. "
            f"Using generic feature names instead. Error: {exc}"
        )

        try:
            feature_count = len(
                preprocessor.transform(
                    pd.DataFrame(
                        [
                            {
                                feature: 0
                                for feature in preprocessor.feature_names_in_
                            }
                        ]
                    )
                ).toarray()[0]
            )

        except Exception:
            feature_count = 0

        return [
            f"feature_{index}"
            for index in range(feature_count)
        ]


# =============================================================================
# PREPROCESSING
# =============================================================================

def transform_dataset(
    X: pd.DataFrame,
    preprocessor,
):
    """
    Transform raw network features using the trained binary preprocessor.

    Parameters
    ----------
    X:
        Raw model input features.

    preprocessor:
        Fitted preprocessing pipeline.

    Returns
    -------
    numpy.ndarray
        Dense transformed feature matrix.
    """

    transformed = preprocessor.transform(X)

    # SHAP explanations for the Random Forest are easier to process with
    # a dense matrix for this project because the transformed feature space
    # is relatively small (194 features).
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed)


# =============================================================================
# SHAP VALUE EXTRACTION
# =============================================================================

def calculate_shap_values(
    model,
    background_data: np.ndarray,
    explanation_data: np.ndarray,
):
    """
    Calculate SHAP values using TreeExplainer.

    Parameters
    ----------
    model:
        Trained Random Forest classifier.

    background_data:
        Background samples used by the explainer.

    explanation_data:
        Samples for which explanations are required.

    Returns
    -------
    numpy.ndarray
        SHAP values corresponding to the attack class.
    """

    print("\n" + "=" * 80)
    print("CALCULATING SHAP VALUES")
    print("=" * 80)

    print(
        f"\nBackground samples : {len(background_data)}"
    )

    print(
        f"Explanation samples: {len(explanation_data)}"
    )

    print(
        f"Feature count       : {explanation_data.shape[1]}"
    )

    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(
        model,
        data=background_data,
    )

    print("TreeExplainer created successfully.")

    print("\nCalculating SHAP values...")

    shap_values = explainer.shap_values(
        explanation_data
    )

    # -------------------------------------------------------------------------
    # SHAP output handling
    # -------------------------------------------------------------------------
    #
    # Depending on the installed SHAP version, binary classification may
    # return either:
    #
    #     shape = (samples, features)
    #
    # or:
    #
    #     shape = (samples, features, classes)
    #
    # We always select the attack class (class 1).
    # -------------------------------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]

        else:
            shap_values = shap_values[-1]

    elif isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:

            # Newer SHAP versions may return:
            # samples x features x classes
            shap_values = shap_values[:, :, 1]

    shap_values = np.asarray(shap_values)

    print(
        f"\nSHAP value matrix shape: {shap_values.shape}"
    )

    return shap_values


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate global feature importance from mean absolute SHAP values.

    Parameters
    ----------
    shap_values:
        SHAP contribution matrix.

    feature_names:
        Transformed feature names.

    Returns
    -------
    pandas.DataFrame
        Ranked SHAP feature importance.
    """

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance_dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": mean_absolute_shap,
        }
    )

    importance_dataframe = (
        importance_dataframe
        .sort_values(
            by="mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_dataframe.insert(
        0,
        "rank",
        np.arange(
            1,
            len(importance_dataframe) + 1,
        ),
    )

    return importance_dataframe


# =============================================================================
# SHAP CONTRIBUTION TABLE
# =============================================================================

def create_shap_contribution_table(
    shap_values: np.ndarray,
    transformed_data: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Create a detailed SHAP contribution table.

    Each row represents a sample-feature combination.

    Parameters
    ----------
    shap_values:
        SHAP contribution matrix.

    transformed_data:
        Transformed feature values.

    feature_names:
        Feature names.

    Returns
    -------
    pandas.DataFrame
        SHAP contribution table.
    """

    rows = []

    number_of_samples = shap_values.shape[0]

    for sample_index in range(number_of_samples):

        sample_shap_values = shap_values[
            sample_index
        ]

        sample_feature_values = transformed_data[
            sample_index
        ]

        sorted_indices = np.argsort(
            np.abs(sample_shap_values)
        )[::-1]

        for feature_index in sorted_indices:

            rows.append(
                {
                    "sample_index": sample_index,
                    "feature": feature_names[feature_index],
                    "feature_value": sample_feature_values[
                        feature_index
                    ],
                    "shap_value": sample_shap_values[
                        feature_index
                    ],
                    "absolute_shap_value": abs(
                        sample_shap_values[
                            feature_index
                        ]
                    ),
                    "effect": (
                        "Increases Attack Probability"
                        if sample_shap_values[feature_index] > 0
                        else "Decreases Attack Probability"
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_shap_summary_plot(
    shap_values: np.ndarray,
    transformed_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Generate the global SHAP beeswarm summary plot.
    """

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        transformed_data,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    plt.title(
        "SHAP Summary — Binary Attack Detection",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "binary_shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSHAP summary plot saved:\n{output_path}"
    )


def generate_shap_bar_plot(
    shap_values: np.ndarray,
    transformed_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Generate the global SHAP bar plot.
    """

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        transformed_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    plt.title(
        "SHAP Feature Importance — Binary Attack Detection",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "binary_shap_bar.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSHAP bar plot saved:\n{output_path}"
    )


# =============================================================================
# HUMAN-READABLE EXPLANATION
# =============================================================================

def print_top_sample_explanation(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_index: int,
    top_n: int = 10,
) -> None:
    """
    Print the strongest SHAP contributors for one network event.

    Parameters
    ----------
    shap_values:
        SHAP contribution matrix.

    feature_names:
        Transformed feature names.

    sample_index:
        Index of the event being explained.

    top_n:
        Number of strongest contributors to display.
    """

    sample_values = shap_values[
        sample_index
    ]

    sorted_indices = np.argsort(
        np.abs(sample_values)
    )[::-1]

    print("\n" + "=" * 80)
    print(
        f"TOP SHAP CONTRIBUTORS FOR SAMPLE {sample_index}"
    )
    print("=" * 80)

    displayed = 0

    for feature_index in sorted_indices:

        shap_value = sample_values[
            feature_index
        ]

        if abs(shap_value) == 0:
            continue

        direction = (
            "↑ Attack probability"
            if shap_value > 0
            else "↓ Attack probability"
        )

        print(
            f"{displayed + 1:02d}. "
            f"{feature_names[feature_index]:40s} "
            f"{shap_value:+.6f}  "
            f"{direction}"
        )

        displayed += 1

        if displayed >= top_n:
            break


# =============================================================================
# MAIN XAI WORKFLOW
# =============================================================================

def run_binary_shap_analysis() -> None:
    """
    Execute the complete binary-model SHAP analysis.
    """

    print("\n")

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load model and preprocessor
    # -------------------------------------------------------------------------

    model, preprocessor = (
        load_binary_model_and_preprocessor()
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    dataframe = load_testing_dataset()

    # -------------------------------------------------------------------------
    # Prepare features and labels
    # -------------------------------------------------------------------------

    X, y = prepare_features(
        dataframe
    )

    # -------------------------------------------------------------------------
    # Transform complete testing dataset
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRANSFORMING TESTING DATA")
    print("=" * 80)

    X_transformed = transform_dataset(
        X,
        preprocessor,
    )

    feature_names = get_transformed_feature_names(
        preprocessor
    )

    print(
        f"\nTransformed dataset shape: "
        f"{X_transformed.shape}"
    )

    print(
        f"Number of feature names   : "
        f"{len(feature_names)}"
    )

    # -------------------------------------------------------------------------
    # Select background samples
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    normal_indices = np.where(
        y.to_numpy() == 0
    )[0]

    if len(normal_indices) == 0:
        raise ValueError(
            "No normal records were found in the testing dataset."
        )

    background_size = min(
        BACKGROUND_SAMPLE_SIZE,
        len(normal_indices),
    )

    background_indices = rng.choice(
        normal_indices,
        size=background_size,
        replace=False,
    )

    background_data = X_transformed[
        background_indices
    ]

    # -------------------------------------------------------------------------
    # Select events to explain
    # -------------------------------------------------------------------------

    explanation_size = min(
        EXPLANATION_SAMPLE_SIZE,
        len(X_transformed),
    )

    explanation_indices = rng.choice(
        len(X_transformed),
        size=explanation_size,
        replace=False,
    )

    explanation_data = X_transformed[
        explanation_indices
    ]

    explanation_labels = y.iloc[
        explanation_indices
    ].to_numpy()

    # -------------------------------------------------------------------------
    # Calculate SHAP values
    # -------------------------------------------------------------------------

    shap_values = calculate_shap_values(
        model=model,
        background_data=background_data,
        explanation_data=explanation_data,
    )

    # -------------------------------------------------------------------------
    # Validate dimensions
    # -------------------------------------------------------------------------

    if shap_values.shape != explanation_data.shape:
        raise ValueError(
            "SHAP value dimensions do not match transformed data.\n"
            f"SHAP shape: {shap_values.shape}\n"
            f"Data shape: {explanation_data.shape}"
        )

    # -------------------------------------------------------------------------
    # Global SHAP importance
    # -------------------------------------------------------------------------

    importance_dataframe = (
        calculate_shap_feature_importance(
            shap_values=shap_values,
            feature_names=feature_names,
        )
    )

    # -------------------------------------------------------------------------
    # Save global feature importance
    # -------------------------------------------------------------------------

    importance_path = (
        OUTPUT_DIRECTORY
        / "binary_feature_importance.csv"
    )

    importance_dataframe.to_csv(
        importance_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Create contribution table
    # -------------------------------------------------------------------------

    contribution_dataframe = (
        create_shap_contribution_table(
            shap_values=shap_values,
            transformed_data=explanation_data,
            feature_names=feature_names,
        )
    )

    contribution_path = (
        OUTPUT_DIRECTORY
        / "binary_shap_values.csv"
    )

    contribution_dataframe.to_csv(
        contribution_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------

    generate_shap_summary_plot(
        shap_values=shap_values,
        transformed_data=explanation_data,
        feature_names=feature_names,
    )

    generate_shap_bar_plot(
        shap_values=shap_values,
        transformed_data=explanation_data,
        feature_names=feature_names,
    )

    # -------------------------------------------------------------------------
    # Print top global features
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP GLOBAL SHAP FEATURES")
    print("=" * 80)

    print(
        importance_dataframe.head(20).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Print individual event explanations
    # -------------------------------------------------------------------------

    for local_index in range(
        min(3, len(explanation_indices))
    ):

        original_index = explanation_indices[
            local_index
        ]

        actual_label = explanation_labels[
            local_index
        ]

        predicted_label = model.predict(
            explanation_data[
                local_index:local_index + 1
            ]
        )[0]

        print("\n" + "=" * 80)
        print(
            f"EVENT EXPLANATION "
            f"{local_index + 1}"
        )
        print("=" * 80)

        print(
            f"Original dataset index : {original_index}"
        )

        print(
            f"Actual label           : "
            f"{'Attack' if actual_label == 1 else 'Normal'}"
        )

        print(
            f"Predicted label        : "
            f"{'Attack' if predicted_label == 1 else 'Normal'}"
        )

        print_top_sample_explanation(
            shap_values=shap_values,
            feature_names=feature_names,
            sample_index=local_index,
            top_n=10,
        )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BINARY SHAP ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated files:")

    print(
        f"1. {importance_path}"
    )

    print(
        f"2. {contribution_path}"
    )

    print(
        f"3. {OUTPUT_DIRECTORY / 'binary_shap_summary.png'}"
    )

    print(
        f"4. {OUTPUT_DIRECTORY / 'binary_shap_bar.png'}"
    )

    print("\nThese outputs will be used for the project's")
    print("Explainable AI / Feature Importance section.")


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_binary_shap_analysis()