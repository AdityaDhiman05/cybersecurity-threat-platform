"""
================================================================================
ATTACK CLASSIFICATION - SHAP EXPLAINABILITY
================================================================================

Project:
    Cybersecurity Threat Platform

Purpose:
    Explain the predictions made by the known-attack multiclass classifier.

Model:
    Random Forest

Dataset:
    UNSW-NB15

This module:
    1. Loads the trained attack-category classifier.
    2. Loads the official UNSW-NB15 testing dataset.
    3. Selects attack records only.
    4. Transforms the selected features using the trained preprocessing pipeline.
    5. Calculates SHAP values using TreeExplainer.
    6. Produces global feature-importance analysis.
    7. Produces SHAP summary and bar plots.
    8. Produces event-level explanations for selected network events.
    9. Saves explainability outputs for the project.

Important:
    This module does NOT retrain the model.

Author:
    Cybersecurity Threat Platform Project
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Number of background records used by SHAP.
BACKGROUND_SAMPLES = 100

# Number of events explained individually.
EXPLANATION_SAMPLES = 10

# Number of top features displayed.
TOP_FEATURES = 20

# Number of contributors displayed for each individual event.
TOP_EVENT_CONTRIBUTORS = 10


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "xai"

TESTING_DATASET_PATH = DATA_DIRECTORY / "UNSW_NB15_testing-set.csv"

ATTACK_MODEL_PATH = MODEL_DIRECTORY / "attack_classifier.pkl"
ATTACK_PREPROCESSOR_PATH = MODEL_DIRECTORY / "attack_preprocessor.pkl"


# =============================================================================
# ATTACK CLASSIFIER FEATURES
# =============================================================================

SELECTED_NUMERICAL_FEATURES = [
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
    SELECTED_NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str) -> None:
    """
    Print a formatted section heading.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def ensure_output_directory() -> None:
    """
    Create the XAI output directory if it does not already exist.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_attack_model():
    """
    Load the trained attack-category classifier and its preprocessor.

    The attack classifier was trained using the selected 15 numerical
    features and 3 categorical features.
    """

    print_section(
        "LOADING ATTACK CLASSIFIER FOR SHAP EXPLAINABILITY"
    )

    if not ATTACK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Attack classifier not found:\n{ATTACK_MODEL_PATH}"
        )

    attack_model = joblib.load(ATTACK_MODEL_PATH)

    print("Attack classifier loaded successfully.")
    print(f"Model type: {type(attack_model).__name__}")

    attack_preprocessor = None

    if ATTACK_PREPROCESSOR_PATH.exists():
        attack_preprocessor = joblib.load(
            ATTACK_PREPROCESSOR_PATH
        )

        print("Attack preprocessor loaded successfully.")
        print(
            f"Preprocessor type: "
            f"{type(attack_preprocessor).__name__}"
        )
    else:
        print(
            "Separate attack preprocessor not found."
        )

    return attack_model, attack_preprocessor


# =============================================================================
# EXTRACT MODEL COMPONENTS
# =============================================================================

def extract_preprocessor_and_classifier(
    attack_model,
    attack_preprocessor,
):
    """
    Extract the preprocessing component and Random Forest classifier.

    The function supports both:
        1. A Pipeline containing preprocessing + classifier.
        2. A standalone RandomForestClassifier with a separate
           preprocessor.

    Returns
    -------
    tuple
        (preprocessor, classifier)
    """

    # -------------------------------------------------------------------------
    # CASE 1: MODEL IS A SKLEARN PIPELINE
    # -------------------------------------------------------------------------

    if isinstance(attack_model, Pipeline):

        print()
        print("Attack model is a sklearn Pipeline.")

        pipeline_steps = attack_model.named_steps

        print(
            "Pipeline steps:",
            list(pipeline_steps.keys())
        )

        preprocessor = None
        classifier = None

        for step_name, step_object in pipeline_steps.items():

            object_name = type(step_object).__name__

            # Detect ColumnTransformer / preprocessing component.
            if object_name == "ColumnTransformer":
                preprocessor = step_object

            # Detect Random Forest classifier.
            if object_name == "RandomForestClassifier":
                classifier = step_object

        # Sometimes the preprocessor may use a different object type.
        if preprocessor is None:
            for step_name, step_object in pipeline_steps.items():

                if hasattr(
                    step_object,
                    "get_feature_names_out",
                ):
                    preprocessor = step_object
                    break

        # Sometimes the classifier can be identified through predict_proba.
        if classifier is None:
            for step_name, step_object in pipeline_steps.items():

                if (
                    hasattr(step_object, "predict")
                    and hasattr(step_object, "predict_proba")
                ):
                    classifier = step_object

        if preprocessor is None:
            raise RuntimeError(
                "Could not identify the preprocessing component "
                "inside attack_classifier.pkl."
            )

        if classifier is None:
            raise RuntimeError(
                "Could not identify the Random Forest classifier "
                "inside attack_classifier.pkl."
            )

        return preprocessor, classifier

    # -------------------------------------------------------------------------
    # CASE 2: MODEL IS A STANDALONE CLASSIFIER
    # -------------------------------------------------------------------------

    print()
    print("Attack model is a standalone classifier.")

    if attack_preprocessor is None:
        raise RuntimeError(
            "Attack classifier is standalone, but no separate "
            "attack preprocessor was found."
        )

    return attack_preprocessor, attack_model


# =============================================================================
# LOAD TESTING DATASET
# =============================================================================

def load_testing_dataset() -> pd.DataFrame:
    """
    Load the official UNSW-NB15 testing dataset.
    """

    print_section(
        "LOADING UNSW-NB15 TESTING DATASET"
    )

    if not TESTING_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n"
            f"{TESTING_DATASET_PATH}"
        )

    dataset = pd.read_csv(
        TESTING_DATASET_PATH
    )

    print(
        f"Dataset shape: {dataset.shape}"
    )

    return dataset


# =============================================================================
# PREPARE ATTACK DATA
# =============================================================================

def prepare_attack_data(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select only attack records because the attack classifier is a
    multiclass model trained on known attack categories.
    """

    print_section(
        "PREPARING KNOWN ATTACK RECORDS"
    )

    if "label" not in dataset.columns:
        raise ValueError(
            "Dataset does not contain the 'label' column."
        )

    if "attack_cat" not in dataset.columns:
        raise ValueError(
            "Dataset does not contain the 'attack_cat' column."
        )

    attack_dataset = dataset[
        dataset["label"] == 1
    ].copy()

    print(
        f"Attack records available: "
        f"{len(attack_dataset):,}"
    )

    print()
    print("Attack category distribution:")

    print(
        attack_dataset["attack_cat"]
        .value_counts()
        .to_string()
    )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in attack_dataset.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing:\n"
            + "\n".join(missing_features)
        )

    return attack_dataset


# =============================================================================
# TRANSFORM FEATURES
# =============================================================================

def transform_attack_features(
    attack_dataset: pd.DataFrame,
    preprocessor,
):
    """
    Transform attack records using the exact preprocessing object used
    by the trained attack classifier.
    """

    print_section(
        "TRANSFORMING ATTACK CLASSIFICATION FEATURES"
    )

    X_attack = attack_dataset[
        MODEL_FEATURES
    ].copy()

    transformed_data = preprocessor.transform(
        X_attack
    )

    print(
        f"Original feature count : "
        f"{X_attack.shape[1]}"
    )

    print(
        f"Transformed shape      : "
        f"{transformed_data.shape}"
    )

    # -------------------------------------------------------------------------
    # GET TRANSFORMED FEATURE NAMES
    # -------------------------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out(
                MODEL_FEATURES
            )
        )

    except Exception:

        feature_names = np.array(
            [
                f"feature_{index}"
                for index in range(
                    transformed_data.shape[1]
                )
            ]
        )

    print(
        f"Number of feature names : "
        f"{len(feature_names)}"
    )

    return (
        X_attack,
        transformed_data,
        np.asarray(feature_names),
    )


# =============================================================================
# CONVERT SPARSE MATRIX
# =============================================================================

def convert_to_dense(matrix):
    """
    Convert a sparse matrix to a dense NumPy array if necessary.

    SHAP TreeExplainer works most reliably with dense arrays for the
    Random Forest classifier.
    """

    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


# =============================================================================
# CREATE SHAP EXPLAINER
# =============================================================================

def create_shap_explainer(
    classifier,
):
    """
    Create a SHAP TreeExplainer for the Random Forest classifier.
    """

    print_section(
        "CREATING SHAP TREE EXPLAINER"
    )

    print(
        f"Classifier type: "
        f"{type(classifier).__name__}"
    )

    if type(classifier).__name__ != "RandomForestClassifier":
        warnings.warn(
            "The classifier is not identified as a "
            "RandomForestClassifier. TreeExplainer may still work."
        )

    explainer = shap.TreeExplainer(
        classifier
    )

    print(
        "TreeExplainer created successfully."
    )

    return explainer


# =============================================================================
# HANDLE SHAP OUTPUT
# =============================================================================

def normalize_shap_output(
    shap_values,
):
    """
    Normalize SHAP output across different SHAP library versions.

    Multiclass Random Forest SHAP values may be returned as:

        list[class] -> (samples, features)

    or:

        (samples, features, classes)

    Returns
    -------
    numpy.ndarray
        Shape:
            (classes, samples, features)
    """

    # -------------------------------------------------------------------------
    # SHAP RETURNS A LIST
    # -------------------------------------------------------------------------

    if isinstance(shap_values, list):

        arrays = [
            np.asarray(values)
            for values in shap_values
        ]

        return np.stack(
            arrays,
            axis=0,
        )

    # -------------------------------------------------------------------------
    # SHAP RETURNS A NUMPY ARRAY
    # -------------------------------------------------------------------------

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:

        # Expected modern format:
        # samples x features x classes

        return np.transpose(
            values,
            (2, 0, 1),
        )

    if values.ndim == 2:

        # Binary/single-output fallback.
        return values[
            np.newaxis,
            :,
            :,
        ]

    raise RuntimeError(
        "Unexpected SHAP value shape: "
        f"{values.shape}"
    )


# =============================================================================
# CALCULATE SHAP VALUES
# =============================================================================

def calculate_shap_values(
    explainer,
    transformed_data,
    classifier,
):
    """
    Calculate SHAP values for a limited explanation sample.

    A small explanation sample is intentionally used so the explainability
    stage remains practical on a local machine.
    """

    print_section(
        "CALCULATING MULTICLASS SHAP VALUES"
    )

    dense_data = convert_to_dense(
        transformed_data
    )

    # -------------------------------------------------------------------------
    # SELECT BACKGROUND DATA
    # -------------------------------------------------------------------------

    background_count = min(
        BACKGROUND_SAMPLES,
        dense_data.shape[0],
    )

    explanation_count = min(
        EXPLANATION_SAMPLES,
        dense_data.shape[0],
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    background_indices = rng.choice(
        dense_data.shape[0],
        size=background_count,
        replace=False,
    )

    explanation_indices = rng.choice(
        dense_data.shape[0],
        size=explanation_count,
        replace=False,
    )

    background_data = dense_data[
        background_indices
    ]

    explanation_data = dense_data[
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
        f"Feature count       : "
        f"{explanation_data.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # CALCULATE SHAP VALUES
    # -------------------------------------------------------------------------

    print()
    print("Calculating SHAP values...")

    shap_values = explainer(
        explanation_data
    )

    # New SHAP Explanation object.
    if hasattr(
        shap_values,
        "values",
    ):

        shap_values = shap_values.values

    normalized_values = normalize_shap_output(
        shap_values
    )

    print(
        "Normalized SHAP matrix shape: "
        f"{normalized_values.shape}"
    )

    return (
        normalized_values,
        explanation_data,
        explanation_indices,
    )


# =============================================================================
# GLOBAL FEATURE IMPORTANCE
# =============================================================================

def calculate_global_importance(
    shap_values,
    feature_names,
):
    """
    Calculate global mean absolute SHAP importance across:
        - explanation samples
        - attack categories
    """

    print_section(
        "CALCULATING GLOBAL SHAP FEATURE IMPORTANCE"
    )

    # Shape:
    # classes x samples x features
    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=(0, 1),
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance.insert(
        0,
        "rank",
        np.arange(
            1,
            len(feature_importance) + 1,
        ),
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "attack_feature_importance.csv"
    )

    feature_importance.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "TOP GLOBAL SHAP FEATURES"
    )

    print(
        feature_importance
        .head(TOP_FEATURES)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Global feature importance saved:"
    )

    print(output_path)

    return feature_importance


# =============================================================================
# CREATE BAR PLOT
# =============================================================================

def create_bar_plot(
    feature_importance: pd.DataFrame,
):
    """
    Create a global SHAP bar chart.
    """

    print_section(
        "CREATING SHAP FEATURE IMPORTANCE BAR PLOT"
    )

    top_features = (
        feature_importance
        .head(TOP_FEATURES)
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(12, 9)
    )

    plt.barh(
        top_features["feature"],
        top_features["mean_absolute_shap"],
    )

    plt.xlabel(
        "mean(|SHAP value|)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Feature Importance — "
        "Known Attack Classification"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "attack_shap_bar.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"SHAP bar plot saved:\n{output_path}"
    )


# =============================================================================
# CREATE SHAP SUMMARY PLOT
# =============================================================================

def create_summary_plot(
    shap_values,
    explanation_data,
    feature_names,
):
    """
    Create a multiclass SHAP summary plot.

    The plot aggregates the SHAP contributions across attack categories.
    """

    print_section(
        "CREATING SHAP SUMMARY PLOT"
    )

    # -------------------------------------------------------------------------
    # AGGREGATE ACROSS CLASSES
    # -------------------------------------------------------------------------

    # classes x samples x features
    #
    # Convert to:
    # samples x features
    #
    # by taking the mean absolute contribution across classes.

    aggregated_values = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    plt.figure(
        figsize=(12, 10)
    )

    shap.summary_plot(
        aggregated_values,
        explanation_data,
        feature_names=feature_names,
        max_display=TOP_FEATURES,
        show=False,
    )

    plt.title(
        "SHAP Summary — "
        "Known Attack Classification"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "attack_shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"SHAP summary plot saved:\n{output_path}"
    )


# =============================================================================
# SAVE RAW SHAP VALUES
# =============================================================================

def save_shap_values(
    shap_values,
    feature_names,
):
    """
    Save flattened SHAP values to CSV.

    Each row represents one sample/class combination.
    """

    print_section(
        "SAVING RAW SHAP VALUES"
    )

    rows = []

    number_of_classes = (
        shap_values.shape[0]
    )

    number_of_samples = (
        shap_values.shape[1]
    )

    for class_index in range(
        number_of_classes
    ):

        for sample_index in range(
            number_of_samples
        ):

            for feature_index, feature_name in enumerate(
                feature_names
            ):

                rows.append(
                    {
                        "class_index": class_index,
                        "sample_index": sample_index,
                        "feature": feature_name,
                        "shap_value": shap_values[
                            class_index,
                            sample_index,
                            feature_index,
                        ],
                    }
                )

    shap_dataframe = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "attack_shap_values.csv"
    )

    shap_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Raw SHAP values saved:\n{output_path}"
    )


# =============================================================================
# EVENT-LEVEL EXPLANATION
# =============================================================================

def explain_individual_events(
    attack_dataset,
    transformed_data,
    explanation_indices,
    shap_values,
    feature_names,
    classifier,
):
    """
    Produce human-readable explanations for selected attack events.

    For each event:
        1. Determine predicted attack category.
        2. Determine actual category.
        3. Select SHAP values belonging to predicted category.
        4. Rank the strongest contributors.
    """

    print_section(
        "EVENT-LEVEL ATTACK CATEGORY EXPLANATIONS"
    )

    dense_data = convert_to_dense(
        transformed_data
    )

    predictions = classifier.predict(
        dense_data[
            explanation_indices
        ]
    )

    # -------------------------------------------------------------------------
    # CLASS LABELS
    # -------------------------------------------------------------------------

    if hasattr(
        classifier,
        "classes_",
    ):

        class_labels = list(
            classifier.classes_
        )

    else:

        class_labels = [
            str(index)
            for index in range(
                shap_values.shape[0]
            )
        ]

    for event_number, (
        dataset_index,
        prediction,
    ) in enumerate(
        zip(
            explanation_indices,
            predictions,
        ),
        start=1,
    ):

        actual_category = (
            attack_dataset.iloc[
                dataset_index
            ]["attack_cat"]
        )

        predicted_category = prediction

        # -------------------------------------------------------------
        # FIND PREDICTED CLASS INDEX
        # -------------------------------------------------------------

        try:

            predicted_class_index = (
                class_labels.index(
                    predicted_category
                )
            )

        except ValueError:

            predicted_class_index = 0

        event_shap_values = (
            shap_values[
                predicted_class_index,
                event_number - 1,
                :,
            ]
        )

        # -------------------------------------------------------------
        # RANK CONTRIBUTIONS
        # -------------------------------------------------------------

        ranked_indices = np.argsort(
            np.abs(
                event_shap_values
            )
        )[::-1]

        print()
        print("=" * 80)
        print(
            f"EVENT EXPLANATION {event_number}"
        )
        print("=" * 80)

        print(
            f"Original dataset index : "
            f"{dataset_index}"
        )

        print(
            f"Actual attack category : "
            f"{actual_category}"
        )

        print(
            f"Predicted category     : "
            f"{predicted_category}"
        )

        if (
            str(actual_category)
            == str(predicted_category)
        ):

            print(
                "Classification result  : CORRECT"
            )

        else:

            print(
                "Classification result  : INCORRECT"
            )

        print()
        print(
            "TOP SHAP CONTRIBUTORS "
            "FOR PREDICTED CATEGORY"
        )

        for rank, feature_index in enumerate(
            ranked_indices[
                :TOP_EVENT_CONTRIBUTORS
            ],
            start=1,
        ):

            feature_name = (
                feature_names[
                    feature_index
                ]
            )

            shap_value = (
                event_shap_values[
                    feature_index
                ]
            )

            if shap_value > 0:

                direction = (
                    f"↑ {predicted_category} probability"
                )

            else:

                direction = (
                    f"↓ {predicted_category} probability"
                )

            print(
                f"{rank:02d}. "
                f"{feature_name:<45} "
                f"{shap_value:+.6f}  "
                f"{direction}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main execution function.
    """

    print()
    print("=" * 80)
    print(
        "KNOWN ATTACK CLASSIFICATION "
        "SHAP EXPLAINABILITY"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PREPARE OUTPUT DIRECTORY
    # -------------------------------------------------------------------------

    ensure_output_directory()

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    attack_model, attack_preprocessor = (
        load_attack_model()
    )

    # -------------------------------------------------------------------------
    # EXTRACT PREPROCESSOR + CLASSIFIER
    # -------------------------------------------------------------------------

    preprocessor, classifier = (
        extract_preprocessor_and_classifier(
            attack_model,
            attack_preprocessor,
        )
    )

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    dataset = load_testing_dataset()

    # -------------------------------------------------------------------------
    # SELECT ATTACK RECORDS
    # -------------------------------------------------------------------------

    attack_dataset = prepare_attack_data(
        dataset
    )

    # -------------------------------------------------------------------------
    # TRANSFORM FEATURES
    # -------------------------------------------------------------------------

    (
        X_attack,
        transformed_data,
        feature_names,
    ) = transform_attack_features(
        attack_dataset,
        preprocessor,
    )

    # -------------------------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # -------------------------------------------------------------------------

    explainer = create_shap_explainer(
        classifier
    )

    # -------------------------------------------------------------------------
    # CALCULATE SHAP
    # -------------------------------------------------------------------------

    (
        shap_values,
        explanation_data,
        explanation_indices,
    ) = calculate_shap_values(
        explainer,
        transformed_data,
        classifier,
    )

    # -------------------------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -------------------------------------------------------------------------

    feature_importance = (
        calculate_global_importance(
            shap_values,
            feature_names,
        )
    )

    # -------------------------------------------------------------------------
    # BAR PLOT
    # -------------------------------------------------------------------------

    create_bar_plot(
        feature_importance
    )

    # -------------------------------------------------------------------------
    # SUMMARY PLOT
    # -------------------------------------------------------------------------

    create_summary_plot(
        shap_values,
        explanation_data,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # SAVE RAW SHAP VALUES
    # -------------------------------------------------------------------------

    save_shap_values(
        shap_values,
        feature_names,
    )

    # -------------------------------------------------------------------------
    # INDIVIDUAL EVENT EXPLANATIONS
    # -------------------------------------------------------------------------

    explain_individual_events(
        attack_dataset,
        transformed_data,
        explanation_indices,
        shap_values,
        feature_names,
        classifier,
    )

    # -------------------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ATTACK CLASSIFICATION SHAP "
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)

    print()
    print("Generated files:")

    print(
        f"1. {OUTPUT_DIRECTORY / 'attack_feature_importance.csv'}"
    )

    print(
        f"2. {OUTPUT_DIRECTORY / 'attack_shap_values.csv'}"
    )

    print(
        f"3. {OUTPUT_DIRECTORY / 'attack_shap_summary.png'}"
    )

    print(
        f"4. {OUTPUT_DIRECTORY / 'attack_shap_bar.png'}"
    )

    print()
    print(
        "These outputs will be used for the project's"
    )

    print(
        "Explainable AI / Attack Classification analysis."
    )

    print()


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()