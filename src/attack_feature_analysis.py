"""
===============================================================================
UNSW-NB15 MULTICLASS ATTACK CLASSIFIER
===============================================================================

Purpose
-------
Train the final supervised multiclass classifier for known cyberattack
categories using the UNSW-NB15 dataset.

This model is responsible for answering:

    "If this network event is an attack, what type of known attack is it?"

Architecture
------------
    UNSW-NB15
         |
         v
    Attack Records Only
         |
         v
    Selected Features
         |
         v
    Preprocessing
         |
         v
    Random Forest
         |
         v
    Attack Category

Selected Numerical Features
---------------------------
The feature-selection experiment identified the following 15 numerical
features as the best configuration:

    1.  sbytes
    2.  smean
    3.  ct_dst_src_ltm
    4.  ct_srv_dst
    5.  ct_srv_src
    6.  sload
    7.  ct_src_dport_ltm
    8.  ct_dst_sport_ltm
    9.  dbytes
    10. dmean
    11. ct_flw_http_mthd
    12. trans_depth
    13. sttl
    14. dur
    15. ct_src_ltm

Categorical Features
--------------------
The following categorical network features are retained:

    - proto
    - service
    - state

Evaluation
----------
The official UNSW-NB15 test set is used only for the final evaluation.

Metrics:
    - Accuracy
    - Balanced Accuracy
    - Macro F1
    - Classification Report
    - Confusion Matrix
    - Per-class Recall

Output
------
The final trained classifier and preprocessing pipeline are saved to:

    models/attack_classifier.pkl
    models/attack_preprocessor.pkl

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_training-set.csv"
)

TESTING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)

MODELS_DIRECTORY = PROJECT_ROOT / "models"

ATTACK_CLASSIFIER_PATH = (
    MODELS_DIRECTORY / "attack_classifier.pkl"
)

ATTACK_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY / "attack_preprocessor.pkl"
)


# =============================================================================
# SELECTED FEATURES
# =============================================================================

# These 15 numerical features were selected from the controlled
# feature-selection experiment based on validation performance.

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


SELECTED_FEATURES = (
    SELECTED_NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# RANDOM FOREST CONFIGURATION
# =============================================================================

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load an UNSW-NB15 CSV dataset.

    Parameters
    ----------
    file_path : Path
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_attack_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract attack records and prepare features/target.

    Only records where label == 1 are retained because this model
    performs multiclass classification among known attack categories.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw UNSW-NB15 dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature dataframe and attack-category target.
    """

    attack_data = dataframe[
        dataframe["label"] == 1
    ].copy()

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in attack_data.columns
    ]

    if missing_features:
        raise ValueError(
            "The following required features are missing:\n"
            + "\n".join(missing_features)
        )

    X = attack_data[SELECTED_FEATURES].copy()

    y = attack_data["attack_cat"].copy()

    return X, y


# =============================================================================
# PREPROCESSOR
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Standard scaling

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding
        - Unknown categories ignored

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                SELECTED_NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# =============================================================================
# MODEL CREATION
# =============================================================================

def build_classifier() -> RandomForestClassifier:
    """
    Create the final Random Forest attack classifier.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model.
    """

    return RandomForestClassifier(
        **RANDOM_FOREST_PARAMETERS
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Evaluate the final model on the official UNSW-NB15 test set.

    Parameters
    ----------
    model : Pipeline
        Trained preprocessing + classifier pipeline.

    X_test : pd.DataFrame
        Official attack test features.

    y_test : pd.Series
        Official attack test labels.
    """

    print("\n")
    print("=" * 80)
    print("FINAL OFFICIAL TEST EVALUATION")
    print("=" * 80)

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print("\nFinal test performance:")
    print(
        f"Accuracy          : {accuracy:.4f}"
    )
    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )
    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # -------------------------------------------------------------------------
    # Per-Class Recall
    # -------------------------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print("=" * 80)
    print("PER-CLASS RECALL")
    print("=" * 80)

    for attack_category in sorted(
        y_test.unique()
    ):
        recall = report.get(
            attack_category,
            {},
        ).get(
            "recall",
            0.0,
        )

        print(
            f"{attack_category:<16} "
            f"{recall:.4f}"
        )

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    class_labels = sorted(
        y_test.unique()
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels,
    )

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=class_labels,
        columns=class_labels,
    )

    print("\nRows = Actual")
    print("Columns = Predicted\n")
    print(confusion_dataframe)


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute the complete final attack-classification pipeline.
    """

    print("=" * 80)
    print("UNSW-NB15 FINAL MULTICLASS ATTACK CLASSIFIER")
    print("=" * 80)

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    training_data = load_dataset(
        TRAINING_DATA_PATH
    )

    testing_data = load_dataset(
        TESTING_DATA_PATH
    )

    print(
        f"\nTraining dataset shape: "
        f"{training_data.shape}"
    )

    print(
        f"Testing dataset shape : "
        f"{testing_data.shape}"
    )

    # =========================================================================
    # PREPARE ATTACK DATA
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("PREPARING ATTACK DATA")
    print("=" * 80)

    X_train, y_train = prepare_attack_data(
        training_data
    )

    X_test, y_test = prepare_attack_data(
        testing_data
    )

    print(
        f"\nAttack training records: "
        f"{len(X_train)}"
    )

    print(
        f"Attack testing records : "
        f"{len(X_test)}"
    )

    # =========================================================================
    # DISPLAY SELECTED FEATURES
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("SELECTED FEATURES")
    print("=" * 80)

    print(
        f"\nNumerical features: "
        f"{len(SELECTED_NUMERICAL_FEATURES)}"
    )

    for feature in SELECTED_NUMERICAL_FEATURES:
        print(f"  - {feature}")

    print(
        f"\nCategorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    for feature in CATEGORICAL_FEATURES:
        print(f"  - {feature}")

    print(
        f"\nTotal raw model features: "
        f"{len(SELECTED_FEATURES)}"
    )

    # =========================================================================
    # DISPLAY TARGET DISTRIBUTION
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("ATTACK CATEGORY DISTRIBUTION")
    print("=" * 80)

    print(
        y_train.value_counts().sort_index()
    )

    # =========================================================================
    # BUILD PREPROCESSOR
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("BUILDING PREPROCESSOR")
    print("=" * 80)

    preprocessor = build_preprocessor()

    # =========================================================================
    # BUILD CLASSIFIER
    # =========================================================================

    classifier = build_classifier()

    print("\nRandom Forest configuration:")

    for parameter, value in (
        RANDOM_FOREST_PARAMETERS.items()
    ):
        print(
            f"  {parameter}: {value}"
        )

    # =========================================================================
    # BUILD COMPLETE PIPELINE
    # =========================================================================

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    # =========================================================================
    # TRAIN FINAL MODEL
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)

    print(
        "\nTraining on ALL attack training records..."
    )

    model_pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed successfully."
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete pipeline as the main classifier.
    joblib.dump(
        model_pipeline,
        ATTACK_CLASSIFIER_PATH,
    )

    # Save the fitted preprocessor separately as well.
    fitted_preprocessor = (
        model_pipeline.named_steps[
            "preprocessor"
        ]
    )

    joblib.dump(
        fitted_preprocessor,
        ATTACK_PREPROCESSOR_PATH,
    )

    print(
        f"\nClassifier saved to:"
        f"\n{ATTACK_CLASSIFIER_PATH}"
    )

    print(
        f"\nPreprocessor saved to:"
        f"\n{ATTACK_PREPROCESSOR_PATH}"
    )

    # =========================================================================
    # FINAL OFFICIAL TEST
    # =========================================================================

    evaluate_model(
        model_pipeline,
        X_test,
        y_test,
    )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("FINAL ATTACK CLASSIFIER COMPLETED")
    print("=" * 80)

    print(
        "\nThe Top-15 feature configuration has now been"
        " trained on the complete attack training set"
        " and evaluated on the official UNSW-NB15 test set."
    )

    print(
        "\nThe attack classifier is now ready for integration"
        " into the unified cybersecurity prediction pipeline."
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()