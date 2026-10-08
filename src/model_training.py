"""
UNSW-NB15 Binary Threat Detection Model
========================================

This module trains a supervised machine-learning model
to distinguish between:

    0 → Normal network traffic
    1 → Attack network traffic

Model:
    Random Forest Classifier

Responsibilities:
    1. Load the official UNSW-NB15 training and testing datasets.
    2. Separate features and binary target.
    3. Identify numerical and categorical features.
    4. Build the preprocessing pipeline.
    5. Preprocess training and testing data.
    6. Train a Random Forest classifier.
    7. Evaluate the classifier.
    8. Display classification metrics.
    9. Display the confusion matrix.
    10. Save the trained model.
    11. Save the preprocessing pipeline.

Important:
    - "id" is excluded from model features.
    - "attack_cat" is excluded to prevent target leakage.
    - "label" is the binary target.
    - The preprocessing pipeline is fitted ONLY on training data.
    - The original CSV files are NEVER modified.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_training-set.csv"
)

TESTING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)

MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

BINARY_MODEL_FILE = (
    MODELS_DIRECTORY
    / "binary_classifier.pkl"
)

PREPROCESSOR_FILE = (
    MODELS_DIRECTORY
    / "binary_preprocessor.pkl"
)


# ============================================================
# RANDOM FOREST CONFIGURATION
# ============================================================

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the official UNSW-NB15 training and testing datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            training_data:
                Training dataset.

            testing_data:
                Testing dataset.
    """

    print("Loading training dataset...")

    training_data = pd.read_csv(
        TRAINING_FILE
    )

    print("Loading testing dataset...")

    testing_data = pd.read_csv(
        TESTING_FILE
    )

    return training_data, testing_data


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

def separate_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the binary target.

    Removed columns:

        id:
            Dataset record identifier.

        attack_cat:
            Attack category / ground-truth information.
            It is removed to prevent target leakage.

        label:
            Binary target.

    Target:

        0 = Normal
        1 = Attack

    Args:
        data:
            UNSW-NB15 dataframe.

    Returns:
        X:
            Model input features.

        y:
            Binary target.
    """

    columns_to_remove = [
        "id",
        "attack_cat",
        "label",
    ]

    X = data.drop(
        columns=columns_to_remove
    )

    y = data["label"]

    return X, y


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def identify_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical features.

    Returns:
        numerical_features:
            Numerical feature names.

        categorical_features:
            String/categorical feature names.
    """

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if pd.api.types.is_string_dtype(
            X[column]
        )
    ]

    return (
        numerical_features,
        categorical_features,
    )


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numerical features:
        1. Median imputation.
        2. Standard scaling.

    Categorical features:
        1. Most-frequent imputation.
        2. One-hot encoding.
        3. Unknown categories ignored.

    Returns:
        ColumnTransformer:
            Complete preprocessing pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODEL CREATION
# ============================================================

def build_model() -> RandomForestClassifier:
    """
    Create the Random Forest binary classifier.

    Returns:
        RandomForestClassifier:
            Configured Random Forest model.
    """

    model = RandomForestClassifier(
        **RANDOM_FOREST_PARAMETERS
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model: RandomForestClassifier,
    X_test_processed,
    y_test: pd.Series,
) -> None:
    """
    Evaluate the trained Random Forest classifier.

    Evaluation metrics:

        Accuracy
        Classification report
        Confusion matrix
        ROC-AUC

    Special attention is given to the ATTACK class because
    missing an actual attack is more serious than incorrectly
    flagging some normal traffic.

    Args:
        model:
            Trained Random Forest model.

        X_test_processed:
            Preprocessed testing features.

        y_test:
            Actual testing labels.
    """

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test_processed
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Attack",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "\n                Predicted"
    )

    print(
        "                 Normal  Attack"
    )

    print(
        f"Actual Normal   {matrix[0][0]:7d} "
        f"{matrix[0][1]:7d}"
    )

    print(
        f"Actual Attack   {matrix[1][0]:7d} "
        f"{matrix[1][1]:7d}"
    )

    # --------------------------------------------------------
    # Confusion matrix interpretation
    # --------------------------------------------------------

    true_negative = matrix[0][0]
    false_positive = matrix[0][1]
    false_negative = matrix[1][0]
    true_positive = matrix[1][1]

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX INTERPRETATION")
    print("-" * 70)

    print(
        f"\nTrue Negatives : {true_negative}"
    )

    print(
        f"False Positives: {false_positive}"
    )

    print(
        f"False Negatives: {false_negative}"
    )

    print(
        f"True Positives : {true_positive}"
    )

    print(
        "\nFalse Negatives are especially important "
        "for cybersecurity because they represent "
        "actual attacks classified as normal."
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "-" * 70)
    print("ROC-AUC")
    print("-" * 70)

    print(
        f"\nROC-AUC Score: {roc_auc:.4f}"
    )


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

def save_model_artifacts(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
) -> None:
    """
    Save the trained model and preprocessing pipeline.

    Files:

        models/binary_classifier.pkl
        models/binary_preprocessor.pkl

    Args:
        model:
            Trained Random Forest classifier.

        preprocessor:
            Fitted preprocessing pipeline.
    """

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 70)

    joblib.dump(
        model,
        BINARY_MODEL_FILE,
    )

    print(
        f"\nBinary classifier saved to:"
        f"\n{BINARY_MODEL_FILE}"
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
    )

    print(
        f"\nPreprocessor saved to:"
        f"\n{PREPROCESSOR_FILE}"
    )


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================

def main() -> None:
    """
    Execute the complete binary classification workflow.
    """

    print("\n" + "=" * 70)
    print("CYBERSECURITY THREAT PLATFORM")
    print("UNSW-NB15 BINARY THREAT DETECTION")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    training_data, testing_data = (
        load_datasets()
    )

    print("\nDatasets loaded successfully.")

    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X_train, y_train = (
        separate_features_and_target(
            training_data
        )
    )

    X_test, y_test = (
        separate_features_and_target(
            testing_data
        )
    )

    # ========================================================
    # FEATURE TYPES
    # ========================================================

    (
        numerical_features,
        categorical_features,
    ) = identify_feature_types(
        X_train
    )

    print("\n" + "=" * 70)
    print("FEATURE CONFIGURATION")
    print("=" * 70)

    print(
        f"\nNumerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    # ========================================================
    # PREPROCESS TRAINING DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING TRAINING DATA")
    print("=" * 70)

    print(
        "\nFitting preprocessing pipeline "
        "on training data..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Training preprocessing completed."
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    # ========================================================
    # PREPROCESS TESTING DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING TESTING DATA")
    print("=" * 70)

    print(
        "\nTransforming testing data "
        "using the training preprocessor..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "Testing preprocessing completed."
    )

    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("RANDOM FOREST CONFIGURATION")
    print("=" * 70)

    print("\nModel parameters:")

    for parameter, value in (
        RANDOM_FOREST_PARAMETERS.items()
    ):
        print(
            f"  {parameter}: {value}"
        )

    model = build_model()

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    print(
        "\nTraining Random Forest..."
    )

    print(
        "This may take some time "
        "because the dataset contains "
        "175,341 training records."
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    print(
        "\nRandom Forest training completed."
    )

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    evaluate_model(
        model,
        X_test_processed,
        y_test,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_model_artifacts(
        model,
        preprocessor,
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 70)
    print("BINARY THREAT DETECTION TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nThe trained model is ready for "
        "the next stage of the platform."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()