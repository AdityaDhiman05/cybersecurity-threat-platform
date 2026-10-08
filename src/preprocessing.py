"""
UNSW-NB15 Data Preprocessing Pipeline
=====================================

This module prepares the UNSW-NB15 network traffic dataset
for machine learning.

Responsibilities:
    1. Load the official training and testing datasets.
    2. Separate features from target labels.
    3. Remove identifiers and target-related columns.
    4. Identify numerical and categorical features.
    5. Check for missing values.
    6. Impute missing values when required.
    7. Encode categorical network features.
    8. Scale numerical features.
    9. Build a reusable preprocessing pipeline.

Important:
    - The original CSV files are NEVER modified.
    - The "attack_cat" column is excluded from the binary
      classifier features to prevent target leakage.
    - The preprocessor is fitted ONLY on training data.
    - The testing data is transformed using the already-fitted
      training preprocessor.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


# ============================================================
# DATA LOADING
# ============================================================

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the official UNSW-NB15 training and testing datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            training_data:
                Official UNSW-NB15 training dataset.

            testing_data:
                Official UNSW-NB15 testing dataset.
    """

    print("Loading training dataset...")
    training_data = pd.read_csv(TRAINING_FILE)

    print("Loading testing dataset...")
    testing_data = pd.read_csv(TESTING_FILE)

    return training_data, testing_data


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

def separate_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate machine-learning features from the binary target.

    Target:
        label
            0 = Normal
            1 = Attack

    Columns removed from the model input:

        id
            Dataset record identifier.

        attack_cat
            Attack category / ground-truth information.

            This column must NOT be supplied to the binary
            classifier because it directly contains information
            about the attack type.

        label
            Binary target that the classifier must predict.

    Args:
        data:
            Input UNSW-NB15 dataframe.

    Returns:
        X:
            Feature dataframe.

        y:
            Binary target series.
    """

    columns_to_remove = [
        "id",
        "attack_cat",
        "label",
    ]

    X = data.drop(columns=columns_to_remove)

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

    Numerical features:
        Columns containing numeric values.

    Categorical features:
        Columns containing text/string values.

    A column-by-column dtype check is used instead of
    select_dtypes(include=["object"]) to avoid the pandas
    string-dtype compatibility warning.

    Args:
        X:
            Feature dataframe.

    Returns:
        numerical_features:
            List of numerical feature names.

        categorical_features:
            List of categorical feature names.
    """

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if pd.api.types.is_string_dtype(X[column])
    ]

    return numerical_features, categorical_features


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the complete preprocessing pipeline.

    Numerical features:
        1. Missing values → median
        2. Standardization → StandardScaler

    Categorical features:
        1. Missing values → most frequent value
        2. One-hot encoding
        3. Unknown categories → ignored

    Args:
        numerical_features:
            List of numerical feature names.

        categorical_features:
            List of categorical feature names.

    Returns:
        ColumnTransformer:
            Configured preprocessing transformer.
    """

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Combine numerical and categorical pipelines
    # --------------------------------------------------------

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
# MAIN PREPROCESSING WORKFLOW
# ============================================================

def main() -> None:
    """
    Execute the complete preprocessing workflow.

    The function:
        1. Loads training and testing data.
        2. Separates features and targets.
        3. Identifies numerical/categorical columns.
        4. Checks missing values.
        5. Builds the preprocessing pipeline.
        6. Fits the pipeline on training data.
        7. Transforms training and testing data.
        8. Displays the resulting shapes and distributions.

    The processed datasets are currently kept in memory only.
    They will be handled by the model-training stage later.
    """

    print("\n" + "=" * 70)
    print("CYBERSECURITY THREAT PLATFORM")
    print("UNSW-NB15 PREPROCESSING")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    training_data, testing_data = load_datasets()

    print("\nDatasets loaded successfully.")

    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X_train, y_train = separate_features_and_target(
        training_data
    )

    X_test, y_test = separate_features_and_target(
        testing_data
    )

    # ========================================================
    # IDENTIFY FEATURE TYPES
    # ========================================================

    numerical_features, categorical_features = (
        identify_feature_types(X_train)
    )

    # ========================================================
    # DISPLAY FEATURE INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE INFORMATION")
    print("=" * 70)

    print(
        f"\nTotal model input features before encoding: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print("\nNumerical features:")

    for feature in numerical_features:
        print(f"  - {feature}")

    print("\nCategorical features:")

    for feature in categorical_features:
        print(f"  - {feature}")

    # ========================================================
    # MISSING VALUE CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    missing_training = X_train.isnull().sum()

    missing_training = missing_training[
        missing_training > 0
    ]

    if missing_training.empty:

        print(
            "\nNo missing values found "
            "in training features."
        )

    else:

        print("\nMissing values found:")

        print(missing_training)

    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    print("\n" + "=" * 70)
    print("PREPROCESSOR")
    print("=" * 70)

    print(
        "\nPreprocessor created successfully."
    )

    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    print(
        "\nFitting preprocessor on training data..."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    print(
        "Training data preprocessing completed."
    )

    # ========================================================
    # TRANSFORM TEST DATA
    # ========================================================

    print(
        "\nTransforming testing data..."
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "Testing data preprocessing completed."
    )

    # ========================================================
    # DISPLAY PROCESSED DATA SHAPES
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSED DATA")
    print("=" * 70)

    print(
        f"\nOriginal training shape : "
        f"{X_train.shape}"
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"\nOriginal testing shape  : "
        f"{X_test.shape}"
    )

    print(
        f"Processed testing shape : "
        f"{X_test_processed.shape}"
    )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")

    print(
        y_train.value_counts()
    )

    print("\nTesting:")

    print(
        y_test.value_counts()
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING INSPECTION COMPLETED")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()