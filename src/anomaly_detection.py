"""
===============================================================================
UNSW-NB15 ANOMALY / UNKNOWN-THREAT DETECTION
===============================================================================

Purpose
-------
Train an Isolation Forest anomaly detector using ONLY normal network traffic.

The objective is different from the supervised classifiers:

    Binary Classifier
        -> Normal vs Attack

    Attack Classifier
        -> Known Attack Category

    Anomaly Detector
        -> Does this network behavior look unusual compared with learned
           normal traffic?

Important Research Limitation
-----------------------------
UNSW-NB15 does not provide a true future zero-day test set.

Therefore, this module does NOT claim to detect actual zero-day attacks.

Instead, it evaluates whether the anomaly detector can identify malicious
network behavior as anomalous when the detector has learned only normal
traffic.

Pipeline
--------
    Normal Training Traffic
            |
            v
      Selected Features
            |
            v
       Preprocessing
            |
            v
      Isolation Forest
            |
            v
     Anomaly / Normal

Output
------
    models/anomaly_detector.pkl
    models/anomaly_preprocessor.pkl

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# PROJECT PATHS
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

ANOMALY_DETECTOR_PATH = (
    MODELS_DIRECTORY
    / "anomaly_detector.pkl"
)

ANOMALY_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY
    / "anomaly_preprocessor.pkl"
)


# =============================================================================
# SELECTED FEATURES
# =============================================================================
#
# These are the 15 numerical features selected during our controlled
# feature-selection experiment.
#

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
# ISOLATION FOREST CONFIGURATION
# =============================================================================

ISOLATION_FOREST_PARAMETERS = {
    "n_estimators": 300,
    "contamination": "auto",
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": -1,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load an UNSW-NB15 dataset.

    Parameters
    ----------
    file_path : Path
        Dataset path.

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

def prepare_normal_training_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract only normal records from the training dataset.

    The anomaly detector must learn the normal traffic profile.

    Parameters
    ----------
    dataframe : pd.DataFrame
        UNSW-NB15 training dataset.

    Returns
    -------
    pd.DataFrame
        Normal traffic feature dataframe.
    """

    normal_data = dataframe[
        dataframe["label"] == 0
    ].copy()

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in normal_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )

    return normal_data[SELECTED_FEATURES].copy()


# =============================================================================
# TEST DATA PREPARATION
# =============================================================================

def prepare_test_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the official test set for anomaly evaluation.

    For evaluation only, the real UNSW-NB15 labels are retained.

    label = 0
        Normal

    label = 1
        Attack

    The anomaly detector itself never sees these labels during training.

    Parameters
    ----------
    dataframe : pd.DataFrame
        UNSW-NB15 testing dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Test features and binary labels.
    """

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing_features)
        )

    X_test = dataframe[
        SELECTED_FEATURES
    ].copy()

    y_test = dataframe[
        "label"
    ].copy()

    return X_test, y_test


# =============================================================================
# PREPROCESSOR
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        - Median imputation
        - Standard scaling

    Categorical:
        - Most frequent imputation
        - One-hot encoding

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
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

def build_anomaly_detector() -> IsolationForest:
    """
    Create the Isolation Forest anomaly detector.

    Returns
    -------
    IsolationForest
        Configured anomaly detection model.
    """

    return IsolationForest(
        **ISOLATION_FOREST_PARAMETERS
    )


# =============================================================================
# ANOMALY SCORE CONVERSION
# =============================================================================

def calculate_anomaly_scores(
    model: IsolationForest,
    X_processed,
) -> np.ndarray:
    """
    Calculate normalized anomaly scores.

    Isolation Forest's decision_function produces:

        Positive -> more normal
        Negative -> more anomalous

    We invert the score so that a higher value means more suspicious.

    Parameters
    ----------
    model : IsolationForest
        Trained anomaly detector.

    X_processed
        Preprocessed feature matrix.

    Returns
    -------
    np.ndarray
        Suspicion scores.
    """

    raw_scores = model.decision_function(
        X_processed
    )

    suspicion_scores = -raw_scores

    return suspicion_scores


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_anomaly_detector(
    model: IsolationForest,
    X_test_processed,
    y_test: pd.Series,
) -> None:
    """
    Evaluate anomaly detection against the official test labels.

    The anomaly model predicts:

        -1 -> anomaly
         1 -> normal

    These are converted to:

         1 -> attack/anomaly
         0 -> normal

    Parameters
    ----------
    model : IsolationForest
        Trained anomaly detector.

    X_test_processed
        Preprocessed official test features.

    y_test : pd.Series
        Official binary labels.
    """

    print("\n")
    print("=" * 80)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 80)

    print("\nGenerating anomaly predictions...")

    raw_predictions = model.predict(
        X_test_processed
    )

    # Isolation Forest:
    #
    #     -1 = anomaly
    #      1 = normal
    #
    # Convert to:
    #
    #      1 = attack/anomaly
    #      0 = normal

    anomaly_predictions = (
        raw_predictions == -1
    ).astype(int)

    # -------------------------------------------------------------------------
    # Basic Metrics
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        anomaly_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            anomaly_predictions,
        )
    )

    precision = precision_score(
        y_test,
        anomaly_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        anomaly_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        anomaly_predictions,
        zero_division=0,
    )

    print("\nAnomaly detection performance:")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
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
            anomaly_predictions,
            target_names=[
                "Normal",
                "Attack / Anomaly",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    confusion = confusion_matrix(
        y_test,
        anomaly_predictions,
    )

    confusion_dataframe = pd.DataFrame(
        confusion,
        index=[
            "Actual Normal",
            "Actual Attack",
        ],
        columns=[
            "Predicted Normal",
            "Predicted Attack",
        ],
    )

    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted\n"
    )

    print(
        confusion_dataframe
    )

    # -------------------------------------------------------------------------
    # Prediction Distribution
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ANOMALY PREDICTION DISTRIBUTION")
    print("=" * 80)

    prediction_distribution = pd.Series(
        anomaly_predictions
    ).map(
        {
            0: "Normal",
            1: "Anomaly",
        }
    ).value_counts()

    print(
        prediction_distribution
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Execute the complete anomaly detection pipeline.
    """

    print("=" * 80)
    print("UNSW-NB15 ANOMALY / UNKNOWN-THREAT DETECTION")
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
    # PREPARE NORMAL TRAINING DATA
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("PREPARING NORMAL TRAINING TRAFFIC")
    print("=" * 80)

    X_train_normal = (
        prepare_normal_training_data(
            training_data
        )
    )

    print(
        f"\nNormal training records: "
        f"{len(X_train_normal)}"
    )

    # =========================================================================
    # PREPARE TEST DATA
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("PREPARING OFFICIAL TEST DATA")
    print("=" * 80)

    X_test, y_test = prepare_test_data(
        testing_data
    )

    print(
        f"\nOfficial test records: "
        f"{len(X_test)}"
    )

    print(
        "\nOfficial test label distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Normal",
                1: "Attack",
            }
        )
    )

    # =========================================================================
    # DISPLAY FEATURES
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("FEATURE CONFIGURATION")
    print("=" * 80)

    print(
        f"\nNumerical features: "
        f"{len(SELECTED_NUMERICAL_FEATURES)}"
    )

    for feature in SELECTED_NUMERICAL_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"\nCategorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    for feature in CATEGORICAL_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"\nTotal raw features: "
        f"{len(SELECTED_FEATURES)}"
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
    # FIT PREPROCESSOR ONLY ON NORMAL TRAINING DATA
    # =========================================================================

    print(
        "\nFitting preprocessor on normal training traffic..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train_normal
        )
    )

    print(
        "Preprocessing completed."
    )

    print(
        f"\nProcessed normal training shape: "
        f"{X_train_processed.shape}"
    )

    # =========================================================================
    # TRANSFORM TEST DATA
    # =========================================================================

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"Processed test shape: "
        f"{X_test_processed.shape}"
    )

    # =========================================================================
    # BUILD ISOLATION FOREST
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("BUILDING ISOLATION FOREST")
    print("=" * 80)

    anomaly_detector = (
        build_anomaly_detector()
    )

    print(
        "\nIsolation Forest configuration:"
    )

    for parameter, value in (
        ISOLATION_FOREST_PARAMETERS.items()
    ):
        print(
            f"  {parameter}: {value}"
        )

    # =========================================================================
    # TRAIN ANOMALY DETECTOR
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("TRAINING ANOMALY DETECTOR")
    print("=" * 80)

    print(
        "\nTraining ONLY on normal traffic..."
    )

    anomaly_detector.fit(
        X_train_processed
    )

    print(
        "Training completed successfully."
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("SAVING ANOMALY MODEL")
    print("=" * 80)

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        anomaly_detector,
        ANOMALY_DETECTOR_PATH,
    )

    joblib.dump(
        preprocessor,
        ANOMALY_PREPROCESSOR_PATH,
    )

    print(
        f"\nAnomaly detector saved to:"
        f"\n{ANOMALY_DETECTOR_PATH}"
    )

    print(
        f"\nAnomaly preprocessor saved to:"
        f"\n{ANOMALY_PREPROCESSOR_PATH}"
    )

    # =========================================================================
    # EVALUATE
    # =========================================================================

    evaluate_anomaly_detector(
        anomaly_detector,
        X_test_processed,
        y_test,
    )

    # =========================================================================
    # ANOMALY SCORE SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("ANOMALY SCORE SUMMARY")
    print("=" * 80)

    suspicion_scores = (
        calculate_anomaly_scores(
            anomaly_detector,
            X_test_processed,
        )
    )

    score_dataframe = pd.DataFrame(
        {
            "Actual_Label": y_test.values,
            "Anomaly_Score": suspicion_scores,
        }
    )

    score_dataframe[
        "Actual_Label"
    ] = score_dataframe[
        "Actual_Label"
    ].map(
        {
            0: "Normal",
            1: "Attack",
        }
    )

    print(
        "\nAverage anomaly score by actual class:"
    )

    print(
        score_dataframe.groupby(
            "Actual_Label"
        )["Anomaly_Score"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
    )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("ANOMALY DETECTION COMPLETED")
    print("=" * 80)

    print(
        "\nThe anomaly detector has been trained"
        " using normal traffic only."
    )

    print(
        "\nIt is now ready to be integrated into"
        " the unified cybersecurity prediction pipeline."
    )

    print(
        "\nImportant:"
    )

    print(
        "This component identifies anomalous behavior"
        " relative to learned normal traffic."
    )

    print(
        "It is not being presented as a guaranteed"
        " zero-day attack detector."
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()