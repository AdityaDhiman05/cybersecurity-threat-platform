"""
================================================================================
CYBERSECURITY THREAT PLATFORM
T2 — EVALUATION INTEGRITY
================================================================================

Purpose
-------
Evaluate the trained cybersecurity models on the official UNSW-NB15
training and testing datasets.

This evaluation deliberately reports three test-set views:

1. Full official test set
2. Test set with exact duplicate feature rows removed
3. Test rows whose 42 model-input features do not appear in training

The evaluation does NOT retrain any model.

Models
------
- Binary Random Forest
- 9-class attack-category Random Forest
- Isolation Forest trained on normal traffic only

Outputs
-------
outputs/evaluation/metrics.json

The script also prints a complete evaluation report.

Important
---------
This is an evaluation script only.

It does not:
- modify model artifacts
- retrain models
- perform SMOTE
- generate synthetic data
- change inference thresholds
- claim zero-day detection

Anomaly scores follow the project's current convention:

    Higher decision_function() score -> more normal
    Lower decision_function() score  -> more anomalous

Therefore anomaly AUC uses the negative normality score as the
anomaly-strength value.
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIRECTORY = PROJECT_ROOT / "models"

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

TRAINING_DATASET_PATH = (
    DATA_DIRECTORY / "UNSW_NB15_training-set.csv"
)

TESTING_DATASET_PATH = (
    DATA_DIRECTORY / "UNSW_NB15_testing-set.csv"
)

EVALUATION_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "evaluation"
)

METRICS_OUTPUT_PATH = (
    EVALUATION_DIRECTORY / "metrics.json"
)


# =============================================================================
# DATASET COLUMNS
# =============================================================================

IDENTIFIER_COLUMNS = [
    "id",
]

TARGET_COLUMNS = [
    "attack_cat",
    "label",
]


# =============================================================================
# ATTACK CATEGORIES
# =============================================================================

ATTACK_CATEGORIES = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]


# =============================================================================
# MODEL ARTIFACT PATHS
# =============================================================================

BINARY_CLASSIFIER_PATH = (
    MODELS_DIRECTORY / "binary_classifier.pkl"
)

BINARY_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY / "binary_preprocessor.pkl"
)

ATTACK_CLASSIFIER_PATH = (
    MODELS_DIRECTORY / "attack_classifier.pkl"
)

ATTACK_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY / "attack_preprocessor.pkl"
)

ANOMALY_DETECTOR_PATH = (
    MODELS_DIRECTORY / "anomaly_detector.pkl"
)

ANOMALY_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY / "anomaly_preprocessor.pkl"
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_artifacts() -> Dict[str, Any]:
    """
    Load the existing trained model artifacts.

    No training or modification is performed.
    """

    required_paths = {
        "binary_classifier": BINARY_CLASSIFIER_PATH,
        "binary_preprocessor": BINARY_PREPROCESSOR_PATH,
        "attack_classifier": ATTACK_CLASSIFIER_PATH,
        "attack_preprocessor": ATTACK_PREPROCESSOR_PATH,
        "anomaly_detector": ANOMALY_DETECTOR_PATH,
        "anomaly_preprocessor": ANOMALY_PREPROCESSOR_PATH,
    }

    print()
    print("=" * 80)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 80)

    for artifact_name, artifact_path in required_paths.items():

        if not artifact_path.exists():

            raise FileNotFoundError(
                f"Required model artifact is missing:\n"
                f"{artifact_name}: {artifact_path}"
            )

        print(
            f"[OK] {artifact_name:<24} "
            f"{artifact_path.name}"
        )

    artifacts = {
        "binary_classifier": joblib.load(
            BINARY_CLASSIFIER_PATH
        ),
        "binary_preprocessor": joblib.load(
            BINARY_PREPROCESSOR_PATH
        ),
        "attack_classifier": joblib.load(
            ATTACK_CLASSIFIER_PATH
        ),
        "attack_preprocessor": joblib.load(
            ATTACK_PREPROCESSOR_PATH
        ),
        "anomaly_detector": joblib.load(
            ANOMALY_DETECTOR_PATH
        ),
        "anomaly_preprocessor": joblib.load(
            ANOMALY_PREPROCESSOR_PATH
        ),
    }

    print()
    print("All model artifacts loaded successfully.")

    return artifacts


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the official UNSW-NB15 training and testing CSV files.
    """

    print()
    print("=" * 80)
    print("LOADING UNSW-NB15 DATASETS")
    print("=" * 80)

    if not TRAINING_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{TRAINING_DATASET_PATH}"
        )

    if not TESTING_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Testing dataset not found:\n"
            f"{TESTING_DATASET_PATH}"
        )

    training_data = pd.read_csv(
        TRAINING_DATASET_PATH
    )

    testing_data = pd.read_csv(
        TESTING_DATASET_PATH
    )

    print(
        f"[OK] Training dataset : "
        f"{len(training_data):,} rows"
    )

    print(
        f"[OK] Testing dataset  : "
        f"{len(testing_data):,} rows"
    )

    return training_data, testing_data


# =============================================================================
# FEATURE DISCOVERY
# =============================================================================

def get_model_input_features(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Determine the 42 raw model-input features.

    The project uses all dataset columns except:
        id
        attack_cat
        label
    """

    excluded_columns = set(
        IDENTIFIER_COLUMNS + TARGET_COLUMNS
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    return feature_columns


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset_structure(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
) -> List[str]:
    """
    Validate that training and testing datasets contain the same
    expected structure.
    """

    if list(training_data.columns) != list(
        testing_data.columns
    ):

        raise ValueError(
            "Training and testing datasets do not have "
            "the same column structure."
        )

    feature_columns = get_model_input_features(
        training_data
    )

    if len(feature_columns) != 42:

        raise ValueError(
            "Expected exactly 42 model-input features, "
            f"but found {len(feature_columns)}."
        )

    required_columns = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + feature_columns
    )

    missing_training = (
        required_columns
        - set(training_data.columns)
    )

    missing_testing = (
        required_columns
        - set(testing_data.columns)
    )

    if missing_training:

        raise ValueError(
            "Training dataset is missing columns:\n"
            + "\n".join(
                sorted(missing_training)
            )
        )

    if missing_testing:

        raise ValueError(
            "Testing dataset is missing columns:\n"
            + "\n".join(
                sorted(missing_testing)
            )
        )

    return feature_columns


# =============================================================================
# DUPLICATE / OVERLAP AUDIT
# =============================================================================

def create_feature_key_frame(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Return only the 42 model-input feature columns.

    A copy is returned so the original dataset remains unchanged.
    """

    return dataframe[
        feature_columns
    ].copy()


def audit_duplicates_and_overlap(
    training_data: pd.DataFrame,
    testing_data: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, Any]:
    """
    Audit exact duplicate feature records and train/test overlap.

    Matching is performed only on the 42 model-input features.

    Returns
    -------
    dict
        Counts for:
        - duplicate rows
        - unique feature records
        - duplicate groups
        - test rows present in training
        - test rows not present in training
    """

    training_features = create_feature_key_frame(
        training_data,
        feature_columns,
    )

    testing_features = create_feature_key_frame(
        testing_data,
        feature_columns,
    )

    # -------------------------------------------------------------------------
    # Exact duplicates inside training
    # -------------------------------------------------------------------------

    training_duplicate_mask = (
        training_features.duplicated(
            keep="first"
        )
    )

    training_duplicate_rows = int(
        training_duplicate_mask.sum()
    )

    training_unique_rows = int(
        training_features.drop_duplicates().shape[0]
    )

    training_duplicate_groups = int(
        training_features.duplicated(
            keep=False
        ).sum()
        - training_duplicate_rows
    )

    # -------------------------------------------------------------------------
    # Exact duplicates inside testing
    # -------------------------------------------------------------------------

    testing_duplicate_mask = (
        testing_features.duplicated(
            keep="first"
        )
    )

    testing_duplicate_rows = int(
        testing_duplicate_mask.sum()
    )

    testing_unique_rows = int(
        testing_features.drop_duplicates().shape[0]
    )

    testing_duplicate_groups = int(
        testing_features.duplicated(
            keep=False
        ).sum()
        - testing_duplicate_rows
    )

    # -------------------------------------------------------------------------
    # Train/test overlap
    # -------------------------------------------------------------------------

    unique_training_features = (
        training_features.drop_duplicates()
    )

    overlap_marker = (
        unique_training_features.assign(
            _present_in_training=1
        )
    )

    overlap_check = testing_features.merge(
        overlap_marker,
        how="left",
        on=feature_columns,
        sort=False,
    )

    test_overlap_mask = (
        overlap_check[
            "_present_in_training"
        ].notna()
    )

    test_rows_in_training = int(
        test_overlap_mask.sum()
    )

    test_rows_not_in_training = int(
        (~test_overlap_mask).sum()
    )

    # -------------------------------------------------------------------------
    # Class-specific duplicate rates
    # -------------------------------------------------------------------------

    testing_with_duplicate_flag = testing_data.copy()

    testing_with_duplicate_flag[
        "_is_duplicate_feature_row"
    ] = testing_duplicate_mask.to_numpy()

    class_duplicate_rates = {}

    for category in ATTACK_CATEGORIES:

        category_mask = (
            testing_with_duplicate_flag[
                "attack_cat"
            ]
            .astype(str)
            .str.strip()
            == category
        )

        category_rows = int(
            category_mask.sum()
        )

        if category_rows == 0:

            class_duplicate_rates[
                category
            ] = {
                "rows": 0,
                "duplicate_rows": 0,
                "duplicate_rate": 0.0,
            }

            continue

        category_duplicates = int(
            testing_with_duplicate_flag.loc[
                category_mask,
                "_is_duplicate_feature_row",
            ].sum()
        )

        class_duplicate_rates[
            category
        ] = {
            "rows": category_rows,
            "duplicate_rows": category_duplicates,
            "duplicate_rate": (
                category_duplicates
                / category_rows
            ),
        }

    normal_mask = (
        testing_data["label"] == 0
    )

    normal_rows = int(
        normal_mask.sum()
    )

    normal_duplicates = int(
        testing_with_duplicate_flag.loc[
            normal_mask,
            "_is_duplicate_feature_row",
        ].sum()
    )

    class_duplicate_rates["Normal"] = {
        "rows": normal_rows,
        "duplicate_rows": normal_duplicates,
        "duplicate_rate": (
            normal_duplicates / normal_rows
            if normal_rows
            else 0.0
        ),
    }

    return {
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": feature_columns,

        "training": {
            "rows": len(training_data),
            "unique_feature_rows": (
                training_unique_rows
            ),
            "duplicate_rows_after_first": (
                training_duplicate_rows
            ),
            "duplicate_groups": (
                training_duplicate_groups
            ),
            "duplicate_rate": (
                training_duplicate_rows
                / len(training_data)
            ),
        },

        "testing": {
            "rows": len(testing_data),
            "unique_feature_rows": (
                testing_unique_rows
            ),
            "duplicate_rows_after_first": (
                testing_duplicate_rows
            ),
            "duplicate_groups": (
                testing_duplicate_groups
            ),
            "duplicate_rate": (
                testing_duplicate_rows
                / len(testing_data)
            ),
            "class_duplicate_rates": (
                class_duplicate_rates
            ),
        },

        "train_test_overlap": {
            "test_rows_present_in_training": (
                test_rows_in_training
            ),
            "test_rows_not_present_in_training": (
                test_rows_not_in_training
            ),
            "overlap_rate": (
                test_rows_in_training
                / len(testing_data)
            ),
            "not_in_training_rate": (
                test_rows_not_in_training
                / len(testing_data)
            ),
        },
    }


# =============================================================================
# TEST-SET VIEWS
# =============================================================================

def build_evaluation_views(
    testing_data: pd.DataFrame,
    training_data: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Build the three required evaluation views.

    View A:
        Full official test set.

    View B:
        Test set after removing exact duplicate feature rows,
        retaining the first occurrence.

    View C:
        Test rows whose 42 feature values do not appear in training.
    """

    # -------------------------------------------------------------------------
    # Full test set
    # -------------------------------------------------------------------------

    full_test = testing_data.copy()

    # -------------------------------------------------------------------------
    # Deduplicated test set
    # -------------------------------------------------------------------------

    deduplicated_test = (
        testing_data.drop_duplicates(
            subset=feature_columns,
            keep="first",
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Test rows not present in training
    # -------------------------------------------------------------------------

    unique_training_features = (
        training_data[
            feature_columns
        ]
        .drop_duplicates()
    )

    test_with_marker = testing_data.merge(
        unique_training_features.assign(
            _present_in_training=1
        ),
        how="left",
        on=feature_columns,
        sort=False,
    )

    not_in_training_mask = (
        test_with_marker[
            "_present_in_training"
        ].isna()
    )

    not_in_training_test = (
        testing_data.loc[
            not_in_training_mask.to_numpy()
        ]
        .reset_index(drop=True)
    )

    return {
        "full_test": full_test,
        "deduplicated_test": deduplicated_test,
        "not_in_training": not_in_training_test,
    }


# =============================================================================
# BINARY EVALUATION
# =============================================================================

def evaluate_binary(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    classifier: Any,
    preprocessor: Any,
) -> Dict[str, Any]:
    """
    Evaluate the binary attack classifier.
    """

    features = dataframe[
        feature_columns
    ].copy()

    processed_features = (
        preprocessor.transform(
            features
        )
    )

    predictions = classifier.predict(
        processed_features
    )

    probabilities = classifier.predict_proba(
        processed_features
    )

    class_labels = list(
        classifier.classes_
    )

    if 1 not in class_labels:

        raise ValueError(
            "Binary classifier does not contain "
            "class 1 = Attack."
        )

    attack_probability_index = (
        class_labels.index(1)
    )

    attack_probabilities = probabilities[
        :,
        attack_probability_index,
    ]

    true_labels = dataframe[
        "label"
    ].astype(int).to_numpy()

    predictions = np.asarray(
        predictions
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    normal_recall = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predictions,
            )
        ),

        "attack_precision": float(
            precision_score(
                true_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "attack_recall": float(
            recall_score(
                true_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "normal_recall": float(
            normal_recall
        ),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "roc_auc": float(
            roc_auc_score(
                true_labels,
                attack_probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                true_labels,
                attack_probabilities,
            )
        ),

        "confusion_matrix": [
            [
                int(tn),
                int(fp),
            ],
            [
                int(fn),
                int(tp),
            ],
        ],

        "confusion_matrix_labels": [
            "Normal",
            "Attack",
        ],

        "row_count": len(
            dataframe
        ),
    }


# =============================================================================
# MULTICLASS EVALUATION
# =============================================================================

def evaluate_multiclass(
    dataframe: pd.DataFrame,
    classifier: Any,
) -> Dict[str, Any]:
    """
    Evaluate the 9-class attack-category classifier.

    Only true attack rows are evaluated.

    This intentionally evaluates the category model independently
    of the binary detector so category accuracy includes attacks
    that the binary classifier might otherwise miss.
    """

    attack_data = dataframe[
        dataframe["label"] == 1
    ].copy()

    if attack_data.empty:

        return {
            "row_count": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "top_2_accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "confusion_matrix_labels": [],
        }

    features = attack_data[
        [
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
            "proto",
            "service",
            "state",
        ]
    ].copy()

    true_categories = (
        attack_data[
            "attack_cat"
        ]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    predictions = classifier.predict(
        features
    )

    predictions = np.asarray(
        predictions
    ).astype(str)

    probabilities = classifier.predict_proba(
        features
    )

    class_labels = np.asarray(
        classifier.classes_
    ).astype(str)

    # -------------------------------------------------------------------------
    # Top-2 accuracy
    # -------------------------------------------------------------------------

    top_two_indices = np.argsort(
        probabilities,
        axis=1,
    )[:, -2:]

    top_two_predictions = (
        class_labels[
            top_two_indices
        ]
    )

    top_two_correct = np.any(
        top_two_predictions
        == true_categories.reshape(-1, 1),
        axis=1,
    )

    top_two_accuracy = float(
        np.mean(
            top_two_correct
        )
    )

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        true_categories,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            true_categories,
            predictions,
        )
    )

    macro_f1 = f1_score(
        true_categories,
        predictions,
        labels=ATTACK_CATEGORIES,
        average="macro",
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # Per-class metrics
    # -------------------------------------------------------------------------

    precision_values, recall_values, f1_values, support_values = (
        precision_recall_fscore_support(
            true_categories,
            predictions,
            labels=ATTACK_CATEGORIES,
            average=None,
            zero_division=0,
        )
    )

    per_class = {}

    for index, category in enumerate(
        ATTACK_CATEGORIES
    ):

        per_class[category] = {
            "precision": float(
                precision_values[index]
            ),
            "recall": float(
                recall_values[index]
            ),
            "f1": float(
                f1_values[index]
            ),
            "support": int(
                support_values[index]
            ),
        }

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    category_confusion_matrix = (
        confusion_matrix(
            true_categories,
            predictions,
            labels=ATTACK_CATEGORIES,
        )
    )

    return {
        "row_count": len(
            attack_data
        ),

        "accuracy": float(
            accuracy
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "top_2_accuracy": float(
            top_two_accuracy
        ),

        "per_class": per_class,

        "confusion_matrix": (
            category_confusion_matrix
            .astype(int)
            .tolist()
        ),

        "confusion_matrix_labels": (
            ATTACK_CATEGORIES
        ),
    }


# =============================================================================
# ANOMALY EVALUATION
# =============================================================================

def evaluate_anomaly(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    detector: Any,
    preprocessor: Any,
) -> Dict[str, Any]:
    """
    Evaluate the Isolation Forest anomaly detector.

    The detector is treated as a binary anomaly flag:

        -1 -> Anomaly
         1 -> Normal

    The raw decision_function score is a normality score:

        higher -> more normal
        lower  -> more anomalous

    Therefore anomaly AUC is calculated using -score.
    """

    features = dataframe[
        feature_columns
    ].copy()

    processed_features = (
        preprocessor.transform(
            features
        )
    )

    detector_predictions = detector.predict(
        processed_features
    )

    normality_scores = detector.decision_function(
        processed_features
    )

    true_attack_labels = (
        dataframe["label"]
        .astype(int)
        .to_numpy()
    )

    anomaly_flags = (
        detector_predictions == -1
    )

    anomaly_strength = (
        -np.asarray(
            normality_scores
        )
    )

    tn = int(
        np.sum(
            (true_attack_labels == 0)
            & (~anomaly_flags)
        )
    )

    fp = int(
        np.sum(
            (true_attack_labels == 0)
            & anomaly_flags
        )
    )

    fn = int(
        np.sum(
            (true_attack_labels == 1)
            & (~anomaly_flags)
        )
    )

    tp = int(
        np.sum(
            (true_attack_labels == 1)
            & anomaly_flags
        )
    )

    attack_recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Per-class anomaly flag rates
    # -------------------------------------------------------------------------

    per_class_flag_rate = {}

    for category in ATTACK_CATEGORIES:

        category_mask = (
            dataframe["attack_cat"]
            .astype(str)
            .str.strip()
            == category
        )

        category_count = int(
            category_mask.sum()
        )

        category_flags = int(
            anomaly_flags[
                category_mask.to_numpy()
            ].sum()
        )

        per_class_flag_rate[
            category
        ] = {
            "rows": category_count,
            "flagged": category_flags,
            "flag_rate": (
                category_flags
                / category_count
                if category_count
                else 0.0
            ),
        }

    normal_mask = (
        dataframe["label"] == 0
    )

    normal_count = int(
        normal_mask.sum()
    )

    normal_flags = int(
        anomaly_flags[
            normal_mask.to_numpy()
        ].sum()
    )

    per_class_flag_rate["Normal"] = {
        "rows": normal_count,
        "flagged": normal_flags,
        "flag_rate": (
            normal_flags / normal_count
            if normal_count
            else 0.0
        ),
    }

    return {
        "row_count": len(
            dataframe
        ),

        "auc": float(
            roc_auc_score(
                true_attack_labels,
                anomaly_strength,
            )
        ),

        "recall": float(
            attack_recall
        ),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "confusion_matrix": [
            [
                tn,
                fp,
            ],
            [
                fn,
                tp,
            ],
        ],

        "confusion_matrix_labels": [
            "Normal",
            "Attack",
        ],

        "per_true_class_flag_rate": (
            per_class_flag_rate
        ),
    }


# =============================================================================
# END-TO-END EVALUATION
# =============================================================================

def evaluate_end_to_end(
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    binary_classifier: Any,
    binary_preprocessor: Any,
    attack_classifier: Any,
    anomaly_detector: Any,
    anomaly_preprocessor: Any,
) -> Dict[str, Any]:
    """
    Evaluate the complete binary -> category -> anomaly -> risk path.

    Category accuracy is calculated independently across ALL true
    attacks, including attacks missed by the binary detector.

    HIGH/CRITICAL percentages are based on the final rule-based
    Threat Priority Score returned by the existing risk engine.
    """

    # Import the existing project risk engine.
    try:

        from risk_assessment import assess_risk

    except ModuleNotFoundError:

        from src.risk_assessment import assess_risk

    features = dataframe[
        feature_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Binary inference
    # -------------------------------------------------------------------------

    binary_processed = (
        binary_preprocessor.transform(
            features
        )
    )

    binary_predictions = (
        binary_classifier.predict(
            binary_processed
        )
    )

    binary_probabilities = (
        binary_classifier.predict_proba(
            binary_processed
        )
    )

    binary_class_labels = list(
        binary_classifier.classes_
    )

    attack_probability_index = (
        binary_class_labels.index(1)
    )

    attack_probabilities = (
        binary_probabilities[
            :,
            attack_probability_index,
        ]
    )

    detection_attack_mask = (
        binary_predictions == 1
    )

    # -------------------------------------------------------------------------
    # Attack-category inference for ALL true attacks
    # -------------------------------------------------------------------------

    attack_features = dataframe[
        [
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
            "proto",
            "service",
            "state",
        ]
    ].copy()

    category_predictions = (
        attack_classifier.predict(
            attack_features
        )
    )

    category_probabilities = (
        attack_classifier.predict_proba(
            attack_features
        )
    )

    category_class_labels = np.asarray(
        attack_classifier.classes_
    ).astype(str)

    category_probability_indices = (
        category_probabilities.argmax(
            axis=1
        )
    )

    predicted_categories = (
        category_class_labels[
            category_probability_indices
        ]
    )

    predicted_category_confidences = (
        category_probabilities[
            np.arange(
                len(category_probabilities)
            ),
            category_probability_indices,
        ]
    )

    # -------------------------------------------------------------------------
    # Anomaly inference
    # -------------------------------------------------------------------------

    anomaly_processed = (
        anomaly_preprocessor.transform(
            attack_features
        )
    )

    anomaly_predictions = (
        anomaly_detector.predict(
            anomaly_processed
        )
    )

    anomaly_scores = (
        anomaly_detector.decision_function(
            anomaly_processed
        )
    )

    # -------------------------------------------------------------------------
    # Category accuracy across ALL true attacks
    # -------------------------------------------------------------------------

    true_attack_mask = (
        dataframe["label"]
        .astype(int)
        .to_numpy()
        == 1
    )

    true_attack_categories = (
        dataframe.loc[
            true_attack_mask,
            "attack_cat",
        ]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    all_attack_predicted_categories = (
        predicted_categories[
            true_attack_mask
        ]
    )

    category_accuracy_all_attacks = (
        accuracy_score(
            true_attack_categories,
            all_attack_predicted_categories,
        )
    )

    # -------------------------------------------------------------------------
    # End-to-end risk evaluation
    # -------------------------------------------------------------------------

    risk_scores = np.zeros(
        len(dataframe),
        dtype=float,
    )

    risk_levels = []

    for index in range(
        len(dataframe)
    ):

        detection_status = (
            "Attack"
            if detection_attack_mask[index]
            else "Normal"
        )

        attack_category = (
            predicted_categories[index]
            if detection_status == "Attack"
            else None
        )

        category_probability = (
            float(
                predicted_category_confidences[index]
            )
            if detection_status == "Attack"
            else None
        )

        anomaly_status = (
            "Anomaly"
            if anomaly_predictions[index] == -1
            else "Normal"
        )

        risk_result = assess_risk(
            detection_status=detection_status,
            attack_category=attack_category,
            anomaly_status=anomaly_status,
            anomaly_score=float(
                anomaly_scores[index]
            ),
            binary_attack_probability=float(
                attack_probabilities[index]
            ),
            attack_category_probability=(
                category_probability
            ),
        )

        risk_scores[index] = (
            float(
                risk_result.risk_score
            )
        )

        risk_levels.append(
            str(
                risk_result.risk_level
            ).upper()
        )

    risk_levels = np.asarray(
        risk_levels
    )

    high_or_critical_mask = np.isin(
        risk_levels,
        [
            "HIGH",
            "CRITICAL",
        ],
    )

    normal_mask = (
        dataframe["label"]
        .astype(int)
        .to_numpy()
        == 0
    )

    attack_mask = (
        dataframe["label"]
        .astype(int)
        .to_numpy()
        == 1
    )

    normal_high_critical = int(
        np.sum(
            normal_mask
            & high_or_critical_mask
        )
    )

    attack_high_critical = int(
        np.sum(
            attack_mask
            & high_or_critical_mask
        )
    )

    normal_count = int(
        normal_mask.sum()
    )

    attack_count = int(
        attack_mask.sum()
    )

    return {
        "row_count": len(
            dataframe
        ),

        "normal_flows": normal_count,

        "attack_flows": attack_count,

        "normal_high_critical_count": (
            normal_high_critical
        ),

        "normal_high_critical_rate": (
            normal_high_critical
            / normal_count
            if normal_count
            else 0.0
        ),

        "attack_high_critical_count": (
            attack_high_critical
        ),

        "attack_high_critical_rate": (
            attack_high_critical
            / attack_count
            if attack_count
            else 0.0
        ),

        "category_accuracy_all_true_attacks": (
            float(
                category_accuracy_all_attacks
            )
        ),

        "note": (
            "Category accuracy is evaluated across "
            "all true attack records, including records "
            "that the binary classifier classified as Normal."
        ),
    }


# =============================================================================
# EVALUATION VIEW
# =============================================================================

def evaluate_view(
    view_name: str,
    dataframe: pd.DataFrame,
    feature_columns: List[str],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate all required models on one test-set view.
    """

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {view_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Rows: {len(dataframe):,}"
    )

    binary_metrics = evaluate_binary(
        dataframe=dataframe,
        feature_columns=feature_columns,
        classifier=artifacts[
            "binary_classifier"
        ],
        preprocessor=artifacts[
            "binary_preprocessor"
        ],
    )

    print(
        f"Binary Accuracy       : "
        f"{binary_metrics['accuracy']:.4f}"
    )

    print(
        f"Binary Attack Recall  : "
        f"{binary_metrics['attack_recall']:.4f}"
    )

    print(
        f"Binary ROC-AUC        : "
        f"{binary_metrics['roc_auc']:.4f}"
    )

    print(
        f"Binary PR-AUC         : "
        f"{binary_metrics['pr_auc']:.4f}"
    )

    multiclass_metrics = evaluate_multiclass(
        dataframe=dataframe,
        classifier=artifacts[
            "attack_classifier"
        ],
    )

    print(
        f"Multiclass Accuracy   : "
        f"{multiclass_metrics['accuracy']:.4f}"
    )

    print(
        f"Multiclass Balanced Acc: "
        f"{multiclass_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Multiclass Macro-F1   : "
        f"{multiclass_metrics['macro_f1']:.4f}"
    )

    print(
        f"Multiclass Top-2 Acc  : "
        f"{multiclass_metrics['top_2_accuracy']:.4f}"
    )

    anomaly_metrics = evaluate_anomaly(
        dataframe=dataframe,
        feature_columns=feature_columns,
        detector=artifacts[
            "anomaly_detector"
        ],
        preprocessor=artifacts[
            "anomaly_preprocessor"
        ],
    )

    print(
        f"Anomaly AUC           : "
        f"{anomaly_metrics['auc']:.4f}"
    )

    print(
        f"Anomaly Recall        : "
        f"{anomaly_metrics['recall']:.4f}"
    )

    print(
        f"Anomaly FPR           : "
        f"{anomaly_metrics['false_positive_rate']:.4f}"
    )

    end_to_end_metrics = evaluate_end_to_end(
        dataframe=dataframe,
        feature_columns=feature_columns,
        binary_classifier=artifacts[
            "binary_classifier"
        ],
        binary_preprocessor=artifacts[
            "binary_preprocessor"
        ],
        attack_classifier=artifacts[
            "attack_classifier"
        ],
        anomaly_detector=artifacts[
            "anomaly_detector"
        ],
        anomaly_preprocessor=artifacts[
            "anomaly_preprocessor"
        ],
    )

    print(
        f"Normal HIGH/CRITICAL   : "
        f"{end_to_end_metrics['normal_high_critical_rate']:.4f}"
    )

    print(
        f"Attack HIGH/CRITICAL   : "
        f"{end_to_end_metrics['attack_high_critical_rate']:.4f}"
    )

    print(
        f"Category Accuracy ALL  : "
        f"{end_to_end_metrics['category_accuracy_all_true_attacks']:.4f}"
    )

    return {
        "row_count": len(
            dataframe
        ),
        "binary": binary_metrics,
        "multiclass": multiclass_metrics,
        "anomaly": anomaly_metrics,
        "end_to_end": end_to_end_metrics,
    }


# =============================================================================
# REPORT DISPLAY
# =============================================================================

def print_duplicate_audit(
    duplicate_audit: Dict[str, Any],
) -> None:
    """
    Print the duplicate and overlap audit.
    """

    print()
    print("=" * 80)
    print("DUPLICATE / TRAIN-TEST OVERLAP AUDIT")
    print("=" * 80)

    training = duplicate_audit[
        "training"
    ]

    testing = duplicate_audit[
        "testing"
    ]

    overlap = duplicate_audit[
        "train_test_overlap"
    ]

    print()
    print("TRAINING DATA")
    print(
        f"Rows                         : "
        f"{training['rows']:,}"
    )

    print(
        f"Unique feature rows          : "
        f"{training['unique_feature_rows']:,}"
    )

    print(
        f"Duplicate rows after first   : "
        f"{training['duplicate_rows_after_first']:,}"
    )

    print(
        f"Duplicate rate               : "
        f"{training['duplicate_rate']:.4%}"
    )

    print()
    print("TESTING DATA")
    print(
        f"Rows                         : "
        f"{testing['rows']:,}"
    )

    print(
        f"Unique feature rows          : "
        f"{testing['unique_feature_rows']:,}"
    )

    print(
        f"Duplicate rows after first   : "
        f"{testing['duplicate_rows_after_first']:,}"
    )

    print(
        f"Duplicate rate               : "
        f"{testing['duplicate_rate']:.4%}"
    )

    print()
    print("TRAIN / TEST OVERLAP")
    print(
        f"Test rows present in train   : "
        f"{overlap['test_rows_present_in_training']:,}"
    )

    print(
        f"Test rows NOT in train       : "
        f"{overlap['test_rows_not_present_in_training']:,}"
    )

    print(
        f"Overlap rate                 : "
        f"{overlap['overlap_rate']:.4%}"
    )


def print_full_test_baseline(
    full_test_metrics: Dict[str, Any],
) -> None:
    """
    Print the key full-test metrics used for T2 verification.
    """

    binary = full_test_metrics[
        "binary"
    ]

    multiclass = full_test_metrics[
        "multiclass"
    ]

    print()
    print("=" * 80)
    print("FULL TEST BASELINE CHECK")
    print("=" * 80)

    print(
        f"Binary Accuracy       : "
        f"{binary['accuracy']:.4%}"
    )

    print(
        f"Binary Attack Recall  : "
        f"{binary['attack_recall']:.4%}"
    )

    print(
        f"Binary ROC-AUC        : "
        f"{binary['roc_auc']:.4f}"
    )

    print(
        f"Multiclass Accuracy   : "
        f"{multiclass['accuracy']:.4%}"
    )

    print(
        f"Multiclass Balanced Acc: "
        f"{multiclass['balanced_accuracy']:.4%}"
    )

    print(
        f"Multiclass Macro-F1   : "
        f"{multiclass['macro_f1']:.4%}"
    )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def make_json_safe(
    value: Any,
) -> Any:
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    """

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): make_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):

        return make_json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):

        return value.item()

    if isinstance(
        value,
        np.bool_,
    ):

        return bool(value)

    return value


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main() -> None:
    """
    Execute the complete T2 evaluation.
    """

    print()
    print("=" * 80)
    print("CYBERSECURITY THREAT PLATFORM")
    print("T2 — EVALUATION INTEGRITY")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    training_data, testing_data = (
        load_datasets()
    )

    # -------------------------------------------------------------------------
    # Validate dataset structure
    # -------------------------------------------------------------------------

    feature_columns = (
        validate_dataset_structure(
            training_data,
            testing_data,
        )
    )

    print()
    print(
        f"Model-input feature count: "
        f"{len(feature_columns)}"
    )

    # -------------------------------------------------------------------------
    # Audit duplicates and train/test overlap
    # -------------------------------------------------------------------------

    duplicate_audit = (
        audit_duplicates_and_overlap(
            training_data=training_data,
            testing_data=testing_data,
            feature_columns=feature_columns,
        )
    )

    print_duplicate_audit(
        duplicate_audit
    )

    # -------------------------------------------------------------------------
    # Build three evaluation views
    # -------------------------------------------------------------------------

    evaluation_views = (
        build_evaluation_views(
            testing_data=testing_data,
            training_data=training_data,
            feature_columns=feature_columns,
        )
    )

    print()
    print("=" * 80)
    print("EVALUATION VIEWS")
    print("=" * 80)

    for view_name, view_dataframe in (
        evaluation_views.items()
    ):

        print(
            f"{view_name:<24} "
            f"{len(view_dataframe):>8,} rows"
        )

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------

    artifacts = (
        load_model_artifacts()
    )

    # -------------------------------------------------------------------------
    # Evaluate all three views
    # -------------------------------------------------------------------------

    evaluations = {}

    for view_name, view_dataframe in (
        evaluation_views.items()
    ):

        evaluations[
            view_name
        ] = evaluate_view(
            view_name=view_name,
            dataframe=view_dataframe,
            feature_columns=feature_columns,
            artifacts=artifacts,
        )

    # -------------------------------------------------------------------------
    # Print full-test baseline
    # -------------------------------------------------------------------------

    print_full_test_baseline(
        evaluations[
            "full_test"
        ]
    )

    # -------------------------------------------------------------------------
    # Build final JSON document
    # -------------------------------------------------------------------------

    metrics = {
        "evaluation_version": "T2",

        "dataset": {
            "training_path": str(
                TRAINING_DATASET_PATH
            ),
            "testing_path": str(
                TESTING_DATASET_PATH
            ),
            "training_rows": len(
                training_data
            ),
            "testing_rows": len(
                testing_data
            ),
            "feature_count": len(
                feature_columns
            ),
            "feature_columns": (
                feature_columns
            ),
        },

        "duplicate_and_overlap_audit": (
            duplicate_audit
        ),

        "evaluation_views": {
            view_name: {
                "rows": len(
                    dataframe
                ),
            }
            for view_name, dataframe in (
                evaluation_views.items()
            )
        },

        "metrics": evaluations,

        "limitations": [
            (
                "The UNSW-NB15 test set contains a substantial "
                "attack prevalence compared with real operational "
                "networks."
            ),
            (
                "Exact duplicate records are present in the "
                "testing dataset."
            ),
            (
                "Train/test feature overlap is reported explicitly "
                "rather than ignored."
            ),
            (
                "Anomaly detection identifies behavior that does "
                "not conform to the learned normal profile and "
                "may require investigation; it is not claimed "
                "to be a zero-day detector."
            ),
            (
                "The reported precision values are dependent on "
                "the class prevalence of the evaluation dataset."
            ),
        ],
    }

    metrics = make_json_safe(
        metrics
    )

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------

    EVALUATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:

        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("T2 EVALUATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Metrics file:"
    )

    print(
        METRICS_OUTPUT_PATH
    )

    print()
    print(
        "No models were retrained."
    )

    print(
        "No synthetic data was generated."
    )

    print(
        "No SMOTE was used."
    )

    print()
    print("=" * 80)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 80)
        print("T2 EVALUATION FAILED")
        print("=" * 80)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        raise