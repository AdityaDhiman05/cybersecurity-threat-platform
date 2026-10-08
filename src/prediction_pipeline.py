"""
================================================================================
CYBERSECURITY THREAT PLATFORM
UNIFIED PREDICTION PIPELINE
================================================================================

Purpose
-------
Integrates the trained cybersecurity models into one unified prediction
pipeline using real UNSW-NB15 network events.

Pipeline
--------
UNSW-NB15 Network Event
        |
        v
Binary Attack Detection
        |
        +---- Normal
        |
        +---- Attack
                 |
                 v
        Known Attack Classification
                 |
                 v
        Anomaly Detection
                 |
                 v
        Threat Priority Assessment
                 |
                 v
        Final Security Result

IMPORTANT
---------
The binary attack probability and attack-category confidence are kept as
SEPARATE values.

binary_attack_probability
    Probability that the event belongs to the Attack class.

attack_category_probability
    Confidence of the predicted known attack category.

The final Threat Priority Score is a rule-based priority score.
It is NOT a probability.

T4 SECURITY VALIDATION
----------------------
Input validation is based on the actual feature contract stored inside the
trained preprocessing artifacts.

The required input features are the UNION of:

    - binary_preprocessor.feature_names_in_
    - attack_preprocessor.feature_names_in_
    - anomaly_preprocessor.feature_names_in_

Numeric fields must contain finite numeric values.

Categorical fields must contain strings.

Missing required fields are rejected before model inference.
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterable, Set
import math

import joblib
import pandas as pd


# Support both:
#     python src/prediction_pipeline.py
# and:
#     import src.prediction_pipeline
try:
    from risk_assessment import assess_risk
except ModuleNotFoundError:
    from src.risk_assessment import assess_risk


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIRECTORY = PROJECT_ROOT / "models"

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

TESTING_DATASET_PATH = (
    DATA_DIRECTORY / "UNSW_NB15_testing-set.csv"
)


# =============================================================================
# MODEL PATHS
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
# BINARY MODEL CONFIGURATION
# =============================================================================

BINARY_EXCLUDED_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]


# =============================================================================
# ATTACK MODEL FEATURES
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

ATTACK_MODEL_FEATURES = (
    SELECTED_NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# EXPECTED UNSW-NB15 ATTACK CATEGORIES
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
# CUSTOM VALIDATION ERROR
# =============================================================================

class ValidationError(ValueError):
    """
    Raised when a network event fails API input validation.

    This exception is intentionally separate from generic model/runtime
    exceptions so the API layer can return HTTP 400 for invalid user input
    and HTTP 500 for genuine server-side failures.
    """

    pass


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models() -> Dict[str, Any]:
    """
    Load all trained cybersecurity models and preprocessing artifacts.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all trained models and preprocessors.

    Raises
    ------
    FileNotFoundError
        If any required model artifact is missing.
    """

    required_files = [
        BINARY_CLASSIFIER_PATH,
        BINARY_PREPROCESSOR_PATH,
        ATTACK_CLASSIFIER_PATH,
        ATTACK_PREPROCESSOR_PATH,
        ANOMALY_DETECTOR_PATH,
        ANOMALY_PREPROCESSOR_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                "Required model artifact was not found:\n"
                f"{file_path}"
            )

    print()
    print("Loading trained cybersecurity models...")

    models = {
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

    print("All models loaded successfully.")

    return models


# =============================================================================
# MODEL FEATURE CONTRACT
# =============================================================================

def _get_preprocessor_features(
    preprocessor: Any,
    preprocessor_name: str,
) -> Set[str]:
    """
    Extract the feature names stored by a trained preprocessor.

    Parameters
    ----------
    preprocessor:
        Fitted preprocessing artifact.

    preprocessor_name:
        Human-readable artifact name used in validation errors.

    Returns
    -------
    Set[str]
        Feature names expected by the fitted preprocessor.

    Raises
    ------
    ValidationError
        If the fitted preprocessor does not expose feature_names_in_.
    """

    feature_names = getattr(
        preprocessor,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        raise ValidationError(
            f"{preprocessor_name} does not expose "
            "feature_names_in_."
        )

    return {
        str(feature)
        for feature in feature_names
    }


def get_required_features(
    models: Dict[str, Any],
) -> Set[str]:
    """
    Return the union of features required by all trained preprocessors.

    The API input contract is deliberately derived from the trained model
    artifacts instead of maintaining a hard-coded feature list.

    This prevents the API from silently accepting an incomplete event when
    the underlying model feature requirements change.
    """

    binary_features = _get_preprocessor_features(
        models["binary_preprocessor"],
        "Binary preprocessor",
    )

    attack_features = _get_preprocessor_features(
        models["attack_preprocessor"],
        "Attack preprocessor",
    )

    anomaly_features = _get_preprocessor_features(
        models["anomaly_preprocessor"],
        "Anomaly preprocessor",
    )

    return (
        binary_features
        | attack_features
        | anomaly_features
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def _validate_numeric_value(
    feature_name: str,
    value: Any,
) -> None:
    """
    Validate one numeric feature value.

    Numeric values must be finite.

    NaN, positive infinity, negative infinity, strings, null values and
    boolean values are rejected.
    """

    if isinstance(value, bool) or not isinstance(value, Real):

        raise ValidationError(
            f"Feature '{feature_name}' must be a finite numeric value."
        )

    if not math.isfinite(float(value)):

        raise ValidationError(
            f"Feature '{feature_name}' must be a finite numeric value."
        )


def _validate_categorical_value(
    feature_name: str,
    value: Any,
) -> None:
    """
    Validate one categorical feature value.

    Categorical network features must be supplied as strings.
    """

    if not isinstance(value, str):

        raise ValidationError(
            f"Feature '{feature_name}' must be a string."
        )

    if not value.strip():

        raise ValidationError(
            f"Feature '{feature_name}' cannot be empty."
        )


def validate_input_event(
    network_event: Dict[str, Any],
    models: Dict[str, Any],
) -> None:
    """
    Validate one network event against the trained model feature contract.

    Validation includes:

        1. Required-feature presence.
        2. Numeric-value validation.
        3. Finite-number validation.
        4. Categorical-string validation.

    Extra fields are allowed because they may contain metadata supplied by
    the API client. Model inference only uses the features required by the
    trained preprocessing/model artifacts.

    Raises
    ------
    ValidationError
        If the event is malformed or missing required model features.
    """

    if not isinstance(network_event, dict):

        raise ValidationError(
            "Network event must be a JSON object."
        )

    if not network_event:

        raise ValidationError(
            "Network event cannot be empty."
        )

    required_features = get_required_features(
        models
    )

    available_features = set(
        network_event.keys()
    )

    missing_features = (
        required_features - available_features
    )

    if missing_features:

        missing_features_sorted = sorted(
            missing_features
        )

        raise ValidationError(
            "Network event is missing required features: "
            + ", ".join(
                missing_features_sorted
            )
        )

    for feature_name in required_features:

        value = network_event[
            feature_name
        ]

        if feature_name in CATEGORICAL_FEATURES:

            _validate_categorical_value(
                feature_name=feature_name,
                value=value,
            )

        else:

            _validate_numeric_value(
                feature_name=feature_name,
                value=value,
            )


# =============================================================================
# DATAFRAME CREATION
# =============================================================================

def create_dataframe(
    network_event: Dict[str, Any],
    models: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convert one validated network event into a one-row DataFrame.
    """

    validate_input_event(
        network_event=network_event,
        models=models,
    )

    return pd.DataFrame(
        [network_event]
    )


# =============================================================================
# MODEL FEATURE SELECTION
# =============================================================================

def _select_preprocessor_features(
    dataframe: pd.DataFrame,
    preprocessor: Any,
) -> pd.DataFrame:
    """
    Select exactly the raw features expected by a fitted preprocessor.
    """

    feature_names = getattr(
        preprocessor,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        raise RuntimeError(
            "The fitted preprocessor does not expose "
            "feature_names_in_."
        )

    return dataframe[
        list(feature_names)
    ].copy()


# =============================================================================
# BINARY ATTACK DETECTION
# =============================================================================

def predict_binary_attack(
    dataframe: pd.DataFrame,
    classifier: Any,
    preprocessor: Any,
) -> Dict[str, Any]:
    """
    Detect whether the network event is normal or an attack.

    Returns
    -------
    Dict[str, Any]

        prediction
            Numeric class predicted by the binary classifier.

        detection_status
            "Attack" or "Normal".

        binary_attack_probability
            Probability of class 1 (Attack).
    """

    features = _select_preprocessor_features(
        dataframe=dataframe,
        preprocessor=preprocessor,
    )

    processed_features = (
        preprocessor.transform(
            features
        )
    )

    prediction = int(
        classifier.predict(
            processed_features
        )[0]
    )

    # -------------------------------------------------------------------------
    # Extract probability specifically for class 1 = Attack.
    # -------------------------------------------------------------------------

    binary_attack_probability = None

    if hasattr(
        classifier,
        "predict_proba",
    ):

        probabilities = (
            classifier.predict_proba(
                processed_features
            )[0]
        )

        class_labels = list(
            classifier.classes_
        )

        if 1 in class_labels:

            attack_index = (
                class_labels.index(1)
            )

            binary_attack_probability = float(
                probabilities[
                    attack_index
                ]
            )

    # -------------------------------------------------------------------------
    # Convert numeric prediction to readable status.
    # -------------------------------------------------------------------------

    if prediction == 1:

        detection_status = "Attack"

    else:

        detection_status = "Normal"

    return {
        "prediction": prediction,
        "detection_status": detection_status,
        "binary_attack_probability": (
            binary_attack_probability
        ),
    }


# =============================================================================
# KNOWN ATTACK CATEGORY CLASSIFICATION
# =============================================================================

def predict_attack_category(
    dataframe: pd.DataFrame,
    classifier: Any,
) -> Dict[str, Any]:
    """
    Predict the known attack category.

    The saved attack classifier contains its own preprocessing pipeline.
    Therefore, the raw selected attack-model features are passed directly
    to the classifier.
    """

    features = dataframe[
        ATTACK_MODEL_FEATURES
    ].copy()

    prediction = classifier.predict(
        features
    )[0]

    attack_category = str(
        prediction
    )

    # -------------------------------------------------------------------------
    # Probability of the predicted attack category.
    # -------------------------------------------------------------------------

    attack_category_probability = None

    if hasattr(
        classifier,
        "predict_proba",
    ):

        probabilities = (
            classifier.predict_proba(
                features
            )[0]
        )

        highest_probability_index = (
            probabilities.argmax()
        )

        attack_category_probability = float(
            probabilities[
                highest_probability_index
            ]
        )

    return {
        "attack_category": attack_category,
        "attack_category_probability": (
            attack_category_probability
        ),
    }


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def predict_anomaly(
    dataframe: pd.DataFrame,
    detector: Any,
    preprocessor: Any,
) -> Dict[str, Any]:
    """
    Detect whether the network event deviates from the learned normal profile.

    Isolation Forest convention
    ----------------------------
    decision_function():

        Higher score -> more normal
        Lower score  -> more anomalous
    """

    features = _select_preprocessor_features(
        dataframe=dataframe,
        preprocessor=preprocessor,
    )

    processed_features = (
        preprocessor.transform(
            features
        )
    )

    prediction = int(
        detector.predict(
            processed_features
        )[0]
    )

    anomaly_score = float(
        detector.decision_function(
            processed_features
        )[0]
    )

    if prediction == -1:

        anomaly_status = "Anomaly"

    else:

        anomaly_status = "Normal"

    return {
        "prediction": prediction,
        "anomaly_status": anomaly_status,
        "anomaly_score": anomaly_score,
    }


# =============================================================================
# UNIFIED NETWORK EVENT PREDICTION
# =============================================================================

def predict_network_event(
    network_event: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run one network event through the complete cybersecurity pipeline.

    The important distinction is:

        binary_attack_probability
            -> probability that traffic is an attack.

        attack_category_probability
            -> confidence in the predicted known attack category.

    Both values are passed independently into the Threat Priority Score
    calculation.
    """

    dataframe = create_dataframe(
        network_event=network_event,
        models=models,
    )

    # =========================================================================
    # STEP 1 — BINARY ATTACK DETECTION
    # =========================================================================

    binary_result = predict_binary_attack(
        dataframe=dataframe,
        classifier=models[
            "binary_classifier"
        ],
        preprocessor=models[
            "binary_preprocessor"
        ],
    )

    detection_status = (
        binary_result[
            "detection_status"
        ]
    )

    binary_attack_probability = (
        binary_result[
            "binary_attack_probability"
        ]
    )

    # =========================================================================
    # STEP 2 — KNOWN ATTACK CLASSIFICATION
    # =========================================================================

    attack_category = None

    attack_category_probability = None

    if detection_status == "Attack":

        attack_result = predict_attack_category(
            dataframe=dataframe,
            classifier=models[
                "attack_classifier"
            ],
        )

        attack_category = (
            attack_result[
                "attack_category"
            ]
        )

        attack_category_probability = (
            attack_result[
                "attack_category_probability"
            ]
        )

    # =========================================================================
    # STEP 3 — ANOMALY DETECTION
    # =========================================================================

    anomaly_result = predict_anomaly(
        dataframe=dataframe,
        detector=models[
            "anomaly_detector"
        ],
        preprocessor=models[
            "anomaly_preprocessor"
        ],
    )

    anomaly_status = (
        anomaly_result[
            "anomaly_status"
        ]
    )

    anomaly_score = (
        anomaly_result[
            "anomaly_score"
        ]
    )

    # =========================================================================
    # STEP 4 — THREAT PRIORITY ASSESSMENT
    # =========================================================================

    risk_result = assess_risk(
        detection_status=detection_status,
        attack_category=attack_category,
        anomaly_status=anomaly_status,
        anomaly_score=anomaly_score,
        binary_attack_probability=(
            binary_attack_probability
        ),
        attack_category_probability=(
            attack_category_probability
        ),
    )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    return {
        "detection_status": detection_status,

        # ---------------------------------------------------------------------
        # Binary attack probability
        # ---------------------------------------------------------------------

        "binary_attack_probability": (
            binary_attack_probability
        ),

        # Backward-compatible field used by the existing API/dashboard.
        # This correctly means binary ATTACK probability.
        "attack_probability": (
            binary_attack_probability
        ),

        # ---------------------------------------------------------------------
        # Known attack category
        # ---------------------------------------------------------------------

        "attack_category": attack_category,

        "attack_category_probability": (
            attack_category_probability
        ),

        # ---------------------------------------------------------------------
        # Anomaly detection
        # ---------------------------------------------------------------------

        "anomaly_status": anomaly_status,

        "anomaly_score": anomaly_score,

        # ---------------------------------------------------------------------
        # Threat Priority Assessment
        # ---------------------------------------------------------------------

        "risk_score": risk_result.risk_score,

        "risk_level": risk_result.risk_level,

        "risk_components": risk_result.components,

        "explanation": risk_result.explanation,

        "recommended_action": (
            risk_result.recommended_action
        ),
    }


# =============================================================================
# SINGLE RESULT DISPLAY
# =============================================================================

def display_prediction_result(
    result: Dict[str, Any],
) -> None:
    """
    Display the complete prediction result.
    """

    print()
    print("=" * 80)
    print("UNIFIED CYBERSECURITY THREAT ASSESSMENT")
    print("=" * 80)

    print()

    print(
        "Detection Status      : "
        f"{result['detection_status']}"
    )

    print(
        "Attack Probability    : "
        f"{result['binary_attack_probability']:.4f}"
        if result["binary_attack_probability"] is not None
        else "Attack Probability    : N/A"
    )

    print(
        "Attack Category       : "
        f"{result['attack_category'] or 'None'}"
    )

    print(
        "Category Confidence   : "
        f"{result['attack_category_probability']:.4f}"
        if result["attack_category_probability"] is not None
        else "Category Confidence   : N/A"
    )

    print(
        "Anomaly Status        : "
        f"{result['anomaly_status']}"
    )

    print(
        "Normality Score       : "
        f"{result['anomaly_score']:.4f}"
    )

    print()

    print("-" * 80)

    print(
        "Threat Priority Score: "
        f"{result['risk_score']}/100"
    )

    print(
        "Risk Level            : "
        f"{result['risk_level']}"
    )

    print()

    if "risk_components" in result:

        print("Score Components:")

        for component, value in (
            result["risk_components"].items()
        ):

            print(
                f"  {component:<32}: {value}"
            )

        print()

    print("Explanation:")

    print(
        result["explanation"]
    )

    print()

    print("Recommended Action:")

    print(
        result["recommended_action"]
    )

    print()

    print("=" * 80)


# =============================================================================
# MULTI-CATEGORY REAL-DATA DEMONSTRATION
# =============================================================================

def run_attack_category_demo(
    dataset: pd.DataFrame,
    models: Dict[str, Any],
) -> pd.DataFrame:
    """
    Run one real UNSW-NB15 test record from each known attack category.

    This is an integration demonstration only.
    It is NOT the official multiclass evaluation.
    """

    print()
    print("=" * 80)
    print(
        "REAL UNSW-NB15 MULTI-CATEGORY PIPELINE DEMONSTRATION"
    )
    print("=" * 80)

    attack_records = dataset[
        dataset["label"] == 1
    ].copy()

    results = []

    for category in ATTACK_CATEGORIES:

        category_records = attack_records[
            attack_records["attack_cat"] == category
        ]

        if category_records.empty:

            print()
            print(
                f"{category:<18} NO RECORD FOUND"
            )

            results.append(
                {
                    "Actual Category": category,
                    "Predicted Category": "N/A",
                    "Detection": "N/A",
                    "Anomaly": "N/A",
                    "Threat Priority": "N/A",
                    "Risk": "N/A",
                    "Status": "No record",
                }
            )

            continue

        selected_record = (
            category_records.iloc[0].to_dict()
        )

        prediction_result = predict_network_event(
            network_event=selected_record,
            models=models,
        )

        actual_category = category

        predicted_category = (
            prediction_result[
                "attack_category"
            ]
        )

        detection_status = (
            prediction_result[
                "detection_status"
            ]
        )

        anomaly_status = (
            prediction_result[
                "anomaly_status"
            ]
        )

        risk_score = (
            prediction_result[
                "risk_score"
            ]
        )

        risk_level = (
            prediction_result[
                "risk_level"
            ]
        )

        if predicted_category == actual_category:

            category_status = "Correct"

        else:

            category_status = "Incorrect"

        print(
            f"{actual_category:<18} "
            f"-> "
            f"{str(predicted_category):<18} "
            f"| Detection: {detection_status:<6} "
            f"| Anomaly: {anomaly_status:<9} "
            f"| Priority: {risk_score:<3} "
            f"| Risk: {risk_level:<8} "
            f"| {category_status}"
        )

        results.append(
            {
                "Actual Category": actual_category,
                "Predicted Category": predicted_category,
                "Detection": detection_status,
                "Anomaly": anomaly_status,
                "Threat Priority": risk_score,
                "Risk": risk_level,
                "Status": category_status,
            }
        )

    return pd.DataFrame(
        results
    )


# =============================================================================
# DEMONSTRATION SUMMARY
# =============================================================================

def display_demo_summary(
    results: pd.DataFrame,
) -> None:
    """
    Display summary statistics for the category demonstration.
    """

    valid_results = results[
        results["Status"].isin(
            [
                "Correct",
                "Incorrect",
            ]
        )
    ].copy()

    if valid_results.empty:

        print()
        print(
            "No valid category results available."
        )

        return

    total_categories = len(
        valid_results
    )

    correct_categories = int(
        (
            valid_results["Status"]
            == "Correct"
        ).sum()
    )

    category_accuracy = (
        correct_categories
        / total_categories
        * 100
    )

    print()
    print("=" * 80)
    print(
        "MULTI-CATEGORY DEMONSTRATION SUMMARY"
    )
    print("=" * 80)

    print()

    print(
        f"Categories tested       : "
        f"{total_categories}"
    )

    print(
        f"Correct category        : "
        f"{correct_categories}"
    )

    print(
        f"Incorrect category      : "
        f"{total_categories - correct_categories}"
    )

    print(
        f"Demo category accuracy  : "
        f"{category_accuracy:.2f}%"
    )

    print()

    print("-" * 80)

    print("NOTE:")

    print(
        "This demonstration uses one real record from each "
        "attack category."
    )

    print(
        "It is NOT the official multiclass evaluation."
    )

    print(
        "Official evaluation uses the complete UNSW-NB15 "
        "testing dataset."
    )

    print("=" * 80)


# =============================================================================
# MAIN REAL-DATA DEMONSTRATION
# =============================================================================

def run_real_dataset_demo() -> None:
    """
    Load the official UNSW-NB15 testing dataset and run one real event from
    each known attack category through the complete pipeline.
    """

    if not TESTING_DATASET_PATH.exists():

        raise FileNotFoundError(
            "UNSW-NB15 testing dataset was not found:\n"
            f"{TESTING_DATASET_PATH}"
        )

    print()
    print("=" * 80)
    print("UNSW-NB15 REAL-DATA PIPELINE TEST")
    print("=" * 80)

    dataset = pd.read_csv(
        TESTING_DATASET_PATH
    )

    print(
        f"\nDataset shape: "
        f"{dataset.shape}"
    )

    models = load_models()

    print()
    print("Model input feature contract:")

    required_features = sorted(
        get_required_features(
            models
        )
    )

    print(
        f"  Required features: "
        f"{len(required_features)}"
    )

    for feature in required_features:

        print(
            f"    - {feature}"
        )

    results = run_attack_category_demo(
        dataset=dataset,
        models=models,
    )

    print()
    print("=" * 80)
    print("RESULT TABLE")
    print("=" * 80)

    print()

    print(
        results.to_string(
            index=False
        )
    )

    display_demo_summary(
        results
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_real_dataset_demo()