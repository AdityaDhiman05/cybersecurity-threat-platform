"""
================================================================================
CYBERSECURITY THREAT PLATFORM
REST API SERVER
================================================================================

Purpose
-------
Provides the REST API for:

    - ML prediction
    - Risk assessment
    - Prediction history
    - Analyst feedback
    - Platform statistics
    - Database health
    - Model information

Security Controls
-----------------
    - API-key authentication for protected endpoints
    - JSON-only request validation
    - Request-size limitation
    - Model-based feature validation
    - Finite numeric validation
    - Generic internal-error responses
    - No internal exception details exposed to API clients

T10 Analyst Feedback
--------------------
The server is the authority for feedback correctness.

The analyst supplies actual ground truth:

    actual_result
    actual_attack_category

The server calculates:

    binary_correct
    category_correct

The client cannot override the server-calculated values.

Training eligibility remains disabled. Analyst feedback is NOT automatically
used for model retraining.
================================================================================
"""

from __future__ import annotations


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import os
import sys

from pathlib import Path
from typing import Any


# =============================================================================
# PROJECT PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

from flask import Flask, jsonify, request


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.prediction_pipeline import (
    ValidationError,
    get_required_features,
    load_models,
    predict_network_event,
)

from database.database_manager import (
    database_manager,
)


# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)


# =============================================================================
# API SECURITY CONFIGURATION
# =============================================================================

# Network prediction requests are small.
# 64 KB is sufficient while preventing unnecessarily large request bodies.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


# API key is supplied through the environment:
#
# Windows PowerShell:
#
#     $env:THREAT_PLATFORM_API_KEY="your-secret-key"
#
API_KEY_ENVIRONMENT_VARIABLE = "THREAT_PLATFORM_API_KEY"


# =============================================================================
# LOAD MACHINE-LEARNING MODELS
# =============================================================================

print("=" * 80)
print("CYBERSECURITY THREAT PLATFORM API")
print("=" * 80)

print()
print("Loading trained machine-learning models...")
print()

MODELS = load_models()

if not MODELS:
    raise RuntimeError(
        "No machine-learning models were loaded."
    )

print()
print("All trained models loaded successfully.")
print()

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Models path  : {PROJECT_ROOT / 'models'}"
)

print(
    f"Database     : {database_manager.database_path}"
)

print(
    "Dataset      : UNSW-NB15"
)

print("=" * 80)


# =============================================================================
# API AUTHENTICATION HELPERS
# =============================================================================

def get_configured_api_key() -> str | None:
    """
    Read the configured API key from the environment.
    """

    api_key = os.getenv(
        API_KEY_ENVIRONMENT_VARIABLE
    )

    if api_key is None:
        return None

    api_key = api_key.strip()

    if not api_key:
        return None

    return api_key


def is_api_key_valid() -> bool:
    """
    Validate the X-API-Key request header.

    Returns True only when:

        1. Server API key is configured.
        2. Request contains X-API-Key.
        3. Both values match exactly.
    """

    configured_api_key = (
        get_configured_api_key()
    )

    request_api_key = request.headers.get(
        "X-API-Key"
    )

    if configured_api_key is None:
        return False

    if request_api_key is None:
        return False

    return request_api_key == configured_api_key


def require_api_key() -> Any:
    """
    Enforce API-key authentication.

    Returns a Flask 401 response when authentication fails.
    Otherwise returns None.
    """

    if not is_api_key_valid():

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unauthorized.",
                }
            ),
            401,
        )

    return None


# =============================================================================
# REQUEST VALIDATION
# =============================================================================

def require_json_object() -> tuple[
    dict[str, Any] | None,
    Any,
]:
    """
    Validate that the request contains a JSON object.

    Returns
    -------
    tuple
        (request_data, error_response)

        request_data:
            Parsed dictionary when valid.

        error_response:
            Flask response when invalid.
    """

    if not request.is_json:

        return (
            None,
            (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Request content type must be "
                            "application/json."
                        ),
                    }
                ),
                400,
            ),
        )

    request_data = request.get_json(
        silent=True
    )

    if request_data is None:

        return (
            None,
            (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Request body must contain "
                            "valid JSON."
                        ),
                    }
                ),
                400,
            ),
        )

    if not isinstance(
        request_data,
        dict,
    ):

        return (
            None,
            (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "JSON request body must be "
                            "an object."
                        ),
                    }
                ),
                400,
            ),
        )

    return (
        request_data,
        None,
    )


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.route(
    "/",
    methods=["GET"],
)
def root() -> Any:
    """
    Return basic API information.
    """

    return jsonify(
        {
            "success": True,
            "service": (
                "Cybersecurity Threat Platform API"
            ),
            "version": "1.0.0",
            "dataset": "UNSW-NB15",
            "endpoints": {
                "health": "/api/v1/health",
                "models": "/api/v1/models",
                "predict": "/api/v1/predict",
                "history": "/api/v1/history",
                "feedback": "/api/v1/feedback",
                "stats": "/api/v1/stats",
            },
        }
    )


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================

@app.route(
    "/api/v1/health",
    methods=["GET"],
)
def health_check() -> Any:
    """
    Return API, database and model health.

    This endpoint remains public so monitoring systems can check whether
    the service is alive.
    """

    try:

        # ---------------------------------------------------------------------
        # MODEL HEALTH
        # ---------------------------------------------------------------------

        models_loaded = (
            isinstance(MODELS, dict)
            and bool(MODELS)
        )

        if not models_loaded:

            return (
                jsonify(
                    {
                        "success": False,
                        "status": "unhealthy",
                        "database": "unknown",
                        "models": "not_loaded",
                        "error": (
                            "Machine-learning models "
                            "are unavailable."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # DATABASE HEALTH
        # ---------------------------------------------------------------------

        health = (
            database_manager.health_check()
        )

        if health["status"] != "operational":

            return (
                jsonify(
                    {
                        "success": False,
                        "status": "unhealthy",
                        "database": "error",
                        "models": "loaded",
                        "error": (
                            "Database unavailable."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------------------

        return jsonify(
            {
                "success": True,
                "status": "healthy",
                "service": (
                    "Cybersecurity Threat Platform API"
                ),
                "database": "operational",
                "models": "loaded",
                "dataset": "UNSW-NB15",
                "prediction_records": health.get(
                    "records",
                    0,
                ),
                "feedback_records": health.get(
                    "feedback_records",
                    0,
                ),
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "status": "unhealthy",
                    "error": (
                        "Service health check failed."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# MODEL INFORMATION ENDPOINT
# =============================================================================

@app.route(
    "/api/v1/models",
    methods=["GET"],
)
def model_information() -> Any:
    """
    Return information about the ML components.

    Authentication is required.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        required_features = sorted(
            get_required_features(
                MODELS
            )
        )

        return jsonify(
            {
                "success": True,
                "dataset": "UNSW-NB15",
                "models": {
                    "binary_classifier": {
                        "type": "Random Forest",
                        "purpose": (
                            "Known attack detection"
                        ),
                    },
                    "attack_classifier": {
                        "type": "Random Forest",
                        "purpose": (
                            "Known attack-category classification"
                        ),
                    },
                    "anomaly_detector": {
                        "type": "Isolation Forest",
                        "purpose": (
                            "Anomalous network behavior detection"
                        ),
                    },
                    "risk_assessment": {
                        "type": "Rule-based risk aggregation",
                        "purpose": (
                            "Risk scoring and severity assessment"
                        ),
                    },
                    "analyst_feedback": {
                        "type": "Human-in-the-loop feedback",
                        "purpose": (
                            "Investigation feedback and "
                            "controlled future evaluation"
                        ),
                    },
                },
                "required_input_features": (
                    required_features
                ),
                "required_feature_count": (
                    len(required_features)
                ),
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Failed to retrieve model information."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# PREDICTION ENDPOINT
# =============================================================================

@app.route(
    "/api/v1/predict",
    methods=["POST"],
)
def predict() -> Any:
    """
    Run the complete cybersecurity prediction pipeline.

    Accepted request formats:

        {
            "event": {
                ...
            }
        }

    or:

        {
            "features": {
                ...
            }
        }

    or directly:

        {
            ...
        }

    Authentication:
        X-API-Key header is required.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        # ---------------------------------------------------------------------
        # READ JSON
        # ---------------------------------------------------------------------

        request_data, request_error = (
            require_json_object()
        )

        if request_error is not None:
            return request_error

        # ---------------------------------------------------------------------
        # EXTRACT NETWORK EVENT
        # ---------------------------------------------------------------------

        if "event" in request_data:

            network_event = request_data[
                "event"
            ]

        elif "features" in request_data:

            network_event = request_data[
                "features"
            ]

        else:

            network_event = request_data

        # ---------------------------------------------------------------------
        # VALIDATE NETWORK EVENT
        # ---------------------------------------------------------------------

        if not isinstance(
            network_event,
            dict,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Network event must be "
                            "a JSON object."
                        ),
                    }
                ),
                400,
            )

        if not network_event:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Network event cannot be empty."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # RUN ML PIPELINE
        # ---------------------------------------------------------------------

        try:

            prediction_result = (
                predict_network_event(
                    network_event,
                    MODELS,
                )
            )

        except ValidationError as exc:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(exc),
                    }
                ),
                400,
            )

        except (
            ValueError,
            TypeError,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Network event contains "
                            "invalid feature values."
                        ),
                    }
                ),
                400,
            )

        except Exception:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Prediction pipeline failed."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # VALIDATE RESULT
        # ---------------------------------------------------------------------

        if not isinstance(
            prediction_result,
            dict,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Prediction pipeline returned "
                            "an invalid result."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # SOURCE
        # ---------------------------------------------------------------------

        source = request_data.get(
            "source",
            "API",
        )

        if source is None:
            source = "API"

        if not isinstance(
            source,
            str,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "source must be a string."
                        ),
                    }
                ),
                400,
            )

        source = source.strip()

        if not source:
            source = "API"

        # ---------------------------------------------------------------------
        # SAVE PREDICTION
        # ---------------------------------------------------------------------

        try:

            prediction_id = (
                database_manager.save_prediction(
                    prediction=prediction_result,
                    network_event=network_event,
                    source=source,
                )
            )

        except Exception:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Prediction completed but "
                            "database storage failed."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # RETURN RESULT
        # ---------------------------------------------------------------------

        return jsonify(
            {
                "success": True,
                "service": (
                    "Cybersecurity Threat Platform API"
                ),
                "dataset": source,
                "database_record_id": prediction_id,
                "prediction_id": prediction_id,
                "prediction": prediction_result,
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unexpected API error.",
                }
            ),
            500,
        )


# =============================================================================
# PREDICTION HISTORY
# =============================================================================

@app.route(
    "/api/v1/history",
    methods=["GET"],
)
def prediction_history() -> Any:
    """
    Return recent prediction history.

    Authentication:
        X-API-Key header is required.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        limit_parameter = request.args.get(
            "limit",
            default="20",
        )

        try:

            limit = int(
                limit_parameter
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Limit must be an integer."
                        ),
                    }
                ),
                400,
            )

        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        records = (
            database_manager
            .get_recent_predictions(
                limit=limit
            )
        )

        return jsonify(
            {
                "success": True,
                "count": len(records),
                "records": records,
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Failed to retrieve "
                        "prediction history."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# ANALYST FEEDBACK
# =============================================================================

@app.route(
    "/api/v1/feedback",
    methods=["POST"],
)
def submit_feedback() -> Any:
    """
    Store analyst feedback.

    T10 IMPORTANT RULE
    ------------------
    The server calculates correctness.

    The client provides:

        prediction_id
        actual_result
        actual_attack_category

    The server calculates:

        binary_correct
        category_correct

    This prevents the dashboard/client from falsely declaring a prediction
    correct or incorrect.

    Rules
    -----
    Binary correctness:

        predicted Attack + actual Attack
            -> Yes

        predicted Normal + actual Normal
            -> Yes

        mismatch
            -> No

        actual Unknown
            -> Unknown

    Category correctness:

        actual Normal
            -> None / N-A

        actual Unknown
            -> Unknown

        actual Attack + category provided
            -> compare with predicted category

        actual Attack + category missing
            -> Unknown

    Contradictory feedback is rejected.

    Example contradiction:

        actual_result = Normal
        actual_attack_category = Exploits

    This is invalid because Normal traffic has no attack category.

    Training eligibility is always False.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        # ---------------------------------------------------------------------
        # READ JSON
        # ---------------------------------------------------------------------

        request_data, request_error = (
            require_json_object()
        )

        if request_error is not None:
            return request_error

        # ---------------------------------------------------------------------
        # PREDICTION ID
        # ---------------------------------------------------------------------

        if "prediction_id" not in request_data:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "prediction_id is required."
                        ),
                    }
                ),
                400,
            )

        try:

            prediction_id = int(
                request_data[
                    "prediction_id"
                ]
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "prediction_id must be an integer."
                        ),
                    }
                ),
                400,
            )

        if prediction_id <= 0:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "prediction_id must be "
                            "greater than zero."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # RETRIEVE STORED PREDICTION
        # ---------------------------------------------------------------------

        prediction = (
            database_manager.get_prediction(
                prediction_id
            )
        )

        if prediction is None:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Prediction not found.",
                    }
                ),
                404,
            )

        predicted_detection_status = (
            prediction.get(
                "detection_status"
            )
        )

        predicted_attack_category = (
            prediction.get(
                "attack_category"
            )
        )

        # ---------------------------------------------------------------------
        # ACTUAL RESULT
        # ---------------------------------------------------------------------

        actual_result = request_data.get(
            "actual_result"
        )

        if actual_result not in {
            "Attack",
            "Normal",
            "Unknown",
        }:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "actual_result must be "
                            "Attack, Normal, or Unknown."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # ACTUAL ATTACK CATEGORY
        # ---------------------------------------------------------------------

        actual_attack_category = (
            request_data.get(
                "actual_attack_category"
            )
        )

        if actual_attack_category is not None:

            if not isinstance(
                actual_attack_category,
                str,
            ):

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "actual_attack_category must "
                                "be a string or null."
                            ),
                        }
                    ),
                    400,
                )

            actual_attack_category = (
                actual_attack_category.strip()
            )

            if not actual_attack_category:

                actual_attack_category = None

        # ---------------------------------------------------------------------
        # CONTRADICTION CHECK
        # ---------------------------------------------------------------------

        if (
            actual_result == "Normal"
            and actual_attack_category is not None
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Contradictory feedback: "
                            "actual_result=Normal cannot have "
                            "an attack category."
                        ),
                    }
                ),
                400,
            )

        if (
            actual_result == "Unknown"
            and actual_attack_category is not None
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Contradictory feedback: "
                            "actual_result=Unknown cannot have "
                            "an attack category."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # SERVER-SIDE BINARY CORRECTNESS
        # ---------------------------------------------------------------------

        if actual_result == "Unknown":

            binary_correct = "Unknown"

        elif predicted_detection_status in {
            "Attack",
            "Normal",
        }:

            if (
                predicted_detection_status
                == actual_result
            ):

                binary_correct = "Yes"

            else:

                binary_correct = "No"

        else:

            binary_correct = "Unknown"

        # ---------------------------------------------------------------------
        # SERVER-SIDE CATEGORY CORRECTNESS
        # ---------------------------------------------------------------------

        if actual_result == "Normal":

            # Category classification does not apply to normal traffic.
            category_correct = None

        elif actual_result == "Unknown":

            category_correct = "Unknown"

        elif actual_result == "Attack":

            if actual_attack_category is None:

                category_correct = "Unknown"

            elif predicted_attack_category is None:

                category_correct = "No"

            else:

                predicted_category_normalized = (
                    str(
                        predicted_attack_category
                    ).strip().lower()
                )

                actual_category_normalized = (
                    str(
                        actual_attack_category
                    ).strip().lower()
                )

                if (
                    predicted_category_normalized
                    == actual_category_normalized
                ):

                    category_correct = "Yes"

                else:

                    category_correct = "No"

        else:

            category_correct = "Unknown"

        # ---------------------------------------------------------------------
        # LEGACY ANALYST VERDICT
        # ---------------------------------------------------------------------
        #
        # This field remains because it already exists in the database.
        #
        # It does NOT determine correctness.
        # ---------------------------------------------------------------------

        supplied_analyst_verdict = (
            request_data.get(
                "analyst_verdict"
            )
        )

        if supplied_analyst_verdict is not None:

            if supplied_analyst_verdict not in {
                "Correct",
                "Incorrect",
                "Uncertain",
            }:

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "analyst_verdict must be "
                                "Correct, Incorrect, or Uncertain."
                            ),
                        }
                    ),
                    400,
                )

            analyst_verdict = (
                supplied_analyst_verdict
            )

        else:

            if binary_correct == "Yes":

                analyst_verdict = "Correct"

            elif binary_correct == "No":

                analyst_verdict = "Incorrect"

            else:

                analyst_verdict = "Uncertain"

        # ---------------------------------------------------------------------
        # OPTIONAL CLIENT CORRECTNESS VALIDATION
        # ---------------------------------------------------------------------
        #
        # Older/newer clients may send these fields.
        #
        # We never trust them.
        #
        # If supplied, they must agree with the server calculation.
        # ---------------------------------------------------------------------

        supplied_binary_correct = (
            request_data.get(
                "binary_correct"
            )
        )

        if (
            supplied_binary_correct is not None
            and supplied_binary_correct
            != binary_correct
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "binary_correct does not match "
                            "server-calculated correctness."
                        ),
                    }
                ),
                400,
            )

        supplied_category_correct = (
            request_data.get(
                "category_correct"
            )
        )

        if (
            supplied_category_correct is not None
            and supplied_category_correct
            != category_correct
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "category_correct does not match "
                            "server-calculated correctness."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # UNKNOWN ATTACK FLAG
        # ---------------------------------------------------------------------

        is_unknown_attack = request_data.get(
            "is_unknown_attack",
            False,
        )

        if not isinstance(
            is_unknown_attack,
            bool,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "is_unknown_attack must be "
                            "true or false."
                        ),
                    }
                ),
                400,
            )

        # ---------------------------------------------------------------------
        # OPTIONAL ROOT CAUSE
        # ---------------------------------------------------------------------

        root_cause = request_data.get(
            "root_cause"
        )

        if root_cause is not None:

            if not isinstance(
                root_cause,
                str,
            ):

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "root_cause must be "
                                "a string or null."
                            ),
                        }
                    ),
                    400,
                )

            root_cause = root_cause.strip()

        # ---------------------------------------------------------------------
        # OPTIONAL ANALYST NOTES
        # ---------------------------------------------------------------------

        analyst_notes = request_data.get(
            "analyst_notes"
        )

        if analyst_notes is not None:

            if not isinstance(
                analyst_notes,
                str,
            ):

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "analyst_notes must be "
                                "a string or null."
                            ),
                        }
                    ),
                    400,
                )

            analyst_notes = analyst_notes.strip()

        # ---------------------------------------------------------------------
        # SAVE FEEDBACK
        # ---------------------------------------------------------------------
        #
        # training_eligible is deliberately forced to False.
        # ---------------------------------------------------------------------

        try:

            feedback_id = (
                database_manager.save_feedback(
                    prediction_id=prediction_id,
                    analyst_verdict=analyst_verdict,
                    actual_result=actual_result,
                    actual_attack_category=(
                        actual_attack_category
                    ),
                    binary_correct=binary_correct,
                    category_correct=category_correct,
                    is_unknown_attack=(
                        is_unknown_attack
                    ),
                    root_cause=root_cause,
                    analyst_notes=analyst_notes,
                    training_eligible=False,
                )
            )

        except ValueError as exc:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(exc),
                    }
                ),
                400,
            )

        except Exception:

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Failed to store "
                            "analyst feedback."
                        ),
                    }
                ),
                500,
            )

        # ---------------------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------------------

        return (
            jsonify(
                {
                    "success": True,
                    "message": (
                        "Analyst feedback stored successfully."
                    ),
                    "feedback_id": feedback_id,
                    "prediction_id": prediction_id,
                    "binary_correct": binary_correct,
                    "category_correct": category_correct,
                    "training_eligible": False,
                }
            ),
            201,
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Unexpected feedback API error."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# ANALYST FEEDBACK - GET
# =============================================================================

@app.route(
    "/api/v1/feedback",
    methods=["GET"],
)
def feedback_history() -> Any:
    """
    Return analyst feedback.

    Optional query parameter:

        ?prediction_id=<ID>

    Authentication:
        X-API-Key header is required.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        prediction_id_parameter = (
            request.args.get(
                "prediction_id"
            )
        )

        # ---------------------------------------------------------------------
        # SINGLE PREDICTION FEEDBACK
        # ---------------------------------------------------------------------

        if prediction_id_parameter is not None:

            try:

                prediction_id = int(
                    prediction_id_parameter
                )

            except (
                TypeError,
                ValueError,
            ):

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "prediction_id must "
                                "be an integer."
                            ),
                        }
                    ),
                    400,
                )

            if prediction_id <= 0:

                return (
                    jsonify(
                        {
                            "success": False,
                            "error": (
                                "prediction_id must be "
                                "greater than zero."
                            ),
                        }
                    ),
                    400,
                )

            feedback = (
                database_manager
                .get_feedback_for_prediction(
                    prediction_id
                )
            )

            return jsonify(
                {
                    "success": True,
                    "count": (
                        1
                        if feedback is not None
                        else 0
                    ),
                    "records": (
                        [feedback]
                        if feedback is not None
                        else []
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # RECENT FEEDBACK
        # ---------------------------------------------------------------------

        limit_parameter = request.args.get(
            "limit",
            default="20",
        )

        try:

            limit = int(
                limit_parameter
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                jsonify(
                    {
                        "success": False,
                        "error": (
                            "Limit must be an integer."
                        ),
                    }
                ),
                400,
            )

        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        records = (
            database_manager
            .get_recent_feedback(
                limit=limit
            )
        )

        return jsonify(
            {
                "success": True,
                "count": len(records),
                "records": records,
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Failed to retrieve "
                        "analyst feedback."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# PLATFORM STATISTICS
# =============================================================================

@app.route(
    "/api/v1/stats",
    methods=["GET"],
)
def platform_statistics() -> Any:
    """
    Return aggregate platform statistics.

    The dashboard can use this endpoint for the Analyst Agreement panel.

    Authentication:
        X-API-Key header is required.
    """

    authentication_error = require_api_key()

    if authentication_error is not None:
        return authentication_error

    try:

        feedback_summary = (
            database_manager
            .get_feedback_summary()
        )

        risk_summary = (
            database_manager
            .get_risk_summary()
        )

        attack_summary = (
            database_manager
            .get_attack_summary()
        )

        prediction_count = (
            database_manager
            .count_predictions()
        )

        return jsonify(
            {
                "success": True,
                "prediction_count": prediction_count,
                "feedback": feedback_summary,
                "risk_distribution": risk_summary,
                "attack_distribution": attack_summary,
            }
        )

    except Exception:

        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Failed to retrieve "
                        "platform statistics."
                    ),
                }
            ),
            500,
        )


# =============================================================================
# REQUEST SIZE ERROR HANDLER
# =============================================================================

@app.errorhandler(413)
def request_entity_too_large(
    _error: Any,
) -> Any:
    """
    Return a controlled response when the request exceeds 64 KB.
    """

    return (
        jsonify(
            {
                "success": False,
                "error": (
                    "Request body is too large. "
                    "Maximum size is 64 KB."
                ),
            }
        ),
        413,
    )


# =============================================================================
# BAD REQUEST HANDLER
# =============================================================================

@app.errorhandler(400)
def bad_request(
    _error: Any,
) -> Any:
    """
    Return a generic JSON response for malformed requests.
    """

    return (
        jsonify(
            {
                "success": False,
                "error": "Bad request.",
            }
        ),
        400,
    )


# =============================================================================
# NOT FOUND HANDLER
# =============================================================================

@app.errorhandler(404)
def not_found(
    _error: Any,
) -> Any:
    """
    Return a generic JSON response for unknown routes.
    """

    return (
        jsonify(
            {
                "success": False,
                "error": "Endpoint not found.",
            }
        ),
        404,
    )


# =============================================================================
# METHOD NOT ALLOWED HANDLER
# =============================================================================

@app.errorhandler(405)
def method_not_allowed(
    _error: Any,
) -> Any:
    """
    Return a generic JSON response for unsupported HTTP methods.
    """

    return (
        jsonify(
            {
                "success": False,
                "error": "HTTP method not allowed.",
            }
        ),
        405,
    )


# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("API SERVER STARTING")
    print("=" * 80)

    print()
    print("Available endpoints:")

    print(
        "  GET  /"
    )

    print(
        "  GET  /api/v1/health"
    )

    print(
        "  GET  /api/v1/models"
    )

    print(
        "  POST /api/v1/predict"
    )

    print(
        "  GET  /api/v1/history"
    )

    print(
        "  POST /api/v1/feedback"
    )

    print(
        "  GET  /api/v1/feedback"
    )

    print(
        "  GET  /api/v1/stats"
    )

    print()

    if get_configured_api_key() is None:

        print(
            "WARNING: THREAT_PLATFORM_API_KEY is not configured."
        )

        print(
            "Protected endpoints will reject requests with HTTP 401."
        )

    else:

        print(
            "API key authentication: CONFIGURED"
        )

    print()

    print(
        "API URL: http://127.0.0.1:5000"
    )

    print()

    print(
        f"Database: {database_manager.database_path}"
    )

    print("=" * 80)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )