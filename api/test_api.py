"""
================================================================================
CYBERSECURITY THREAT PLATFORM
API INTEGRATION TEST
================================================================================

Purpose
-------
Tests the complete cybersecurity threat-detection API using one real
UNSW-NB15 testing-dataset event.

The test verifies:

1. The UNSW-NB15 testing dataset can be loaded.
2. A real network event can be extracted.
3. The event contains the expected 42 model input features.
4. The Flask API accepts the authenticated request.
5. The complete prediction pipeline executes successfully.
6. The API returns the prediction and database record ID.

Authentication
--------------
The Flask API protects the prediction endpoint with the X-API-Key header.

The API key is read from:

    THREAT_PLATFORM_API_KEY

For local development, the default value is:

    test-key-123

This matches the development key used when starting the Flask server.

Important
---------
This script does NOT train or modify any machine-learning model.
It only performs an end-to-end API integration test using an existing
UNSW-NB15 testing record.
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests


# ==============================================================================
# PROJECT PATH CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

TESTING_DATASET_PATH = (
    DATA_DIRECTORY
    / "UNSW_NB15_testing-set.csv"
)


# ==============================================================================
# API CONFIGURATION
# ==============================================================================

API_BASE_URL = (
    "http://127.0.0.1:5000"
)

PREDICTION_ENDPOINT = (
    f"{API_BASE_URL}/api/v1/predict"
)


# ==============================================================================
# API AUTHENTICATION
# ==============================================================================

API_KEY = os.getenv(
    "THREAT_PLATFORM_API_KEY",
    "test-key-123",
)

API_HEADERS = {
    "X-API-Key": API_KEY,
}


# ==============================================================================
# TEST CONFIGURATION
# ==============================================================================

# Real UNSW-NB15 testing-dataset row used for the integration test.
TEST_DATASET_INDEX = 247

# Dataset columns that must never be sent as model input features.
EXCLUDED_COLUMNS = {
    "id",
    "label",
    "attack_cat",
}


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_testing_dataset() -> pd.DataFrame:
    """
    Load the official UNSW-NB15 testing dataset.

    Returns
    -------
    pandas.DataFrame
        Complete UNSW-NB15 testing dataset.

    Raises
    ------
    FileNotFoundError
        If the testing dataset does not exist.
    """

    if not TESTING_DATASET_PATH.exists():

        raise FileNotFoundError(
            "UNSW-NB15 testing dataset was not found:\n"
            f"{TESTING_DATASET_PATH}"
        )

    dataset = pd.read_csv(
        TESTING_DATASET_PATH
    )

    return dataset


# ==============================================================================
# NETWORK EVENT PREPARATION
# ==============================================================================

def prepare_network_event(
    dataset: pd.DataFrame,
    dataset_index: int,
) -> tuple[Dict[str, Any], Any, str]:
    """
    Extract one real UNSW-NB15 event and prepare it for the API.

    Parameters
    ----------
    dataset : pandas.DataFrame
        UNSW-NB15 testing dataset.

    dataset_index : int
        Row index to use for the integration test.

    Returns
    -------
    tuple
        network_event
        actual_label
        actual_attack_category

    Notes
    -----
    The fields id, label and attack_cat are dataset metadata and are not
    model input features. They are therefore removed before sending the
    event to the API.
    """

    if dataset_index < 0 or dataset_index >= len(dataset):

        raise IndexError(
            "Requested dataset index is outside the dataset range: "
            f"{dataset_index}"
        )

    selected_row = dataset.iloc[
        dataset_index
    ]

    actual_label = selected_row.get(
        "label"
    )

    actual_attack_category = str(
        selected_row.get(
            "attack_cat",
            "Unknown",
        )
    ).strip()

    # --------------------------------------------------------------------------
    # Remove dataset metadata and target columns.
    # --------------------------------------------------------------------------

    network_event = (
        selected_row
        .drop(
            labels=list(EXCLUDED_COLUMNS),
            errors="ignore",
        )
        .to_dict()
    )

    # --------------------------------------------------------------------------
    # Convert NumPy scalar values into native Python values.
    # This makes the dictionary safely JSON serializable.
    # --------------------------------------------------------------------------

    cleaned_event: Dict[str, Any] = {}

    for feature_name, feature_value in network_event.items():

        if pd.isna(feature_value):

            cleaned_event[
                feature_name
            ] = None

        elif hasattr(
            feature_value,
            "item",
        ):

            cleaned_event[
                feature_name
            ] = feature_value.item()

        else:

            cleaned_event[
                feature_name
            ] = feature_value

    return (
        cleaned_event,
        actual_label,
        actual_attack_category,
    )


# ==============================================================================
# API REQUEST
# ==============================================================================

def send_prediction_request(
    network_event: Dict[str, Any],
) -> requests.Response:
    """
    Send the network event to the Flask prediction API.

    Parameters
    ----------
    network_event : dict
        Prepared UNSW-NB15 network event.

    Returns
    -------
    requests.Response
        HTTP response returned by the Flask API.
    """

    payload = {
        "event": network_event,
    }

    response = requests.post(
        PREDICTION_ENDPOINT,
        json=payload,
        headers=API_HEADERS,
        timeout=30,
    )

    return response


# ==============================================================================
# RESPONSE DISPLAY
# ==============================================================================

def display_success_response(
    response_data: Dict[str, Any],
) -> None:
    """
    Display the successful prediction returned by the API.
    """

    prediction = response_data.get(
        "prediction",
        {},
    )

    print()
    print("=" * 80)
    print("API TEST SUCCESSFUL")
    print("=" * 80)

    print()

    print(
        "Database record ID : "
        f"{response_data.get('database_record_id', 'N/A')}"
    )

    print()

    print("Prediction result:")

    print(
        f"  Detection status           : "
        f"{prediction.get('detection_status', 'N/A')}"
    )

    print(
        f"  Binary attack probability  : "
        f"{prediction.get('binary_attack_probability', prediction.get('attack_probability', 'N/A'))}"
    )

    print(
        f"  Attack category            : "
        f"{prediction.get('attack_category', 'N/A')}"
    )

    print(
        f"  Category probability       : "
        f"{prediction.get('attack_category_probability', 'N/A')}"
    )

    print(
        f"  Anomaly status             : "
        f"{prediction.get('anomaly_status', 'N/A')}"
    )

    print(
        f"  Anomaly score              : "
        f"{prediction.get('anomaly_score', 'N/A')}"
    )

    print(
        f"  Risk score                 : "
        f"{prediction.get('risk_score', 'N/A')}"
    )

    print(
        f"  Risk level                 : "
        f"{prediction.get('risk_level', 'N/A')}"
    )

    print()

    print(
        f"  Explanation                : "
        f"{prediction.get('explanation', 'N/A')}"
    )

    print()

    print(
        f"  Recommended action         : "
        f"{prediction.get('recommended_action', 'N/A')}"
    )

    print()
    print("=" * 80)


# ==============================================================================
# RESPONSE ERROR HANDLING
# ==============================================================================

def display_api_error(
    response: requests.Response,
) -> None:
    """
    Display an API error response in a readable format.
    """

    print()
    print("=" * 80)
    print("API TEST FAILED")
    print("=" * 80)

    print()

    try:

        response_data = response.json()

        print(
            "Error: API returned HTTP "
            f"{response.status_code}:"
        )

        print(
            response_data
        )

    except ValueError:

        print(
            "Error: API returned HTTP "
            f"{response.status_code}:"
        )

        print(
            response.text
        )

    print()


# ==============================================================================
# MAIN TEST
# ==============================================================================

def run_api_test() -> None:
    """
    Run the complete real-data API integration test.
    """

    print()
    print("=" * 80)
    print("LOADING REAL UNSW-NB15 TESTING EVENT")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------

    dataset = load_testing_dataset()

    print(
        f"Dataset shape: {dataset.shape}"
    )

    # --------------------------------------------------------------------------
    # Prepare selected event
    # --------------------------------------------------------------------------

    (
        network_event,
        actual_label,
        actual_attack_category,
    ) = prepare_network_event(
        dataset=dataset,
        dataset_index=TEST_DATASET_INDEX,
    )

    print()
    print("Selected test event:")

    print(
        f"  Dataset index       : "
        f"{TEST_DATASET_INDEX}"
    )

    print(
        f"  Actual label        : "
        f"{actual_label}"
    )

    print(
        f"  Actual attack class : "
        f"{actual_attack_category}"
    )

    print()
    print(
        "Network features sent: "
        f"{len(network_event)}"
    )

    # --------------------------------------------------------------------------
    # Verify the expected 42-feature event.
    # --------------------------------------------------------------------------

    if len(network_event) != 42:

        raise ValueError(
            "Unexpected number of network features. "
            f"Expected 42, received {len(network_event)}."
        )

    # --------------------------------------------------------------------------
    # Send event to API
    # --------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "SENDING EVENT TO CYBERSECURITY THREAT PLATFORM API"
    )
    print("=" * 80)

    try:

        response = send_prediction_request(
            network_event
        )

    except requests.ConnectionError as exc:

        print()
        print("=" * 80)
        print("API TEST FAILED")
        print("=" * 80)

        print()

        print(
            "Could not connect to the Flask API."
        )

        print(
            "Make sure the API server is running at:"
        )

        print(
            API_BASE_URL
        )

        print()

        print(
            f"Error: {exc}"
        )

        return

    except requests.Timeout as exc:

        print()
        print("=" * 80)
        print("API TEST FAILED")
        print("=" * 80)

        print()

        print(
            "The API request timed out."
        )

        print(
            f"Error: {exc}"
        )

        return

    except requests.RequestException as exc:

        print()
        print("=" * 80)
        print("API TEST FAILED")
        print("=" * 80)

        print()

        print(
            f"Request error: {exc}"
        )

        return

    # --------------------------------------------------------------------------
    # Process response
    # --------------------------------------------------------------------------

    if response.ok:

        try:

            response_data = response.json()

        except ValueError:

            print()
            print("=" * 80)
            print("API TEST FAILED")
            print("=" * 80)

            print()

            print(
                "The API returned a successful HTTP status "
                "but the response was not valid JSON."
            )

            print()
            print(
                response.text
            )

            return

        display_success_response(
            response_data
        )

        return

    # --------------------------------------------------------------------------
    # API returned an error.
    # --------------------------------------------------------------------------

    display_api_error(
        response
    )


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    try:

        run_api_test()

    except Exception as exc:

        print()
        print("=" * 80)
        print("API TEST FAILED")
        print("=" * 80)

        print()

        print(
            f"Error: {exc}"
        )