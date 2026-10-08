"""
================================================================================
CYBERSECURITY THREAT PLATFORM
LIGHT + COLORFUL SOC DASHBOARD
================================================================================

Dashboard
---------
Professional Streamlit interface for the Cybersecurity Threat Platform.

Backend
-------
Flask REST API
SQLite
Random Forest Binary Classifier
Random Forest Attack Classifier
Isolation Forest
Risk Assessment

Important
---------
The dashboard does not train models.
The dashboard does not modify ML models.
Analyst feedback does not automatically retrain models.
The platform does not claim zero-day detection.
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:5000",
)

API_KEY = os.getenv(
    "THREAT_PLATFORM_API_KEY",
    "test-key-123",
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Cybersecurity Threat Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# COLOR PALETTE
# =============================================================================

BLUE = "#2F80ED"
RED = "#FF5B5B"
GREEN = "#20B26B"
ORANGE = "#FF9F43"
PURPLE = "#8E5DE7"
TEAL = "#18B7A0"
YELLOW = "#F2C94C"

DARK = "#172B4D"
MEDIUM = "#526581"
LIGHT = "#71839A"

BACKGROUND = "#F5F8FC"
WHITE = "#FFFFFF"
BORDER = "#DFE7F0"


# =============================================================================
# ATTACK CATEGORY COLORS
# =============================================================================

CATEGORY_COLORS = {
    "Generic": BLUE,
    "Exploits": RED,
    "Fuzzers": YELLOW,
    "DoS": PURPLE,
    "Reconnaissance": TEAL,
    "Analysis": "#4D9DE0",
    "Backdoor": ORANGE,
    "Shellcode": "#6C5CE7",
    "Worms": "#E85AAD",
}


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

MODEL_PERFORMANCE = {
    "Binary Attack Detection": {
        "Accuracy": "87.06%",
        "Attack Recall": "98.61%",
        "ROC-AUC": "0.9792",
    },
    "Known Attack Classification": {
        "Accuracy": "74.29%",
        "Balanced Accuracy": "62.52%",
        "Macro F1": "52.01%",
    },
    "Anomaly Detection": {
        "Accuracy": "71.48%",
        "Attack Recall": "56.84%",
        "Attack Precision": "86.81%",
    },
}


# =============================================================================
# SESSION STATE
# =============================================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

if "selected_prediction_id" not in st.session_state:
    st.session_state.selected_prediction_id = None

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "latest_api_response" not in st.session_state:
    st.session_state.latest_api_response = None

if "latest_event" not in st.session_state:
    st.session_state.latest_event = None

if "latest_actual_label" not in st.session_state:
    st.session_state.latest_actual_label = None

if "latest_actual_category" not in st.session_state:
    st.session_state.latest_actual_category = None


# =============================================================================
# STREAMLIT CSS
# =============================================================================
#
# IMPORTANT:
# We ONLY use CSS here.
#
# There is NO custom HTML dashboard markup anymore.
#
# This prevents the HTML-code-block problem visible in your screenshot.
# =============================================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------
       GLOBAL BACKGROUND
       ------------------------------------------------------------ */

    .stApp {
        background-color: #F5F8FC;
    }

    .main {
        background-color: #F5F8FC;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }


    /* ------------------------------------------------------------
       REMOVE STREAMLIT TOP HEADER / BLUE-DARK BAR
       ------------------------------------------------------------ */

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0rem !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }


    /* ------------------------------------------------------------
       TYPOGRAPHY
       ------------------------------------------------------------ */

    h1 {
        color: #172B4D !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
    }

    h2,
    h3,
    h4 {
        color: #172B4D !important;
        font-weight: 750 !important;
    }

    p {
        color: #526581;
    }


    /* ------------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------------ */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #DFE7F0 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #526581 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #172B4D !important;
    }

    section[data-testid="stSidebar"] span {
        color: #172B4D !important;
    }


    /* ------------------------------------------------------------
       METRIC CARDS
       ------------------------------------------------------------ */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DFE7F0 !important;
        border-radius: 13px !important;
        padding: 1rem !important;
        min-height: 110px !important;
        box-shadow: 0 5px 18px rgba(35, 65, 100, 0.06) !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #526581 !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172B4D !important;
        font-weight: 800 !important;
    }


    /* ------------------------------------------------------------
       BUTTONS
       ------------------------------------------------------------ */

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: 1px solid #B9D5F7 !important;
        color: #1677FF !important;
        background-color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background-color: #EEF6FF !important;
        border-color: #1677FF !important;
    }


    /* ------------------------------------------------------------
       PRIMARY BUTTON
       ------------------------------------------------------------ */

    button[kind="primary"] {
        background-color: #1677FF !important;
        color: #FFFFFF !important;
        border-color: #1677FF !important;
    }


    /* ------------------------------------------------------------
       INPUTS
       ------------------------------------------------------------ */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #D8E2ED !important;
        border-radius: 8px !important;
    }

    input,
    textarea {
        background-color: #FFFFFF !important;
        color: #172B4D !important;
        border-color: #D8E2ED !important;
    }


    /* ------------------------------------------------------------
       DATAFRAME
       ------------------------------------------------------------ */

    div[data-testid="stDataFrame"] {
        border: 1px solid #DFE7F0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }


    /* ------------------------------------------------------------
       EXPANDERS
       ------------------------------------------------------------ */

    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DFE7F0 !important;
        border-radius: 10px !important;
    }


    /* ------------------------------------------------------------
       PROGRESS BAR
       ------------------------------------------------------------ */

    div[data-testid="stProgress"] > div > div > div {
        background-color: #2F80ED !important;
    }


    /* ------------------------------------------------------------
       ALERTS
       ------------------------------------------------------------ */

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }


    /* ------------------------------------------------------------
       DIVIDERS
       ------------------------------------------------------------ */

    hr {
        border-color: #DFE7F0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# API FUNCTIONS
# =============================================================================

def api_headers():
    """Return API authentication headers."""

    return {
        "X-API-Key": API_KEY,
    }


def api_get(
    endpoint,
    timeout=8,
):
    """Perform authenticated GET request."""

    try:

        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            headers=api_headers(),
            timeout=timeout,
        )

        try:
            data = response.json()

        except ValueError:
            data = {}

        if response.ok:
            return data

    except requests.RequestException:
        return None

    return None


def api_post(
    endpoint,
    payload,
    timeout=30,
):
    """Perform authenticated POST request."""

    try:

        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            headers={
                **api_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )

        try:
            data = response.json()

        except ValueError:
            data = {}

        if response.ok:

            return (
                True,
                data,
                None,
            )

        return (
            False,
            data,
            data.get(
                "error",
                f"HTTP {response.status_code}",
            ),
        )

    except requests.RequestException as exc:

        return (
            False,
            {},
            str(exc),
        )


# =============================================================================
# DATASET
# =============================================================================

@st.cache_data
def load_dataset():
    """Load official UNSW-NB15 testing dataset."""

    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            DATA_PATH
        )

    except Exception:

        return pd.DataFrame()


dataset = load_dataset()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_int(
    value,
    default=0,
):
    """Safely convert to integer."""

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(
    value,
    default=0.0,
):
    """Safely convert to float."""

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def find_stat(
    source,
    keys,
    default=0,
):
    """Find a statistic using multiple possible key names."""

    if not isinstance(
        source,
        dict,
    ):
        return default

    for key in keys:

        if key in source:
            return source[key]

    return default


def get_records(
    response,
    possible_keys,
):
    """Extract records from an API response."""

    if not isinstance(
        response,
        dict,
    ):
        return []

    for key in possible_keys:

        value = response.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return value

    return []


def format_timestamp(
    value,
):
    """Format timestamp for display."""

    if value is None:
        return "—"

    try:

        timestamp = pd.to_datetime(
            value
        )

        return timestamp.strftime(
            "%d %b %Y · %H:%M"
        )

    except Exception:

        return str(value)


# =============================================================================
# API STATUS
# =============================================================================

health_response = api_get(
    "/api/v1/health"
)

api_online = (
    health_response is not None
)


# =============================================================================
# API DATA
# =============================================================================

if api_online:

    stats_response = (
        api_get(
            "/api/v1/stats"
        )
        or {}
    )

    history_response = (
        api_get(
            "/api/v1/history"
        )
        or {}
    )

    feedback_response = (
        api_get(
            "/api/v1/feedback"
        )
        or {}
    )

else:

    stats_response = {}
    history_response = {}
    feedback_response = {}


history_records = get_records(
    history_response,
    [
        "records",
        "history",
        "data",
        "predictions",
    ],
)


feedback_records = get_records(
    feedback_response,
    [
        "records",
        "feedback",
        "data",
    ],
)


# =============================================================================
# DATASET STATISTICS
# =============================================================================

if dataset.empty:

    total_test_events = 0
    attack_test_events = 0
    normal_test_events = 0
    category_counts = {}

else:

    total_test_events = len(
        dataset
    )

    if "label" in dataset.columns:

        attack_test_events = int(
            (
                dataset["label"] == 1
            ).sum()
        )

        normal_test_events = int(
            (
                dataset["label"] == 0
            ).sum()
        )

    else:

        attack_test_events = 0
        normal_test_events = 0


    if "attack_cat" in dataset.columns:

        category_series = (
            dataset["attack_cat"]
            .astype(str)
            .str.strip()
        )

        category_series = category_series[
            ~category_series.str.lower().isin(
                [
                    "normal",
                    "nan",
                    "",
                ]
            )
        ]

        category_counts = (
            category_series
            .value_counts()
            .to_dict()
        )

    else:

        category_counts = {}


# =============================================================================
# RISK STATISTICS
# =============================================================================

risk_summary = stats_response.get(
    "risk_summary",
    {},
)


high_count = safe_int(
    find_stat(
        risk_summary,
        [
            "HIGH",
            "high",
            "high_risk",
        ],
        0,
    )
)


critical_count = safe_int(
    find_stat(
        risk_summary,
        [
            "CRITICAL",
            "critical",
            "critical_risk",
        ],
        0,
    )
)


# Fallback to actual prediction history.

if (
    high_count == 0
    and critical_count == 0
    and history_records
):

    high_count = sum(
        1
        for record in history_records
        if str(
            record.get(
                "risk_level",
                "",
            )
        ).upper()
        == "HIGH"
    )

    critical_count = sum(
        1
        for record in history_records
        if str(
            record.get(
                "risk_level",
                "",
            )
        ).upper()
        == "CRITICAL"
    )


high_critical_count = (
    high_count
    +
    critical_count
)


anomaly_count = sum(
    1
    for record in history_records
    if str(
        record.get(
            "anomaly_status",
            "",
        )
    ).upper()
    == "ANOMALY"
)


prediction_count = safe_int(
    find_stat(
        stats_response,
        [
            "prediction_count",
            "total_predictions",
            "count_predictions",
            "predictions",
        ],
        len(history_records),
    )
)


# =============================================================================
# CHART: ATTACK CATEGORIES
# =============================================================================

def create_category_chart(
    counts,
):
    """Create attack-category distribution chart."""

    if not counts:
        return None

    dataframe = pd.DataFrame(
        list(
            counts.items()
        ),
        columns=[
            "Category",
            "Count",
        ],
    )

    dataframe = dataframe.sort_values(
        "Count",
        ascending=False,
    )

    figure, axis = plt.subplots(
        figsize=(10, 4.5)
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.set_facecolor(
        WHITE
    )

    colors = [
        CATEGORY_COLORS.get(
            category,
            BLUE,
        )
        for category in dataframe[
            "Category"
        ]
    ]

    bars = axis.bar(
        dataframe["Category"],
        dataframe["Count"],
        color=colors,
        width=0.68,
    )

    axis.spines[
        "top"
    ].set_visible(False)

    axis.spines[
        "right"
    ].set_visible(False)

    axis.spines[
        "left"
    ].set_color(
        "#DCE5EF"
    )

    axis.spines[
        "bottom"
    ].set_color(
        "#DCE5EF"
    )

    axis.grid(
        axis="y",
        color="#E9EEF5",
        linewidth=0.8,
    )

    axis.set_axisbelow(
        True
    )

    axis.tick_params(
        colors=MEDIUM,
        labelsize=8.5,
    )

    axis.set_ylabel(
        "Events",
        color=MEDIUM,
        fontsize=9,
    )

    axis.set_title(
        "Attack Category Distribution",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        pad=12,
    )

    maximum = max(
        dataframe["Count"]
    )

    for bar, count in zip(
        bars,
        dataframe["Count"],
    ):

        axis.text(
            bar.get_x()
            +
            bar.get_width()
            /
            2,
            bar.get_height()
            +
            maximum * 0.015,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=DARK,
        )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    return figure


# =============================================================================
# CHART: ATTACK VS NORMAL
# =============================================================================

def create_traffic_chart(
    attack_count,
    normal_count,
):
    """Create attack-versus-normal donut chart."""

    total = (
        attack_count
        +
        normal_count
    )

    if total <= 0:
        return None

    figure, axis = plt.subplots(
        figsize=(5.5, 4.5)
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.set_facecolor(
        WHITE
    )

    wedges, _ = axis.pie(
        [
            attack_count,
            normal_count,
        ],
        colors=[
            RED,
            BLUE,
        ],
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": 0.35,
            "edgecolor": WHITE,
            "linewidth": 3,
        },
    )

    axis.text(
        0,
        0.05,
        f"{total:,}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=DARK,
    )

    axis.text(
        0,
        -0.14,
        "Total Events",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MEDIUM,
    )

    attack_percentage = (
        attack_count
        /
        total
        *
        100
    )

    normal_percentage = (
        normal_count
        /
        total
        *
        100
    )

    axis.legend(
        wedges,
        [
            f"Attack · {attack_percentage:.2f}%",
            f"Normal · {normal_percentage:.2f}%",
        ],
        loc="center left",
        bbox_to_anchor=(
            0.94,
            0.5,
        ),
        frameon=False,
        fontsize=9,
    )

    axis.set_title(
        "Traffic Composition",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=DARK,
        pad=12,
    )

    plt.tight_layout()

    return figure


# =============================================================================
# EVENT CLEANING
# =============================================================================

def clean_event(
    event,
):
    """
    Remove target columns and convert pandas values
    into JSON-safe Python values.
    """

    raw_event = (
        event
        .drop(
            labels=[
                "attack_cat",
                "label",
            ],
            errors="ignore",
        )
        .to_dict()
    )

    cleaned = {}

    for key, value in raw_event.items():

        if pd.isna(value):

            cleaned[key] = None

        elif hasattr(
            value,
            "item",
        ):

            cleaned[key] = value.item()

        else:

            cleaned[key] = value

    return cleaned


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "# 🛡️ SECURITY OPERATIONS"
    )

    st.markdown(
        "**CYBERSECURITY THREAT PLATFORM**"
    )

    st.caption(
        "ML-based network threat detection & risk assessment"
    )

    st.divider()

    pages = [
        "Overview",
        "Threat Analysis",
        "Investigation",
        "Analyst Review",
        "Model Performance",
    ]

    if (
        st.session_state.current_page
        not in pages
    ):

        st.session_state.current_page = (
            "Overview"
        )

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.current_page
        ),
        label_visibility="collapsed",
    )

    st.session_state.current_page = (
        selected_page
    )

    st.divider()

    st.markdown(
        "### SYSTEM STATUS"
    )

    if api_online:

        st.success(
            "● API ONLINE"
        )

    else:

        st.error(
            "● API OFFLINE"
        )

    st.caption(
        f"Endpoint: {API_BASE_URL}"
    )

    st.caption(
        "Dataset: UNSW-NB15"
    )

    st.caption(
        "Models: Binary RF · Category RF · Isolation Forest"
    )


# =============================================================================
# MAIN HEADER
# =============================================================================

header_left, header_right = st.columns(
    [
        3.5,
        1,
    ]
)


with header_left:

    st.markdown(
        "**AI / ML SECURITY OPERATIONS CENTER**"
    )

    st.title(
        "Cybersecurity Threat Platform"
    )

    st.caption(
        "Network threat detection · known attack classification · "
        "anomaly detection · risk assessment"
    )


with header_right:

    if api_online:

        st.success(
            "● SYSTEM OPERATIONAL"
        )

        st.caption(
            "Flask API connected"
        )

    else:

        st.error(
            "● SYSTEM OFFLINE"
        )

        st.caption(
            "Flask API unavailable"
        )


st.divider()


# =============================================================================
# OVERVIEW
# =============================================================================

if selected_page == "Overview":

    st.header(
        "Security Overview"
    )

    st.caption(
        "High-level view of the UNSW-NB15 testing environment and stored platform activity."
    )

    # =========================================================================
    # KPI ROW
    # =========================================================================

    k1, k2, k3, k4, k5 = st.columns(
        5
    )

    with k1:

        st.metric(
            "🗄️ Total Predictions",
            f"{prediction_count:,}",
            help="Number of prediction records stored by the platform.",
        )

    with k2:

        st.metric(
            "🚨 Attack Events",
            f"{attack_test_events:,}",
            help="Attack records in the official UNSW-NB15 testing dataset.",
        )

    with k3:

        st.metric(
            "🛡️ Normal Events",
            f"{normal_test_events:,}",
            help="Normal records in the official UNSW-NB15 testing dataset.",
        )

    with k4:

        st.metric(
            "⚠️ High / Critical",
            f"{high_critical_count:,}",
            help="Stored predictions with High or Critical risk.",
        )

    with k5:

        st.metric(
            "🔮 Anomalies",
            f"{anomaly_count:,}",
            help="Stored predictions classified as anomalous.",
        )


    st.write("")


    # =========================================================================
    # THREAT LANDSCAPE
    # =========================================================================

    chart_left, chart_right = st.columns(
        [
            1.65,
            1,
        ]
    )


    with chart_left:

        st.subheader(
            "📊 Threat Landscape"
        )

        st.caption(
            "Known attack categories represented in the official UNSW-NB15 testing dataset."
        )

        figure = create_category_chart(
            category_counts
        )

        if figure is not None:

            st.pyplot(
                figure,
                use_container_width=True,
            )

            plt.close(
                figure
            )

        else:

            st.info(
                "No attack-category data available."
            )


    with chart_right:

        st.subheader(
            "📈 Traffic Composition"
        )

        st.caption(
            "Attack versus normal traffic in the official testing dataset."
        )

        figure = create_traffic_chart(
            attack_test_events,
            normal_test_events,
        )

        if figure is not None:

            st.pyplot(
                figure,
                use_container_width=True,
            )

            plt.close(
                figure
            )

        else:

            st.info(
                "No traffic data available."
            )


    st.divider()


    # =========================================================================
    # RECENT ACTIVITY
    # =========================================================================

    st.subheader(
        "🛡️ Recent Security Activity"
    )

    st.caption(
        "Latest prediction records returned by the threat platform API."
    )


    if not history_records:

        st.info(
            "No prediction records are currently stored."
        )

    else:

        activity_rows = []

        for record in history_records[:10]:

            prediction_id = (
                record.get(
                    "id"
                )
                or
                record.get(
                    "prediction_id"
                )
            )

            timestamp = (
                record.get(
                    "timestamp"
                )
                or
                record.get(
                    "created_at"
                )
                or
                record.get(
                    "prediction_timestamp"
                )
            )

            detection = (
                record.get(
                    "detection_status"
                )
                or
                record.get(
                    "detection"
                )
                or
                "Unknown"
            )

            category = (
                record.get(
                    "attack_category"
                )
                or
                record.get(
                    "category"
                )
                or
                "—"
            )

            anomaly = (
                record.get(
                    "anomaly_status"
                )
                or
                "Unknown"
            )

            risk = (
                record.get(
                    "risk_level"
                )
                or
                "Unknown"
            )

            score = record.get(
                "risk_score"
            )

            probability = record.get(
                "attack_probability"
            )

            if probability is None:

                probability_text = "—"

            else:

                probability_text = (
                    f"{safe_float(probability) * 100:.1f}%"
                )

            activity_rows.append(
                {
                    "ID": prediction_id,
                    "Timestamp": format_timestamp(
                        timestamp
                    ),
                    "Detection": detection,
                    "Category": category,
                    "Anomaly": anomaly,
                    "Risk": risk,
                    "Score": (
                        safe_int(score)
                        if score is not None
                        else "—"
                    ),
                    "Attack Probability": probability_text,
                }
            )


        activity_dataframe = pd.DataFrame(
            activity_rows
        )


        st.dataframe(
            activity_dataframe,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "Select a prediction ID below to investigate the complete finding."
        )


        prediction_options = [
            str(
                row["ID"]
            )
            for row in activity_rows
        ]


        if prediction_options:

            selected_activity_id = st.selectbox(
                "Prediction to investigate",
                prediction_options,
            )


            if st.button(
                "🔎 VIEW SELECTED FINDING",
                type="primary",
            ):

                st.session_state.selected_prediction_id = (
                    selected_activity_id
                )

                st.session_state.current_page = (
                    "Investigation"
                )

                st.rerun()


# =============================================================================
# THREAT ANALYSIS
# =============================================================================

elif selected_page == "Threat Analysis":

    st.header(
        "Threat Analysis"
    )

    st.caption(
        "Submit a real UNSW-NB15 network event to the complete Flask ML pipeline."
    )


    if dataset.empty:

        st.error(
            "UNSW-NB15 testing dataset could not be loaded."
        )

    elif not api_online:

        st.error(
            "Flask API is offline. Start api/server.py first."
        )

    else:

        selection_mode = st.radio(
            "Event selection method",
            [
                "Attack Category",
                "Dataset Row",
            ],
            horizontal=True,
        )


        selected_event = None


        if selection_mode == "Attack Category":

            available_categories = sorted(
                category_counts.keys()
            )


            if available_categories:

                selected_category = st.selectbox(
                    "Attack category",
                    available_categories,
                )


                category_records = dataset[
                    dataset[
                        "attack_cat"
                    ]
                    .astype(str)
                    .str.strip()
                    ==
                    selected_category
                ].reset_index(
                    drop=True
                )


                if not category_records.empty:

                    record_index = st.number_input(
                        "Record within category",
                        min_value=0,
                        max_value=len(
                            category_records
                        ) - 1,
                        value=0,
                        step=1,
                    )


                    selected_event = (
                        category_records.iloc[
                            int(record_index)
                        ]
                    )

        else:

            record_index = st.number_input(
                "Dataset row",
                min_value=0,
                max_value=len(
                    dataset
                ) - 1,
                value=0,
                step=1,
            )


            selected_event = dataset.iloc[
                int(record_index)
            ]


        if selected_event is not None:

            actual_result = (
                "Attack"
                if safe_int(
                    selected_event.get(
                        "label",
                        0,
                    )
                )
                ==
                1
                else
                "Normal"
            )


            actual_category = str(
                selected_event.get(
                    "attack_cat",
                    "Normal",
                )
            ).strip()


            if (
                not actual_category
                or actual_category.lower()
                ==
                "nan"
            ):

                actual_category = "Normal"


            c1, c2, c3 = st.columns(
                3
            )


            with c1:

                st.metric(
                    "Actual Result",
                    actual_result,
                )


            with c2:

                st.metric(
                    "Actual Category",
                    actual_category,
                )


            with c3:

                st.metric(
                    "Model Input Features",
                    "42",
                )


            st.info(
                "The ground-truth label and attack category are removed before the event is submitted to the API."
            )


            if st.button(
                "🔎 RUN THREAT ANALYSIS",
                type="primary",
                use_container_width=True,
            ):

                network_event = clean_event(
                    selected_event
                )


                with st.spinner(
                    "Running threat detection pipeline..."
                ):

                    (
                        success,
                        response,
                        error,
                    ) = api_post(
                        "/api/v1/predict",
                        {
                            "event": network_event
                        },
                        timeout=30,
                    )


                if not success:

                    st.error(
                        f"Threat analysis failed: {error}"
                    )

                else:

                    prediction = response.get(
                        "prediction",
                        {},
                    )


                    st.session_state.latest_prediction = (
                        prediction
                    )

                    st.session_state.latest_api_response = (
                        response
                    )

                    st.session_state.latest_event = (
                        network_event
                    )

                    st.session_state.latest_actual_label = (
                        actual_result
                    )

                    st.session_state.latest_actual_category = (
                        actual_category
                    )

                    st.session_state.selected_prediction_id = (
                        response.get(
                            "database_record_id"
                        )
                    )


                    st.success(
                        "Threat analysis completed successfully."
                    )


                    st.session_state.current_page = (
                        "Investigation"
                    )

                    st.rerun()


# =============================================================================
# INVESTIGATION
# =============================================================================

elif selected_page == "Investigation":

    st.header(
        "Threat Investigation"
    )

    st.caption(
        "Detailed investigation view for a stored security finding."
    )


    selected_id = (
        st.session_state.selected_prediction_id
    )


    selected_record = None


    if selected_id is not None:

        for record in history_records:

            record_id = (
                record.get(
                    "id"
                )
                or
                record.get(
                    "prediction_id"
                )
            )


            if str(
                record_id
            ) == str(
                selected_id
            ):

                selected_record = record
                break


    if selected_record is None:

        selected_record = (
            st.session_state.latest_prediction
        )


    if selected_record is None:

        st.info(
            "No security finding is selected. "
            "Run Threat Analysis or select a prediction from Overview."
        )


    else:

        prediction = selected_record


        prediction_id = (
            prediction.get(
                "id"
            )
            or
            prediction.get(
                "prediction_id"
            )
            or
            selected_id
        )


        detection = prediction.get(
            "detection_status",
            "Unknown",
        )


        category = prediction.get(
            "attack_category",
            "Unknown",
        )


        anomaly = prediction.get(
            "anomaly_status",
            "Unknown",
        )


        risk_level = prediction.get(
            "risk_level",
            "UNKNOWN",
        )


        risk_score = prediction.get(
            "risk_score",
            0,
        )


        st.subheader(
            f"Security Finding #{prediction_id}"
        )


        st.caption(
            "Machine-learning assessment of the selected network event."
        )


        # =====================================================================
        # FINDING CARDS
        # =====================================================================

        f1, f2, f3, f4 = st.columns(
            4
        )


        with f1:

            st.metric(
                "🚨 Detection",
                detection,
            )


        with f2:

            st.metric(
                "🏷️ Attack Category",
                category,
            )


        with f3:

            st.metric(
                "🔮 Anomaly",
                anomaly,
            )


        with f4:

            st.metric(
                "⚠️ Risk Score",
                f"{safe_int(risk_score)}/100",
            )


        st.write("")


        # =====================================================================
        # RISK STATUS
        # =====================================================================

        risk_upper = str(
            risk_level
        ).upper()


        if risk_upper == "CRITICAL":

            st.error(
                f"🔴 CRITICAL RISK · {safe_int(risk_score)}/100"
            )

        elif risk_upper == "HIGH":

            st.warning(
                f"🟠 HIGH RISK · {safe_int(risk_score)}/100"
            )

        elif risk_upper == "MEDIUM":

            st.warning(
                f"🟡 MEDIUM RISK · {safe_int(risk_score)}/100"
            )

        else:

            st.success(
                f"🟢 LOW RISK · {safe_int(risk_score)}/100"
            )


        st.progress(
            min(
                max(
                    safe_float(
                        risk_score
                    )
                    /
                    100,
                    0.0,
                ),
                1.0,
            )
        )


        st.divider()


        # =====================================================================
        # DETECTION EVIDENCE
        # =====================================================================

        st.subheader(
            "Detection Evidence"
        )


        attack_probability = safe_float(
            prediction.get(
                "attack_probability"
            )
        )


        category_probability = prediction.get(
            "attack_category_probability"
        )


        anomaly_score = prediction.get(
            "anomaly_score"
        )


        e1, e2, e3 = st.columns(
            3
        )


        with e1:

            st.metric(
                "Binary Attack Probability",
                f"{attack_probability * 100:.2f}%",
            )


        with e2:

            if category_probability is None:

                confidence_text = "N/A"

            else:

                confidence_text = (
                    f"{safe_float(category_probability) * 100:.2f}%"
                )


            st.metric(
                "Category Confidence",
                confidence_text,
            )


        with e3:

            if anomaly_score is None:

                anomaly_text = "N/A"

            else:

                anomaly_text = (
                    f"{safe_float(anomaly_score):.4f}"
                )


            st.metric(
                "Normality Score",
                anomaly_text,
            )


        st.caption(
            "Lower Isolation Forest normality score means more anomalous behaviour."
        )


        if (
            category_probability is not None
            and
            safe_float(
                category_probability
            )
            <
            0.60
        ):

            st.warning(
                "Category uncertain: attack-category confidence is below 60%."
            )


        # =====================================================================
        # RISK COMPONENTS
        # =====================================================================

        risk_components = prediction.get(
            "risk_components",
            {},
        )


        if risk_components:

            st.subheader(
                "Risk Assessment"
            )


            risk_dataframe = pd.DataFrame(
                {
                    "Evidence Component": [
                        "Binary attack",
                        "Attack category",
                        "Anomaly",
                        "Severity",
                    ],
                    "Contribution": [
                        safe_float(
                            risk_components.get(
                                "binary_attack"
                            )
                        ),
                        safe_float(
                            risk_components.get(
                                "attack_category"
                            )
                        ),
                        safe_float(
                            risk_components.get(
                                "anomaly"
                            )
                        ),
                        safe_float(
                            risk_components.get(
                                "severity"
                            )
                        ),
                    ],
                    "Maximum": [
                        40,
                        25,
                        25,
                        10,
                    ],
                }
            )


            st.dataframe(
                risk_dataframe,
                use_container_width=True,
                hide_index=True,
            )


        st.divider()


        # =====================================================================
        # EXPLANATION
        # =====================================================================

        st.subheader(
            "Why This Finding Was Produced"
        )


        st.info(
            prediction.get(
                "explanation",
                "No explanation returned by the API.",
            )
        )


        st.subheader(
            "Recommended Security Action"
        )


        recommendation = prediction.get(
            "recommended_action",
            "No recommendation returned.",
        )


        if risk_upper in {
            "CRITICAL",
            "HIGH",
        }:

            st.warning(
                recommendation
            )

        else:

            st.info(
                recommendation
            )


        # =====================================================================
        # GROUND TRUTH
        # =====================================================================

        if (
            st.session_state.latest_actual_label
            is not None
        ):

            st.subheader(
                "Dataset Ground Truth"
            )


            g1, g2 = st.columns(
                2
            )


            with g1:

                st.metric(
                    "Actual Result",
                    st.session_state.latest_actual_label,
                )


            with g2:

                st.metric(
                    "Actual Category",
                    st.session_state.latest_actual_category,
                )


        # =====================================================================
        # RAW RECORD
        # =====================================================================

        with st.expander(
            "View complete prediction record"
        ):

            st.json(
                prediction
            )


        st.write("")


        if st.button(
            "📝 OPEN ANALYST REVIEW",
            type="primary",
        ):

            st.session_state.current_page = (
                "Analyst Review"
            )

            st.rerun()


# =============================================================================
# ANALYST REVIEW
# =============================================================================

elif selected_page == "Analyst Review":

    st.header(
        "Analyst Review"
    )

    st.caption(
        "Human validation of a machine-learning prediction."
    )


    selected_id = (
        st.session_state.selected_prediction_id
    )


    if selected_id is None:

        st.info(
            "Select a prediction from Overview first."
        )


    else:

        selected_prediction = None


        for record in history_records:

            record_id = (
                record.get(
                    "id"
                )
                or
                record.get(
                    "prediction_id"
                )
            )


            if str(
                record_id
            ) == str(
                selected_id
            ):

                selected_prediction = record
                break


        if selected_prediction is None:

            selected_prediction = (
                st.session_state.latest_prediction
            )


        if selected_prediction is None:

            st.error(
                "Selected prediction could not be loaded."
            )


        else:

            st.subheader(
                f"Security Finding #{selected_id}"
            )


            p1, p2 = st.columns(
                2
            )


            with p1:

                st.metric(
                    "System Binary Prediction",
                    selected_prediction.get(
                        "detection_status",
                        "Unknown",
                    ),
                )


            with p2:

                st.metric(
                    "System Category Prediction",
                    selected_prediction.get(
                        "attack_category",
                        "Unknown",
                    ),
                )


            st.divider()


            # =================================================================
            # EXISTING FEEDBACK
            # =================================================================

            matching_feedback = [
                feedback
                for feedback in feedback_records
                if str(
                    feedback.get(
                        "prediction_id",
                        "",
                    )
                )
                ==
                str(
                    selected_id
                )
            ]


            if matching_feedback:

                feedback = matching_feedback[0]


                st.success(
                    "Analyst feedback already exists for this prediction."
                )


                r1, r2, r3, r4 = st.columns(
                    4
                )


                with r1:

                    st.metric(
                        "Binary Correct",
                        feedback.get(
                            "binary_correct",
                            "Unknown",
                        ),
                    )


                with r2:

                    st.metric(
                        "Category Correct",
                        feedback.get(
                            "category_correct",
                            "N/A",
                        ),
                    )


                with r3:

                    st.metric(
                        "Actual Result",
                        feedback.get(
                            "actual_result",
                            "Unknown",
                        ),
                    )


                with r4:

                    st.metric(
                        "Training Eligible",
                        str(
                            feedback.get(
                                "training_eligible",
                                False,
                            )
                        ),
                    )


                if feedback.get(
                    "actual_attack_category"
                ):

                    st.write(
                        "**Actual attack category:** "
                        f"{feedback.get('actual_attack_category')}"
                    )


                if feedback.get(
                    "root_cause"
                ):

                    st.write(
                        "**Root cause:** "
                        f"{feedback.get('root_cause')}"
                    )


                if feedback.get(
                    "analyst_notes"
                ):

                    st.write(
                        "**Analyst notes:** "
                        f"{feedback.get('analyst_notes')}"
                    )


            else:

                # =============================================================
                # FEEDBACK FORM
                # =============================================================

                st.subheader(
                    "Analyst Ground Truth"
                )


                with st.form(
                    "analyst_feedback_form"
                ):

                    actual_result = st.selectbox(
                        "Actual traffic result",
                        [
                            "Attack",
                            "Normal",
                            "Unknown",
                        ],
                    )


                    actual_category = st.selectbox(
                        "Actual attack category",
                        [
                            "None",
                            "Analysis",
                            "Backdoor",
                            "DoS",
                            "Exploits",
                            "Fuzzers",
                            "Generic",
                            "Reconnaissance",
                            "Shellcode",
                            "Worms",
                        ],
                    )


                    unknown_attack = st.checkbox(
                        "Possible unknown / previously unseen attack"
                    )


                    root_cause = st.text_input(
                        "Root cause",
                        placeholder=(
                            "Describe the likely cause of the event."
                        ),
                    )


                    analyst_notes = st.text_area(
                        "Analyst notes",
                        placeholder=(
                            "Record investigation evidence and observations."
                        ),
                        height=140,
                    )


                    submit_feedback = st.form_submit_button(
                        "📝 SUBMIT ANALYST REVIEW",
                        type="primary",
                        use_container_width=True,
                    )


                if submit_feedback:

                    if actual_result in {
                        "Normal",
                        "Unknown",
                    }:

                        category_value = None

                    else:

                        category_value = (
                            None
                            if actual_category == "None"
                            else actual_category
                        )


                    payload = {
                        "prediction_id": int(
                            selected_id
                        ),
                        "actual_result": actual_result,
                        "actual_attack_category": category_value,
                        "is_unknown_attack": bool(
                            unknown_attack
                        ),
                        "root_cause": (
                            root_cause.strip()
                            if root_cause.strip()
                            else None
                        ),
                        "analyst_notes": (
                            analyst_notes.strip()
                            if analyst_notes.strip()
                            else None
                        ),
                    }


                    with st.spinner(
                        "Submitting analyst feedback..."
                    ):

                        (
                            success,
                            response,
                            error,
                        ) = api_post(
                            "/api/v1/feedback",
                            payload,
                            timeout=10,
                        )


                    if success:

                        st.success(
                            "Analyst feedback stored successfully."
                        )


                        a1, a2, a3 = st.columns(
                            3
                        )


                        with a1:

                            st.metric(
                                "Binary Correct",
                                response.get(
                                    "binary_correct",
                                    "Unknown",
                                ),
                            )


                        with a2:

                            st.metric(
                                "Category Correct",
                                response.get(
                                    "category_correct",
                                    "N/A",
                                ),
                            )


                        with a3:

                            st.metric(
                                "Training Eligible",
                                str(
                                    response.get(
                                        "training_eligible",
                                        False,
                                    )
                                ),
                            )


                        st.info(
                            "Correctness is calculated by the Flask server. Analyst feedback does not automatically retrain the model."
                        )


                        st.rerun()


                    else:

                        st.error(
                            f"Feedback submission failed: {error}"
                        )


    st.divider()


    # =========================================================================
    # FEEDBACK HISTORY
    # =========================================================================

    st.subheader(
        "Analyst Feedback History"
    )


    if feedback_records:

        feedback_dataframe = pd.DataFrame(
            feedback_records
        )


        preferred_columns = [
            "feedback_id",
            "prediction_id",
            "feedback_timestamp",
            "binary_correct",
            "category_correct",
            "actual_result",
            "actual_attack_category",
            "is_unknown_attack",
            "root_cause",
            "analyst_notes",
            "training_eligible",
        ]


        available_columns = [
            column
            for column in preferred_columns
            if column
            in feedback_dataframe.columns
        ]


        if available_columns:

            feedback_dataframe = (
                feedback_dataframe[
                    available_columns
                ]
            )


        st.dataframe(
            feedback_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No analyst feedback has been recorded yet."
        )


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

elif selected_page == "Model Performance":

    st.header(
        "Model Performance"
    )

    st.caption(
        "Official evaluation results for the cybersecurity machine-learning pipeline."
    )


    p1, p2, p3 = st.columns(
        3
    )


    with p1:

        st.subheader(
            "Binary Attack Detection"
        )

        st.metric(
            "Accuracy",
            MODEL_PERFORMANCE[
                "Binary Attack Detection"
            ][
                "Accuracy"
            ],
        )

        st.metric(
            "Attack Recall",
            MODEL_PERFORMANCE[
                "Binary Attack Detection"
            ][
                "Attack Recall"
            ],
        )

        st.metric(
            "ROC-AUC",
            MODEL_PERFORMANCE[
                "Binary Attack Detection"
            ][
                "ROC-AUC"
            ],
        )


    with p2:

        st.subheader(
            "Known Attack Classification"
        )

        st.metric(
            "Accuracy",
            MODEL_PERFORMANCE[
                "Known Attack Classification"
            ][
                "Accuracy"
            ],
        )

        st.metric(
            "Balanced Accuracy",
            MODEL_PERFORMANCE[
                "Known Attack Classification"
            ][
                "Balanced Accuracy"
            ],
        )

        st.metric(
            "Macro F1",
            MODEL_PERFORMANCE[
                "Known Attack Classification"
            ][
                "Macro F1"
            ],
        )


    with p3:

        st.subheader(
            "Anomaly Detection"
        )

        st.metric(
            "Accuracy",
            MODEL_PERFORMANCE[
                "Anomaly Detection"
            ][
                "Accuracy"
            ],
        )

        st.metric(
            "Attack Recall",
            MODEL_PERFORMANCE[
                "Anomaly Detection"
            ][
                "Attack Recall"
            ],
        )

        st.metric(
            "Attack Precision",
            MODEL_PERFORMANCE[
                "Anomaly Detection"
            ][
                "Attack Precision"
            ],
        )


    st.divider()


    # =========================================================================
    # ARCHITECTURE
    # =========================================================================

    st.subheader(
        "Detection Architecture"
    )


    architecture_dataframe = pd.DataFrame(
        {
            "Stage": [
                "1",
                "2",
                "3",
                "4",
                "5",
            ],
            "Component": [
                "Input Validation",
                "Binary Random Forest",
                "Attack Category Random Forest",
                "Isolation Forest",
                "Risk Assessment",
            ],
            "Purpose": [
                "Validate incoming network event.",
                "Determine Attack versus Normal.",
                "Identify known attack category.",
                "Measure anomalous behaviour.",
                "Combine evidence into a 0–100 risk score.",
            ],
        }
    )


    st.dataframe(
        architecture_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    # =========================================================================
    # RISK MODEL
    # =========================================================================

    st.subheader(
        "Threat Priority Score"
    )


    risk_dataframe = pd.DataFrame(
        {
            "Evidence Component": [
                "Binary attack evidence",
                "Attack category evidence",
                "Anomaly evidence",
                "Severity evidence",
            ],
            "Weight": [
                "40%",
                "25%",
                "25%",
                "10%",
            ],
            "Maximum Contribution": [
                "40 points",
                "25 points",
                "25 points",
                "10 points",
            ],
        }
    )


    st.dataframe(
        risk_dataframe,
        use_container_width=True,
        hide_index=True,
    )


    st.info(
        "The threat priority score is an evidence-aggregation score, not a probability."
    )


    st.warning(
        "The platform does not claim zero-day detection. "
        "The known-attack classifier predicts categories represented in the "
        "training data, while Isolation Forest identifies anomalous behaviour."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Cybersecurity Threat Platform · UNSW-NB15 · "
    "Random Forest + Isolation Forest · Flask REST API · SQLite"
)