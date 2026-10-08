"""
================================================================================
CYBERSECURITY THREAT PLATFORM
RISK ASSESSMENT MODULE
================================================================================

Purpose
-------
Converts the outputs of the machine-learning components into an interpretable
Threat Priority Score.

The score combines:

1. Binary attack probability
2. Known attack-category probability
3. Isolation Forest anomaly strength
4. Attack-category severity

IMPORTANT
---------
The Threat Priority Score is a rule-based priority score.

It is NOT:
    - a probability
    - a prediction probability
    - a guarantee of attack severity

Maximum score = 100.
================================================================================
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# SCORE WEIGHTS
# =============================================================================

BINARY_ATTACK_WEIGHT = 40
ATTACK_CATEGORY_WEIGHT = 25
ANOMALY_WEIGHT = 25
SEVERITY_WEIGHT = 10


# =============================================================================
# ANOMALY SCORE SCALE
# =============================================================================

# Current Isolation Forest decision-function values for this project are
# approximately within the range -0.15 to +0.15.
#
# Lower raw decision_function values mean stronger anomaly behaviour.
ANOMALY_SCORE_SCALE = 0.15


# =============================================================================
# ATTACK CATEGORY SEVERITY
# =============================================================================

ATTACK_CATEGORY_SEVERITY = {
    "Analysis": 5,
    "Backdoor": 9,
    "DoS": 8,
    "Exploits": 10,
    "Fuzzers": 6,
    "Generic": 8,
    "Reconnaissance": 5,
    "Shellcode": 9,
    "Worms": 10,
}


# =============================================================================
# RISK LEVEL THRESHOLDS
# =============================================================================

LOW_RISK_MAX = 24
MEDIUM_RISK_MAX = 49
HIGH_RISK_MAX = 74


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class RiskAssessment:
    """
    Stores the complete Threat Priority Score assessment.

    Attributes
    ----------
    risk_score : int
        Final rule-based score between 0 and 100.

    risk_level : str
        LOW, MEDIUM, HIGH, or CRITICAL.

    anomaly_status : str
        Isolation Forest status:
            NORMAL
            ANOMALY

    components : dict
        Individual score contributions.

    explanation : str
        Human-readable explanation.

    recommended_action : str
        Recommended security response.
    """

    risk_score: int
    risk_level: str
    anomaly_status: str
    components: dict
    explanation: str
    recommended_action: str


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================

def normalize_anomaly_status(
    anomaly_status: str,
) -> str:
    """
    Normalize anomaly status to either NORMAL or ANOMALY.
    """

    normalized_status = str(
        anomaly_status
    ).strip().upper()

    if normalized_status == "ANOMALY":
        return "ANOMALY"

    return "NORMAL"


# =============================================================================
# PROBABILITY CLAMPING
# =============================================================================

def clamp_probability(
    probability: Optional[float],
) -> float:
    """
    Safely constrain a probability to the range [0, 1].

    None is treated as 0.0.
    """

    if probability is None:
        return 0.0

    try:
        probability = float(probability)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            probability,
        ),
    )


# =============================================================================
# ANOMALY STRENGTH
# =============================================================================

def normalize_anomaly_score(
    anomaly_score: float,
) -> float:
    """
    Convert the raw Isolation Forest decision-function score into anomaly
    strength in the range [0, 1].

    Current convention
    ------------------
    Isolation Forest decision_function:

        Higher value  -> more normal
        Lower value   -> more anomalous

    Therefore:

        anomaly strength = clip(
            -anomaly_score / 0.15,
            0,
            1
        )

    Examples
    --------
    anomaly_score = 0.10
        -> 0.0 anomaly strength

    anomaly_score = 0.00
        -> 0.0 anomaly strength

    anomaly_score = -0.075
        -> 0.5 anomaly strength

    anomaly_score = -0.15
        -> 1.0 anomaly strength

    This function does NOT determine whether the event is anomalous.
    The Isolation Forest status remains authoritative.
    """

    try:
        score = float(anomaly_score)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    normalized_strength = (
        -score
        / ANOMALY_SCORE_SCALE
    )

    return max(
        0.0,
        min(
            1.0,
            normalized_strength,
        ),
    )


# =============================================================================
# RISK LEVEL
# =============================================================================

def get_risk_level(
    risk_score: int,
) -> str:
    """
    Convert the numerical Threat Priority Score into a risk level.
    """

    if risk_score <= LOW_RISK_MAX:
        return "LOW"

    if risk_score <= MEDIUM_RISK_MAX:
        return "MEDIUM"

    if risk_score <= HIGH_RISK_MAX:
        return "HIGH"

    return "CRITICAL"


# =============================================================================
# RECOMMENDED SECURITY ACTION
# =============================================================================

def get_recommended_action(
    risk_level: str,
    detection_status: str,
    anomaly_status: str,
) -> str:
    """
    Generate a recommended security response.

    The system does not automatically isolate traffic.

    Critical events are escalated for analyst review.
    """

    normalized_detection = str(
        detection_status
    ).strip().upper()

    normalized_anomaly = str(
        anomaly_status
    ).strip().upper()

    if risk_level == "CRITICAL":

        return (
            "Escalate for immediate analyst review and initiate "
            "appropriate incident-response procedures."
        )

    if risk_level == "HIGH":

        if normalized_anomaly == "ANOMALY":

            return (
                "Prioritize investigation of the anomalous network "
                "event and review the affected host, source, and "
                "destination."
            )

        return (
            "Investigate the detected attack behavior and review "
            "the affected network communication."
        )

    if risk_level == "MEDIUM":

        return (
            "Monitor the network event closely and investigate if "
            "suspicious behavior persists or increases."
        )

    if normalized_detection == "ATTACK":

        return (
            "Continue monitoring the event and validate the detected "
            "attack behavior using additional security telemetry."
        )

    return (
        "No immediate action is required. Continue normal monitoring."
    )


# =============================================================================
# EXPLANATION GENERATION
# =============================================================================

def generate_explanation(
    detection_status: str,
    attack_category: Optional[str],
    anomaly_status: str,
    anomaly_score: float,
    binary_attack_probability: float,
    attack_category_probability: float,
) -> str:
    """
    Generate a human-readable explanation of the assessment.
    """

    normalized_detection = str(
        detection_status
    ).strip().upper()

    normalized_anomaly = normalize_anomaly_status(
        anomaly_status
    )

    explanation_parts = []

    # -------------------------------------------------------------------------
    # Binary detection
    # -------------------------------------------------------------------------

    if normalized_detection == "ATTACK":

        explanation_parts.append(
            "The supervised classifier detected attack behavior."
        )

        explanation_parts.append(
            f"Attack probability is "
            f"{binary_attack_probability * 100:.2f}%."
        )

        if attack_category:

            explanation_parts.append(
                f"The predicted known attack category is "
                f"{attack_category}."
            )

            explanation_parts.append(
                f"Category confidence is "
                f"{attack_category_probability * 100:.2f}%."
            )

    else:

        explanation_parts.append(
            "The supervised classifier classified the network "
            "behavior as normal."
        )

        explanation_parts.append(
            f"Attack probability is "
            f"{binary_attack_probability * 100:.2f}%."
        )

    # -------------------------------------------------------------------------
    # Anomaly detection
    # -------------------------------------------------------------------------

    if normalized_anomaly == "ANOMALY":

        explanation_parts.append(
            "The anomaly detector classified the behavior as "
            "anomalous relative to the learned normal traffic profile."
        )

        explanation_parts.append(
            "This behavior may represent an unusual or previously "
            "unseen pattern and requires further investigation."
        )

    else:

        explanation_parts.append(
            "The anomaly detector classified the behavior as normal "
            "relative to the learned normal traffic profile."
        )

    # -------------------------------------------------------------------------
    # Raw Isolation Forest score
    # -------------------------------------------------------------------------

    explanation_parts.append(
        f"Isolation Forest normality score: {anomaly_score:.4f} "
        "(lower values indicate more anomalous behavior)."
    )

    return " ".join(
        explanation_parts
    )


# =============================================================================
# MAIN RISK ASSESSMENT FUNCTION
# =============================================================================

def assess_risk(
    detection_status: str,
    attack_category: Optional[str],
    anomaly_status: str,
    anomaly_score: float,
    binary_attack_probability: Optional[float] = None,
    attack_category_probability: Optional[float] = None,
) -> RiskAssessment:
    """
    Calculate the final Threat Priority Score.

    Score Components
    ----------------

    1. Binary attack confidence
       Maximum = 40 points

       40 × P(Attack)

    2. Known attack-category confidence
       Maximum = 25 points

       25 × P(Category)

       Applied only when the binary classifier detects an attack.

    3. Anomaly strength
       Maximum = 25 points

       25 × clip(
           -anomaly_score / 0.15,
           0,
           1
       )

       Applied only when Isolation Forest reports Anomaly.

    4. Attack severity
       Maximum = 10 points

       10 × (severity / 10)

       Applied only when the event is detected as an attack and
       has a known attack category.

    Maximum total = 100.

    Parameters
    ----------
    detection_status : str
        Binary classifier result.

    attack_category : Optional[str]
        Predicted known attack category.

    anomaly_status : str
        Isolation Forest status.

    anomaly_score : float
        Raw Isolation Forest decision-function score.

    binary_attack_probability : Optional[float]
        Probability of the Attack class.

    attack_category_probability : Optional[float]
        Probability/confidence of the predicted attack category.

    Returns
    -------
    RiskAssessment
        Complete rule-based threat priority assessment.
    """

    # =========================================================================
    # NORMALIZE INPUTS
    # =========================================================================

    normalized_detection = str(
        detection_status
    ).strip().upper()

    normalized_anomaly = normalize_anomaly_status(
        anomaly_status
    )

    normalized_attack_category = (
        str(attack_category).strip()
        if attack_category
        else None
    )

    binary_probability = clamp_probability(
        binary_attack_probability
    )

    category_probability = clamp_probability(
        attack_category_probability
    )

    # =========================================================================
    # COMPONENT 1 — BINARY ATTACK CONFIDENCE
    # =========================================================================

    binary_component = 0.0

    if normalized_detection == "ATTACK":

        binary_component = (
            BINARY_ATTACK_WEIGHT
            * binary_probability
        )

    # =========================================================================
    # COMPONENT 2 — KNOWN ATTACK CATEGORY CONFIDENCE
    # =========================================================================

    category_component = 0.0

    if (
        normalized_detection == "ATTACK"
        and normalized_attack_category
    ):

        category_component = (
            ATTACK_CATEGORY_WEIGHT
            * category_probability
        )

    # =========================================================================
    # COMPONENT 3 — ANOMALY STRENGTH
    # =========================================================================

    anomaly_strength = normalize_anomaly_score(
        anomaly_score
    )

    anomaly_component = 0.0

    if normalized_anomaly == "ANOMALY":

        anomaly_component = (
            ANOMALY_WEIGHT
            * anomaly_strength
        )

    # =========================================================================
    # COMPONENT 4 — ATTACK SEVERITY
    # =========================================================================

    severity_value = 0

    severity_component = 0.0

    if (
        normalized_detection == "ATTACK"
        and normalized_attack_category
    ):

        severity_value = ATTACK_CATEGORY_SEVERITY.get(
            normalized_attack_category,
            5,
        )

        severity_component = (
            SEVERITY_WEIGHT
            * (
                severity_value
                / 10
            )
        )

    # =========================================================================
    # FINAL SCORE
    # =========================================================================

    raw_risk_score = (
        binary_component
        + category_component
        + anomaly_component
        + severity_component
    )

    risk_score = int(
        round(
            max(
                0.0,
                min(
                    100.0,
                    raw_risk_score,
                ),
            )
        )
    )

    # =========================================================================
    # RISK LEVEL
    # =========================================================================

    risk_level = get_risk_level(
        risk_score
    )

    # =========================================================================
    # SCORE COMPONENTS
    # =========================================================================

    components = {
        "binary_attack": round(
            binary_component,
            2,
        ),
        "attack_category": round(
            category_component,
            2,
        ),
        "anomaly": round(
            anomaly_component,
            2,
        ),
        "severity": round(
            severity_component,
            2,
        ),
        "binary_attack_probability": round(
            binary_probability,
            4,
        ),
        "attack_category_probability": round(
            category_probability,
            4,
        ),
        "anomaly_strength": round(
            anomaly_strength,
            4,
        ),
        "severity_value": severity_value,
    }

    # =========================================================================
    # EXPLANATION
    # =========================================================================

    explanation = generate_explanation(
        detection_status=normalized_detection,
        attack_category=normalized_attack_category,
        anomaly_status=normalized_anomaly,
        anomaly_score=anomaly_score,
        binary_attack_probability=binary_probability,
        attack_category_probability=category_probability,
    )

    # =========================================================================
    # RECOMMENDED ACTION
    # =========================================================================

    recommended_action = get_recommended_action(
        risk_level=risk_level,
        detection_status=normalized_detection,
        anomaly_status=normalized_anomaly,
    )

    # =========================================================================
    # RETURN RESULT
    # =========================================================================

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly_status=normalized_anomaly,
        components=components,
        explanation=explanation,
        recommended_action=recommended_action,
    )


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("RISK ASSESSMENT MODULE TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1 — Normal traffic
    # -------------------------------------------------------------------------

    normal_result = assess_risk(
        detection_status="Normal",
        attack_category=None,
        anomaly_status="Normal",
        anomaly_score=0.10,
        binary_attack_probability=0.05,
        attack_category_probability=0.0,
    )

    print()
    print("-" * 80)
    print("TEST 1 — NORMAL TRAFFIC")
    print("-" * 80)
    print(
        f"Risk Score : "
        f"{normal_result.risk_score}/100"
    )
    print(
        f"Risk Level : "
        f"{normal_result.risk_level}"
    )
    print(
        f"Components : "
        f"{normal_result.components}"
    )

    # -------------------------------------------------------------------------
    # Test 2 — Medium-confidence exploit
    # -------------------------------------------------------------------------

    medium_confidence_result = assess_risk(
        detection_status="Attack",
        attack_category="Exploits",
        anomaly_status="Normal",
        anomaly_score=0.05,
        binary_attack_probability=0.55,
        attack_category_probability=0.60,
    )

    print()
    print("-" * 80)
    print("TEST 2 — MEDIUM-CONFIDENCE EXPLOIT")
    print("-" * 80)
    print(
        f"Risk Score : "
        f"{medium_confidence_result.risk_score}/100"
    )
    print(
        f"Risk Level : "
        f"{medium_confidence_result.risk_level}"
    )
    print(
        f"Components : "
        f"{medium_confidence_result.components}"
    )

    # -------------------------------------------------------------------------
    # Test 3 — High-confidence exploit
    # -------------------------------------------------------------------------

    high_confidence_result = assess_risk(
        detection_status="Attack",
        attack_category="Exploits",
        anomaly_status="Normal",
        anomaly_score=0.05,
        binary_attack_probability=0.99,
        attack_category_probability=0.99,
    )

    print()
    print("-" * 80)
    print("TEST 3 — HIGH-CONFIDENCE EXPLOIT")
    print("-" * 80)
    print(
        f"Risk Score : "
        f"{high_confidence_result.risk_score}/100"
    )
    print(
        f"Risk Level : "
        f"{high_confidence_result.risk_level}"
    )
    print(
        f"Components : "
        f"{high_confidence_result.components}"
    )

    # -------------------------------------------------------------------------
    # Test 4 — Highly anomalous DoS
    # -------------------------------------------------------------------------

    anomalous_dos_result = assess_risk(
        detection_status="Attack",
        attack_category="DoS",
        anomaly_status="Anomaly",
        anomaly_score=-0.12,
        binary_attack_probability=0.99,
        attack_category_probability=0.99,
    )

    print()
    print("-" * 80)
    print("TEST 4 — HIGHLY ANOMALOUS DOS")
    print("-" * 80)
    print(
        f"Risk Score : "
        f"{anomalous_dos_result.risk_score}/100"
    )
    print(
        f"Risk Level : "
        f"{anomalous_dos_result.risk_level}"
    )
    print(
        f"Components : "
        f"{anomalous_dos_result.components}"
    )

    # -------------------------------------------------------------------------
    # T1 VALIDATION
    # -------------------------------------------------------------------------

    assert (
        medium_confidence_result.risk_score
        < high_confidence_result.risk_score
    ), (
        "T1 FAILED: higher binary/category confidence "
        "must produce a higher score."
    )

    assert (
        high_confidence_result.risk_score
        < 75
    ), (
        "T1 FAILED: high-confidence non-anomalous exploit "
        "should not automatically become CRITICAL."
    )

    assert (
        anomalous_dos_result.risk_score
        > high_confidence_result.risk_score
    ), (
        "T1 FAILED: strong anomaly evidence should increase "
        "the Threat Priority Score."
    )

    assert (
        normal_result.risk_score
        == 0
    ), (
        "T1 FAILED: normal traffic with no attack/anomaly "
        "evidence should have a zero priority score."
    )

    assert (
        0
        <= anomalous_dos_result.risk_score
        <= 100
    ), (
        "T1 FAILED: score must remain between 0 and 100."
    )

    assert (
        anomalous_dos_result.risk_level
        == "CRITICAL"
    ), (
        "T1 FAILED: highly anomalous DoS should be CRITICAL."
    )

    assert (
        "Escalate for immediate analyst review"
        in anomalous_dos_result.recommended_action
    ), (
        "T1 FAILED: CRITICAL recommendation must require "
        "immediate analyst review."
    )

    print()
    print("=" * 80)
    print("T1 RISK ASSESSMENT TEST PASSED")
    print("=" * 80)