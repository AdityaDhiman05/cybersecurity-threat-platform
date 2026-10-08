"""
================================================================================
CYBERSECURITY THREAT PLATFORM
DATABASE MANAGEMENT MODULE
================================================================================

Purpose
-------
Provides persistent storage for cybersecurity threat-analysis results and
analyst investigation feedback.

The database manager supports:

    1. Database initialization
    2. Safe schema migration
    3. Prediction storage
    4. Prediction retrieval
    5. Prediction counting
    6. Risk summaries
    7. Attack summaries
    8. Analyst feedback storage
    9. Feedback retrieval
    10. Feedback summaries
    11. Training-eligible feedback retrieval
    12. Database health checks

Database Design
---------------
The module uses SQLite and preserves existing records when a database already
exists.

New databases are created using the final schema.

Existing databases are safely migrated by checking the current schema with
PRAGMA table_info() and adding missing columns where required.

Analyst feedback is linked to its prediction using a foreign key with
ON DELETE CASCADE.

A prediction can have only one analyst-feedback record because prediction_id
is UNIQUE in the analyst_feedback table.

T10 Analyst Feedback
--------------------
Analyst feedback distinguishes between:

    - Binary prediction correctness
    - Attack-category prediction correctness

The older analyst_verdict field is retained for backward compatibility.

Analyst feedback never automatically becomes training data.
"""

from __future__ import annotations


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PROJECT PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIRECTORY = PROJECT_ROOT / "database"

DATABASE_PATH = (
    DATABASE_DIRECTORY / "threat_platform.db"
)


# =============================================================================
# MODEL VERSION
# =============================================================================

MODEL_VERSION = "v1.0-2026-10"


# =============================================================================
# DATABASE TABLE NAMES
# =============================================================================

PREDICTION_TABLE_NAME = "threat_predictions"

FEEDBACK_TABLE_NAME = "analyst_feedback"


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """
    Manage persistent storage for the cybersecurity threat platform.

    The manager handles:

        1. Database initialization
        2. Safe schema migration
        3. Prediction storage
        4. Prediction retrieval
        5. Prediction counting
        6. Risk summaries
        7. Attack summaries
        8. Analyst feedback storage
        9. Feedback retrieval
        10. Feedback summaries
        11. Training-eligible feedback
        12. Database health checks

    Existing records are preserved during safe schema migration.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        database_path: Path | str = DATABASE_PATH,
    ) -> None:

        self.database_path = Path(
            database_path
        )

        self.database_directory = (
            self.database_path.parent
        )

        self.database_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize_database()


    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def get_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection.

        Foreign-key enforcement is enabled for every connection because
        SQLite does not enable foreign keys automatically.
        """

        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        return connection


    # =========================================================================
    # DATABASE INITIALIZATION
    # =========================================================================

    def initialize_database(self) -> None:
        """
        Initialize the database schema.

        Creates the final schema for new databases and safely adds newly
        introduced columns to existing databases.

        Existing records are preserved during migration.
        """

        connection = self.get_connection()

        try:

            # -----------------------------------------------------------------
            # PREDICTION TABLE
            # -----------------------------------------------------------------

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE_NAME} (

                    id
                        INTEGER PRIMARY KEY AUTOINCREMENT,

                    timestamp
                        TEXT NOT NULL,

                    dataset
                        TEXT NOT NULL,

                    detection_status
                        TEXT,

                    attack_category
                        TEXT,

                    attack_probability
                        REAL,

                    attack_category_probability
                        REAL,

                    anomaly_status
                        TEXT,

                    anomaly_score
                        REAL,

                    risk_score
                        INTEGER,

                    risk_level
                        TEXT,

                    risk_components
                        TEXT,

                    model_version
                        TEXT,

                    explanation
                        TEXT,

                    recommended_action
                        TEXT,

                    network_event
                        TEXT NOT NULL
                );
                """
            )

            # -----------------------------------------------------------------
            # SAFE MIGRATION FOR EXISTING PREDICTION TABLES
            # -----------------------------------------------------------------

            existing_columns = {
                row["name"]
                for row in connection.execute(
                    f"PRAGMA table_info({PREDICTION_TABLE_NAME});"
                ).fetchall()
            }

            required_columns = {
                "model_version": "TEXT",
                "attack_category_probability": "REAL",
                "risk_components": "TEXT",
            }

            for column_name, column_type in required_columns.items():

                if column_name not in existing_columns:

                    connection.execute(
                        f"""
                        ALTER TABLE {PREDICTION_TABLE_NAME}
                        ADD COLUMN {column_name} {column_type};
                        """
                    )

            # -----------------------------------------------------------------
            # SAFE MIGRATION FOR EXISTING ANALYST FEEDBACK TABLE
            # -----------------------------------------------------------------

            feedback_table_exists = connection.execute(
                f"""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = ?;
                """,
                (FEEDBACK_TABLE_NAME,),
            ).fetchone()

            if feedback_table_exists is not None:

                feedback_existing_columns = {
                    row["name"]
                    for row in connection.execute(
                        f"PRAGMA table_info({FEEDBACK_TABLE_NAME});"
                    ).fetchall()
                }

                if "binary_correct" not in feedback_existing_columns:

                    connection.execute(
                        f"""
                        ALTER TABLE {FEEDBACK_TABLE_NAME}
                        ADD COLUMN binary_correct TEXT;
                        """
                    )

            # -----------------------------------------------------------------
            # ANALYST FEEDBACK TABLE
            # -----------------------------------------------------------------

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE_NAME} (

                    feedback_id
                        INTEGER PRIMARY KEY AUTOINCREMENT,

                    prediction_id
                        INTEGER NOT NULL UNIQUE,

                    feedback_timestamp
                        TEXT NOT NULL,

                    analyst_verdict
                        TEXT NOT NULL,

                    binary_correct
                        TEXT,

                    actual_result
                        TEXT NOT NULL,

                    actual_attack_category
                        TEXT,

                    category_correct
                        TEXT,

                    is_unknown_attack
                        INTEGER NOT NULL DEFAULT 0,

                    root_cause
                        TEXT,

                    analyst_notes
                        TEXT,

                    training_eligible
                        INTEGER NOT NULL DEFAULT 0,

                    FOREIGN KEY (prediction_id)
                        REFERENCES threat_predictions(id)
                        ON DELETE CASCADE
                );
                """
            )

            connection.commit()

        finally:

            connection.close()


    # =========================================================================
    # SAVE PREDICTION
    # =========================================================================

    def save_prediction(
        self,
        prediction: dict[str, Any],
        network_event: dict[str, Any] | None = None,
        source: str = "API",
        dataset_index: int | None = None,
        actual_label: int | None = None,
        actual_attack_category: str | None = None,
    ) -> int:
        """
        Store one cybersecurity prediction.

        Parameters
        ----------
        prediction
            Prediction generated by the ML pipeline.

        network_event
            Raw network-event features.

        source
            Source identifier.

        dataset_index
            UNSW-NB15 record index when applicable.

        actual_label
            Ground-truth label when available.

        actual_attack_category
            Ground-truth category when available.

        Notes
        -----
        The existing database has a single `dataset` column instead of
        separate `source` and `dataset_index` columns.

        Therefore:

            source + dataset_index

        are combined into the existing `dataset` field.

        The current model version is stored with every prediction.
        """

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # ---------------------------------------------------------------------
        # Determine dataset/source value.
        # ---------------------------------------------------------------------

        if dataset_index is not None:

            dataset_value = (
                f"{source}:{dataset_index}"
            )

        else:

            dataset_value = str(
                source
            )

        # ---------------------------------------------------------------------
        # Serialize network event.
        # ---------------------------------------------------------------------

        if network_event is None:

            network_event_json = "{}"

        else:

            network_event_json = json.dumps(
                network_event,
                default=str,
            )

        connection = self.get_connection()

        try:

            # -----------------------------------------------------------------
            # Serialize risk components.
            # -----------------------------------------------------------------

            risk_components = prediction.get(
                "risk_components"
            )

            if risk_components is None:

                risk_components_json = None

            else:

                risk_components_json = json.dumps(
                    risk_components,
                    default=str,
                )

            # -----------------------------------------------------------------
            # Insert prediction.
            # -----------------------------------------------------------------

            cursor = connection.execute(
                f"""
                INSERT INTO {PREDICTION_TABLE_NAME} (

                    timestamp,
                    dataset,
                    detection_status,
                    attack_category,
                    attack_probability,
                    attack_category_probability,
                    anomaly_status,
                    anomaly_score,
                    risk_score,
                    risk_level,
                    risk_components,
                    model_version,
                    explanation,
                    recommended_action,
                    network_event

                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    timestamp,

                    dataset_value,

                    prediction.get(
                        "detection_status"
                    ),

                    prediction.get(
                        "attack_category"
                    ),

                    prediction.get(
                        "attack_probability"
                    ),

                    prediction.get(
                        "attack_category_probability"
                    ),

                    prediction.get(
                        "anomaly_status"
                    ),

                    prediction.get(
                        "anomaly_score"
                    ),

                    prediction.get(
                        "risk_score"
                    ),

                    prediction.get(
                        "risk_level"
                    ),

                    risk_components_json,

                    MODEL_VERSION,

                    prediction.get(
                        "explanation"
                    ),

                    prediction.get(
                        "recommended_action"
                    ),

                    network_event_json,
                ),
            )

            connection.commit()

            prediction_id = cursor.lastrowid

        finally:

            connection.close()

        return int(
            prediction_id
        )


    # =========================================================================
    # RETRIEVE RECENT PREDICTIONS
    # =========================================================================

    def get_recent_predictions(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent prediction records.

        The database uses `id`.

        For API/dashboard compatibility, the returned dictionary also exposes:

            prediction_id
            source
        """

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        connection = self.get_connection()

        try:

            rows = connection.execute(
                f"""
                SELECT
                    id,
                    timestamp,
                    dataset,
                    detection_status,
                    attack_category,
                    attack_probability,
                    attack_category_probability,
                    anomaly_status,
                    anomaly_score,
                    risk_score,
                    risk_level,
                    risk_components,
                    model_version,
                    explanation,
                    recommended_action,
                    network_event
                FROM {PREDICTION_TABLE_NAME}
                ORDER BY id DESC
                LIMIT ?;
                """,
                (
                    limit,
                ),
            ).fetchall()

            records = []

            for row in rows:

                record = dict(
                    row
                )

                # -------------------------------------------------------------
                # Compatibility aliases.
                # -------------------------------------------------------------

                record["prediction_id"] = (
                    record["id"]
                )

                record["source"] = (
                    record["dataset"]
                )

                # -------------------------------------------------------------
                # Deserialize risk components.
                # -------------------------------------------------------------

                risk_components = record.get(
                    "risk_components"
                )

                if risk_components:

                    try:

                        record["risk_components"] = (
                            json.loads(
                                risk_components
                            )
                        )

                    except (
                        json.JSONDecodeError,
                        TypeError,
                    ):

                        record["risk_components"] = {}

                else:

                    record["risk_components"] = {}

                records.append(
                    record
                )

            return records

        finally:

            connection.close()


    # =========================================================================
    # RETRIEVE SINGLE PREDICTION
    # =========================================================================

    def get_prediction(
        self,
        prediction_id: int,
    ) -> dict[str, Any] | None:
        """
        Retrieve one prediction using the database `id` column.
        """

        connection = self.get_connection()

        try:

            row = connection.execute(
                f"""
                SELECT
                    id,
                    timestamp,
                    dataset,
                    detection_status,
                    attack_category,
                    attack_probability,
                    attack_category_probability,
                    anomaly_status,
                    anomaly_score,
                    risk_score,
                    risk_level,
                    risk_components,
                    model_version,
                    explanation,
                    recommended_action,
                    network_event
                FROM {PREDICTION_TABLE_NAME}
                WHERE id = ?;
                """,
                (
                    prediction_id,
                ),
            ).fetchone()

            if row is None:

                return None

            record = dict(
                row
            )

            # -----------------------------------------------------------------
            # Compatibility aliases.
            # -----------------------------------------------------------------

            record["prediction_id"] = (
                record["id"]
            )

            record["source"] = (
                record["dataset"]
            )

            # -----------------------------------------------------------------
            # Deserialize risk components.
            # -----------------------------------------------------------------

            risk_components = record.get(
                "risk_components"
            )

            if risk_components:

                try:

                    record["risk_components"] = (
                        json.loads(
                            risk_components
                        )
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    record["risk_components"] = {}

            else:

                record["risk_components"] = {}

            return record

        finally:

            connection.close()


    # =========================================================================
    # COUNT PREDICTIONS
    # =========================================================================

    def count_predictions(self) -> int:
        """
        Return the number of stored predictions.
        """

        connection = self.get_connection()

        try:

            row = connection.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM {PREDICTION_TABLE_NAME};
                """
            ).fetchone()

            return int(
                row["total"]
            )

        finally:

            connection.close()


    # =========================================================================
    # RISK SUMMARY
    # =========================================================================

    def get_risk_summary(
        self,
    ) -> dict[str, int]:
        """
        Return prediction counts grouped by risk level.
        """

        connection = self.get_connection()

        try:

            rows = connection.execute(
                f"""
                SELECT
                    risk_level,
                    COUNT(*) AS total
                FROM {PREDICTION_TABLE_NAME}
                GROUP BY risk_level;
                """
            ).fetchall()

            return {
                str(
                    row["risk_level"]
                ): int(
                    row["total"]
                )
                for row in rows
                if row["risk_level"] is not None
            }

        finally:

            connection.close()


    # =========================================================================
    # ATTACK SUMMARY
    # =========================================================================

    def get_attack_summary(
        self,
    ) -> dict[str, int]:
        """
        Return detected attack counts grouped by category.
        """

        connection = self.get_connection()

        try:

            rows = connection.execute(
                f"""
                SELECT
                    attack_category,
                    COUNT(*) AS total
                FROM {PREDICTION_TABLE_NAME}
                WHERE detection_status = 'Attack'
                GROUP BY attack_category
                ORDER BY total DESC;
                """
            ).fetchall()

            return {
                str(
                    row["attack_category"]
                ): int(
                    row["total"]
                )
                for row in rows
                if row["attack_category"] is not None
            }

        finally:

            connection.close()


    # =========================================================================
    # SAVE ANALYST FEEDBACK
    # =========================================================================

    def save_feedback(
        self,
        prediction_id: int,
        analyst_verdict: str,
        actual_result: str,
        actual_attack_category: str | None = None,
        binary_correct: str | None = None,
        category_correct: str | None = None,
        is_unknown_attack: bool = False,
        root_cause: str | None = None,
        analyst_notes: str | None = None,
        training_eligible: bool = False,
    ) -> int:
        """
        Store analyst feedback for one prediction.

        T10 introduces two explicit correctness fields:

            binary_correct
                Yes / No / Unknown

            category_correct
                Yes / No / Unknown / None

        The older `analyst_verdict` field is retained for compatibility.

        Feedback is NOT automatically eligible for model training.
        """

        # ---------------------------------------------------------------------
        # Normalize values.
        # ---------------------------------------------------------------------

        analyst_verdict = str(
            analyst_verdict
        ).strip()

        actual_result = str(
            actual_result
        ).strip()

        if actual_attack_category is not None:

            actual_attack_category = str(
                actual_attack_category
            ).strip()

        if binary_correct is not None:

            binary_correct = str(
                binary_correct
            ).strip()

        if category_correct is not None:

            category_correct = str(
                category_correct
            ).strip()

        # ---------------------------------------------------------------------
        # Validate analyst verdict.
        # ---------------------------------------------------------------------

        valid_verdicts = {
            "Correct",
            "Incorrect",
            "Uncertain",
        }

        if analyst_verdict not in valid_verdicts:

            raise ValueError(
                "analyst_verdict must be one of: "
                "Correct, Incorrect, Uncertain."
            )

        # ---------------------------------------------------------------------
        # Validate actual result.
        # ---------------------------------------------------------------------

        valid_results = {
            "Attack",
            "Normal",
            "Unknown",
        }

        if actual_result not in valid_results:

            raise ValueError(
                "actual_result must be one of: "
                "Attack, Normal, Unknown."
            )

        # ---------------------------------------------------------------------
        # Validate binary correctness.
        # ---------------------------------------------------------------------

        valid_binary_values = {
            "Yes",
            "No",
            "Unknown",
            None,
        }

        if binary_correct not in valid_binary_values:

            raise ValueError(
                "binary_correct must be one of: "
                "Yes, No, Unknown, or None."
            )

        # ---------------------------------------------------------------------
        # Validate category correctness.
        # ---------------------------------------------------------------------

        valid_category_values = {
            None,
            "Yes",
            "No",
            "Unknown",
        }

        if category_correct not in valid_category_values:

            raise ValueError(
                "category_correct must be one of: "
                "Yes, No, Unknown, or None."
            )

        # ---------------------------------------------------------------------
        # Normalize booleans.
        # ---------------------------------------------------------------------

        is_unknown_attack = bool(
            is_unknown_attack
        )

        # ---------------------------------------------------------------------
        # Analyst feedback should not automatically become training data.
        #
        # Even if the caller passes True, the platform deliberately keeps
        # this false until a separate controlled review process approves it.
        #
        # This prevents accidental model poisoning.
        # ---------------------------------------------------------------------

        training_eligible = False

        # ---------------------------------------------------------------------
        # Verify prediction exists.
        # ---------------------------------------------------------------------

        prediction = self.get_prediction(
            prediction_id
        )

        if prediction is None:

            raise ValueError(
                f"Prediction ID {prediction_id} "
                "does not exist."
            )

        connection = self.get_connection()

        try:

            # -----------------------------------------------------------------
            # Prevent duplicate feedback.
            #
            # The database UNIQUE constraint also enforces this at the
            # database level.
            # -----------------------------------------------------------------

            existing_feedback = connection.execute(
                f"""
                SELECT feedback_id
                FROM {FEEDBACK_TABLE_NAME}
                WHERE prediction_id = ?;
                """,
                (
                    prediction_id,
                ),
            ).fetchone()

            if existing_feedback is not None:

                raise ValueError(
                    f"Feedback already exists for "
                    f"prediction ID {prediction_id}."
                )

            # -----------------------------------------------------------------
            # Insert feedback.
            # -----------------------------------------------------------------

            feedback_timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            cursor = connection.execute(
                f"""
                INSERT INTO {FEEDBACK_TABLE_NAME} (

                    prediction_id,
                    feedback_timestamp,
                    analyst_verdict,
                    binary_correct,
                    actual_result,
                    actual_attack_category,
                    category_correct,
                    is_unknown_attack,
                    root_cause,
                    analyst_notes,
                    training_eligible

                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    prediction_id,

                    feedback_timestamp,

                    analyst_verdict,

                    binary_correct,

                    actual_result,

                    actual_attack_category,

                    category_correct,

                    int(
                        is_unknown_attack
                    ),

                    root_cause,

                    analyst_notes,

                    int(
                        training_eligible
                    ),
                ),
            )

            connection.commit()

            feedback_id = cursor.lastrowid

        finally:

            connection.close()

        return int(
            feedback_id
        )


    # =========================================================================
    # RETRIEVE FEEDBACK FOR ONE PREDICTION
    # =========================================================================

    def get_feedback_for_prediction(
        self,
        prediction_id: int,
    ) -> dict[str, Any] | None:
        """
        Retrieve feedback belonging ONLY to the requested prediction.
        """

        connection = self.get_connection()

        try:

            row = connection.execute(
                f"""
                SELECT
                    feedback_id,
                    prediction_id,
                    feedback_timestamp,
                    analyst_verdict,
                    binary_correct,
                    actual_result,
                    actual_attack_category,
                    category_correct,
                    is_unknown_attack,
                    root_cause,
                    analyst_notes,
                    training_eligible
                FROM {FEEDBACK_TABLE_NAME}
                WHERE prediction_id = ?;
                """,
                (
                    prediction_id,
                ),
            ).fetchone()

            if row is None:

                return None

            return dict(
                row
            )

        finally:

            connection.close()


    # =========================================================================
    # RETRIEVE RECENT FEEDBACK
    # =========================================================================

    def get_recent_feedback(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent analyst feedback.
        """

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        connection = self.get_connection()

        try:

            rows = connection.execute(
                f"""
                SELECT
                    feedback_id,
                    prediction_id,
                    feedback_timestamp,
                    analyst_verdict,
                    binary_correct,
                    actual_result,
                    actual_attack_category,
                    category_correct,
                    is_unknown_attack,
                    root_cause,
                    analyst_notes,
                    training_eligible
                FROM {FEEDBACK_TABLE_NAME}
                ORDER BY feedback_id DESC
                LIMIT ?;
                """,
                (
                    limit,
                ),
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()


    # =========================================================================
    # COUNT FEEDBACK
    # =========================================================================

    def count_feedback(self) -> int:
        """
        Return the total number of feedback records.
        """

        connection = self.get_connection()

        try:

            row = connection.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM {FEEDBACK_TABLE_NAME};
                """
            ).fetchone()

            return int(
                row["total"]
            )

        finally:

            connection.close()


    # =========================================================================
    # FEEDBACK SUMMARY
    # =========================================================================

    def get_feedback_summary(
        self,
    ) -> dict[str, int]:
        """
        Return high-level analyst feedback statistics.

        T10 separately reports binary prediction agreement and
        attack-category agreement.
        """

        connection = self.get_connection()

        try:

            row = connection.execute(
                f"""
                SELECT

                    COUNT(*) AS total_feedback,

                    SUM(
                        CASE
                            WHEN analyst_verdict = 'Correct'
                            THEN 1
                            ELSE 0
                        END
                    ) AS correct_predictions,

                    SUM(
                        CASE
                            WHEN analyst_verdict = 'Incorrect'
                            THEN 1
                            ELSE 0
                        END
                    ) AS incorrect_predictions,

                    SUM(
                        CASE
                            WHEN analyst_verdict = 'Uncertain'
                            THEN 1
                            ELSE 0
                        END
                    ) AS uncertain_predictions,

                    SUM(
                        CASE
                            WHEN binary_correct = 'Yes'
                            THEN 1
                            ELSE 0
                        END
                    ) AS binary_correct,

                    SUM(
                        CASE
                            WHEN binary_correct = 'No'
                            THEN 1
                            ELSE 0
                        END
                    ) AS binary_incorrect,

                    SUM(
                        CASE
                            WHEN binary_correct = 'Unknown'
                                 OR binary_correct IS NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS binary_unknown,

                    SUM(
                        CASE
                            WHEN actual_result = 'Attack'
                            THEN 1
                            ELSE 0
                        END
                    ) AS actual_attacks,

                    SUM(
                        CASE
                            WHEN actual_result = 'Normal'
                            THEN 1
                            ELSE 0
                        END
                    ) AS actual_normal,

                    SUM(
                        CASE
                            WHEN actual_result = 'Unknown'
                            THEN 1
                            ELSE 0
                        END
                    ) AS actual_unknown,

                    SUM(
                        CASE
                            WHEN category_correct = 'Yes'
                            THEN 1
                            ELSE 0
                        END
                    ) AS category_correct,

                    SUM(
                        CASE
                            WHEN category_correct = 'No'
                            THEN 1
                            ELSE 0
                        END
                    ) AS category_incorrect,

                    SUM(
                        CASE
                            WHEN category_correct = 'Unknown'
                                 OR category_correct IS NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS category_unknown,

                    SUM(
                        CASE
                            WHEN is_unknown_attack = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS unknown_attacks,

                    SUM(
                        CASE
                            WHEN training_eligible = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS training_eligible

                FROM {FEEDBACK_TABLE_NAME};
                """
            ).fetchone()

            return {

                "total_feedback": int(
                    row["total_feedback"] or 0
                ),

                "correct_predictions": int(
                    row["correct_predictions"] or 0
                ),

                "incorrect_predictions": int(
                    row["incorrect_predictions"] or 0
                ),

                "uncertain_predictions": int(
                    row["uncertain_predictions"] or 0
                ),

                "binary_correct": int(
                    row["binary_correct"] or 0
                ),

                "binary_incorrect": int(
                    row["binary_incorrect"] or 0
                ),

                "binary_unknown": int(
                    row["binary_unknown"] or 0
                ),

                "actual_attacks": int(
                    row["actual_attacks"] or 0
                ),

                "actual_normal": int(
                    row["actual_normal"] or 0
                ),

                "actual_unknown": int(
                    row["actual_unknown"] or 0
                ),

                "category_correct": int(
                    row["category_correct"] or 0
                ),

                "category_incorrect": int(
                    row["category_incorrect"] or 0
                ),

                "category_unknown": int(
                    row["category_unknown"] or 0
                ),

                "unknown_attacks": int(
                    row["unknown_attacks"] or 0
                ),

                "training_eligible": int(
                    row["training_eligible"] or 0
                ),
            }

        finally:

            connection.close()


    # =========================================================================
    # TRAINING-ELIGIBLE FEEDBACK
    # =========================================================================

    def get_training_eligible_feedback(
        self,
    ) -> list[dict[str, Any]]:
        """
        Retrieve feedback explicitly approved for future controlled
        model evaluation/retraining.

        Feedback is never automatically marked eligible.
        """

        connection = self.get_connection()

        try:

            rows = connection.execute(
                f"""
                SELECT
                    feedback_id,
                    prediction_id,
                    feedback_timestamp,
                    analyst_verdict,
                    binary_correct,
                    actual_result,
                    actual_attack_category,
                    category_correct,
                    is_unknown_attack,
                    root_cause,
                    analyst_notes,
                    training_eligible
                FROM {FEEDBACK_TABLE_NAME}
                WHERE training_eligible = 1
                ORDER BY feedback_id ASC;
                """
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            connection.close()


    # =========================================================================
    # DATABASE HEALTH CHECK
    # =========================================================================

    def health_check(
        self,
    ) -> dict[str, Any]:
        """
        Check whether the database is accessible and operational.
        """

        try:

            connection = self.get_connection()

            connection.execute(
                "SELECT 1;"
            )

            connection.close()

            return {
                "status": "operational",

                "database": str(
                    self.database_path
                ),

                "records": (
                    self.count_predictions()
                ),

                "feedback_records": (
                    self.count_feedback()
                ),
            }

        except Exception as exc:

            return {
                "status": "error",

                "database": str(
                    self.database_path
                ),

                "error": str(
                    exc
                ),
            }


# =============================================================================
# MODULE-LEVEL DATABASE INSTANCE
# =============================================================================

database_manager = DatabaseManager()


# =============================================================================
# DATABASE TEST
# =============================================================================

def run_database_test() -> None:
    """
    Verify database accessibility without inserting test data.
    """

    print("=" * 80)

    print(
        "CYBERSECURITY THREAT PLATFORM"
    )

    print(
        "DATABASE INTEGRATION TEST"
    )

    print("=" * 80)

    print()

    print(
        f"Database path : "
        f"{DATABASE_PATH}"
    )

    print()

    print(
        "Checking database..."
    )

    health = (
        database_manager.health_check()
    )

    print(
        f"Status        : "
        f"{health['status']}"
    )

    print(
        f"Predictions   : "
        f"{health.get('records', 0)}"
    )

    print(
        f"Feedback      : "
        f"{health.get('feedback_records', 0)}"
    )

    if health["status"] != "operational":

        print()

        print(
            "Database integration test FAILED."
        )

        print(
            health.get(
                "error",
                "Unknown database error.",
            )
        )

        return

    # -------------------------------------------------------------------------
    # Database schema verification.
    # -------------------------------------------------------------------------

    print()

    print(
        "Database schema:"
    )

    connection = database_manager.get_connection()

    try:

        prediction_columns = connection.execute(
            f"""
            PRAGMA table_info(
                {PREDICTION_TABLE_NAME}
            );
            """
        ).fetchall()

        prediction_column_names = [
            row["name"]
            for row in prediction_columns
        ]

        print(
            "  Prediction columns:"
        )

        for column_name in prediction_column_names:

            print(
                f"    - {column_name}"
            )

        feedback_columns = connection.execute(
            f"""
            PRAGMA table_info(
                {FEEDBACK_TABLE_NAME}
            );
            """
        ).fetchall()

        feedback_column_names = [
            row["name"]
            for row in feedback_columns
        ]

        print(
            "  Feedback columns:"
        )

        for column_name in feedback_column_names:

            print(
                f"    - {column_name}"
            )

        foreign_keys = connection.execute(
            f"""
            PRAGMA foreign_key_list(
                {FEEDBACK_TABLE_NAME}
            );
            """
        ).fetchall()

        print()

        print(
            "Foreign keys:"
        )

        for foreign_key in foreign_keys:

            print(
                f"  {dict(foreign_key)}"
            )

    finally:

        connection.close()

    # -------------------------------------------------------------------------
    # Recent predictions.
    # -------------------------------------------------------------------------

    print()

    print(
        "Recent predictions:"
    )

    predictions = (
        database_manager
        .get_recent_predictions(
            limit=5
        )
    )

    for prediction in predictions:

        print(
            f"  ID={prediction['prediction_id']} | "
            f"Detection={prediction['detection_status']} | "
            f"Category={prediction['attack_category']} | "
            f"Risk={prediction['risk_level']} | "
            f"Model={prediction.get('model_version')}"
        )

    # -------------------------------------------------------------------------
    # Feedback summary.
    # -------------------------------------------------------------------------

    print()

    print(
        "Analyst feedback summary:"
    )

    feedback_summary = (
        database_manager
        .get_feedback_summary()
    )

    for key, value in (
        feedback_summary.items()
    ):

        print(
            f"  {key:<24}: {value}"
        )

    # -------------------------------------------------------------------------
    # Training eligibility.
    # -------------------------------------------------------------------------

    print()

    eligible_feedback = (
        database_manager
        .get_training_eligible_feedback()
    )

    print(
        "Training-eligible feedback:"
        f" {len(eligible_feedback)}"
    )

    print()

    print("=" * 80)

    print(
        "DATABASE INTEGRATION TEST COMPLETED"
    )

    print("=" * 80)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_database_test()