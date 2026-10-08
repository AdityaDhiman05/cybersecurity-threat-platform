# =============================================================================
# ATTACK CLASSIFIER - CONTROLLED MODEL IMPROVEMENT EXPERIMENT
# =============================================================================
#
# Purpose:
#   1. Load the UNSW-NB15 training and testing datasets.
#   2. Keep only attack traffic for attack-category classification.
#   3. Split the official training attack data into:
#        - Training subset
#        - Validation subset
#   4. Compare controlled Random Forest configurations.
#   5. Select the best configuration using:
#        - Balanced Accuracy
#        - Macro F1
#        - Per-class recall
#   6. Retrain the selected configuration on ALL attack training data.
#   7. Evaluate the final selected model once on the official test set.
#   8. Save the final classifier and preprocessor.
#
# Dataset:
#   UNSW-NB15
#
# Target:
#   attack_cat
#
# Important:
#   The official test set is NOT used for model selection.
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_training-set.csv"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "UNSW_NB15_testing-set.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIRECTORY / "attack_classifier.pkl"
PREPROCESSOR_PATH = MODEL_DIRECTORY / "attack_preprocessor.pkl"


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20


# =============================================================================
# RANDOM FOREST EXPERIMENT CONFIGURATIONS
# =============================================================================
#
# Configuration 1:
#   Current baseline.
#
# Configuration 2:
#   Limits tree depth to reduce overly complex trees.
#
# Configuration 3:
#   Uses balanced_subsample so class weights are recalculated
#   for individual bootstrap samples.
#
# =============================================================================

MODEL_CONFIGURATIONS = {
    "Baseline": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    "Depth_20": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    "Balanced_SubSample": {
        "n_estimators": 500,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_datasets():
    """
    Load the official UNSW-NB15 training and testing datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Training and testing dataframes.
    """

    print("=" * 80)
    print("LOADING UNSW-NB15 DATASETS")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape : {test_df.shape}")

    return train_df, test_df


# =============================================================================
# ATTACK DATA PREPARATION
# =============================================================================


def prepare_attack_data(train_df, test_df):
    """
    Extract attack traffic and separate features from target.

    Only records where label == 1 are retained because this model
    classifies the category of an already-detected attack.

    Returns:
        tuple:
            X_train_attack
            y_train_attack
            X_test_attack
            y_test_attack
    """

    print("\n" + "=" * 80)
    print("PREPARING ATTACK-ONLY DATA")
    print("=" * 80)

    train_attack = train_df[train_df["label"] == 1].copy()
    test_attack = test_df[test_df["label"] == 1].copy()

    print(f"Attack training records: {len(train_attack)}")
    print(f"Attack testing records : {len(test_attack)}")

    print("\nTraining attack distribution:")
    print(train_attack["attack_cat"].value_counts().sort_index())

    X_train_attack = train_attack.drop(
        columns=["id", "attack_cat", "label"]
    )

    y_train_attack = train_attack["attack_cat"]

    X_test_attack = test_attack.drop(
        columns=["id", "attack_cat", "label"]
    )

    y_test_attack = test_attack["attack_cat"]

    return (
        X_train_attack,
        y_train_attack,
        X_test_attack,
        y_test_attack,
    )


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================


def build_preprocessor(X):
    """
    Build the preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Standard scaling

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding

    Returns:
        ColumnTransformer:
            Configured preprocessing pipeline.
    """

    categorical_features = [
        column
        for column in X.columns
        if pd.api.types.is_string_dtype(X[column])
    ]

    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    print("\nNumerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    print("\nCategorical columns:")
    for column in categorical_features:
        print(f"  - {column}")

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                SimpleImputer(strategy="most_frequent"),
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
        ]
    )

    return preprocessor


# =============================================================================
# MODEL CREATION
# =============================================================================


def build_model(parameters):
    """
    Create a Random Forest classifier using the supplied parameters.

    Args:
        parameters (dict):
            Random Forest configuration.

    Returns:
        RandomForestClassifier:
            Configured Random Forest model.
    """

    return RandomForestClassifier(
        **parameters,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# =============================================================================
# MODEL EXPERIMENT
# =============================================================================


def run_experiment(
    configuration_name,
    parameters,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train and evaluate one Random Forest configuration.

    Evaluation is performed only on the validation dataset.

    Returns:
        dict:
            Model performance metrics.
    """

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {configuration_name}")
    print("=" * 80)

    print("\nRandom Forest parameters:")

    for parameter, value in parameters.items():
        print(f"  {parameter}: {value}")

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    print("\nProcessed training shape:")
    print(X_train_processed.shape)

    print("\nProcessed validation shape:")
    print(X_validation_processed.shape)

    model = build_model(parameters)

    print("\nTraining model...")

    model.fit(
        X_train_processed,
        y_train,
    )

    print("Training completed.")

    validation_predictions = model.predict(
        X_validation_processed
    )

    accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        validation_predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
    )

    print("\nValidation results:")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    print("\nPer-class recall:")

    report = classification_report(
        y_validation,
        validation_predictions,
        output_dict=True,
        zero_division=0,
    )

    class_recalls = {}

    for class_name in sorted(y_validation.unique()):
        recall = report[class_name]["recall"]

        class_recalls[class_name] = recall

        print(
            f"{class_name:<16} {recall:.4f}"
        )

    return {
        "configuration": configuration_name,
        "parameters": parameters,
        "preprocessor": preprocessor,
        "model": model,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "class_recalls": class_recalls,
    }


# =============================================================================
# BEST MODEL SELECTION
# =============================================================================


def select_best_result(results):
    """
    Select the best model using balanced accuracy as the primary metric.

    Macro F1 is used as the secondary metric.

    Accuracy is used as the final tie-breaker.
    """

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison_rows = []

    for result in results:

        comparison_rows.append(
            {
                "Configuration": result["configuration"],
                "Accuracy": result["accuracy"],
                "Balanced Accuracy": result["balanced_accuracy"],
                "Macro F1": result["macro_f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df = comparison_df.sort_values(
        by=[
            "Balanced Accuracy",
            "Macro F1",
            "Accuracy",
        ],
        ascending=False,
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_configuration = comparison_df.iloc[0]["Configuration"]

    best_result = next(
        result
        for result in results
        if result["configuration"] == best_configuration
    )

    print("\nSelected configuration:")
    print(best_configuration)

    print(
        f"Balanced Accuracy: "
        f"{best_result['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best_result['macro_f1']:.4f}"
    )

    return best_result


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================


def train_final_model(
    best_result,
    X_train_attack,
    y_train_attack,
):
    """
    Retrain the selected Random Forest configuration using ALL
    available attack training data.

    Returns:
        tuple:
            Final preprocessor
            Final model
    """

    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)

    parameters = best_result["parameters"]

    print("Selected configuration:")
    print(best_result["configuration"])

    final_preprocessor = build_preprocessor(
        X_train_attack
    )

    X_train_processed = final_preprocessor.fit_transform(
        X_train_attack
    )

    print(
        f"\nFinal processed training shape: "
        f"{X_train_processed.shape}"
    )

    final_model = build_model(parameters)

    print("\nTraining final model on all attack training data...")

    final_model.fit(
        X_train_processed,
        y_train_attack,
    )

    print("Final model training completed.")

    return (
        final_preprocessor,
        final_model,
    )


# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================


def evaluate_final_model(
    preprocessor,
    model,
    X_test_attack,
    y_test_attack,
):
    """
    Evaluate the final selected model on the official UNSW-NB15
    attack test set.

    This is the first time the official test set is used for
    model selection/evaluation after configuration selection.
    """

    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON OFFICIAL TEST SET")
    print("=" * 80)

    X_test_processed = preprocessor.transform(
        X_test_attack
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test_attack,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_attack,
        predictions,
    )

    macro_f1 = f1_score(
        y_test_attack,
        predictions,
        average="macro",
    )

    print("\nFINAL PERFORMANCE")
    print("-" * 80)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_attack,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    labels = sorted(
        y_test_attack.unique()
    )

    matrix = confusion_matrix(
        y_test_attack,
        predictions,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(confusion_df)

    print("\nPER-CLASS RECALL")
    print("-" * 80)

    report = classification_report(
        y_test_attack,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    for class_name in labels:
        print(
            f"{class_name:<16} "
            f"{report[class_name]['recall']:.4f}"
        )


# =============================================================================
# MODEL SAVING
# =============================================================================


def save_model(
    preprocessor,
    model,
):
    """
    Save the final attack classifier and preprocessing pipeline.
    """

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print("\n" + "=" * 80)
    print("MODEL SAVED")
    print("=" * 80)

    print(f"Classifier   : {MODEL_PATH}")
    print(f"Preprocessor : {PREPROCESSOR_PATH}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Execute the complete controlled model-improvement experiment.
    """

    print("=" * 80)
    print("UNSW-NB15 ATTACK CATEGORY CLASSIFIER")
    print("CONTROLLED MODEL IMPROVEMENT EXPERIMENT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1 - Load datasets
    # -------------------------------------------------------------------------

    train_df, test_df = load_datasets()

    # -------------------------------------------------------------------------
    # STEP 2 - Prepare attack-only data
    # -------------------------------------------------------------------------

    (
        X_train_attack,
        y_train_attack,
        X_test_attack,
        y_test_attack,
    ) = prepare_attack_data(
        train_df,
        test_df,
    )

    # -------------------------------------------------------------------------
    # STEP 3 - Create validation split
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    # The official test set remains untouched.
    #
    # Stratification ensures that each attack category is represented
    # proportionally in both training and validation subsets.
    # -------------------------------------------------------------------------

    (
        X_train_split,
        X_validation,
        y_train_split,
        y_validation,
    ) = train_test_split(
        X_train_attack,
        y_train_attack,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_attack,
    )

    print("\n" + "=" * 80)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 80)

    print(
        f"Training subset   : {X_train_split.shape}"
    )

    print(
        f"Validation subset : {X_validation.shape}"
    )

    print("\nTraining class distribution:")
    print(
        y_train_split.value_counts().sort_index()
    )

    print("\nValidation class distribution:")
    print(
        y_validation.value_counts().sort_index()
    )

    # -------------------------------------------------------------------------
    # STEP 4 - Run controlled experiments
    # -------------------------------------------------------------------------

    results = []

    for configuration_name, parameters in MODEL_CONFIGURATIONS.items():

        result = run_experiment(
            configuration_name,
            parameters,
            X_train_split,
            y_train_split,
            X_validation,
            y_validation,
        )

        results.append(result)

    # -------------------------------------------------------------------------
    # STEP 5 - Select best configuration
    # -------------------------------------------------------------------------

    best_result = select_best_result(
        results
    )

    # -------------------------------------------------------------------------
    # STEP 6 - Retrain best configuration on ALL attack training data
    # -------------------------------------------------------------------------

    (
        final_preprocessor,
        final_model,
    ) = train_final_model(
        best_result,
        X_train_attack,
        y_train_attack,
    )

    # -------------------------------------------------------------------------
    # STEP 7 - Evaluate ONCE on official test set
    # -------------------------------------------------------------------------

    evaluate_final_model(
        final_preprocessor,
        final_model,
        X_test_attack,
        y_test_attack,
    )

    # -------------------------------------------------------------------------
    # STEP 8 - Save final model
    # -------------------------------------------------------------------------

    save_model(
        final_preprocessor,
        final_model,
    )

    print("\n" + "=" * 80)
    print("ATTACK CLASSIFIER EXPERIMENT COMPLETED")
    print("=" * 80)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()