from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_FILE = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TESTING_FILE = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")
training_data = pd.read_csv(TRAINING_FILE)

print("Loading testing dataset...")
testing_data = pd.read_csv(TESTING_FILE)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DATA")
print("=" * 60)

print("Rows:", training_data.shape[0])
print("Columns:", training_data.shape[1])

print("\nColumn names:")
print(training_data.columns.tolist())


print("\n" + "=" * 60)
print("TESTING DATA")
print("=" * 60)

print("Rows:", testing_data.shape[0])
print("Columns:", testing_data.shape[1])


# ============================================================
# ATTACK / NORMAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LABEL DISTRIBUTION")
print("=" * 60)

print(training_data["label"].value_counts())


# ============================================================
# ATTACK CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ATTACK CATEGORIES")
print("=" * 60)

print(training_data["attack_cat"].value_counts())