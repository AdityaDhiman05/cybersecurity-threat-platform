"""
Seed the Cybersecurity Threat Platform with demo predictions.

Runs in the background when the API container starts. It waits for the API,
and if the prediction database is empty it sends a balanced sample of real
UNSW-NB15 testing events to /api/v1/predict, so every dashboard panel
(cards, charts, recent activity) has data to show.

Environment variables
---------------------
THREAT_PLATFORM_API_KEY  API key (same one the server uses)
DEMO_SEED                "0" disables seeding (default "1")
DEMO_SEED_COUNT          number of events to send (default 120)
SEED_API_URL             API base URL (default http://127.0.0.1:5000)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

API_URL = os.getenv("SEED_API_URL", "http://127.0.0.1:5000").rstrip("/")
API_KEY = os.getenv("THREAT_PLATFORM_API_KEY", "test-key-123")
HEADERS = {"X-API-Key": API_KEY}
TOTAL = int(os.getenv("DEMO_SEED_COUNT", "120"))
EXCLUDED = ["id", "label", "attack_cat"]


def log(message: str) -> None:
    print(f"[seed] {message}", flush=True)


def wait_for_api(timeout_seconds: int = 300) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if requests.get(f"{API_URL}/api/v1/health", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def prediction_count() -> int:
    response = requests.get(f"{API_URL}/api/v1/stats", headers=HEADERS, timeout=10)
    response.raise_for_status()
    return int(response.json().get("prediction_count", 0))


def choose_events(frame: pd.DataFrame, total: int) -> pd.DataFrame:
    """Balanced, repeatable sample: ~1/3 normal, the rest spread over attack types."""
    category = frame["attack_cat"].astype(str).str.strip()
    normal = frame[category == "Normal"]
    attacks = frame[category != "Normal"]

    normal_n = min(len(normal), max(1, total // 3))
    chosen = [normal.sample(normal_n, random_state=42)]

    categories = sorted(attacks["attack_cat"].astype(str).str.strip().unique())
    remaining = total - normal_n
    per_category = max(1, remaining // max(1, len(categories)))
    for name in categories:
        pool = attacks[attacks["attack_cat"].astype(str).str.strip() == name]
        chosen.append(pool.sample(min(len(pool), per_category), random_state=42))

    return pd.concat(chosen).sample(frac=1, random_state=7).head(total)


def to_event(row: pd.Series) -> dict:
    event = {}
    for name, value in row.drop(labels=EXCLUDED, errors="ignore").items():
        if pd.isna(value):
            event[name] = None
        elif hasattr(value, "item"):
            event[name] = value.item()
        else:
            event[name] = value
    return event


def main() -> int:
    if os.getenv("DEMO_SEED", "1") == "0":
        log("DEMO_SEED=0, skipping.")
        return 0
    if not DATASET.exists():
        log(f"dataset not found at {DATASET}, skipping.")
        return 0
    if not wait_for_api():
        log("API did not become healthy in time, skipping.")
        return 1

    try:
        existing = prediction_count()
    except Exception as error:
        log(f"could not read stats ({error}), skipping.")
        return 1
    if existing > 0:
        log(f"database already has {existing} predictions, skipping.")
        return 0

    frame = pd.read_csv(DATASET)
    events = choose_events(frame, TOTAL)
    log(f"sending {len(events)} events...")

    sent = failed = 0
    for _, row in events.iterrows():
        try:
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                json={"event": to_event(row)},
                headers=HEADERS,
                timeout=60,
            )
            if response.ok:
                sent += 1
            else:
                failed += 1
                if failed <= 3:
                    log(f"HTTP {response.status_code}: {response.text[:200]}")
        except requests.RequestException as error:
            failed += 1
            if failed <= 3:
                log(f"request failed: {error}")
        time.sleep(0.05)

    log(f"done: {sent} stored, {failed} failed.")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
