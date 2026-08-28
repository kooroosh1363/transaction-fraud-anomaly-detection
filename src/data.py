from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CACHE = DATA_DIR / "creditcard.csv"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OpenML column-name casing without weakening schema validation."""
    by_lower = {str(c).lower(): c for c in df.columns}
    missing = [c for c in EXPECTED_COLUMNS if c.lower() not in by_lower]
    if missing:
        raise ValueError(f"Unexpected credit-card fraud schema; missing columns: {missing}")
    rename = {by_lower[c.lower()]: c for c in EXPECTED_COLUMNS}
    return df.rename(columns=rename)[EXPECTED_COLUMNS].copy()


def load_dataset() -> tuple[pd.DataFrame, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        bunch = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
        df = bunch.frame.copy()

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(df)}")

    df = _normalize_schema(df)
    df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
    if int(df["Class"].sum()) != EXPECTED_FRAUD:
        raise ValueError("Unexpected fraud count")

    # Cache only after the source passes row-count, target-count, and schema checks.
    if not CACHE.exists():
        df.to_csv(CACHE, index=False)

    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    audit = {
        "rows": int(len(df)),
        "fraud_rows": int(df["Class"].sum()),
        "legitimate_rows": int((df["Class"] == 0).sum()),
        "fraud_rate": float(df["Class"].mean()),
        "features": int(df.shape[1] - 1),
        "time_min_seconds": float(df["Time"].min()),
        "time_max_seconds": float(df["Time"].max()),
    }
    return df, audit


def temporal_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    if not (train["Time"].max() <= val["Time"].min() <= val["Time"].max() <= test["Time"].min()):
        raise ValueError("Temporal ordering failed")
    meta = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_fraud": int(train["Class"].sum()),
        "validation_fraud": int(val["Class"].sum()),
        "test_fraud": int(test["Class"].sum()),
        "train_end_time": float(train["Time"].max()),
        "validation_end_time": float(val["Time"].max()),
    }
    return train, val, test, meta
