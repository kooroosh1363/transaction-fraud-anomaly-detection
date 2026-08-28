from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CACHE = DATA_DIR / "creditcard.csv"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


def load_dataset() -> tuple[pd.DataFrame, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        bunch = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df.to_csv(CACHE, index=False)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(df)}")
    if "Class" not in df.columns:
        raise ValueError("Target column Class is missing")

    df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
    if int(df["Class"].sum()) != EXPECTED_FRAUD:
        raise ValueError("Unexpected fraud count")

    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if not required.issubset(df.columns):
        raise ValueError("Unexpected credit-card fraud schema")

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
