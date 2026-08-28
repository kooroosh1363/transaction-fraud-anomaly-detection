from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
CACHE = DATA_DIR / "creditcard.csv"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
BASE_COLUMNS = [*[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def _normalize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Normalize OpenML column casing and explicitly handle the variant that omits Time."""
    by_lower = {str(c).lower(): c for c in df.columns}
    missing_base = [c for c in BASE_COLUMNS if c.lower() not in by_lower]
    if missing_base:
        raise ValueError(f"Unexpected credit-card fraud schema; missing columns: {missing_base}")

    rename = {by_lower[c.lower()]: c for c in BASE_COLUMNS}
    has_time = "time" in by_lower
    if has_time:
        rename[by_lower["time"]] = "Time"
        ordered = ["Time", *BASE_COLUMNS]
        out = df.rename(columns=rename)[ordered].copy()
        out["SequenceIndex"] = np.arange(len(out), dtype=np.int64)
    else:
        # The OpenML distribution currently used by sklearn omits the original Time field
        # but preserves the dataset row order. We keep that order explicitly as SequenceIndex
        # for forward-only splitting and do NOT pretend that it is a real timestamp feature.
        out = df.rename(columns=rename)[BASE_COLUMNS].copy()
        out.insert(0, "SequenceIndex", np.arange(len(out), dtype=np.int64))
    return out, has_time


def load_dataset() -> tuple[pd.DataFrame, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        df = pd.read_csv(CACHE)
        has_time = "Time" in df.columns
        if "SequenceIndex" not in df.columns:
            df.insert(0, "SequenceIndex", np.arange(len(df), dtype=np.int64))
    else:
        bunch = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
        raw = bunch.frame.copy()
        if len(raw) != EXPECTED_ROWS:
            raise ValueError(f"Unexpected row count: {len(raw)}")
        df, has_time = _normalize_schema(raw)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(df)}")

    df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
    if int(df["Class"].sum()) != EXPECTED_FRAUD:
        raise ValueError("Unexpected fraud count")

    required = {"SequenceIndex", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if not required.issubset(df.columns):
        raise ValueError("Unexpected normalized fraud schema")

    if not CACHE.exists():
        df.to_csv(CACHE, index=False)

    df = df.sort_values("SequenceIndex", kind="stable").reset_index(drop=True)
    audit = {
        "rows": int(len(df)),
        "fraud_rows": int(df["Class"].sum()),
        "legitimate_rows": int((df["Class"] == 0).sum()),
        "fraud_rate": float(df["Class"].mean()),
        "model_features": 29,
        "source_time_available": bool(has_time),
        "split_order_field": "Time" if has_time else "SequenceIndex",
        "sequence_min": int(df["SequenceIndex"].min()),
        "sequence_max": int(df["SequenceIndex"].max()),
    }
    if has_time:
        audit["time_min_seconds"] = float(df["Time"].min())
        audit["time_max_seconds"] = float(df["Time"].max())
    return df, audit


def temporal_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    order_field = "Time" if "Time" in df.columns else "SequenceIndex"
    if not (train[order_field].max() <= val[order_field].min() <= val[order_field].max() <= test[order_field].min()):
        raise ValueError("Forward-order split failed")

    meta = {
        "split_policy": "forward-only 60/20/20 using source Time when available, otherwise preserved source row order",
        "order_field": order_field,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_fraud": int(train["Class"].sum()),
        "validation_fraud": int(val["Class"].sum()),
        "test_fraud": int(test["Class"].sum()),
        "train_end_order": float(train[order_field].max()),
        "validation_end_order": float(val[order_field].max()),
    }
    return train, val, test, meta
