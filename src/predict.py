"""Load the saved pipeline and predict for one row or a batch."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH, NUM_COLS, THRESH


@lru_cache(maxsize=4)
def load_bundle(path: str = str(MODEL_PATH)) -> dict:
    """Load model bundle (pipeline + metadata) once per path."""
    p = Path(path)  # model file
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}. Run `make train` first.")
    return joblib.load(p)


def to_df(rows: list[dict]) -> pd.DataFrame:
    """List of dicts -> model-ready df (TotalCharges may be missing; it gets imputed)."""
    df = pd.DataFrame(rows)  # raw input
    miss = [c for c in FEATURES if c not in df.columns and c != "TotalCharges"]  # missing cols
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    if "TotalCharges" not in df:
        df["TotalCharges"] = float("nan")
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    return df[FEATURES]


def predict_batch(rows: list[dict], path: str = str(MODEL_PATH)) -> list[dict]:
    """Predict churn for many customers."""
    b = load_bundle(path)  # model bundle
    thr = b.get("threshold", THRESH)  # decision threshold
    proba = b["pipeline"].predict_proba(to_df(rows))[:, 1]  # churn proba
    return [{"churn_probability": round(float(p), 4), "churn": bool(p >= thr),
             "label": "Yes" if p >= thr else "No"} for p in proba]


def predict_one(row: dict, path: str = str(MODEL_PATH)) -> dict:
    """Predict churn for one customer."""
    return predict_batch([row], path)[0]
