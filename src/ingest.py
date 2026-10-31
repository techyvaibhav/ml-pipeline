"""Load + clean raw data."""
from pathlib import Path

import pandas as pd

from data.download_data import fetch
from src.config import ID_COL, RAW_PATH, SRC_PATH, TARGET


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read raw csv; download (or synthesize) it first if missing."""
    if not path.exists():
        fetch(path)
    return pd.read_csv(path)


def data_source() -> str:
    """Return 'real' or 'synthetic' (unknown if no marker file)."""
    return SRC_PATH.read_text().strip() if SRC_PATH.exists() else "unknown"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known issues: whitespace, blank TotalCharges, duplicates, target to 0/1."""
    out = df.copy()  # working copy
    for col in out.columns:  # strip text whitespace
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].str.strip()
    if "TotalCharges" in out:  # blank -> NaN; brand-new customers (tenure 0) have 0 charges
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if "tenure" in out:
            new = out["TotalCharges"].isna() & (out["tenure"] == 0)  # new-customer mask
            out.loc[new, "TotalCharges"] = 0.0
    if ID_COL in out:
        out = out.drop_duplicates(subset=ID_COL)
    if TARGET in out and not pd.api.types.is_numeric_dtype(out[TARGET]):
        tgt = out[TARGET].map({"Yes": 1, "No": 0})  # unknown labels -> NaN -> caught by validate
        out[TARGET] = tgt.astype("Int64") if tgt.isna().any() else tgt.astype(int)
    return out.reset_index(drop=True)


