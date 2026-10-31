"""Simple drift check: PSI (all features) + KS test (numeric) vs the training data."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import FEATURES, NUM_COLS, PROC_DIR

EPS = 1e-4  # avoids log(0)


def _psi(e: np.ndarray, a: np.ndarray) -> float:
    """PSI from expected (e) and actual (a) proportions."""
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_num(ref: pd.Series, new: pd.Series, bins: int = 10) -> float:
    """PSI for a numeric column using reference quantile bins."""
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))  # bin edges
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref.dropna(), edges)[0] / ref.notna().sum()  # expected props
    a = np.histogram(new.dropna(), edges)[0] / max(new.notna().sum(), 1)  # actual props
    return _psi(e, a)


def psi_cat(ref: pd.Series, new: pd.Series) -> float:
    """PSI for a categorical column."""
    cats = sorted(set(ref.dropna()) | set(new.dropna()))  # all categories
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = new.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(e, a)


def status(psi: float) -> str:
    """Rule of thumb: <0.1 stable, 0.1-0.2 moderate, >=0.2 drift."""
    return "drift" if psi >= 0.2 else "moderate" if psi >= 0.1 else "stable"


def drift_report(ref: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per-feature drift table, sorted by PSI (highest first)."""
    rows = []  # report rows
    for c in FEATURES:
        if c in NUM_COLS:
            p = psi_num(ref[c], new[c])  # psi
            ks = ks_2samp(ref[c].dropna(), new[c].dropna())  # ks result
            rows.append({"feature": c, "psi": p, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue})
        else:
            rows.append({"feature": c, "psi": psi_cat(ref[c], new[c]),
                         "ks_stat": np.nan, "ks_pvalue": np.nan})
    out = pd.DataFrame(rows)  # report df
    out["status"] = out["psi"].map(status)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    """CLI: python -m src.drift new_data.csv"""
    from src.ingest import clean  # local import

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.drift <new_data.csv>")
    ref = pd.read_csv(PROC_DIR / "train.csv")  # training reference
    new = clean(pd.read_csv(sys.argv[1]))  # new data
    rep = drift_report(ref, new)  # report
    print(rep.round(4).to_string(index=False))
    print(f"\nFeatures with drift: {(rep.status == 'drift').sum()} / {len(rep)}")


if __name__ == "__main__":
    main()
