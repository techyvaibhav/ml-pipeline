"""Download Telco churn csv; fall back to a synthetic dataset with the same schema."""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATS, DATA_URL, ID_COL, NUM_COLS, RAW_PATH, SRC_PATH, TARGET


def make_synthetic(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Build a realistic fake telco dataset (same schema/format as the real csv)."""
    rng = np.random.default_rng(seed)  # random generator
    pick = lambda vals, p=None: rng.choice(vals, size=n, p=p)  # noqa: E731  # sample helper

    ten = np.where(rng.random(n) < 0.4, rng.integers(0, 12, n), rng.integers(0, 73, n))  # months
    phone = pick(["Yes", "No"], [0.9, 0.1])  # has phone
    net = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])  # internet type
    no_net = net == "No"  # no internet mask

    def addon(p_yes: float) -> np.ndarray:
        """Internet add-on column: Yes/No, or 'No internet service'."""
        arr = np.where(rng.random(n) < p_yes, "Yes", "No")  # raw yes/no
        return np.where(no_net, "No internet service", arr)

    df = pd.DataFrame({ID_COL: [f"{i:05d}-SYNTH" for i in range(n)]})
    df["gender"] = pick(["Male", "Female"])
    df["SeniorCitizen"] = (rng.random(n) < 0.16).astype(int)
    df["Partner"] = pick(["Yes", "No"], [0.48, 0.52])
    df["Dependents"] = pick(["Yes", "No"], [0.3, 0.7])
    df["tenure"] = ten.astype(int)
    df["PhoneService"] = phone
    df["MultipleLines"] = np.where(
        phone == "No", "No phone service", np.where(rng.random(n) < 0.47, "Yes", "No")
    )
    df["InternetService"] = net
    for col, p in [("OnlineSecurity", 0.29), ("OnlineBackup", 0.34), ("DeviceProtection", 0.34),
                   ("TechSupport", 0.29), ("StreamingTV", 0.38), ("StreamingMovies", 0.39)]:
        df[col] = addon(p)
    df["Contract"] = pick(CATS["Contract"], [0.55, 0.21, 0.24])
    df["PaperlessBilling"] = pick(["Yes", "No"], [0.59, 0.41])
    df["PaymentMethod"] = pick(CATS["PaymentMethod"], [0.34, 0.23, 0.22, 0.21])

    n_add = sum((df[c] == "Yes").astype(int) for c in
                ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"])  # count of add-ons
    mon = (18 + 22 * (net == "DSL") + 50 * (net == "Fiber optic") + 5 * n_add
           + 4 * (phone == "Yes") + rng.normal(0, 3, n))  # monthly charge
    df["MonthlyCharges"] = np.clip(mon, 18.25, 118.75).round(2)
    tot = df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.95, 1.05, n)  # total charge
    df["TotalCharges"] = np.where(df["tenure"] == 0, " ", tot.round(2).astype(str))  # blank=real

    z = (-0.5 + 1.2 * (df["Contract"] == "Month-to-month") - 0.9 * (df["Contract"] == "Two year")
         - 0.03 * df["tenure"] + 0.015 * (df["MonthlyCharges"] - 65) + 0.7 * (net == "Fiber optic")
         + 0.5 * (df["PaymentMethod"] == "Electronic check")
         - 0.5 * (df["TechSupport"] == "Yes") - 0.4 * (df["OnlineSecurity"] == "Yes")
         + 0.3 * df["SeniorCitizen"] + 0.2 * (df["PaperlessBilling"] == "Yes")).to_numpy()  # logit
    lo, hi = -5.0, 5.0  # bisection bounds for intercept shift (target churn rate ~26.5%)
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.265:
            hi = mid
        else:
            lo = mid
    p_churn = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))  # churn probability
    df[TARGET] = np.where(rng.random(n) < p_churn, "Yes", "No")
    return df[[ID_COL, *[c for c in df.columns if c not in (ID_COL, TARGET)], TARGET]]


def fetch(path: Path = RAW_PATH, url: str = DATA_URL, force_synth: bool = False) -> str:
    """Download csv to path (or generate synthetic). Returns 'real' or 'synthetic'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    src = "synthetic"  # data source label
    df = None  # loaded dataframe
    if not force_synth:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
            need = {ID_COL, TARGET, *NUM_COLS, *CATS}  # expected columns
            if not need.issubset(df.columns):
                raise ValueError("downloaded csv has unexpected columns")
            src = "real"
        except Exception as e:  # no internet / bad file -> synthetic
            print(f"[download] failed ({e}); using synthetic data")
            df = None
    if df is None:
        df = make_synthetic()
    df.to_csv(path, index=False)
    SRC_PATH.write_text(src)
    print(f"[download] saved {len(df)} rows ({src}) -> {path}")
    return src


if __name__ == "__main__":
    ap = argparse.ArgumentParser()  # cli parser
    ap.add_argument("--synthetic", action="store_true", help="skip download, make fake data")
    fetch(force_synth=ap.parse_args().synthetic)
