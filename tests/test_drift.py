from src.config import FEATURES
from src.drift import drift_report
from src.ingest import clean


def test_no_drift_on_same_data(raw):
    df = clean(raw)[FEATURES]
    assert (drift_report(df, df)["psi"] < 0.01).all()


def test_detects_shift(raw):
    df = clean(raw)[FEATURES]
    new = df.copy()
    new["MonthlyCharges"] = new["MonthlyCharges"] + 40
    new["Contract"] = "Month-to-month"
    rep = drift_report(df, new).set_index("feature")
    assert rep.loc["MonthlyCharges", "status"] == "drift"
    assert rep.loc["Contract", "status"] == "drift"
