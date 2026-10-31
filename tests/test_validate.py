import pytest

from src.ingest import clean
from src.validate import validate


def test_good_data_passes(raw):
    df = validate(clean(raw))
    assert len(df) == len(raw)


def test_blank_total_charges_cleaned(raw):
    df = clean(raw)
    assert (raw["TotalCharges"].str.strip() == "").any()  # raw has blanks
    assert df["TotalCharges"].isna().sum() == 0
    assert df["TotalCharges"].dtype == float


def test_negative_tenure_fails(raw):
    df = clean(raw)
    df.loc[0, "tenure"] = -5
    with pytest.raises(ValueError, match="validation failed"):
        validate(df)


def test_unknown_category_fails(raw):
    df = clean(raw)
    df.loc[0, "Contract"] = "Five year"
    with pytest.raises(ValueError):
        validate(df)


def test_null_in_required_col_fails(raw):
    df = clean(raw)
    df.loc[0, "MonthlyCharges"] = None
    with pytest.raises(ValueError):
        validate(df)
