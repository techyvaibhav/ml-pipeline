"""Pandera schema: fail loudly on bad data."""
import pandas as pd

try:  # newer pandera
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

from src.config import CATS, ID_COL, TARGET


def build_schema() -> "pa.DataFrameSchema":
    """Schema for the cleaned dataset: types, ranges, allowed categories, nulls."""
    cols = {ID_COL: pa.Column(str, nullable=False, unique=True)}  # column specs
    for name, vals in CATS.items():
        cols[name] = pa.Column(str, pa.Check.isin(vals), nullable=False)
    cols["SeniorCitizen"] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    cols["tenure"] = pa.Column(int, pa.Check.in_range(0, 120), nullable=False)
    cols["MonthlyCharges"] = pa.Column(float, pa.Check.in_range(0, 500), nullable=False)
    cols["TotalCharges"] = pa.Column(float, pa.Check.in_range(0, 50000), nullable=True)
    cols[TARGET] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(cols, strict=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate cleaned df. Raises ValueError listing the first problems."""
    try:
        return build_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        bad = e.failure_cases[["column", "check", "failure_case"]].head(15)  # first failures
        raise ValueError(f"Data validation failed ({len(e.failure_cases)} issues):\n{bad}") from e
