"""One sklearn Pipeline: impute + scale numeric, impute + one-hot categorical + model."""
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CAT_COLS, NUM_COLS


def build_prep() -> ColumnTransformer:
    """Preprocessing only (unfitted)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])  # numeric
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])  # categorical
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Full pipeline = preprocessing + model (same object used for train and serve)."""
    return Pipeline([("prep", build_prep()), ("model", model)])


def feat_names(pipe: Pipeline) -> list[str]:
    """Output feature names of a fitted pipeline's preprocessing step."""
    return [str(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
