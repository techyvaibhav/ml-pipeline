import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import FEATURES, NUM_COLS, TARGET
from src.features import build_pipeline, build_prep, feat_names
from src.ingest import clean


def _xy(raw):
    df = clean(raw)
    return df[FEATURES], df[TARGET].astype(int)


def test_prep_output_has_no_nans(raw):
    X, _ = _xy(raw)
    out = build_prep().fit_transform(X)
    assert out.shape[0] == len(X) and not np.isnan(out).any()


def test_handles_missing_and_unseen(raw):
    X, y = _xy(raw)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["TotalCharges"] = np.nan
    row["PaymentMethod"] = "Crypto"  # unseen category
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_scaler_fit_on_train_only(raw):
    X, y = _xy(raw)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=1)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(Xtr, ytr)
    sc = pipe.named_steps["prep"].named_transformers_["num"]["sc"]
    i = NUM_COLS.index("tenure")
    assert np.isclose(sc.mean_[i], Xtr["tenure"].mean())
    assert not np.isclose(sc.mean_[i], X["tenure"].mean())  # not fit on all data


def test_feature_names_match_width(raw):
    X, y = _xy(raw)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert len(feat_names(pipe)) == pipe.named_steps["model"].coef_.shape[1]
