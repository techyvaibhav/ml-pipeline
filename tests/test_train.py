import json

from src.config import FEATURES
from src.ingest import clean
from src.predict import load_bundle, predict_batch


def test_artifacts_saved(trained):
    d, _ = trained
    for f in ["model.joblib", "metrics.json", "model_card.md"]:
        assert (d / f).exists()
    for f in ["roc_curve.png", "confusion_matrix.png", "feature_importance.png"]:
        assert (d / "reports" / f).exists()


def test_metrics_sane(trained):
    d, m = trained
    t = m["test_metrics"]
    assert set(t) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert t["roc_auc"] > 0.65  # synthetic data has real signal
    assert json.loads((d / "metrics.json").read_text())["version"] == m["version"]


def test_saved_model_predicts(trained, raw):
    d, _ = trained
    assert "version" in load_bundle(str(d / "model.joblib"))
    rows = clean(raw)[FEATURES].head(5).to_dict("records")
    out = predict_batch(rows, str(d / "model.joblib"))
    assert len(out) == 5 and all(0 <= o["churn_probability"] <= 1 for o in out)
