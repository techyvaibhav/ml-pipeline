import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained[0] / "model.joblib"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["version"] and "roc_auc" in j["metrics"] and len(j["features"]) == 19


def test_predict(client):
    r = client.post("/predict", json=EXAMPLE)
    j = r.json()
    assert r.status_code == 200
    assert 0 <= j["churn_probability"] <= 1 and j["label"] in ("Yes", "No")


def test_predict_without_total_charges(client):
    body = {k: v for k, v in EXAMPLE.items() if k != "TotalCharges"}
    assert client.post("/predict", json=body).status_code == 200


@pytest.mark.parametrize("patch", [
    {"tenure": -1}, {"Contract": "Weekly"}, {"MonthlyCharges": "abc"}, {"extra": 1},
])
def test_predict_invalid_422(client, patch):
    r = client.post("/predict", json={**EXAMPLE, **patch})
    assert r.status_code == 422 and r.json()["errors"]


def test_predict_missing_field_422(client):
    body = {k: v for k, v in EXAMPLE.items() if k != "tenure"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, EXAMPLE]})
    j = r.json()
    assert r.status_code == 200 and j["count"] == 2 and len(j["predictions"]) == 2


def test_batch_empty_422(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_missing_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=EXAMPLE).status_code == 503


def test_500_is_clean(client, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "predict_batch", lambda *a, **k: 1 / 0)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/predict", json=EXAMPLE)
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
