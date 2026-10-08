"""Unit and integration tests for the Sentiment Analysis API."""
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app


@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan (model loading)
    with TestClient(app) as c:
        yield c


# ------------------------------- Health ----------------------------------- #
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_docs_available(client):
    assert client.get("/docs").status_code == 200


# ------------------------------- Predict: happy path ---------------------- #
def test_predict_positive(client):
    response = client.post("/predict", json={"text": "I love this platform!"})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"sentiment", "score"}
    assert body["sentiment"] == "positive"
    assert isinstance(body["score"], float)
    assert 0.0 <= body["score"] <= 1.0


def test_predict_negative(client):
    response = client.post("/predict", json={"text": "This is absolutely terrible and awful"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"


# ------------------------------- Predict: validation ---------------------- #
def test_predict_empty_string(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in (400, 422)


def test_predict_blank_string(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_integer_instead_of_string(client):
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422


def test_predict_missing_key(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_too_long(client):
    response = client.post("/predict", json={"text": "a" * 5001})
    assert response.status_code == 422


# ------------------------------- Predict: error handling ------------------ #
class _BrokenModel:
    def predict_proba(self, _):
        raise RuntimeError("secret internal failure details")


def test_inference_failure_returns_generic_500(client, monkeypatch):
    monkeypatch.setattr(main, "ml_model", _BrokenModel())
    response = client.post("/predict", json={"text": "hello world"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_model_not_loaded_returns_500(client, monkeypatch):
    monkeypatch.setattr(main, "ml_model", None)
    response = client.post("/predict", json={"text": "hello world"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
