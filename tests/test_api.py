"""Test the real FastAPI route with a small local model, no Docker service."""

import math

import pytest
from fastapi.testclient import TestClient

import api
from train import save_model


@pytest.fixture
def client(fitted_pipeline, tmp_path, monkeypatch):
    # The app's lifespan requires a fitted model before serving requests.
    model_path = tmp_path / "model.pkl"
    save_model(fitted_pipeline, model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def listing_payload():
    return {
        "latitude": 51.511,
        "longitude": -0.13,
        "accommodates": 2,
        "bathrooms": 1.0,
        "bedrooms": 1.0,
        "beds": 1.0,
        "minimum_nights": 2.0,
        "neighbourhood_cleansed": "Westminster",
        "property_type": "Entire rental unit",
        "room_type": "Entire home/apt",
    }


def test_predict_returns_number(client, listing_payload):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True}

    response = client.post("/predict", json=listing_payload)
    assert response.status_code == 200
    prediction = response.json()["predicted_price"]
    assert isinstance(prediction, (int, float))
    assert math.isfinite(prediction)


def test_predict_rejects_missing_field(client, listing_payload):
    del listing_payload["latitude"]
    response = client.post("/predict", json=listing_payload)
    assert response.status_code == 422


def test_predict_allows_missing_optional_numbers(client, listing_payload):
    listing_payload["bathrooms"] = None
    response = client.post("/predict", json=listing_payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["predicted_price"])
