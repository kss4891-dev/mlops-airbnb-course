"""Standard-library smoke checks for a running API; run from a second terminal."""

import json
import math
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SAMPLE = {
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


def post_json(payload):
    request = Request(
        BASE_URL + "/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=15)


def main():
    with urlopen(BASE_URL + "/health", timeout=15) as response:
        health = json.load(response)
        assert response.status == 200, health
        assert health == {"status": "ok", "model_loaded": True}, health
    print("GET /health: OK")

    with post_json(SAMPLE) as response:
        result = json.load(response)
        assert response.status == 200, result
        assert isinstance(result.get("predicted_price"), (int, float)), result
        assert math.isfinite(result["predicted_price"]), result
    print(f"POST /predict: OK; predicted_price={result['predicted_price']:.2f}")

    invalid = dict(SAMPLE)
    invalid.pop("latitude")
    try:
        with post_json(invalid):
            raise AssertionError("Expected HTTP 422 for missing latitude")
    except HTTPError as error:
        assert error.code == 422, f"Expected 422, got {error.code}"
    print("POST /predict validation (missing feature): OK, HTTP 422")
    print("API smoke checks passed")


if __name__ == "__main__":
    main()
