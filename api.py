"""FastAPI service for predictions from the pretrained Airbnb sklearn Pipeline.

The Pipeline is produced by the separate `train` Docker Compose service.
This API only loads and uses the artifact; it never fits a model.
"""

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model.pkl"))


class ListingFeatures(BaseModel):
    """One Airbnb listing: exactly the features used during model training."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accommodates: int = Field(ge=1)
    bathrooms: float | None = Field(default=None, ge=0)
    bedrooms: float | None = Field(default=None, ge=0)
    beds: float | None = Field(default=None, ge=0)
    minimum_nights: float | None = Field(default=None, ge=0)
    neighbourhood_cleansed: str = Field(min_length=1)
    property_type: str = Field(min_length=1)
    room_type: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_price: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted saved Pipeline ONCE at API startup."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Trained Pipeline not found at {MODEL_PATH}")

    # pickle must only load a trusted artifact produced by our train service.
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, Pipeline):
        raise RuntimeError("The saved model is not a sklearn Pipeline")
    check_is_fitted(model)
    app.state.model = model
    print(f"Pretrained Pipeline loaded from {MODEL_PATH}", flush=True)
    yield
    del app.state.model


app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Predicts Airbnb listing prices using an already trained sklearn Pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Simple check that the API has loaded the fitted model."""
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingFeatures):
    """Receive one JSON object, return its price prediction."""
    # Convert nullable numeric inputs to float columns, as in training.
    dataframe = pd.DataFrame([listing.model_dump()])
    numeric_columns = [
        "latitude", "longitude", "accommodates", "bathrooms",
        "bedrooms", "beds", "minimum_nights",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise").astype(float)

    try:
        result = app.state.model.predict(dataframe)
        value = float(result[0])
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed") from error

    if not np.isfinite(value):
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction")
    return PredictionResponse(predicted_price=value)
