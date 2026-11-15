"""Batch training for the Airbnb price-regression project.

The feature set, grouped split and model parameters match 02_pipeline.ipynb.
Run with: python train.py (requires AWS credentials and access to Yandex Object Storage).
"""

import os
import pickle
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
]
CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"
GROUP_COLUMN = "listing_id"


def read_s3_data() -> pd.DataFrame:
    """Read the source CSV directly from S3; no local download required."""
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        raise RuntimeError(
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment."
        )

    bucket = os.environ.get("S3_BUCKET", "mlops26-course-bucket")
    key = os.environ.get("S3_KEY", "prepared/listings.csv")
    endpoint_url = os.environ.get(
        "S3_ENDPOINT_URL", "https://storage.yandexcloud.net"
    )
    region = os.environ.get("AWS_DEFAULT_REGION", "ru-central1")

    client = boto3.client(
        "s3", endpoint_url=endpoint_url, region_name=region
    )
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        dataframe = pd.read_csv(body)

    print(f"S3 data loaded: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")
    return dataframe


def split_data(df: pd.DataFrame):
    """Make train/test splits by listing ID to prevent repeated-listing leakage."""
    required_columns = FEATURES + [TARGET, GROUP_COLUMN]
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("The S3 CSV has no rows")
    if df[TARGET].isna().any() or df[GROUP_COLUMN].isna().any():
        raise ValueError("price and listing_id must not contain missing values")
    if df[GROUP_COLUMN].nunique() < 2:
        raise ValueError("At least two distinct listing_id values are needed")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=df[GROUP_COLUMN])
    )
    train_groups = set(df.iloc[train_idx][GROUP_COLUMN])
    test_groups = set(df.iloc[test_idx][GROUP_COLUMN])
    if train_groups & test_groups:
        raise RuntimeError("The same listing_id appears in train and test")

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    print(f"Train: {X_train.shape}; test: {X_test.shape}; shared listings: 0")
    return X_train, X_test, y_train, y_test


def make_pipeline() -> Pipeline:
    """Reproduce the notebook preprocessing + RandomForestRegressor."""
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, model_path: Path) -> None:
    """Save the trained Pipeline to the mounted model volume."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    if model_path.stat().st_size == 0:
        raise RuntimeError("model.pkl is unexpectedly empty")
    print(f"Saved trained Pipeline to {model_path} ({model_path.stat().st_size} bytes)")


def main() -> None:
    data = read_s3_data()
    X_train, X_test, y_train, y_test = split_data(data)
    model = make_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"Test metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    model_path = Path(os.environ.get("MODEL_PATH", "/models/model.pkl"))
    save_model(model, model_path)

    # Verify the artifact can be loaded and predicts identically without refitting.
    with model_path.open("rb") as file:
        loaded_model = pickle.load(file)
    loaded_predictions = loaded_model.predict(X_test)
    if not np.allclose(predictions, loaded_predictions):
        raise RuntimeError("Predictions changed after saving/loading the model")
    print("Artifact verification successful: loaded Pipeline predictions match")
    print("Training finished successfully")


if __name__ == "__main__":
    main()
