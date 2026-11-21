"""Small reusable dataset and fitted model for isolated unit tests."""

import numpy as np
import pandas as pd
import pytest

from train import make_pipeline, split_data


@pytest.fixture
def sample_data():
    """Twelve rows with repeated listing IDs, missing values and categories."""
    return pd.DataFrame(
        {
            "listing_id": [101, 101, 102, 102, 103, 103, 104, 104, 105, 105, 106, 106],
            "latitude": [51.50, 51.50, 51.51, 51.51, 51.52, 51.52,
                         51.53, 51.53, 51.54, 51.54, 51.55, 51.55],
            "longitude": [-0.12, -0.12, -0.13, -0.13, -0.14, -0.14,
                          -0.15, -0.15, -0.16, -0.16, -0.17, -0.17],
            "accommodates": [2, 2, 4, 4, 1, 1, 3, 3, 5, 5, 2, 2],
            "bathrooms": [1.0, 1.0, 2.0, 2.0, np.nan, 1.0,
                          1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            "bedrooms": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0,
                         2.0, 2.0, 3.0, 3.0, 1.0, 1.0],
            "beds": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0,
                     2.0, 2.0, 3.0, 3.0, 1.0, 1.0],
            "minimum_nights": [1, 2, 2, 3, 1, 2, 4, 4, 2, 3, 1, 2],
            "neighbourhood_cleansed": [
                "Westminster", "Westminster", "Camden", "Camden",
                "Hackney", "Hackney", "Westminster", "Westminster",
                "Camden", "Camden", "Hackney", "Hackney",
            ],
            "property_type": [
                "Entire rental unit", "Entire rental unit", "Private room", "Private room",
                "Entire rental unit", "Entire rental unit", "Private room", "Private room",
                "Entire rental unit", "Entire rental unit", "Private room", "Private room",
            ],
            "room_type": [
                "Entire home/apt", "Entire home/apt", "Private room", "Private room",
                "Entire home/apt", "Entire home/apt", "Private room", "Private room",
                "Entire home/apt", "Entire home/apt", "Private room", "Private room",
            ],
            "price": [100, 110, 175, 180, 80, 85, 155, 160, 230, 235, 120, 125],
        }
    )


@pytest.fixture
def fitted_pipeline(sample_data):
    """Fit on only a few rows, not on the real S3 dataset."""
    X_train, _, y_train, _ = split_data(sample_data)
    model = make_pipeline()
    # Keep the same structure as production, but use only 3 trees for fast tests.
    model.set_params(model__n_estimators=3, model__max_depth=3, model__n_jobs=1)
    model.fit(X_train, y_train)
    return model
