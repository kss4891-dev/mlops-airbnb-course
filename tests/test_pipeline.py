"""Tests for preprocessing, prediction, and Pipeline serialization."""

import pickle

import numpy as np

from train import FEATURES, make_pipeline, save_model, split_data


def test_pipeline_transforms_and_predicts(sample_data):
    X_train, X_test, y_train, _ = split_data(sample_data)
    pipeline = make_pipeline()
    pipeline.set_params(model__n_estimators=3, model__max_depth=3, model__n_jobs=1)
    pipeline.fit(X_train, y_train)

    # The imputer and encoder are inside the fitted Pipeline.
    transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    assert transformed.shape[0] == len(X_test)
    assert transformed.shape[1] >= len(FEATURES)

    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.isfinite(predictions).all()

    # Unseen categories and missing numeric values should also be supported.
    new_listing = X_test.iloc[[0]].copy()
    new_listing.loc[:, "neighbourhood_cleansed"] = "New neighbourhood"
    new_listing.loc[:, "bathrooms"] = np.nan
    new_prediction = pipeline.predict(new_listing)
    assert len(new_prediction) == 1
    assert np.isfinite(new_prediction).all()


def test_saved_pipeline_predicts_identically(fitted_pipeline, sample_data, tmp_path):
    X = sample_data[FEATURES].iloc[:3]
    before = fitted_pipeline.predict(X)

    path = tmp_path / "models" / "model.pkl"
    save_model(fitted_pipeline, path)
    assert path.is_file() and path.stat().st_size > 0

    with path.open("rb") as file:
        restored = pickle.load(file)
    after = restored.predict(X)
    np.testing.assert_allclose(after, before, rtol=0, atol=0)
