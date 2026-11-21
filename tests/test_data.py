"""Tests for S3 ingestion and the group-aware train/test split."""

from io import BytesIO
from unittest.mock import Mock

import pandas as pd
import pytest

import train


def test_read_s3_data_with_mock(sample_data, monkeypatch):
    """Read a tiny CSV using a fake boto3 client, without any S3 request."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret-key")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_KEY", "test/listings.csv")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ru-central1")

    csv_bytes = sample_data.to_csv(index=False).encode("utf-8")
    fake_s3 = Mock()
    fake_s3.get_object.return_value = {"Body": BytesIO(csv_bytes)}
    mock_client = Mock(return_value=fake_s3)
    monkeypatch.setattr(train.boto3, "client", mock_client)

    result = train.read_s3_data()

    pd.testing.assert_frame_equal(result, sample_data, check_dtype=False)
    mock_client.assert_called_once_with(
        "s3", endpoint_url="https://example.invalid", region_name="ru-central1"
    )
    fake_s3.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test/listings.csv"
    )


def test_split_data_separates_listing_ids(sample_data):
    """Repeated rows of one listing must never appear in both subsets."""
    X_train, X_test, y_train, y_test = train.split_data(sample_data)

    train_ids = set(sample_data.loc[X_train.index, "listing_id"])
    test_ids = set(sample_data.loc[X_test.index, "listing_id"])
    assert train_ids.isdisjoint(test_ids)
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert list(X_train.columns) == train.FEATURES
    assert list(X_test.columns) == train.FEATURES


def test_split_data_rejects_missing_price(sample_data):
    """Missing required columns should fail before model training."""
    with pytest.raises(ValueError, match="Missing required columns"):
        train.split_data(sample_data.drop(columns=["price"]))
