"""
tests/test_dataset.py — Basic sanity checks on the SDSS dataset and pipeline.
Run with: pytest tests/
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.environ.get("DATA_PATH", "data.csv")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


# ── Dataset integrity ───────────────────────────────────────────────────────

def test_file_exists():
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"


def test_required_columns(df):
    required = {"u", "g", "r", "i", "z", "redshift", "class"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"


def test_no_empty_dataframe(df):
    assert len(df) > 0, "Dataset is empty"


def test_no_nulls_in_features(df):
    feature_cols = ["u", "g", "r", "i", "z", "redshift"]
    null_counts = df[feature_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Nulls found:\n{null_counts[null_counts > 0]}"


def test_valid_class_labels(df):
    valid = {"Galaxy", "Star", "QSO"}
    found = set(df["class"].unique())
    assert found.issubset(valid), f"Unexpected labels: {found - valid}"


def test_redshift_range(df):
    assert df["redshift"].min() >= 0, "Negative redshift values found"
    assert df["redshift"].max() < 10, "Suspiciously large redshift values"


def test_photometric_values_reasonable(df):
    for band in ["u", "g", "r", "i", "z"]:
        assert df[band].between(10, 35).all(), f"Band {band} has out-of-range magnitudes"


# ── Preprocessing ───────────────────────────────────────────────────────────

def test_classification_split(df):
    from src.preprocessing import get_classification_data
    X_train, X_test, y_train, y_test, le, _ = get_classification_data(df)
    total = len(X_train) + len(X_test)
    assert total == len(df)
    assert abs(len(X_test) / total - 0.30) < 0.02


def test_regression_split(df):
    from src.preprocessing import get_regression_data
    X_train, X_test, y_train, y_test, _ = get_regression_data(df)
    total = len(X_train) + len(X_test)
    assert total == len(df)


def test_clustering_data_shape(df):
    from src.preprocessing import get_clustering_data
    X, labels, _ = get_clustering_data(df)
    assert X.shape == (len(df), 5)
    assert len(labels) == len(df)


# ── Model smoke tests ────────────────────────────────────────────────────────

def test_knn_trains_and_predicts(df):
    from src.preprocessing import get_classification_data
    from src.classification import train_knn
    X_train, X_test, y_train, y_test, le, _ = get_classification_data(df)
    model = train_knn(X_train, y_train, k=5)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_linear_regression_trains(df):
    from src.preprocessing import get_regression_data
    from src.regression import train_linear_regression
    X_train, X_test, y_train, y_test, _ = get_regression_data(df)
    model = train_linear_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_kmeans_trains(df):
    from src.preprocessing import get_clustering_data
    from src.clustering import train_kmeans
    X, _, _ = get_clustering_data(df)
    model = train_kmeans(X, n_clusters=3)
    assert len(np.unique(model.labels_)) == 3
