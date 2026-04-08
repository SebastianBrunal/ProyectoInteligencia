"""
preprocessing.py — Load and prepare the SDSS dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


FEATURE_COLS = ["u", "g", "r", "i", "z"]
TARGET_CLASS = "class"
TARGET_REDSHIFT = "redshift"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Preprocessing] Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def get_classification_data(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test for KNN classification."""
    X = df[FEATURE_COLS + ["redshift"]].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_CLASS].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Classification — train: {len(X_train)}, test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, le, scaler


def get_regression_data(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42):
    """Return X_train, X_test, y_train, y_test for linear regression on redshift."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_REDSHIFT].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"[Preprocessing] Regression — train: {len(X_train)}, test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


def get_clustering_data(df: pd.DataFrame):
    """Return scaled photometric magnitudes for KMeans clustering."""
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df[TARGET_CLASS].values, scaler
