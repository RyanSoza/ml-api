"""Load dataset from sklearn and split into training and testing sets"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET = "MedHouseVal"


def load_data() -> pd.DataFrame:
    """Returns the full dataset."""

    return fetch_california_housing(as_frame=True).frame


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 16):
    """Return X_train, X_test, y_train, y_test."""
    # Select features and target from data
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test
