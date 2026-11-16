"""Load the model and make predictions"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_api.data import FEATURES


def load_model(path: str | Path = "models/model.joblib") -> Pipeline:
    return joblib.load(path)


def predict(model: Pipeline, rows: pd.DataFrame) -> list[float]:
    predictions = model.predict(rows[FEATURES])
    return predictions.tolist()
