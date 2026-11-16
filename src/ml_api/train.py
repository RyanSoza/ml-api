"""Train, evlauate and save the model"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

from ml_api.data import load_data, split
from ml_api.features import build_preprocessor

MODEL_DIR = Path("models")


def build_model(seed: int = 16) -> Pipeline:
    pipe = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", HistGradientBoostingRegressor(random_state=seed)),
        ]
    )

    return pipe


def train(df: pd.DataFrame, seed: int = 16) -> tuple[Pipeline, dict]:
    """Return (model, metrics)"""
    metrics = {}
    X_train, X_test, y_train, y_test = split(df)

    # Train model
    model = build_model(seed=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get baseline score with DummyRegressor RMSE
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    # Save metrics to dict
    metrics["rmse"] = root_mean_squared_error(y_test, y_pred)
    metrics["r2"] = model.score(X_test, y_test)
    metrics["baseline_rmse"] = root_mean_squared_error(y_test, y_pred_baseline)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    return (model, metrics)


def save(model: Pipeline, metrics: dict, out_dir: Path = MODEL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    model, metrics = train(load_data())
    save(model, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
