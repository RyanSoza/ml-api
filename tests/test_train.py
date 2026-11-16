import pytest

from ml_api.predict import load_model, predict
from ml_api.train import save, train


def test_model_beats_baseline(tiny_df):
    _, metrics = train(tiny_df)
    assert metrics["rmse"] < metrics["baseline_rmse"]


def test_save_and_load_round_trip(tiny_df, tmp_path):
    model, metrics = train(tiny_df)
    save(model, metrics, out_dir=tmp_path)
    loaded = load_model(tmp_path / "model.joblib")
    assert predict(loaded, tiny_df.head(5)) == pytest.approx(predict(model, tiny_df.head(5)))
    assert (tmp_path / "metrics.json").exists()


def test_training_is_deterministic(tiny_df):
    _, metrics_1 = train(tiny_df, seed=1)
    _, metrics_2 = train(tiny_df, seed=1)
    assert metrics_1 == metrics_2


def test_predict_returns_one_float_per_row(tiny_df):
    model, _ = train(tiny_df)
    result = predict(model, tiny_df.head(3))
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(i, float) for i in result)


@pytest.mark.parametrize("missing", ["MedInc", "Latitude"])
def test_predict_fails_loudly_on_missing_column(tiny_df, missing):
    model, _ = train(tiny_df)
    rows = tiny_df.head(3).drop(columns=missing)
    with pytest.raises(KeyError):
        predict(model, rows)
