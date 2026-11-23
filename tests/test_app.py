import pytest
from fastapi.testclient import TestClient

from ml_api.app import create_app
from ml_api.config import Settings
from ml_api.train import save, train

VALID = {
    "median_income": 3.5,
    "house_age": 20,
    "avg_rooms": 5.0,
    "avg_bedrooms": 1.0,
    "population": 1200,
    "avg_occupancy": 3.0,
    "latitude": 34.0,
    "longitude": -118.0,
}


@pytest.fixture
def client(tiny_df, tmp_path):
    model, metrics = train(tiny_df)
    save(model, metrics, out_dir=tmp_path)
    settings = Settings(artifact_path=str(tmp_path / "model.joblib"), version="test")
    with TestClient(create_app(settings)) as c:  # "with" runs the startup code
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "test"}


def test_predict_returns_a_number(client):
    # POST VALID to /predict; expect 200, a float prediction, version "test"
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json()["version"] == "test"


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID, "median_income": -1},  # out of range
        {**VALID, "latitude": "north"},  # wrong type
        {k: v for k, v in VALID.items() if k != "population"},  # missing field
    ],
)
def test_bad_input_is_rejected(client, bad):
    # expect status 422
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_missing_model_fails_at_startup(tmp_path):
    # Settings pointing at a file that does not exist
    settings = Settings(artifact_path=str(tmp_path / "missing.joblib"), version="test")
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(settings)):
            pass
