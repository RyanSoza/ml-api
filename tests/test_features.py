import numpy as np

from ml_api.data import FEATURES
from ml_api.features import build_preprocessor


def test_preprocessor_fills_missing_values(tiny_df):
    X = tiny_df[FEATURES].copy()
    X.iloc[0, 0] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert X.shape == out.shape
