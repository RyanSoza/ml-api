import numpy as np
import pandas as pd
import pytest

from ml_api.data import FEATURES, TARGET


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """200 fake rows with learnable feature for testing"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(1, 10, size=(200, len(FEATURES))), columns=FEATURES)
    df[TARGET] = 0.5 * df["MedInc"] + rng.normal(0, 0.1, size=200)
    return df
