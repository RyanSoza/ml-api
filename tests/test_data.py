from ml_api.data import FEATURES, split


def test_splits_are_right_size(tiny_df):
    X_train, X_test, _, _ = split(tiny_df)
    assert len(X_train) == 160
    assert len(X_test) == 40


def test_column_list_equals_features(tiny_df):
    X_train, X_test, _, _ = split(tiny_df)
    assert list(X_train.columns) == FEATURES


def test_no_index_in_both_splits(tiny_df):
    X_train, X_test, _, _ = split(tiny_df)
    assert set(X_train.index).isdisjoint(X_test.index)
