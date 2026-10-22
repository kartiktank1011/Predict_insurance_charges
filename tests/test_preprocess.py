"""
tests/test_preprocess.py
------------------------
Unit tests for src/preprocess.py

Covers:
  - build_preprocessor returns a ColumnTransformer
  - load_data returns expected shape and columns
  - prepare_data splits correctly, outputs correct shapes
  - log-transform behaviour
  - feature name extraction after fitting
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Make sure the project root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.preprocess import (
    build_preprocessor,
    load_data,
    prepare_data,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

DATA_PATH = os.path.join(ROOT, "data", "insurance.csv")

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def prepared_data_log():
    return prepare_data(DATA_PATH, log_transform_target=True)


@pytest.fixture(scope="module")
def prepared_data_nolog():
    return prepare_data(DATA_PATH, log_transform_target=False)


# ------------------------------------------------------------------ #
# Tests: load_data
# ------------------------------------------------------------------ #

class TestLoadData:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_row_count(self, raw_df):
        assert len(raw_df) == 1338, f"Expected 1338 rows, got {len(raw_df)}"

    def test_column_count(self, raw_df):
        assert raw_df.shape[1] == 7

    def test_expected_columns(self, raw_df):
        expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
        assert expected == set(raw_df.columns)

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_charges_positive(self, raw_df):
        assert (raw_df[TARGET] > 0).all()


# ------------------------------------------------------------------ #
# Tests: build_preprocessor
# ------------------------------------------------------------------ #

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        ct = build_preprocessor()
        assert isinstance(ct, ColumnTransformer)

    def test_transformer_names(self):
        ct = build_preprocessor()
        names = [t[0] for t in ct.transformers]
        assert "num" in names
        assert "cat" in names

    def test_fit_transform_output_shape(self, raw_df):
        ct = build_preprocessor()
        X = raw_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_t = ct.fit_transform(X)
        # 3 numeric + (1 sex + 1 smoker + 3 region) = 8 features with drop='first'
        assert X_t.shape == (1338, 8), f"Unexpected shape: {X_t.shape}"


# ------------------------------------------------------------------ #
# Tests: prepare_data
# ------------------------------------------------------------------ #

class TestPrepareData:
    def test_split_sizes_log(self, prepared_data_log):
        X_train, X_test, y_train, y_test, _, _ = prepared_data_log
        total = len(y_train) + len(y_test)
        assert total == 1338
        assert len(y_test) == pytest.approx(268, abs=2)  # 20% of 1338

    def test_split_sizes_nolog(self, prepared_data_nolog):
        X_train, X_test, y_train, y_test, _, _ = prepared_data_nolog
        total = len(y_train) + len(y_test)
        assert total == 1338

    def test_feature_matrix_dtype(self, prepared_data_log):
        X_train, X_test, *_ = prepared_data_log
        assert X_train.dtype in [np.float32, np.float64]

    def test_no_nan_in_features(self, prepared_data_log):
        X_train, X_test, y_train, y_test, *_ = prepared_data_log
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_log_transform_flag_true(self, prepared_data_log):
        *_, log_flag = prepared_data_log
        assert log_flag is True

    def test_log_transform_flag_false(self, prepared_data_nolog):
        *_, log_flag = prepared_data_nolog
        assert log_flag is False

    def test_log_target_range(self, prepared_data_log):
        _, _, y_train, y_test, _, _ = prepared_data_log
        # log1p(charges): charges > 0 so log1p > 0; max is ~log1p(63770) ≈ 11
        assert y_train.min() > 0
        assert y_train.max() < 15

    def test_nolog_target_range(self, prepared_data_nolog):
        _, _, y_train, y_test, _, _ = prepared_data_nolog
        assert y_train.min() > 0
        assert y_train.max() > 1000  # charges in dollars

    def test_reproducibility(self):
        """Same split produced with same random_state."""
        result1 = prepare_data(DATA_PATH, random_state=42)
        result2 = prepare_data(DATA_PATH, random_state=42)
        np.testing.assert_array_equal(result1[2], result2[2])  # y_train

    def test_different_random_state_different_split(self):
        r1 = prepare_data(DATA_PATH, random_state=42)
        r2 = prepare_data(DATA_PATH, random_state=99)
        # With very high probability the splits differ
        assert not np.array_equal(r1[2], r2[2])


# ------------------------------------------------------------------ #
# Tests: get_feature_names
# ------------------------------------------------------------------ #

class TestGetFeatureNames:
    def test_feature_count(self, prepared_data_log):
        _, _, _, _, preprocessor, _ = prepared_data_log
        names = get_feature_names(preprocessor)
        assert len(names) == 8  # 3 numeric + 5 one-hot (drop first)

    def test_numeric_names_present(self, prepared_data_log):
        _, _, _, _, preprocessor, _ = prepared_data_log
        names = get_feature_names(preprocessor)
        for n in NUMERIC_FEATURES:
            assert n in names

    def test_returns_list(self, prepared_data_log):
        _, _, _, _, preprocessor, _ = prepared_data_log
        names = get_feature_names(preprocessor)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
