"""
tests/test_prediction.py
------------------------
Unit tests for the prediction pipeline:
  - Model bundle structure
  - Single and batch predictions
  - Boundary conditions
  - Smoker vs non-smoker ordering
  - Positive output guarantee
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def predict_charges(bundle: dict, user_input: dict) -> float:
    """Mirror the logic in app/streamlit_app.py."""
    df = pd.DataFrame([user_input])
    X = bundle["preprocessor"].transform(df)
    raw = bundle["model"].predict(X)[0]
    if bundle.get("log_transform_target", False):
        return float(np.expm1(raw))
    return float(raw)


def make_sample(
    age=30,
    sex="male",
    bmi=25.0,
    children=0,
    smoker="no",
    region="northeast",
) -> dict:
    return dict(age=age, sex=sex, bmi=bmi, children=children,
                smoker=smoker, region=region)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def bundle():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            "models/best_model.pkl not found. "
            "Run `python src/tune_evaluate.py` first."
        )
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------------ #
# Tests: bundle structure
# ------------------------------------------------------------------ #

class TestBundleStructure:
    def test_required_keys(self, bundle):
        required = {"model", "preprocessor", "feature_names",
                    "log_transform_target", "model_name", "test_metrics"}
        assert required.issubset(set(bundle.keys()))

    def test_metrics_keys(self, bundle):
        assert "R²" in bundle["test_metrics"]
        assert "MAE" in bundle["test_metrics"]
        assert "RMSE" in bundle["test_metrics"]

    def test_r2_reasonable(self, bundle):
        r2 = bundle["test_metrics"]["R²"]
        assert 0.70 <= r2 <= 1.0, f"R² out of expected range: {r2}"

    def test_model_has_predict(self, bundle):
        assert hasattr(bundle["model"], "predict")

    def test_preprocessor_has_transform(self, bundle):
        assert hasattr(bundle["preprocessor"], "transform")

    def test_feature_names_is_list(self, bundle):
        assert isinstance(bundle["feature_names"], list)
        assert len(bundle["feature_names"]) > 0


# ------------------------------------------------------------------ #
# Tests: prediction values
# ------------------------------------------------------------------ #

class TestPrediction:
    def test_returns_float(self, bundle):
        charge = predict_charges(bundle, make_sample())
        assert isinstance(charge, float)

    def test_positive_output(self, bundle):
        charge = predict_charges(bundle, make_sample())
        assert charge > 0, f"Charge should be positive, got {charge}"

    def test_smoker_more_expensive(self, bundle):
        """Smokers should be predicted higher charges than non-smokers, all else equal."""
        smoker_charge = predict_charges(bundle, make_sample(smoker="yes"))
        non_smoker_charge = predict_charges(bundle, make_sample(smoker="no"))
        assert smoker_charge > non_smoker_charge, (
            f"Expected smoker ({smoker_charge:.0f}) > non-smoker ({non_smoker_charge:.0f})"
        )

    def test_higher_age_higher_charge(self, bundle):
        """Older individuals should generally have higher charges."""
        young = predict_charges(bundle, make_sample(age=20))
        old = predict_charges(bundle, make_sample(age=60))
        assert old > young, f"Expected age 60 ({old:.0f}) > age 20 ({young:.0f})"

    def test_high_bmi_higher_charge(self, bundle):
        """Higher BMI should produce higher predicted charge."""
        low_bmi = predict_charges(bundle, make_sample(bmi=18.0))
        high_bmi = predict_charges(bundle, make_sample(bmi=45.0))
        assert high_bmi > low_bmi, f"Expected BMI 45 ({high_bmi:.0f}) > BMI 18 ({low_bmi:.0f})"

    def test_all_regions_produce_valid_outputs(self, bundle):
        regions = ["northeast", "northwest", "southeast", "southwest"]
        for region in regions:
            charge = predict_charges(bundle, make_sample(region=region))
            assert charge > 0, f"Non-positive charge for region {region}: {charge}"

    def test_male_and_female_both_valid(self, bundle):
        for sex in ["male", "female"]:
            charge = predict_charges(bundle, make_sample(sex=sex))
            assert charge > 0

    def test_zero_children(self, bundle):
        charge = predict_charges(bundle, make_sample(children=0))
        assert charge > 0

    def test_max_children(self, bundle):
        charge = predict_charges(bundle, make_sample(children=5))
        assert charge > 0

    def test_charge_realistic_range(self, bundle):
        """Predicted charges should be within a realistic dollar range."""
        charge = predict_charges(bundle, make_sample(age=40, bmi=30, smoker="no"))
        assert 500 < charge < 200_000, f"Charge out of realistic range: {charge}"

    def test_extreme_smoker_charge(self, bundle):
        """Elderly obese smoker should produce very high charge."""
        charge = predict_charges(
            bundle, make_sample(age=64, bmi=50, smoker="yes", children=0)
        )
        assert charge > 20_000, f"Expected high charge for extreme case, got {charge:.0f}"


# ------------------------------------------------------------------ #
# Tests: batch consistency
# ------------------------------------------------------------------ #

class TestBatchPrediction:
    def test_batch_vs_single(self, bundle):
        """Batch prediction should match individual predictions."""
        samples = [
            make_sample(age=25, smoker="no"),
            make_sample(age=40, smoker="yes", bmi=32),
            make_sample(age=55, sex="female", children=3),
        ]
        # Single predictions
        singles = [predict_charges(bundle, s) for s in samples]

        # Batch prediction
        df = pd.DataFrame(samples)
        X = bundle["preprocessor"].transform(df)
        raw_batch = bundle["model"].predict(X)
        if bundle.get("log_transform_target", False):
            batch = list(np.expm1(raw_batch))
        else:
            batch = list(raw_batch)

        for i, (s, b) in enumerate(zip(singles, batch)):
            assert abs(s - b) < 1e-6, f"Mismatch at sample {i}: {s} vs {b}"
