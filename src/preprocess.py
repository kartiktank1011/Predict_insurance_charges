"""
src/preprocess.py
-----------------
Preprocessing pipeline for the insurance charges dataset.

Steps:
  - One-hot encode categorical columns: sex, smoker, region
  - Standard-scale numerical columns: age, bmi, children
  - Optional log-transform of the target (charges) to reduce right-skew
  - 80/20 train-test split with random_state=42
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------
# Column groups
# -----------------------------------------------------------------
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"


def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer.

    Numeric columns  → StandardScaler
    Categorical cols → OneHotEncoder (drop first to avoid multicollinearity)
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_data(filepath: str = "data/insurance.csv") -> pd.DataFrame:
    """Load the raw insurance CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting the preprocessor."""
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )
    return num_names + cat_names


def prepare_data(
    filepath: str = "data/insurance.csv",
    log_transform_target: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end data preparation.

    Parameters
    ----------
    filepath : str
        Path to insurance.csv.
    log_transform_target : bool
        If True, model targets will be log1p(charges); predictions must be
        expm1'd to get back to dollar values.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test : np.ndarray
        Preprocessed feature matrices.
    y_train, y_test : np.ndarray
        Target vectors (log-transformed if requested).
    preprocessor : fitted ColumnTransformer
    log_transform_target : bool
        Passed through so callers know whether to back-transform.
    """
    df = load_data(filepath)

    # Validate expected columns
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    if log_transform_target:
        y = np.log1p(y)

    # Split first, then fit preprocessor on train only (no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor, log_transform_target


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, prep, log_t = prepare_data()
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train range : {y_train.min():.4f} – {y_train.max():.4f}")
    print(f"Log-transform : {log_t}")
    print(f"Feature names : {get_feature_names(prep)}")
