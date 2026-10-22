"""
src/train.py
------------
Train and compare six regression models on the insurance charges dataset.

Models:
  - Linear Regression
  - Ridge
  - Lasso
  - Random Forest
  - Gradient Boosting
  - XGBoost

Evaluation:
  - 5-fold cross-validation on training set
  - Reports R², MAE, and RMSE for each model
  - Saves comparison table to reports/model_comparison.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# Ensure src is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import prepare_data, get_feature_names

RANDOM_STATE = 42


def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_models() -> dict:
    """Return a dict of model name -> instantiated estimator."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        ),
    }


def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> pd.DataFrame:
    """Run 5-fold CV on every model and return a comparison DataFrame."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "r2": "r2",
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    }

    rows = []
    for name, model in models.items():
        print(f"  Cross-validating: {name} …", flush=True)
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        rows.append(
            {
                "Model": name,
                "CV R²": cv_results["test_r2"].mean().round(4),
                "CV MAE": (-cv_results["test_mae"]).mean().round(4),
                "CV RMSE": (-cv_results["test_rmse"]).mean().round(4),
                "R² Std": cv_results["test_r2"].std().round(4),
            }
        )

    df = pd.DataFrame(rows).sort_values("CV R²", ascending=False).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("PHASE 4 – Baseline Model Comparison")
    print("=" * 60)

    print("\n[1] Loading & preprocessing data …")
    X_train, X_test, y_train, y_test, preprocessor, log_t = prepare_data()
    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"    Log-transform target: {log_t}")

    print("\n[2] Running 5-fold cross-validation …")
    models = get_models()
    results_df = cross_validate_models(models, X_train, y_train)

    print("\n[3] Cross-Validation Results:")
    print(results_df.to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n    Saved -> reports/model_comparison.csv")

    return results_df, models, X_train, X_test, y_train, y_test, preprocessor, log_t


if __name__ == "__main__":
    main()
