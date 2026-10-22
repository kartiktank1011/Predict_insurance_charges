"""
src/tune_evaluate.py
--------------------
Hyperparameter tuning and final evaluation of the two best models
(Gradient Boosting & Random Forest) using RandomizedSearchCV.

Outputs:
  - reports/tuning_results.csv
  - reports/actual_vs_predicted.png
  - reports/residuals.png
  - reports/feature_importance.png
  - models/best_model.pkl  (joblib)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless rendering
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import prepare_data, get_feature_names

RANDOM_STATE = 42


# ------------------------------------------------------------------
# Hyperparameter grids
# ------------------------------------------------------------------
PARAM_GRIDS = {
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [200, 300, 500],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.8, 0.9, 1.0],
        },
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "params": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
    },
}


def tune_model(name: str, config: dict, X_train, y_train, n_iter: int = 30):
    """Run RandomizedSearchCV and return the best estimator + CV score."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        config["estimator"],
        config["params"],
        n_iter=n_iter,
        scoring="r2",
        cv=kf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best CV R² for {name}: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_, search.best_params_


def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_transform: bool = True,
) -> dict:
    """Compute R², MAE, RMSE on the held-out test set.

    If log_transform is True, back-transforms predictions via expm1.
    """
    y_pred_raw = model.predict(X_test)

    if log_transform:
        y_pred = np.expm1(y_pred_raw)
        y_true = np.expm1(y_test)
    else:
        y_pred = y_pred_raw
        y_true = y_test

    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "y_pred": y_pred,
        "y_true": y_true,
    }


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------
PLOT_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3d4058",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#a0a0b0",
    "ytick.color": "#a0a0b0",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2d3e",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, save_path: str):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.45, s=22, color="#7c83fd", edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#ff6b6b", lw=2, label="Perfect fit")
    ax.set_xlabel("Actual Charges ($)", fontsize=13)
    ax.set_ylabel("Predicted Charges ($)", fontsize=13)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=15, fontweight="bold")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {save_path}")


def plot_residuals(y_true, y_pred, model_name: str, save_path: str):
    _apply_style()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=22, color="#4ecdc4", edgecolors="none")
    axes[0].axhline(0, color="#ff6b6b", lw=2)
    axes[0].set_xlabel("Predicted Charges ($)", fontsize=12)
    axes[0].set_ylabel("Residual ($)", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    axes[0].grid(True)

    # Residual histogram
    axes[1].hist(residuals, bins=40, color="#7c83fd", edgecolor="#0f1117", alpha=0.85)
    axes[1].axvline(0, color="#ff6b6b", lw=2)
    axes[1].set_xlabel("Residual ($)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")
    axes[1].grid(True)

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {save_path}")


def plot_feature_importance(model, feature_names: list, model_name: str, save_path: str):
    _apply_style()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"    {model_name} does not have feature_importances_; skipping plot.")
        return

    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    indices = indices[:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))[::-1]
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color=colors,
        edgecolor="none",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PHASE 5 – Hyperparameter Tuning & Evaluation")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n[1] Loading & preprocessing data …")
    X_train, X_test, y_train, y_test, preprocessor, log_t = prepare_data()
    feature_names = get_feature_names(preprocessor)
    print(f"    Features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # Tune both candidate models
    # ------------------------------------------------------------------
    tuning_rows = []
    best_score = -np.inf
    best_model = None
    best_name = ""

    for model_name, config in PARAM_GRIDS.items():
        print(f"\n[2] Tuning {model_name} …")
        estimator, cv_score, best_params = tune_model(
            model_name, config, X_train, y_train, n_iter=30
        )
        # Test set evaluation
        metrics = evaluate_on_test(estimator, X_test, y_test, log_transform=log_t)
        print(
            f"    Test  ->  R²={metrics['R²']:.4f}  MAE=${metrics['MAE']:,.0f}  "
            f"RMSE=${metrics['RMSE']:,.0f}"
        )
        tuning_rows.append(
            {
                "Model": model_name,
                "CV R²": round(cv_score, 4),
                "Test R²": round(metrics["R²"], 4),
                "Test MAE ($)": round(metrics["MAE"], 2),
                "Test RMSE ($)": round(metrics["RMSE"], 2),
                "Best Params": str(best_params),
            }
        )
        if metrics["R²"] > best_score:
            best_score = metrics["R²"]
            best_model = estimator
            best_name = model_name
            best_metrics = metrics

    # Save tuning table
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv("reports/tuning_results.csv", index=False)
    print(f"\n    Tuning results saved -> reports/tuning_results.csv")

    # ------------------------------------------------------------------
    # Plots for the best model
    # ------------------------------------------------------------------
    print(f"\n[3] Best model: {best_name}  (Test R²={best_score:.4f})")
    y_true = best_metrics["y_true"]
    y_pred = best_metrics["y_pred"]

    plot_actual_vs_predicted(
        y_true, y_pred, best_name, "reports/actual_vs_predicted.png"
    )
    plot_residuals(y_true, y_pred, best_name, "reports/residuals.png")
    plot_feature_importance(
        best_model, feature_names, best_name, "reports/feature_importance.png"
    )

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    model_bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "log_transform_target": log_t,
        "model_name": best_name,
        "test_metrics": {
            "R²": best_score,
            "MAE": best_metrics["MAE"],
            "RMSE": best_metrics["RMSE"],
        },
    }
    joblib.dump(model_bundle, "models/best_model.pkl")
    print(f"\n    Best model bundle saved -> models/best_model.pkl")
    print("\n" + "=" * 60)
    print(f"DONE — {best_name}: R²={best_score:.4f}, "
          f"MAE=${best_metrics['MAE']:,.0f}, RMSE=${best_metrics['RMSE']:,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
