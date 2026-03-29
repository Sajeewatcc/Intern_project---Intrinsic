"""
train.py
--------
Step 3 of the pipeline.

Loads engineered train/test splits, trains Linear Regression and
Random Forest baselines with hyperparameter tuning, evaluates both
on the test set, plots results, and saves trained models.

Outputs:
    models/lr_model.h5
    models/rf_model.h5
    Plots and Visualizations/
        14_lr_pred_vs_actual.png
        15_rf_pred_vs_actual.png
        16_residuals.png
        17_metrics_comparison.png
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    get_lr_model, get_lr_param_grid,
    get_rf_model, get_rf_param_grid,
    evaluate
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOT_DIR   = os.path.join(ROOT_DIR, "Plots and Visualizations")

TARGET = "Appliances"


# ── Load ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load engineered train and test splits."""
    train = pd.read_csv(os.path.join(PROC_DIR, "train_engineered.csv"))
    test  = pd.read_csv(os.path.join(PROC_DIR, "test_engineered.csv"))

    drop = ["date", TARGET]
    X_train = train.drop(columns=drop).values
    y_train = train[TARGET].values
    X_test  = test.drop(columns=drop).values
    y_test  = test[TARGET].values

    feature_cols = [c for c in train.columns if c not in drop]

    print(f"[load]  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, feature_cols


# ── Training with Hyperparameter Tuning ──────────────────────────────────────

def train_lr(X_train, y_train):
    """
    Train Ridge Regression with GridSearchCV.
    GridSearch is suitable here — the alpha search space is small (6 values).
    Uses 5-fold cross-validation, scoring on negative RMSE.
    """
    print("\n[LR]  GridSearchCV over alpha (TimeSeriesSplit, 5 folds)...")
    grid = GridSearchCV(
        estimator  = get_lr_model(),
        param_grid = get_lr_param_grid(),
        cv         = TimeSeriesSplit(n_splits=5),
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = -1,
        verbose    = 0
    )
    grid.fit(X_train, y_train)
    print(f"[LR]  Best params : {grid.best_params_}")
    print(f"[LR]  Best CV RMSE: {-grid.best_score_:.4f}")
    return grid.best_estimator_


def train_rf(X_train, y_train):
    """
    Train Random Forest with RandomizedSearchCV.
    RandomizedSearch is used because the RF search space is large
    (combinatorial explosion over n_estimators, depth, splits, etc.).
    Samples 20 random combinations, 3-fold CV.
    """
    print("\n[RF]  RandomizedSearchCV (20 iterations, TimeSeriesSplit 3 folds)...")
    search = RandomizedSearchCV(
        estimator          = get_rf_model(),
        param_distributions= get_rf_param_grid(),
        n_iter             = 20,
        cv                 = TimeSeriesSplit(n_splits=3),
        scoring            = "neg_root_mean_squared_error",
        random_state       = 42,
        n_jobs             = -1,
        verbose            = 1
    )
    search.fit(X_train, y_train)
    print(f"[RF]  Best params : {search.best_params_}")
    print(f"[RF]  Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_estimator_


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pred_vs_actual(y_test, y_pred, model_name: str, filename: str):
    """Scatter plot of predicted vs actual values + identity line."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Scatter
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=8, color="steelblue")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
    axes[0].set(title=f"{model_name} — Predicted vs Actual",
                xlabel="Actual (Wh)", ylabel="Predicted (Wh)")
    axes[0].legend()

    # Time-series overlay (first 500 points for clarity)
    n = min(500, len(y_test))
    axes[1].plot(y_test[:n],  label="Actual",    linewidth=0.8, color="steelblue")
    axes[1].plot(y_pred[:n],  label="Predicted", linewidth=0.8, color="salmon", alpha=0.8)
    axes[1].set(title=f"{model_name} — First {n} Test Predictions",
                xlabel="Test index", ylabel="Energy (Wh)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()



def plot_metrics_comparison(lr_metrics: dict, rf_metrics: dict):
    """Bar chart comparing MAE, RMSE, MAPE and R² for both models."""
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    lr_vals = [lr_metrics[m] for m in metrics]
    rf_vals = [rf_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, lr_vals, width, label="Linear Regression", color="steelblue")
    ax.bar(x + width/2, rf_vals, width, label="Random Forest",      color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Model Performance Comparison — LR vs RF")
    ax.set_ylabel("Metric Value")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "17_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ── Save ─────────────────────────────────────────────────────────────────────

def save_models(lr_model, rf_model):
    """Persist trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.h5"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.h5"))
    print(f"\n[save]  lr_model.h5  →  models/")
    print(f"[save]  rf_model.h5  →  models/")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run():
    """Train, evaluate, and compare LR and RF baselines."""
    print("=" * 50)
    print("STEP 3 — Baseline Model Training")
    print("=" * 50)

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test, feature_cols = load_data()

    # ── Train ──
    lr_model = train_lr(X_train, y_train)
    rf_model = train_rf(X_train, y_train)

    # ── Predict ──
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # ── Evaluate ──
    print("\n" + "=" * 50)
    print("Evaluation on Test Set")
    print("=" * 50)
    lr_metrics = evaluate(y_test, lr_pred, "Linear Regression")
    rf_metrics = evaluate(y_test, rf_pred, "Random Forest")

    # ── Comparison table ──
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    comparison = pd.DataFrame([lr_metrics, rf_metrics]).set_index("Model").round(4)
    print(comparison.to_string())

    # ── Plots ──
    plot_pred_vs_actual(y_test, lr_pred, "Linear Regression", "14_lr_pred_vs_actual.png")
    plot_pred_vs_actual(y_test, rf_pred, "Random Forest",     "15_rf_pred_vs_actual.png")
    plot_metrics_comparison(lr_metrics, rf_metrics)

    # ── Feature importance (RF only) ──
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
    feat_imp.plot(kind="bar", figsize=(13, 4), color="steelblue",
                  title="Random Forest — Top 15 Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "18_rf_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── Save models ──
    save_models(lr_model, rf_model)

    print("\n" + "=" * 50)
    print("Baseline training complete.")
    print("=" * 50)

    return lr_model, rf_model, lr_metrics, rf_metrics


if __name__ == "__main__":
    run()
