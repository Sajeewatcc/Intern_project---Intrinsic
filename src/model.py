"""
model.py
--------
Defines all model architectures, hyperparameter grids,
and the shared evaluation function.

Baseline models:
    - Linear Regression  (via Ridge — adds alpha as a tunable hyperparameter)
    - Random Forest Regressor

Deep Learning models:
    - LSTM
    - GRU
    - CNN-LSTM hybrid

Evaluation metrics:
    - MAE   (Mean Absolute Error)
    - RMSE  (Root Mean Squared Error)
    - MAPE  (Mean Absolute Percentage Error)
    - R²    (Coefficient of Determination)
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, GRU, Conv1D, MaxPooling1D,
                                     Dense, Dropout, Flatten, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ── Model Definitions ────────────────────────────────────────────────────────

def get_lr_model() -> Ridge:
    """
    Ridge Regression as the linear baseline.
    Ridge extends plain Linear Regression with an L2 penalty (alpha),
    giving us a meaningful hyperparameter to tune while keeping
    the model fully interpretable and linear.
    """
    return Ridge()


def get_lr_param_grid() -> dict:
    """Hyperparameter search space for Ridge Regression."""
    return {
        "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    }


def get_rf_model() -> RandomForestRegressor:
    """Random Forest Regressor — ensemble tree-based baseline."""
    return RandomForestRegressor(random_state=42, n_jobs=-1)


def get_rf_param_grid() -> dict:
    """Hyperparameter search space for Random Forest (used with RandomizedSearchCV)."""
    return {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2"]
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """
    Compute regression evaluation metrics.

    Returns a dict with:
        MAE   — average absolute error
        RMSE  — penalises large errors more than MAE
        MAPE  — percentage error (zero targets excluded to avoid div/0)
        R²    — proportion of variance explained (1.0 = perfect)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — exclude zero targets to prevent division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    print(f"\n[{model_name}]")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────

# Sequence length: 24 steps × 10 min = 4 hours of history fed into each sample
SEQ_LEN = 24


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    """
    Reshape flat (n_samples, n_features) arrays into overlapping 3-D sequences.

    Window : X[i - seq_len + 1 : i + 1]  — seq_len rows ending AT row i
    Target : y[i]                          — Appliances at the last row of the window

    Why this alignment?
        The last row of the window is the prediction row itself.
        Its lag_1 feature = Appliances[i-1] — the most recent known energy value.
        This is identical to what LR/RF see: features at row t predict Appliances[t].
        Previous (broken) code used y[i + seq_len] as target, which meant the window
        ended 1 row before the prediction row — the LSTM was missing lag_1[t] and
        effectively predicting further ahead than LR, causing LR to appear best.
    """
    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)



def get_callbacks(patience: int = 10) -> list:
    """
    Standard callbacks for all DL models:
        EarlyStopping     — stops training when val_loss stops improving
        ReduceLROnPlateau — halves LR after 5 stagnant epochs
    """
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
    ]


# ── LSTM ─────────────────────────────────────────────────────────────────────

def build_lstm(input_shape: tuple, units: int = 256,
               dropout: float = 0.2, lr: float = 0.001) -> Sequential:
    """
    Two-layer stacked LSTM with BatchNormalization and gradient clipping.
        Layer 1: LSTM(units)      — wider first layer captures richer short-term patterns
        BatchNorm + Dropout
        Layer 2: LSTM(units//2)   — distils into a single context vector
        Dropout
        Dense(64) + Dense(32)     — deeper projection head
        Dense(1)                  — regression output (energy in Wh)

    Improvements over original:
        - units increased 128→256 (layer 1) and 64→128 (layer 2) for more capacity
        - BatchNormalization after layer 1 stabilises activations, speeds convergence
        - clipnorm=1.0 in Adam prevents exploding gradients
        - Wider Dense(64) head replaces original Dense(32)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="LSTM")
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
    return model


# ── GRU ──────────────────────────────────────────────────────────────────────

def build_gru(input_shape: tuple, units: int = 64,
              dropout: float = 0.2, lr: float = 0.001) -> Sequential:
    """
    Two-layer stacked GRU — fewer parameters than LSTM, often trains faster.
        Layer 1: GRU(units)
        Layer 2: GRU(units//2)
        Output : Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=True),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="GRU")
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model


# ── CNN-LSTM ─────────────────────────────────────────────────────────────────

def build_cnn_lstm(input_shape: tuple, filters: int = 64, units: int = 64,
                   dropout: float = 0.2, lr: float = 0.001) -> Sequential:
    """
    CNN-LSTM hybrid:
        Conv1D + MaxPooling  — extracts local temporal patterns (feature maps)
        LSTM                 — captures long-range dependencies on those features
        Dense(1)             — regression output

    Why hybrid?  CNN reduces sequence length before the LSTM, lowering
    computational cost while preserving multi-scale temporal structure.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="CNN_LSTM")
    model.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return model
