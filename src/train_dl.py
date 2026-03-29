"""
train_dl.py
-----------
Step 4 of the pipeline.

Trains LSTM, GRU, and CNN-LSTM with the best hyperparameters found
during tuning, evaluates on the test set, and compares all models.

Best params (from tuning run):
    LSTM     : units=128, dropout=0.3, lr=0.001  (3-layer + BN + clipnorm)
    GRU      : units=64,  dropout=0.2, lr=0.001
    CNN-LSTM : filters=64, units=64, dropout=0.2, lr=0.001

Outputs:
    models/lstm_model.h5
    models/gru_model.h5
    models/cnn_lstm_model.h5
    Plots and Visualizations/
        19_training_curves.png
        20_dl_pred_vs_actual.png
        21_dl_residuals.png
        22_all_models_comparison.png
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    create_sequences, SEQ_LEN,
    build_lstm, build_gru, build_cnn_lstm,
    get_callbacks, evaluate
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOT_DIR   = os.path.join(ROOT_DIR, "Plots and Visualizations")

TARGET = "Appliances"

# ── Best hyperparameters (from tuning) ───────────────────────────────────────
LSTM_PARAMS     = {"units": 256, "dropout": 0.2, "lr": 0.001}
GRU_PARAMS      = {"units": 64,  "dropout": 0.2, "lr": 0.001}
CNN_LSTM_PARAMS = {"filters": 64, "units": 64, "dropout": 0.2, "lr": 0.001}


# ── Load & Sequence ──────────────────────────────────────────────────────────

def load_sequences():
    """Load engineered splits and reshape into 3-D sequences for DL models."""
    train = pd.read_csv(os.path.join(PROC_DIR, "train_engineered.csv"))
    test  = pd.read_csv(os.path.join(PROC_DIR, "test_engineered.csv"))

    drop    = ["date", TARGET]
    X_train = train.drop(columns=drop).values
    y_train = train[TARGET].values
    X_test  = test.drop(columns=drop).values
    y_test  = test[TARGET].values

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)

    # Scale target — fit on train only to prevent leakage
    target_scaler = MinMaxScaler()
    y_train_seq = target_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_test_seq  = target_scaler.transform(y_test_seq.reshape(-1, 1)).flatten()

    print(f"[load]  X_train_seq: {X_train_seq.shape}  |  X_test_seq: {X_test_seq.shape}")
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler


# ── Train ────────────────────────────────────────────────────────────────────

def train_model(build_fn, params: dict, X_train_seq, y_train_seq, model_name: str):
    """Train a single DL model with early stopping and LR reduction."""
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"\n[{model_name}]  Training with params: {params}")

    model = build_fn(input_shape, **params)
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split = 0.05,
        epochs           = 100,
        batch_size       = 64,
        callbacks        = get_callbacks(patience=15),
        verbose          = 1
    )
    return model, history


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict):
    """Loss curves (train vs val) for all three DL models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.history["loss"],     label="Train loss")
        ax.plot(history.history["val_loss"], label="Val loss")
        ax.set_title(f"{name} — Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "19_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_dl_pred_vs_actual(y_test, predictions: dict):
    """Predicted vs actual overlay for all DL models (first 500 points)."""
    n = min(500, len(y_test))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.plot(y_test[:n], label="Actual",    linewidth=0.8, color="steelblue")
        ax.plot(y_pred[:n], label="Predicted", linewidth=0.8, color="salmon", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("Test index")
        ax.set_ylabel("Energy (Wh)")
        ax.legend()
    plt.suptitle("DL Models — Predicted vs Actual (first 500 test points)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "20_dl_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.show()



def plot_dl_residuals(y_test, predictions: dict):
    """Residuals vs fitted values for all DL models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=5, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name} — Residuals")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "21_dl_residuals.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_models_comparison(all_metrics: list):
    """Bar chart comparing all 5 models (LR, RF, LSTM, GRU, CNN-LSTM)."""
    df      = pd.DataFrame(all_metrics).set_index("Model")
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    colors  = ["steelblue", "salmon", "seagreen", "darkorange", "mediumpurple"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df.index, df[metric], color=colors[:len(df)])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("All Models — Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "22_all_models_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ── Save ─────────────────────────────────────────────────────────────────────

def save_dl_models(models: dict):
    """Save all three DL models in Keras format."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    name_map = {"LSTM": "lstm_model.h5", "GRU": "gru_model.h5", "CNN-LSTM": "cnn_lstm_model.h5"}
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, name_map[name])
        model.save(path)
        print(f"[save]  {name_map[name]}  →  models/")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run():
    """Train, evaluate, and compare LSTM, GRU, and CNN-LSTM."""
    print("=" * 50)
    print("STEP 4 — Deep Learning Model Training")
    print("=" * 50)

    os.makedirs(PLOT_DIR, exist_ok=True)

    X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler = load_sequences()

    # ── Train ──
    lstm_model,     lstm_history     = train_model(build_lstm,     LSTM_PARAMS,     X_train_seq, y_train_seq, "LSTM")
    gru_model,      gru_history      = train_model(build_gru,      GRU_PARAMS,      X_train_seq, y_train_seq, "GRU")
    cnn_lstm_model, cnn_lstm_history = train_model(build_cnn_lstm, CNN_LSTM_PARAMS, X_train_seq, y_train_seq, "CNN-LSTM")

    # ── Predict & inverse-transform back to Wh ──
    def inv(raw_pred):
        return target_scaler.inverse_transform(raw_pred.reshape(-1, 1)).flatten()

    lstm_pred     = inv(lstm_model.predict(X_test_seq,     verbose=0).flatten())
    gru_pred      = inv(gru_model.predict(X_test_seq,      verbose=0).flatten())
    cnn_lstm_pred = inv(cnn_lstm_model.predict(X_test_seq, verbose=0).flatten())
    y_test_wh     = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # ── Evaluate ──
    print("\n" + "=" * 50)
    print("DL Evaluation on Test Set")
    print("=" * 50)
    lstm_metrics     = evaluate(y_test_wh, lstm_pred,     "LSTM")
    gru_metrics      = evaluate(y_test_wh, gru_pred,      "GRU")
    cnn_lstm_metrics = evaluate(y_test_wh, cnn_lstm_pred, "CNN-LSTM")

    # ── Load baselines — evaluate on the SAME rows as DL models ──
    # DL sequences start at row SEQ_LEN-1, so align LR/RF to rows [SEQ_LEN-1 :]
    try:
        lr_model  = joblib.load(os.path.join(MODELS_DIR, "lr_model.h5"))
        rf_model  = joblib.load(os.path.join(MODELS_DIR, "rf_model.h5"))
        test_df   = pd.read_csv(os.path.join(PROC_DIR, "test_engineered.csv"))
        drop      = ["date", TARGET]
        X_test_2d = test_df.drop(columns=drop).values[SEQ_LEN - 1:]
        y_test_2d = test_df[TARGET].values[SEQ_LEN - 1:]
        lr_metrics = evaluate(y_test_2d, lr_model.predict(X_test_2d), "Linear Regression")
        rf_metrics = evaluate(y_test_2d, rf_model.predict(X_test_2d), "Random Forest")
        all_metrics = [lr_metrics, rf_metrics, lstm_metrics, gru_metrics, cnn_lstm_metrics]
    except FileNotFoundError:
        print("[warn]  Baseline models not found — skipping combined comparison.")
        all_metrics = [lstm_metrics, gru_metrics, cnn_lstm_metrics]

    # ── Comparison table ──
    print("\n" + "=" * 50)
    print("Full Model Comparison")
    print("=" * 50)
    print(pd.DataFrame(all_metrics).set_index("Model").round(4).to_string())

    # ── Plots ──
    histories   = {"LSTM": lstm_history, "GRU": gru_history, "CNN-LSTM": cnn_lstm_history}
    predictions = {"LSTM": lstm_pred,    "GRU": gru_pred,    "CNN-LSTM": cnn_lstm_pred}

    plot_training_curves(histories)
    plot_dl_pred_vs_actual(y_test_wh, predictions)
    plot_dl_residuals(y_test_wh, predictions)
    plot_all_models_comparison(all_metrics)

    # ── Save ──
    save_dl_models({"LSTM": lstm_model, "GRU": gru_model, "CNN-LSTM": cnn_lstm_model})
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler.h5"))
    print("[save]  target_scaler.h5  →  models/")

    print("\n" + "=" * 50)
    print("Deep learning training complete.")
    print("=" * 50)
    return lstm_model, gru_model, cnn_lstm_model, all_metrics


if __name__ == "__main__":
    run()
