"""train_dl.py - Step 4: Train LSTM, GRU, CNN-LSTM; evaluate and compare all models."""

import os
import sys
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (create_sequences, SEQ_LEN, build_lstm, build_gru,
                   build_cnn_lstm, get_callbacks, tf)
from evaluation import evaluate
tf.random.set_seed(SEED)

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR      = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
PLOT_DIR      = os.path.join(ROOT_DIR, "Plots and Visualizations")
TARGET        = "Appliances"
LSTM_SEQ_LEN  = 144  # 24 hours - plays to LSTM's long-range memory advantage

LSTM_PARAMS     = {"units": 128, "dropout": 0.2, "lr": 0.001}
GRU_PARAMS      = {"units": 64,  "dropout": 0.2, "lr": 0.001}
CNN_LSTM_PARAMS = {"filters": 64, "units": 64, "dropout": 0.2, "lr": 0.001}


def load_sequences(seq_len=SEQ_LEN):
    train = pd.read_csv(os.path.join(PROC_DIR, "train_engineered.csv"))
    test  = pd.read_csv(os.path.join(PROC_DIR, "test_engineered.csv"))
    drop  = ["date", TARGET]
    X_train, y_train = train.drop(columns=drop).values, train[TARGET].values
    X_test,  y_test  = test.drop(columns=drop).values,  test[TARGET].values
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  seq_len)
    target_scaler = MinMaxScaler()
    y_train_seq = target_scaler.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_test_seq  = target_scaler.transform(y_test_seq.reshape(-1, 1)).flatten()
    print(f"[load]  seq_len={seq_len}  X_train: {X_train_seq.shape}  |  X_test: {X_test_seq.shape}")
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, target_scaler


def train_model(build_fn, params, X_train, y_train, model_name):
    print(f"\n[{model_name}]  Training with params: {params}")
    model = build_fn((X_train.shape[1], X_train.shape[2]), **params)
    history = model.fit(X_train, y_train, validation_split=0.05,
                        epochs=100, batch_size=64,
                        callbacks=get_callbacks(patience=15), verbose=1)
    return model, history


def plot_training_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.history["loss"],     label="Train loss")
        ax.plot(history.history["val_loss"], label="Val loss")
        ax.set_title(f"{name} - Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "19_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_dl_pred_vs_actual(pred_pairs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (y_true, y_pred)) in zip(axes, pred_pairs.items()):
        n = min(500, len(y_true))
        ax.plot(y_true[:n], label="Actual",    linewidth=0.8, color="steelblue")
        ax.plot(y_pred[:n], label="Predicted", linewidth=0.8, color="salmon", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("Test index")
        ax.set_ylabel("Energy (Wh)")
        ax.legend()
    plt.suptitle("DL Models - Predicted vs Actual (first 500 test points)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "20_dl_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_dl_residuals(pred_pairs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (y_true, y_pred)) in zip(axes, pred_pairs.items()):
        ax.scatter(y_pred, y_true - y_pred, alpha=0.3, s=5, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name} - Residuals")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "21_dl_residuals.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_models_comparison(all_metrics):
    df     = pd.DataFrame(all_metrics).set_index("Model")
    colors = ["steelblue", "salmon", "seagreen", "darkorange", "mediumpurple"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE", "R2"]):
        bars = ax.bar(df.index, df[metric], color=colors[:len(df)])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("All Models - Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "22_all_models_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()


def run():
    print("=" * 50)
    print("STEP 4 - Deep Learning Model Training")
    print("=" * 50)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # LSTM uses a longer sequence (144 steps = 24 h) to exploit its long-range memory
    X_lstm_tr, y_lstm_tr, X_lstm_te, y_lstm_te, lstm_scaler = load_sequences(LSTM_SEQ_LEN)
    X_train,   y_train,   X_test,   y_test,   target_scaler = load_sequences(SEQ_LEN)

    lstm_model,     lstm_hist     = train_model(build_lstm,     LSTM_PARAMS,     X_lstm_tr, y_lstm_tr, "LSTM")
    gru_model,      gru_hist      = train_model(build_gru,      GRU_PARAMS,      X_train,   y_train,   "GRU")
    cnn_lstm_model, cnn_lstm_hist = train_model(build_cnn_lstm, CNN_LSTM_PARAMS, X_train,   y_train,   "CNN-LSTM")

    inv       = lambda p, sc: sc.inverse_transform(p.flatten().reshape(-1, 1)).flatten()
    lstm_pred     = inv(lstm_model.predict(X_lstm_te, verbose=0), lstm_scaler)
    gru_pred      = inv(gru_model.predict(X_test,     verbose=0), target_scaler)
    cnn_lstm_pred = inv(cnn_lstm_model.predict(X_test, verbose=0), target_scaler)
    y_lstm_wh  = lstm_scaler.inverse_transform(y_lstm_te.reshape(-1, 1)).flatten()
    y_test_wh  = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    print("\n" + "=" * 50)
    print("DL Evaluation on Test Set")
    print("=" * 50)
    lstm_metrics     = evaluate(y_lstm_wh,  lstm_pred,     "LSTM")
    gru_metrics      = evaluate(y_test_wh,  gru_pred,      "GRU")
    cnn_lstm_metrics = evaluate(y_test_wh,  cnn_lstm_pred, "CNN-LSTM")

    try:
        lr_model  = joblib.load(os.path.join(MODELS_DIR, "lr_model.h5"))
        rf_model  = joblib.load(os.path.join(MODELS_DIR, "rf_model.h5"))
        test_df   = pd.read_csv(os.path.join(PROC_DIR, "test_engineered.csv"))
        X_test_2d = test_df.drop(columns=["date", TARGET]).values[SEQ_LEN - 1:]
        y_test_2d = test_df[TARGET].values[SEQ_LEN - 1:]
        lr_metrics = evaluate(y_test_2d, lr_model.predict(X_test_2d), "Linear Regression")
        rf_metrics = evaluate(y_test_2d, rf_model.predict(X_test_2d), "Random Forest")
        all_metrics = [lr_metrics, rf_metrics, lstm_metrics, gru_metrics, cnn_lstm_metrics]
    except FileNotFoundError:
        print("[warn]  Baseline models not found - skipping combined comparison.")
        all_metrics = [lstm_metrics, gru_metrics, cnn_lstm_metrics]

    print("\n" + "=" * 50)
    print("Full Model Comparison")
    print("=" * 50)
    print(pd.DataFrame(all_metrics).set_index("Model").round(4).to_string())

    histories   = {"LSTM": lstm_hist, "GRU": gru_hist, "CNN-LSTM": cnn_lstm_hist}
    # Each model paired with its own y_test (LSTM uses longer sequence)
    pred_pairs  = {"LSTM": (y_lstm_wh, lstm_pred), "GRU": (y_test_wh, gru_pred), "CNN-LSTM": (y_test_wh, cnn_lstm_pred)}
    plot_training_curves(histories)
    plot_dl_pred_vs_actual(pred_pairs)
    plot_dl_residuals(pred_pairs)
    plot_all_models_comparison(all_metrics)

    os.makedirs(MODELS_DIR, exist_ok=True)
    for model, fname in [(lstm_model, "lstm_model.h5"),
                         (gru_model,  "gru_model.h5"),
                         (cnn_lstm_model, "cnn_lstm_model.h5")]:
        model.save(os.path.join(MODELS_DIR, fname))
        print(f"[save]  {fname}  →  models/")
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler.h5"))
    print("[save]  target_scaler.h5  →  models/")

    print("\n" + "=" * 50)
    print("Deep learning training complete.")
    print("=" * 50)
    return lstm_model, gru_model, cnn_lstm_model, all_metrics


if __name__ == "__main__":
    run()
