"""model.py - Model architectures, sequence creation, evaluation, and callbacks."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, GRU, Conv1D, MaxPooling1D,
                                     Dense, Dropout, BatchNormalization,
                                     LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


#  Baseline models 

def get_lr_model():
    return Ridge()

def get_lr_param_grid():
    return {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}

def get_rf_model():
    return RandomForestRegressor(random_state=42, n_jobs=-1)

def get_rf_param_grid():
    return {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2"],
    }



#  Sequences & callbacks 

SEQ_LEN = 24  # 24 steps × 10 min = 4 hours of history


def create_sequences(X, y, seq_len=SEQ_LEN):
    """Reshape flat arrays into overlapping 3-D sequences ending at the prediction row."""
    Xs, ys = [], []
    for i in range(seq_len - 1, len(X)):
        Xs.append(X[i - seq_len + 1 : i + 1])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def get_callbacks(patience=10):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
    ]


#  Deep learning models

def build_lstm(input_shape, units=128, dropout=0.2, lr=0.001):
    """Two-layer LSTM with LayerNormalization, recurrent dropout, and gradient clipping."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True, recurrent_dropout=0.1),
        LayerNormalization(),
        Dropout(dropout),
        LSTM(units // 2, recurrent_dropout=0.1),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="LSTM")
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape, units=64, dropout=0.2, lr=0.001):
    """Two-layer GRU with BatchNormalization and gradient clipping."""
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="GRU")
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape, filters=64, units=64, dropout=0.2, lr=0.001):
    """CNN-LSTM hybrid: Conv1D extracts local patterns, LSTM captures long-range dependencies."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="CNN_LSTM")
    model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
    return model
