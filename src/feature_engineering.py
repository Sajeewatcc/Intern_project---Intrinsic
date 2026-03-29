"""feature_engineering.py - Step 2: Cyclical encoding, lags, rolling, interactions, MinMaxScaling."""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
TARGET     = "Appliances"


def engineer(df, split_name=""):
    """Apply all feature engineering steps to one split independently."""
    df = df.copy()
    # [A] Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["NSM_sin"]  = np.sin(2 * np.pi * df["NSM"]  / 86400)
    df["NSM_cos"]  = np.cos(2 * np.pi * df["NSM"]  / 86400)
    df.drop(columns=["hour", "NSM"], inplace=True)
    # [B] Lag features
    df["lag_1"]   = df[TARGET].shift(1)
    df["lag_3"]   = df[TARGET].shift(3)
    df["lag_6"]   = df[TARGET].shift(6)
    df["lag_144"] = df[TARGET].shift(144)
    # [C] Rolling features (shift first to prevent leakage)
    past = df[TARGET].shift(1)
    df["roll_mean_6"]  = past.rolling(6).mean()
    df["roll_std_6"]   = past.rolling(6).std()
    df["roll_mean_18"] = past.rolling(18).mean()
    df["roll_std_18"]  = past.rolling(18).std()
    # [D] Interaction features
    df["T_out_x_RH_out"] = df["T_out"] * df["RH_out"]
    df["T1_x_RH_1"]      = df["T1"]    * df["RH_1"]
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[engineer:{split_name}]  {before} → {len(df)} rows  (dropped {before - len(df)} NaN rows from lags)")
    return df


def run():
    print("=" * 50)
    print("STEP 2 — Feature Engineering")
    print("=" * 50)
    train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"), parse_dates=["date"])
    test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"),  parse_dates=["date"])
    print(f"[load]  train: {train.shape}  |  test: {test.shape}")

    train = engineer(train, "train")
    test  = engineer(test,  "test")

    drop_cols    = ["date", TARGET]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    cyclical   = [c for c in feature_cols if c.endswith(("_sin", "_cos"))]
    lag_cols   = [c for c in feature_cols if c.startswith("lag_")]
    roll_cols  = [c for c in feature_cols if c.startswith("roll_")]
    inter_cols = [c for c in feature_cols if "_x_" in c]
    other_cols = [c for c in feature_cols if c not in cyclical + lag_cols + roll_cols + inter_cols]
    print(f"\n[scale]  Feature groups:")
    print(f"         [A] Cyclical      ({len(cyclical):2d}) : {cyclical}")
    print(f"         [B] Lag           ({len(lag_cols):2d}) : {lag_cols}")
    print(f"         [C] Rolling       ({len(roll_cols):2d}) : {roll_cols}")
    print(f"         [D] Interaction   ({len(inter_cols):2d}) : {inter_cols}")
    print(f"         [original sensor] ({len(other_cols):2d}) : {len(other_cols)} columns")
    print(f"         Total features    : {len(feature_cols)}")

    os.makedirs(PROC_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROC_DIR, "train_engineered.csv"), index=False)
    test.to_csv( os.path.join(PROC_DIR, "test_engineered.csv"),  index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "minmax_scaler.pkl"))
    print(f"\n[save]  train_engineered.csv  →  {train.shape}")
    print(f"[save]  test_engineered.csv   →  {test.shape}")
    print(f"[save]  minmax_scaler.pkl     →  models/")
    print("=" * 50)
    print("Feature engineering complete.")
    print("=" * 50)
    return train, test, scaler, feature_cols


if __name__ == "__main__":
    run()
