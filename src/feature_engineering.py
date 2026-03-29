"""
feature_engineering.py
-----------------------
Step 2 of the pipeline.

Loads the cleaned train/test splits from data/processed/,
performs all feature engineering **separately on each split**
to prevent data leakage, then applies MinMaxScaling
(fit on train only) and saves the final feature matrices.

Feature groups:
    [A] Cyclical encoding  — sin/cos pairs for hour and NSM (both time-of-day).
                             day_of_week, month, is_weekend removed —
                             low predictive value for energy consumption.
    [B] Lag features       — past Appliances values (10 min, 30 min, 1 hr, 24 hr)
    [C] Rolling windows    — 1-hr and 3-hr rolling mean & std
    [D] Interaction        — temperature × humidity cross terms
    [E] MinMaxScaler       — fit on X_train only, transform X_train & X_test
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

TARGET = "Appliances"


# ── Load ─────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned train and test CSVs produced by data_preprocessing.py."""
    train = pd.read_csv(os.path.join(PROC_DIR, "train.csv"), parse_dates=["date"])
    test  = pd.read_csv(os.path.join(PROC_DIR, "test.csv"),  parse_dates=["date"])
    print(f"[load]  train: {train.shape}  |  test: {test.shape}")
    return train, test


# ── [A] Cyclical Encoding ────────────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cyclical time features as sin/cos pairs.

    Kept features (both cyclical, both time-of-day):
        hour : period = 24     (0–23)
        NSM  : period = 86,400 (seconds since midnight, finer resolution than hour)

    Removed: day_of_week, month, is_weekend — low predictive value for energy.
    Raw columns are dropped after encoding.
    """
    # hour (0–23)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # NSM — seconds since midnight (0–86,400)
    df["NSM_sin"]  = np.sin(2 * np.pi * df["NSM"] / 86400)
    df["NSM_cos"]  = np.cos(2 * np.pi * df["NSM"] / 86400)

    # Drop raw columns — replaced by sin/cos pairs
    df.drop(columns=["hour", "NSM"], inplace=True)

    return df


# ── [B] Lag Features ─────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lagged Appliances values at key autocorrelation intervals:
        lag_1   → 10 min ago
        lag_3   → 30 min ago
        lag_6   → 1 hour ago
        lag_144 → 24 hours ago  (strong daily periodicity)
    """
    df["lag_1"]   = df[TARGET].shift(1)
    df["lag_3"]   = df[TARGET].shift(3)
    df["lag_6"]   = df[TARGET].shift(6)
    df["lag_144"] = df[TARGET].shift(144)
    return df


# ── [C] Rolling Window Features ──────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std over 1-hr (6 steps) and 3-hr (18 steps).
    shift(1) ensures only past values enter the window — no leakage.
    """
    past = df[TARGET].shift(1)
    df["roll_mean_6"]  = past.rolling(6).mean()
    df["roll_std_6"]   = past.rolling(6).std()
    df["roll_mean_18"] = past.rolling(18).mean()
    df["roll_std_18"]  = past.rolling(18).std()
    return df


# ── [D] Interaction Features ─────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross terms between correlated feature pairs.
    Captures combined thermal-humidity effect on energy consumption.
    """
    df["T_out_x_RH_out"] = df["T_out"] * df["RH_out"]
    df["T1_x_RH_1"]      = df["T1"]    * df["RH_1"]
    return df


# ── Engineer (applied separately to each split) ──────────────────────────────

def engineer(df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
    """Apply all feature engineering steps to one split independently."""
    df = df.copy()
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[engineer:{split_name}]  {before} → {len(df)} rows  (dropped {before - len(df)} NaN rows from lags)")
    return df


# ── [E] MinMax Scaling ───────────────────────────────────────────────────────

def scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, list[str]]:
    """
    Fit MinMaxScaler on X_train only, then transform both splits.
    date and target are excluded from scaling.
    """
    drop_cols    = ["date", TARGET]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols]  = scaler.transform(test[feature_cols])

    # Feature group summary
    cyclical_cols    = [c for c in feature_cols if c.endswith(("_sin", "_cos"))]
    lag_cols         = [c for c in feature_cols if c.startswith("lag_")]
    roll_cols        = [c for c in feature_cols if c.startswith("roll_")]
    interaction_cols = [c for c in feature_cols if "_x_" in c]
    other_cols       = [c for c in feature_cols
                        if c not in cyclical_cols + lag_cols + roll_cols + interaction_cols]

    print(f"\n[scale]  Feature groups:")
    print(f"         [A] Cyclical      ({len(cyclical_cols):2d}) : {cyclical_cols}")
    print(f"         [B] Lag           ({len(lag_cols):2d}) : {lag_cols}")
    print(f"         [C] Rolling       ({len(roll_cols):2d}) : {roll_cols}")
    print(f"         [D] Interaction   ({len(interaction_cols):2d}) : {interaction_cols}")
    print(f"         [original sensor] ({len(other_cols):2d}) : {len(other_cols)} columns")
    print(f"         Total features    : {len(feature_cols)}")

    return train, test, scaler, feature_cols


# ── Save ─────────────────────────────────────────────────────────────────────

def save(train: pd.DataFrame, test: pd.DataFrame, scaler: MinMaxScaler) -> None:
    """Save engineered splits and the fitted scaler."""
    os.makedirs(PROC_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train.to_csv(os.path.join(PROC_DIR, "train_engineered.csv"), index=False)
    test.to_csv( os.path.join(PROC_DIR, "test_engineered.csv"),  index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "minmax_scaler.pkl"))

    print(f"\n[save]  train_engineered.csv  →  {train.shape}")
    print(f"[save]  test_engineered.csv   →  {test.shape}")
    print(f"[save]  minmax_scaler.pkl     →  models/")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run():
    """Execute the full feature engineering pipeline."""
    print("=" * 50)
    print("STEP 2 — Feature Engineering")
    print("=" * 50)

    train, test = load_splits()

    # Applied separately on each split — no leakage
    train = engineer(train, split_name="train")
    test  = engineer(test,  split_name="test")

    train, test, scaler, feature_cols = scale(train, test)
    save(train, test, scaler)

    print("=" * 50)
    print("Feature engineering complete.")
    print("=" * 50)
    return train, test, scaler, feature_cols


if __name__ == "__main__":
    run()
