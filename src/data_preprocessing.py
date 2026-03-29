"""
data_preprocessing.py
---------------------
Step 1 of the pipeline.

Steps:
    1. Load raw data, drop noise columns
    2. Handle missing values (interpolation safeguard)
    3. Detect & cap outliers on target (IQR Winsorization)
    4. Derive time-based features
    5. Train / test split (80 / 20, temporal — no shuffle)
    6. Save train and test sets to data/processed/
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH  = os.path.join(ROOT_DIR, "data", "raw",       "energy_data_set.csv")
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV, parse date, sort chronologically, drop noise columns."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.drop(columns=["rv1", "rv2"], inplace=True)
    print(f"[load]    shape={df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing values and apply linear interpolation.
    Forward-fill / backward-fill handles any edge NaNs.
    No-op when dataset is already complete.
    """
    n_missing = df.isnull().sum().sum()
    print(f"[missing] missing cells before: {n_missing}")

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    print(f"[missing] missing cells after : {df.isnull().sum().sum()}")
    return df


def cap_outliers(df: pd.DataFrame, target: str = "Appliances") -> pd.DataFrame:
    """
    Detect outliers on the target column using IQR and Z-score,
    then cap (Winsorize) at the IQR upper bound.

    High-energy spikes are real events, not sensor errors,
    so rows are kept but clipped rather than removed.
    """
    Q1  = df[target].quantile(0.25)
    Q3  = df[target].quantile(0.75)
    IQR = Q3 - Q1
    upper_iqr = Q3 + 1.5 * IQR
    lower_iqr = Q1 - 1.5 * IQR

    n_iqr    = ((df[target] < lower_iqr) | (df[target] > upper_iqr)).sum()
    n_zscore = (np.abs(zscore(df[target])) > 3).sum()

    print(f"[outlier] IQR bounds : [{lower_iqr:.1f}, {upper_iqr:.1f}] Wh")
    print(f"[outlier] IQR outliers   : {n_iqr}  ({n_iqr/len(df)*100:.1f}%)")
    print(f"[outlier] Z-score outliers (|z|>3): {n_zscore}  ({n_zscore/len(df)*100:.1f}%)")

    df[target] = df[target].clip(upper=upper_iqr)
    print(f"[outlier] capped at {upper_iqr:.1f} Wh  |  new max={df[target].max():.1f} Wh")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-based columns from the date field."""
    df["hour"] = df["date"].dt.hour
    df["NSM"]  = df["hour"] * 3600 + df["date"].dt.minute * 60  # seconds since midnight
    print(f"[features] added: hour, NSM")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.80):
    """
    Temporal 80/20 split — no shuffling to prevent data leakage.
    First 80% of the time series → train.
    Last  20%                    → test.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test  = df.iloc[split_idx:].reset_index(drop=True)

    print(f"[split]   train: {len(train)} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"[split]   test : {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, test


def save(train: pd.DataFrame, test: pd.DataFrame, out_dir: str) -> None:
    """Save train and test DataFrames to data/processed/."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path  = os.path.join(out_dir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path,  index=False)
    print(f"[save]    train → {train_path}")
    print(f"[save]    test  → {test_path}")


def run():
    """Execute the full preprocessing pipeline."""
    print("=" * 50)
    print("STEP 1 — Data Preprocessing")
    print("=" * 50)

    df = load_data(RAW_PATH)
    df = handle_missing(df)
    df = cap_outliers(df)
    df = add_time_features(df)
    train, test = split_data(df)
    save(train, test, PROC_DIR)

    print("=" * 50)
    print("Preprocessing complete.")
    print("=" * 50)
    return train, test


if __name__ == "__main__":
    run()
