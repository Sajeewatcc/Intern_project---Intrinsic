"""data_preprocessing.py - Step 1: Load → clean → outlier cap → time features → split → save."""

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "energy_data_set.csv")
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.drop(columns=["rv1", "rv2"], inplace=True)
    print(f"[load]    shape={df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def handle_missing(df):
    n_missing = df.isnull().sum().sum()
    print(f"[missing] missing cells before: {n_missing}")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    print(f"[missing] missing cells after : {df.isnull().sum().sum()}")
    return df


def cap_outliers(df, target="Appliances"):
    Q1, Q3  = df[target].quantile(0.25), df[target].quantile(0.75)
    IQR     = Q3 - Q1
    upper_iqr, lower_iqr = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR
    n_iqr    = ((df[target] < lower_iqr) | (df[target] > upper_iqr)).sum()
    n_zscore = (np.abs(zscore(df[target])) > 3).sum()
    print(f"[outlier] IQR bounds : [{lower_iqr:.1f}, {upper_iqr:.1f}] Wh")
    print(f"[outlier] IQR outliers   : {n_iqr}  ({n_iqr/len(df)*100:.1f}%)")
    print(f"[outlier] Z-score outliers (|z|>3): {n_zscore}  ({n_zscore/len(df)*100:.1f}%)")
    df[target] = df[target].clip(upper=upper_iqr)
    print(f"[outlier] capped at {upper_iqr:.1f} Wh  |  new max={df[target].max():.1f} Wh")
    return df


def add_time_features(df):
    df["hour"] = df["date"].dt.hour
    df["NSM"]  = df["hour"] * 3600 + df["date"].dt.minute * 60
    print(f"[features] added: hour, NSM")
    return df


def split_data(df, train_ratio=0.80):
    idx   = int(len(df) * train_ratio)
    train = df.iloc[:idx].reset_index(drop=True)
    test  = df.iloc[idx:].reset_index(drop=True)
    print(f"[split]   train: {len(train)} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"[split]   test : {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, test


def run():
    print("=" * 50)
    print("STEP 1 - Data Preprocessing")
    print("=" * 50)
    os.makedirs(PROC_DIR, exist_ok=True)
    df = load_data(RAW_PATH)
    df = handle_missing(df)
    df = cap_outliers(df)
    df = add_time_features(df)
    train, test = split_data(df)
    train_path = os.path.join(PROC_DIR, "train.csv")
    test_path  = os.path.join(PROC_DIR, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path,  index=False)
    print(f"[save]    train → {train_path}")
    print(f"[save]    test  → {test_path}")
    print("=" * 50)
    print("Preprocessing complete.")
    print("=" * 50)
    return train, test


if __name__ == "__main__":
    run()
