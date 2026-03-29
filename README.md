# Appliance Energy Prediction - Deep Learning Internship Project

Multivariate time-series forecasting of household appliance energy consumption using the [Energy Prediction Dataset](https://drive.google.com/file/d/1ZkR70gAxSUoU5b1KqvTW4bdFM0il-nZz/view?usp=sharing).


## Project Overview

The pipeline trains and compares five models Ridge Regression, Random Forest, LSTM, GRU, and CNN-LSTM on 10-minute interval energy data recorded across a Belgian residence from January to May 2016.

**Best result:** GRU - MAE ≈ 13 Wh, R² ≈ 0.71

## Project Structure

```
My_Intern_Project/
├── data/
│   ├── raw/                     # Raw dataset (energy_data_set.csv)
│   └── processed/               # Cleaned and engineered CSVs
├── models/                      # Saved model files (.h5, .pkl)
├── notebooks/
│   ├── EDA.ipynb                # Exploratory data analysis
│   └── Run_pipeline.ipynb       # End-to-end pipeline runner
├── src/
│   ├── data_preprocessing.py    # Step 1 - Load, clean, split
│   ├── feature_engineering.py   # Step 2 - Feature creation and scaling
│   ├── model.py                 # Model architectures and callbacks
│   ├── evaluation.py            # Shared evaluation function for all models
│   ├── train.py                 # Step 3 - Baseline model training (LR, RF)
│   └── train_dl.py              # Step 4 - DL model training (LSTM, GRU, CNN-LSTM)

├── Plots and Visualizations/    # All generated plots
├── reports/
│   └── report.pdf               # PDF report
└── requirements.txt             # List all dependencies and their versions.
```


## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv tf_env
# Windows
tf_env\Scripts\activate
# macOS / Linux
source tf_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.20 requires Python 3.9–3.12. It is not compatible with Python 3.13+.

### 3. Place the dataset

Put `energy_data_set.csv` in `data/raw/`:

```
data/raw/energy_data_set.csv
```

## How to Run

### Option A - Jupyter Notebooks (recommended)

**Start with EDA** - open `notebooks/EDA.ipynb` to explore the dataset, distributions, correlations, and temporal patterns before running the pipeline.

**Then run the full pipeline** - open `notebooks/Run_pipeline.ipynb` and run all cells in order:

```
Step 1 → Data Preprocessing
Step 2 → Feature Engineering
Step 3 → Baseline Training (LR + RF)
Step 4 → DL Architecture Review
Step 5 → DL Training (LSTM, GRU, CNN-LSTM)
Step 6 → All Models Comparison DataFrame
```

### Option B - Run each script individually

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train.py
python src/train_dl.py
```

## Models

| Model             | Type        | Key design                                  |
|-------------------|-------------|---------------------------------------------|
| Ridge Regression  | Linear      | L2 regularisation, GridSearchCV tuning      |
| Random Forest     | Ensemble    | 300 trees, RandomizedSearchCV tuning        |
| LSTM              | Deep Learning | 2-layer + BatchNorm + gradient clipping   |
| GRU               | Deep Learning | 2-layer + BatchNorm + gradient clipping   |
| CNN-LSTM          | Deep Learning | Conv1D feature extractor + LSTM           |

All DL models use:
- Sequence length: 24 steps (4 hours of history)
- Optimizer: Adam with `clipnorm=1.0`
- Callbacks: EarlyStopping (patience=15) + ReduceLROnPlateau
- Random seed: 42 (fully reproducible)

## Results

| Model            | MAE (Wh) | RMSE (Wh) | MAPE (%) | R²   |
|------------------|----------|-----------|----------|------|
| Ridge Regression | ~13.6    | ~22.0     | ~17.0    | 0.68 |
| Random Forest    | ~13.0    | ~20.6     | ~16.0    | 0.71 |
| LSTM             | ~13.5    | ~21.2     | ~16.8    | 0.70 |
| **GRU**          | **~13.0**| **~20.6** | **~16.0**| **0.71** |
| CNN-LSTM         | ~14.1    | ~21.4     | ~17.0    | 0.69 |

## Dataset

- **Source:** Energy Prediction Dataset
- **Records:** 19,735 rows at 10-minute intervals
- **Period:** January-May 2016
- **Target:** `Appliances` - energy consumption in Wh
- **Features:** 26 environmental sensors (temperature, humidity, lighting, weather)
