#!/usr/bin/env python3
"""
etl/train_baseline.py

Train an Elastic-Net model on gold.features → gold.labels.
Saves the fitted model to models/elasticnet_baseline.pkl.
"""

import os
import sys
import duckdb
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ─── Ensure src/ is importable ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Configuration ───────────────────────────────────────────────────────────────
DB_PATH       = os.path.join("data", "punta.duckdb")
FEATURE_TABLE = "gold.features"
LABEL_TABLE   = "gold.labels"
MODEL_PATH    = os.path.join("models", "elasticnet_baseline.pkl")
N_SPLITS      = 5  # for time-series CV folds

# ─── Connect to DuckDB ───────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH)

# ─── Load features into pandas ──────────────────────────────────────────────────
df_feat = con.execute(f"SELECT * FROM {FEATURE_TABLE}").fetchdf()

# Sanity-check we have at least 'date' and numeric cols
assert "date" in df_feat.columns, "Expected 'date' column in gold.features"

# Drop only the 'date' column for X (we don't need ticker here)
X = df_feat.drop(columns=["date"])


# ─── Load labels into pandas ─────────────────────────────────────────────────────
df_lab = con.execute(f"SELECT excess_return_12m FROM {LABEL_TABLE}").fetchdf()
y = df_lab["excess_return_12m"]

# Align indexes and drop any rows with NaNs
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
mask = ~(X.isna().any(axis=1) | y.isna())
X, y = X[mask], y[mask]

print(f"🔍 Training on {len(X)} rows with {X.shape[1]} features")

# ─── TimeSeries cross-validation ─────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# ─── Elastic-Net with built-in CV ────────────────────────────────────────────────
model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=[0.01, 0.1, 1.0],
    cv=tscv,
    max_iter=10000,
    n_jobs=-1,
    random_state=42
)
model.fit(X, y)

# ─── Quick evaluation on last fold ───────────────────────────────────────────────
train_idx, test_idx = list(tscv.split(X))[-1]
y_pred = model.predict(X.iloc[test_idx])
mse = mean_squared_error(y.iloc[test_idx], y_pred)
r2  = r2_score(y.iloc[test_idx], y_pred)
print(f"✅ ElasticNetCV done · Best alpha={model.alpha_:.4f}, l1_ratio={model.l1_ratio_:.4f}")
print(f"   Test MSE: {mse:.6f}, R²: {r2:.4f}")

# ─── Save the model artifact ─────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
