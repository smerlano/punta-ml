#!/usr/bin/env python3
"""
etl/shap_baseline.py

Compute SHAP values for the Elastic-Net baseline on the last CV fold,
and output a summary of mean absolute SHAP values per feature.
"""

import os
import sys
import duckdb
import pandas as pd
import joblib
import shap
import numpy as np

# ─── Ensure we can import from src/ ───────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Config ───────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join("data", "punta.duckdb")
FEATURE_TABLE = "gold.features"
MODEL_PATH    = os.path.join("models", "elasticnet_baseline.pkl")
OUTPUT_CSV    = os.path.join("models", "shap_baseline_summary.csv")
N_SPLITS      = 5

# ─── Load model & data ───────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
con   = duckdb.connect(DB_PATH)
df   = con.execute(f"SELECT * FROM {FEATURE_TABLE}").fetchdf()

# Prepare X: drop non-features
X = df.drop(columns=["date", "ticker"], errors="ignore")
X = X.reset_index(drop=True)

# ─── TimeSeriesSplit to get last fold ────────────────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
splits = list(tscv.split(X))
_, test_idx = splits[-1]

X_test = X.iloc[test_idx]

# ─── Compute SHAP values ─────────────────────────────────────────────────────────
import numpy as np  # add at top with other imports

# old line:
# explainer = shap.LinearExplainer(model, X_test, feature_dependence="independent")
# new line:
explainer = shap.LinearExplainer(
    model, 
    X_test, 
   feature_perturbation="interventional"
)
shap_vals = explainer.shap_values(X_test)

# ─── Summarize mean absolute SHAP per feature ────────────────────────────────────
shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)


# Save summary
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
shap_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ SHAP summary saved to {OUTPUT_CSV}")
print(shap_df)
