#!/usr/bin/env python3
"""
etl/train_lightgbm.py

Train a LightGBM model on gold.features → gold.labels using Optuna for hyperparameter tuning.
Saves the best model to models/lightgbm_optuna.pkl.
"""

import os
import sys
import duckdb
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Config ───────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join("data", "punta.duckdb")
FEATURE_TABLE = "gold.features"
LABEL_TABLE   = "gold.labels"
MODEL_PATH    = os.path.join("models", "lightgbm_optuna.pkl")
N_SPLITS      = 5
N_TRIALS      = 50  # adjust as needed

# ─── Load data ───────────────────────────────────────────────────────────────────
con     = duckdb.connect(DB_PATH)
df_feat = con.execute(f"SELECT * FROM {FEATURE_TABLE}").fetchdf()
X       = df_feat.drop(columns=["date"], errors="ignore")
y       = con.execute(f"SELECT excess_return_12m FROM {LABEL_TABLE}").fetchdf()["excess_return_12m"]

# reset indexes & drop null rows
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
mask = ~(X.isna().any(axis=1) | y.isna())
X, y = X[mask], y[mask]

# ─── TimeSeries cross-validation splits ──────────────────────────────────────────
tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
splits = list(tscv.split(X))

def objective(trial):
    # define hyperparameter search space
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }

    rmses = []
    for train_idx, test_idx in splits:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_te, label=y_te, reference=train_data)

        bst = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ]
        )

        y_pred = bst.predict(X_te)
        # compute RMSE since mean_squared_error no longer accepts squared=False
        mse = mean_squared_error(y_te, y_pred)
        rmse = mse ** 0.5
        rmses.append(rmse)


    # average RMSE across folds
    return sum(rmses) / len(rmses)

# ─── Run the Optuna study ─────────────────────────────────────────────────────────
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

print("✅ Best trial parameters:", study.best_params)
print("   Best RMSE:", study.best_value)

# ─── Train final model on full data ──────────────────────────────────────────────
final_data  = lgb.Dataset(X, label=y)
final_model = lgb.train(study.best_params, final_data, num_boost_round=study.best_trial.number * 10)

# ─── Save the model ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
print(f"💾 LightGBM model saved to {MODEL_PATH}")
