#!/usr/bin/env python3
"""
etl/silver_transform.py

Transforms bronze.prices → silver.prices by:
 1. Recording point-in-time validity.
 2. Winsorising price columns to ±3σ.
"""

import duckdb
import pandas as pd
import os

# ─── Configuration ─────────────────────────────────────────────────────────────
DB_PATH   = os.path.join("data", "punta.duckdb")
SRC_TABLE = "bronze.prices"
DST_TABLE = "silver.prices"
SIGMA     = 3  # winsorisation threshold

# ─── Connect & load bronze data ────────────────────────────────────────────────
con = duckdb.connect(DB_PATH)
# Ensure the silver schema exists
con.execute("CREATE SCHEMA IF NOT EXISTS silver")

df = con.execute(f"SELECT * FROM {SRC_TABLE}").fetchdf()
print(f"🔍 Loaded {len(df)} rows from {SRC_TABLE}")

# ─── Stamp point-in-time validity ──────────────────────────────────────────────
# Add a column showing that this row was valid exactly at 'date'
df["valid_from"] = df["date"]
df["valid_to"]   = pd.NaT  # open-ended; training code will fill valid_to = next row

# ─── Winsorise price columns ──────────────────────────────────────────────────
num_cols = ["open", "high", "low", "close", "adj_close"]
for col in num_cols:
    mu = df[col].mean()
    sigma = df[col].std()
    lower, upper = mu - SIGMA * sigma, mu + SIGMA * sigma
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"• {col}: clipped to [{lower:.2f}, {upper:.2f}]")

# ─── Write out silver.prices ──────────────────────────────────────────────────
# Drop old table
con.execute(f"DROP TABLE IF EXISTS {DST_TABLE}")
# Write from DataFrame
con.register("tmp", df)
con.execute(f"CREATE TABLE {DST_TABLE} AS SELECT * FROM tmp")
con.unregister("tmp")
print(f"✅ Wrote {len(df)} rows to {DST_TABLE}")
