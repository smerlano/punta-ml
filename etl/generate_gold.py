#!/usr/bin/env python3
"""
etl/generate_gold.py

Reads silver.prices, computes features & labels, writes to gold.features and gold.labels.
"""

import sys, os
# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import duckdb
import pandas as pd
from src.features import momentum_12m, compute_excess_return

# ─── Config ──────────────────────────────────────────────────────────────────────
DB_PATH        = os.path.join("data", "punta.duckdb")
GOLD_FEATURES  = "gold.features"
GOLD_LABELS    = "gold.labels"
BENCHMARK_TKR  = "SPY"
RETURN_DAYS    = 252  # trading days ≈ 1 year
HIT_THRESHOLD  = 0.02  # 2% excess return

# ─── Load silver data ─────────────────────────────────────────────────────────────
con = duckdb.connect(DB_PATH)
con.execute("CREATE SCHEMA IF NOT EXISTS gold")
df = con.execute("SELECT * FROM silver.prices ORDER BY ticker, date").fetchdf()

# ─── PRECOMPUTE SPY benchmark series ──────────────────────────────────────────────
spy_df = df[df["ticker"] == BENCHMARK_TKR].copy()
spy_df.set_index("date", inplace=True)
# stock forward return for SPY
spy_df["stock_fwd_ret"] = spy_df["adj_close"].pct_change(periods=RETURN_DAYS).shift(-RETURN_DAYS)
# keep only the benchmark return column and rename
spy_df = spy_df[["stock_fwd_ret"]].rename(columns={"stock_fwd_ret": "spy_fwd_ret"})

# ─── Compute Features & Labels ────────────────────────────────────────────────────
features_list = []
labels_list   = []

for tkr, group in df.groupby("ticker"):
    if tkr == BENCHMARK_TKR:
        continue  # skip SPY itself

    grp = group.copy()
    grp.set_index("date", inplace=True)

    # 1) momentum feature
    grp["momentum_12m"] = momentum_12m(grp)

    # 2) stock 12m forward return
    grp["stock_fwd_ret"] = grp["adj_close"].pct_change(periods=RETURN_DAYS).shift(-RETURN_DAYS)

    # 3) merge in SPY forward return
    grp = grp.join(spy_df, how="left")

    # 4) compute excess return label
    grp["excess_return_12m"] = compute_excess_return(grp["stock_fwd_ret"], grp["spy_fwd_ret"])
    grp["hit_2pct"] = (grp["excess_return_12m"] >= HIT_THRESHOLD).astype(int)

    # 5) collect feature columns
    features_list.append(grp[[
        "open", "high", "low", "close",
        "adj_close", "volume", "momentum_12m"
    ]])

    # 6) collect label columns
    labels_list.append(grp[[
        # keep ticker & date as columns
        "ticker",
        "stock_fwd_ret", "spy_fwd_ret",
        "excess_return_12m", "hit_2pct"
    ]])

# ─── Concatenate & reset index ───────────────────────────────────────────────────
features_df = pd.concat(features_list).reset_index()
labels_df   = pd.concat(labels_list).reset_index()

# ─── Write to DuckDB ─────────────────────────────────────────────────────────────
# Features
con.execute(f"DROP TABLE IF EXISTS {GOLD_FEATURES}")
con.register("f_df", features_df)
con.execute(f"CREATE TABLE {GOLD_FEATURES} AS SELECT * FROM f_df")
con.unregister("f_df")

# Labels
con.execute(f"DROP TABLE IF EXISTS {GOLD_LABELS}")
con.register("l_df", labels_df)
con.execute(f"CREATE TABLE {GOLD_LABELS} AS SELECT * FROM l_df")
con.unregister("l_df")

print(f"✅ Written {len(features_df)} rows to {GOLD_FEATURES}")
print(f"✅ Written {len(labels_df)} rows to {GOLD_LABELS}")
