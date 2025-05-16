#!/usr/bin/env python3
"""
etl/ingest_raw.py

Fetches daily OHLCV prices via yfinance (unadjusted) and loads them into
DuckDB under schema `raw.prices`, dropping any prior table to avoid schema mismatches.
"""

import duckdb
import yfinance as yf
import os

# === CONFIGURATION ===
DB_PATH    = os.path.join("data", "punta.duckdb")
TICKERS    = ["AAPL", "MSFT", "SPY"]   # initial test universe
START_DATE = "2023-01-01"             # adjust as needed

# === CONNECT TO DUCKDB ===
con = duckdb.connect(DB_PATH)
con.execute("CREATE SCHEMA IF NOT EXISTS raw")

# Drop old table (so we always match schema)
con.execute("DROP TABLE IF EXISTS raw.prices")

# Create table with exactly 8 columns
con.execute("""
    CREATE TABLE raw.prices (
        date       DATE,
        open       DOUBLE,
        high       DOUBLE,
        low        DOUBLE,
        close      DOUBLE,
        adj_close  DOUBLE,
        volume     BIGINT,
        ticker     VARCHAR
    )
""")

# === FETCH & LOAD ===
for ticker in TICKERS:
    print(f"📥 Fetching {ticker}…")
    # force unadjusted so we get the "Adj Close" column
    df = yf.download(ticker,
                     start=START_DATE,
                     progress=False,
                     auto_adjust=False)
    df = (
        df
        .reset_index()
        .rename(columns={
            "Date":      "date",
            "Open":      "open",
            "High":      "high",
            "Low":       "low",
            "Close":     "close",
            "Adj Close": "adj_close",
            "Volume":    "volume"
        })
    )
    df["ticker"] = ticker

    # Insert rows (duckdb will match by column order)
    con.register("tmp", df)
    con.execute("INSERT INTO raw.prices SELECT * FROM tmp")
    con.unregister("tmp")
    print(f"✅ Loaded {len(df)} rows for {ticker}")

print("🎉 RAW ingestion complete.")
