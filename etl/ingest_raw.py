#!/usr/bin/env python3
"""
etl/ingest_raw.py

Backfill 2013–2024 raw price data for your 20‐ticker universe.
"""

import os
import duckdb
import yfinance as yf

# === CONFIGURATION ===
DB_PATH    = os.path.join("data", "punta.duckdb")
TICKERS = [
    "SPY","NVDA","AMD","TSLA","AVGO","NFLX","NOW","PGR","LLY","ISRG","AMZN",
    "MSFT","1211.HK","META","RMS","600519.SS","ASML","TMUS","COST",
    "2330.TW","RELIANCE.NS"
]
START_DATE = "2013-01-01"
END_DATE   = "2024-10-31"

# === CONNECT TO DUCKDB ===
con = duckdb.connect(DB_PATH)
con.execute("CREATE SCHEMA IF NOT EXISTS raw")

# Drop and recreate raw.prices to get a clean 2013–2024 window
con.execute("DROP TABLE IF EXISTS raw.prices")
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
    print(f"📥 Fetching {ticker} from {START_DATE} to {END_DATE}…")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=False
    )
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

    con.register("tmp", df)
    con.execute("INSERT INTO raw.prices SELECT * FROM tmp")
    con.unregister("tmp")
    print(f"✅ Loaded {len(df)} rows for {ticker}")

print("🎉 FULL RAW backfill complete.")
