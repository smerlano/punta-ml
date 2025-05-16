import duckdb
import pandas as pd
import pytest

# === CONFIG ===
DB_PATH = "data/punta.duckdb"
MIN_ROWS = 500
EXPECTED_COLUMNS = [
    "date", "open", "high", "low",
    "close", "adj_close", "volume", "ticker"
]

def test_raw_prices_quality():
    # 1) Load data via DuckDB into a pandas DataFrame
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM raw.prices").df()

    # 2a) Row count >= MIN_ROWS
    assert len(df) >= MIN_ROWS, f"Only {len(df)} rows; expected ≥ {MIN_ROWS}"

    # 2b) No nulls in 'ticker'
    nulls = df['ticker'].isna().sum()
    assert nulls == 0, f"Found {nulls} null(s) in 'ticker'"

    # 2c) Exact columns
    cols = list(df.columns)
    assert cols == EXPECTED_COLUMNS, f"Columns {cols} ≠ expected {EXPECTED_COLUMNS}"
