#!/usr/bin/env python3
"""
etl/bronze_transform.py

Clean raw.prices into bronze.prices:
 - enforce types
 - remove duplicates
 - drop rows with volume<=0 or price<=0
"""

import duckdb
import os

DB_PATH = os.path.join("data", "punta.duckdb")

# Connect & ensure schema
con = duckdb.connect(DB_PATH)
con.execute("CREATE SCHEMA IF NOT EXISTS bronze")

# Count raw rows
raw_count = con.execute("SELECT COUNT(*) FROM raw.prices").fetchone()[0]
print(f"🔍 raw.prices row count: {raw_count}")

# Drop old bronze.prices
con.execute("DROP TABLE IF EXISTS bronze.prices")

# Create bronze.prices with cleaning
con.execute("""
CREATE TABLE bronze.prices AS
SELECT
  CAST(date       AS DATE)    AS date,
  CAST(open       AS DOUBLE)  AS open,
  CAST(high       AS DOUBLE)  AS high,
  CAST(low        AS DOUBLE)  AS low,
  CAST(close      AS DOUBLE)  AS close,
  CAST(adj_close  AS DOUBLE)  AS adj_close,
  CAST(volume     AS BIGINT)  AS volume,
  CAST(ticker     AS VARCHAR) AS ticker
FROM raw.prices
WHERE volume > 0
  AND close  > 0
  AND open   > 0
  AND high   > 0
  AND low    > 0
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY ticker, date
    ORDER BY date
) = 1
""")

# Count bronze rows
bronze_count = con.execute("SELECT COUNT(*) FROM bronze.prices").fetchone()[0]
print(f"✅ bronze.prices row count: {bronze_count}")
