# src/features.py

import pandas as pd

def momentum_12m(df: pd.DataFrame) -> pd.Series:
    """
    12-month momentum: (adj_close today / adj_close 252 trading days ago) – 1.
    Assumes df is sorted by date ascending with a DatetimeIndex.
    """
    return df["adj_close"].pct_change(periods=252)

# Stub for label computation (we’ll do ETL in generate_gold.py)
def compute_excess_return(stock_ret: float, spy_ret: float) -> float:
    """
    Excess return = stock_return − SPY_return over same horizon.
    """
    return stock_ret - spy_ret
