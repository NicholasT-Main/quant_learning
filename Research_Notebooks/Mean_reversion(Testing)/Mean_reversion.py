import pandas as pd

def mean_reversion_strategy(data: pd.DataFrame, lookback: int = 20):
    """
    Very basic mean reversion strategy with safe lookback handling.
    Works even if yfinance returns multi-indexed columns.
    """
    # Handle case where 'Close' is multi-indexed (e.g. ("Close","NVDA"))
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]  # pick first ticker's close
    else:
        close = data["Close"]

    # Calculate moving average
    ma = close.rolling(window=lookback).mean()

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Only set signals where MA is valid
    valid = ma.notna()
    signals[valid & (close < ma)] = 1
    signals[valid & (close > ma)] = -1

    return signals


