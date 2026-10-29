"""Simple moving average, matching MT5/TradingView `ta.sma`."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series | np.ndarray, length: int) -> np.ndarray:
    """
    Arithmetic mean of the last `length` values, NaN until `length` bars exist.

    Uses only bars up to and including t, so it is safe to evaluate on bar t.
    """
    s = pd.Series(np.asarray(series, dtype=float))
    return s.rolling(window=length, min_periods=length).mean().to_numpy()
