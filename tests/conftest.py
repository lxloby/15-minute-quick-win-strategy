import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_m15(ohlc, start="2024-01-02 12:00", spread_price=0.0):
    """Build an M15 frame from a list of (open, high, low, close) tuples."""
    arr = np.asarray(ohlc, dtype=float)
    idx = pd.date_range(start, periods=len(arr), freq="15min", tz="UTC")
    return pd.DataFrame({
        "time_utc": idx.tz_localize(None),
        "open": arr[:, 0], "high": arr[:, 1],
        "low": arr[:, 2], "close": arr[:, 3],
        "tick_volume": 100.0,
        "spread": 0.0,
        "_spread_price": spread_price,
    })


def random_ohlc(n=2000, seed=0, start=100.0, vol=0.3):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, vol, n))
    open_ = close + rng.normal(0, vol / 3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, vol / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol / 2, n))
    return open_, high, low, close


@pytest.fixture
def rand_series():
    return random_ohlc()
