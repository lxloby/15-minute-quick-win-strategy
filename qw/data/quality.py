"""
Bad-tick detection.

A broker feed occasionally carries a print that never traded - a single bar
whose high or low is far outside everything around it, with the next bar
carrying on as if nothing happened. One real example in this data:

    2015-01-14 22:00  CHFJPY  open 115.156  high 139.043  low 113.820  close 115.078

Every neighbouring bar sits at ~115.0, and the 139.0 print is eleven hours
before the SNB actually unpegged the franc. The spike is not a market move.

This matters more than it looks. The backtest reads highs and lows to decide
whether a stop or target was hit, so one phantom print can close a trade that
in reality was never touched - and a 24-pip-wide bad tick on a 15-pip stop is
a guaranteed fake loss.

The test is deliberately conservative, because a genuine shock looks similar
on the bar it happens: a bar is suspect only when its range dwarfs the local
median AND the extreme does not persist into the neighbouring bars. The real
unpeg bar (2015-01-15 09:30) passes that second test - price stayed up there -
so it is correctly left alone.

Nothing here alters data by default. `find_suspect_bars` reports, and the
caller decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_suspect_bars(df: pd.DataFrame, window: int = 96,
                      range_mult: float = 8.0,
                      persist_tol: float = 0.25) -> pd.DataFrame:
    """
    Flag bars that look like bad prints rather than moves.

    window      bars either side used for the local median range
    range_mult  how many times the local median range counts as extreme
    persist_tol fraction of the spike that must survive into the next bar
                for the move to be considered real

    Returns one row per suspect bar with the evidence attached.
    """
    if len(df) < window * 2 + 2:
        return pd.DataFrame()

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    rng = high - low

    local = (pd.Series(rng).rolling(window * 2 + 1, center=True, min_periods=window)
                           .median().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(local > 0, rng / local, 0.0)

    extreme = ratio >= range_mult
    idx = np.flatnonzero(extreme)
    if idx.size == 0:
        return pd.DataFrame()

    rows = []
    for i in idx:
        if i == 0 or i >= len(df) - 1:
            continue
        prev_c, next_c = close[i - 1], close[i + 1]
        # How far did the bar reach beyond where it started and ended?
        up = high[i] - max(prev_c, next_c, close[i])
        dn = min(prev_c, next_c, close[i]) - low[i]
        spike = max(up, dn)
        if spike <= 0:
            continue
        # Did any of it stick? Compare the move the neighbours actually made.
        carried = abs(next_c - prev_c)
        persisted = carried / spike if spike > 0 else 1.0
        if persisted >= persist_tol:
            continue            # a real shock - price stayed there
        rows.append({
            "time_utc": df["time_utc"].iloc[i],
            "bar": int(i),
            "open": df["open"].iloc[i], "high": high[i],
            "low": low[i], "close": close[i],
            "prev_close": prev_c, "next_close": next_c,
            "range": rng[i], "local_median_range": local[i],
            "range_x_median": round(float(ratio[i]), 1),
            "spike": float(spike),
            "persisted_frac": round(float(persisted), 3),
        })
    return pd.DataFrame(rows)


def clip_suspect_bars(df: pd.DataFrame, suspect: pd.DataFrame) -> pd.DataFrame:
    """
    Pull a suspect bar's high/low back to the body of its neighbours.

    Used only when the caller explicitly asks for it. The bar is kept - only
    the phantom extreme is trimmed - so bar counts and indices never shift.
    """
    if suspect is None or suspect.empty:
        return df
    out = df.copy()
    hi = out["high"].to_numpy(float).copy()
    lo = out["low"].to_numpy(float).copy()
    op = out["open"].to_numpy(float)
    cl = out["close"].to_numpy(float)
    for i in suspect["bar"].to_numpy(int):
        if i <= 0 or i >= len(out) - 1:
            continue
        ceiling = max(op[i], cl[i], cl[i - 1], cl[i + 1])
        floor = min(op[i], cl[i], cl[i - 1], cl[i + 1])
        hi[i] = min(hi[i], ceiling)
        lo[i] = max(lo[i], floor)
    out["high"], out["low"] = hi, lo
    return out
