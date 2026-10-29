"""
Performance statistics, all R-based so they survive changes in account size.

Drawdown is reported two ways because they answer different questions:
  * R drawdown  - the worst peak-to-trough run of the cumulative R curve.
  * % drawdown  - the same sequence compounded at the configured risk per
                  trade, which is what the account would actually have felt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _streak(flags) -> int:
    """Longest run of True."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def summarise(trades: pd.DataFrame, risk_pct: float = 1.0) -> dict:
    """One row of statistics for a set of trades."""
    n = len(trades)
    base = {
        "trades": n, "wins": 0, "losses": 0, "win_rate": np.nan,
        "avg_r": np.nan, "expectancy_r": np.nan, "total_r": 0.0,
        "profit_factor": np.nan, "max_dd_r": 0.0, "max_dd_pct": 0.0,
        "longest_losing_streak": 0, "avg_trades_per_week": np.nan,
        "avg_win_r": np.nan, "avg_loss_r": np.nan, "avg_sl_pips": np.nan,
        "total_pips": 0.0,
    }
    if n == 0:
        return base

    r = trades["r_multiple"].to_numpy(float)
    wins, losses = r > 0, r < 0
    gross_win, gross_loss = r[wins].sum(), -r[losses].sum()

    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd_r = float((peak - equity).max()) if n else 0.0

    # Compounded at a fixed fraction of equity.
    frac = risk_pct / 100.0
    curve = np.cumprod(1.0 + frac * r)
    cpeak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    dd_pct = float(((cpeak - curve) / cpeak).max() * 100.0) if n else 0.0

    span_weeks = np.nan
    if "entry_time" in trades and n > 1:
        t = pd.DatetimeIndex(trades["entry_time"])
        days = (t.max() - t.min()).total_seconds() / 86400.0
        span_weeks = max(days / 7.0, 1e-9)

    return {
        "trades": n,
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "win_rate": float(wins.mean() * 100.0),
        "avg_r": float(r.mean()),
        "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "max_dd_r": dd_r,
        "max_dd_pct": dd_pct,
        "longest_losing_streak": _streak(losses),
        "avg_trades_per_week": float(n / span_weeks) if span_weeks == span_weeks else np.nan,
        "avg_win_r": float(r[wins].mean()) if wins.any() else np.nan,
        "avg_loss_r": float(r[losses].mean()) if losses.any() else np.nan,
        "avg_sl_pips": float(trades["sl_pips"].mean()) if "sl_pips" in trades else np.nan,
        "total_pips": float(trades["pips"].sum()) if "pips" in trades else 0.0,
    }


def summary_table(trades: pd.DataFrame, risk_pct: float = 1.0,
                  by=("symbol", "method", "direction")) -> pd.DataFrame:
    """
    Per pair / per method / per direction, plus 'combined' roll-ups for
    direction and for symbol, so the headline numbers sit in the same table.
    """
    if trades.empty:
        return pd.DataFrame()

    rows = []

    def add(label: dict, sub: pd.DataFrame):
        rows.append({**label, **summarise(sub, risk_pct)})

    for method, m_df in trades.groupby("method"):
        for symbol, s_df in m_df.groupby("symbol"):
            for direction, d_df in s_df.groupby("direction"):
                add({"method": method, "symbol": symbol, "direction": direction}, d_df)
            add({"method": method, "symbol": symbol, "direction": "combined"}, s_df)
        for direction, d_df in m_df.groupby("direction"):
            add({"method": method, "symbol": "ALL", "direction": direction}, d_df)
        add({"method": method, "symbol": "ALL", "direction": "combined"}, m_df)

    cols = ["method", "symbol", "direction"]
    out = pd.DataFrame(rows)
    return out[cols + [c for c in out.columns if c not in cols]]


def breakdown(trades: pd.DataFrame, key: str, risk_pct: float = 1.0,
              display_tz: str = "Asia/Singapore") -> pd.DataFrame:
    """
    Slice performance by 'year', 'quarter' or 'hour'.

    Hours are reported in the display timezone (SGT by default) because the
    session window was specified that way; entry_time stays UTC everywhere else.
    """
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    t = pd.DatetimeIndex(df["entry_time"])
    if t.tz is None:
        t = t.tz_localize("UTC")

    if key == "year":
        df["_bucket"] = t.year
    elif key == "quarter":
        df["_bucket"] = [f"{ts.year}Q{ts.quarter}" for ts in t]
    elif key == "hour":
        df["_bucket"] = t.tz_convert(display_tz).hour
    else:
        raise ValueError(f"unknown breakdown key {key!r}")

    rows = []
    for method, m_df in df.groupby("method"):
        for bucket, b_df in m_df.groupby("_bucket"):
            rows.append({"method": method, key: bucket, **summarise(b_df, risk_pct)})
    out = pd.DataFrame(rows)
    return out.sort_values(["method", key]).reset_index(drop=True)


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative R against entry time, for plotting."""
    if trades.empty:
        return pd.DataFrame(columns=["entry_time", "cum_r"])
    df = trades.sort_values("entry_time")[["entry_time", "r_multiple"]].copy()
    df["cum_r"] = df["r_multiple"].cumsum()
    return df


def sl_distribution(sl_pips, bins=None) -> pd.DataFrame:
    """Histogram of stop sizes - the evidence behind the 10-20 pip filter."""
    s = pd.Series(list(sl_pips), dtype=float).dropna()
    if s.empty:
        return pd.DataFrame()
    if bins is None:
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, np.inf]
    cut = pd.cut(s, bins=bins, right=False)
    out = cut.value_counts().sort_index().rename("count").to_frame()
    out["pct"] = (out["count"] / len(s) * 100).round(2)
    out.index = out.index.astype(str)
    return out.reset_index(names="sl_pips_bin")
