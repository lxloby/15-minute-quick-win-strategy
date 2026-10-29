"""
Entry-time filters: trading session and high-impact news.

Both answer the same question - "may a NEW trade be opened at this UTC
instant?" - and neither touches a position that is already running.
"""

from __future__ import annotations

from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

# Which currencies a symbol is exposed to, for the news filter.
def symbol_currencies(symbol: str) -> list:
    s = symbol.upper()
    base, quote = s[:3], s[3:6]
    return sorted({base, quote})


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def _parse_hhmm(text: str) -> dtime:
    h, m = str(text).split(":")
    return dtime(int(h), int(m))


def session_mask(times_utc, start_utc: str, end_utc: str) -> np.ndarray:
    """
    Boolean mask of timestamps inside [start, end) in UTC.

    A window that wraps past midnight (end <= start) is handled, so 22:00-06:00
    would work as well as the configured 12:00-23:00.
    """
    idx = pd.DatetimeIndex(times_utc)
    start, end = _parse_hhmm(start_utc), _parse_hhmm(end_utc)
    minutes = idx.hour * 60 + idx.minute
    s = start.hour * 60 + start.minute
    e = end.hour * 60 + end.minute
    if s <= e:
        return (minutes >= s) & (minutes < e)
    return (minutes >= s) | (minutes < e)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
class NewsFilter:
    """
    Blocks entries within +/- a window of a high-impact event for either
    currency in the pair.

    The calendar must carry UTC timestamps. `data/news_calendar.csv` columns:
        utc_time,currency,impact,title
    """

    REQUIRED = {"utc_time", "currency", "impact", "title"}

    def __init__(self, events: pd.DataFrame, minutes_before: int,
                 minutes_after: int, impacts, currencies):
        self.before = pd.Timedelta(minutes=minutes_before)
        self.after = pd.Timedelta(minutes=minutes_after)
        ev = events.copy()
        ev["impact"] = ev["impact"].astype(str).str.strip().str.title()
        ev["currency"] = ev["currency"].astype(str).str.strip().str.upper()
        ev = ev[ev["impact"].isin({str(i).title() for i in impacts})]
        ev = ev[ev["currency"].isin({str(c).upper() for c in currencies})]
        self.events = ev.sort_values("utc_time").reset_index(drop=True)
        self._by_ccy = {
            ccy: np.sort(g["utc_time"].to_numpy(dtype="datetime64[ns]"))
            for ccy, g in self.events.groupby("currency")
        }

    @classmethod
    def from_csv(cls, path, minutes_before=30, minutes_after=30,
                 impacts=("High",), currencies=("EUR", "USD", "JPY", "CHF")):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"News calendar not found at {p}. Build it with "
                f"scripts/build_news.py before running the backtest - the news "
                f"filter is not allowed to silently no-op."
            )
        df = pd.read_csv(p)
        missing = cls.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns {sorted(missing)}")
        df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True).dt.tz_localize(None)
        return cls(df, minutes_before, minutes_after, impacts, currencies)

    def coverage(self):
        if self.events.empty:
            return None, None
        return self.events["utc_time"].min(), self.events["utc_time"].max()

    def missing_currencies(self, symbols) -> dict:
        """
        Currencies a symbol is exposed to that the calendar has no events for.

        This is the failure that would otherwise pass unnoticed: add a new
        pair, forget to rebuild the calendar, and the news filter quietly
        does nothing for half of it while still reporting as 'enabled'.
        """
        have = set(self._by_ccy)
        out = {}
        for sym in symbols:
            gap = [c for c in symbol_currencies(sym) if c not in have]
            if gap:
                out[sym] = gap
        return out

    def blocked_mask(self, times_utc, symbol: str) -> np.ndarray:
        """True where a new entry is forbidden by news for this symbol."""
        idx = pd.DatetimeIndex(times_utc)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        t = idx.to_numpy(dtype="datetime64[ns]")
        blocked = np.zeros(len(t), dtype=bool)

        for ccy in symbol_currencies(symbol):
            ev = self._by_ccy.get(ccy)
            if ev is None or not len(ev):
                continue
            # An entry at time t is blocked if some event e satisfies
            #   e - before <= t <= e + after
            # i.e. the nearest event at or before (t + before) is within reach.
            pos = np.searchsorted(ev, t + np.timedelta64(self.before), "right") - 1
            valid = pos >= 0
            if valid.any():
                nearest = ev[np.clip(pos, 0, len(ev) - 1)]
                within = (t >= nearest - np.timedelta64(self.before)) & (
                    t <= nearest + np.timedelta64(self.after)
                )
                blocked |= valid & within
        return blocked
