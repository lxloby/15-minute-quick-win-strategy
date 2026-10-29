"""
Trade simulation: given an entry, walk forward and decide how it ended.

Price convention. MT5 bars are BID. A long is filled at the ask (bid+spread)
and exits on the bid; a short is filled at the bid and exits on the ask. So
every trade pays the spread exactly once, and the risk distance
`entry -> stop` already contains it. Working in "exit-price space" keeps the
two directions symmetric:

    long :  fill = zone_top + spread      exits compared against raw high/low
    short:  fill = zone_bottom            exits compared against high/low + spread

Sequencing. Inside one M15 bar the order of stop, partial and target is
unknowable from OHLC alone, so M1 bars resolve it when they exist. Where they
do not - or where a single M1 bar still contains two of them - the WORSE
outcome is assumed. Every such trade is tagged `intrabar_source = M15` (or
`M1/M15` for a trade that straddles the boundary) so the two populations can
be reported apart.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

LONG, SHORT = 1, -1


@dataclass
class Fill:
    """Everything known at the moment the limit order fills."""

    direction: int
    entry_bar: int
    entry_time: object
    entry_price: float       # already spread-adjusted for longs
    stop: float
    target: float
    risk: float              # price distance entry -> stop, always > 0
    pip: float
    spread_at_entry: float


@dataclass
class Outcome:
    exit_bar: int = -1
    exit_time: object = None
    exit_price: float = float("nan")
    exit_reason: str = "open"
    r_multiple: float = 0.0
    pips: float = 0.0
    partial_time: object = None
    partial_price: float = float("nan")
    intrabar_source: str = "M15"
    bars_held: int = 0
    used_worst_case: bool = False
    notes: list = field(default_factory=list)


class M1Window(NamedTuple):
    """The M1 bars inside one M15 bar, as plain arrays."""

    times: np.ndarray
    high: np.ndarray
    low: np.ndarray
    spread: np.ndarray


class M1Index:
    """
    Whole-series M1 lookup, held in memory.

    Fine for tests and short series. For a multi-year backtest use
    `YearlyM1Index` instead - see the note there.
    """

    def __init__(self, m1: "object | None"):
        self.empty = m1 is None or len(m1) == 0
        if self.empty:
            self.times = np.array([], dtype="datetime64[ns]")
            return
        self.times = m1["time_utc"].to_numpy(dtype="datetime64[ns]")
        self.high = m1["high"].to_numpy(float)
        self.low = m1["low"].to_numpy(float)
        # `_spread_price` is the spread already converted from points to price,
        # so the two timeframes speak the same units.
        self.spread = (
            m1["_spread_price"].to_numpy(float) if "_spread_price" in m1
            else np.zeros(len(m1))
        )

    def covers(self, ts) -> bool:
        if self.empty:
            return False
        t = np.datetime64(ts, "ns")
        return bool(self.times[0] <= t <= self.times[-1])

    def window(self, start, end):
        """M1 bars with start <= time < end, or None."""
        if self.empty:
            return None
        a = np.searchsorted(self.times, np.datetime64(start, "ns"), "left")
        b = np.searchsorted(self.times, np.datetime64(end, "ns"), "left")
        if b <= a:
            return None
        return M1Window(self.times[a:b], self.high[a:b],
                        self.low[a:b], self.spread[a:b])


class YearlyM1Index:
    """
    M1 read one year at a time, straight off the year-partitioned cache.

    Nineteen years of M1 is ~7M rows per symbol, about 292MB as a DataFrame
    and the same again once it is turned into arrays. On a 6GB machine that
    does not fit beside everything else, and the first attempt at it died of
    a MemoryError.

    The backtest asks for M1 windows in chronological order, so keeping only
    the year in play (plus the previous one, for trades that straddle New
    Year) costs ~18MB instead and never touches the rest.
    """

    def __init__(self, cache_dir, symbol: str, timeframe: str, point: float,
                 keep: int = 2, clean: bool = False):
        from pathlib import Path
        self._dir = Path(cache_dir) / f"{symbol}_{timeframe}"
        self._point = float(point)
        self._keep = keep
        # M1 carries the same daily-rollover bad prints as M15, so it gets the
        # same treatment - applied per year as each one is read.
        self._clean = clean
        self.cleaned = 0
        self._cache: "OrderedDict[int, M1Window]" = OrderedDict()
        self._years = {}
        if self._dir.exists():
            for f in self._dir.glob("*.parquet"):
                try:
                    self._years[int(f.stem)] = f
                except ValueError:
                    continue
        self.empty = not self._years

    @property
    def year_range(self):
        return (min(self._years), max(self._years)) if self._years else (None, None)

    def covers(self, ts) -> bool:
        return (not self.empty) and pd.Timestamp(ts).year in self._years

    def _load(self, year: int):
        if year in self._cache:
            self._cache.move_to_end(year)
            return self._cache[year]
        path = self._years.get(year)
        if path is None:
            return None
        cols = ["time_utc", "high", "low", "spread"]
        if self._clean:
            cols = ["time_utc", "open", "high", "low", "close", "spread"]
        df = pd.read_parquet(path, columns=cols)
        if self._clean:
            from ..data.quality import clip_suspect_bars, find_suspect_bars
            suspect = find_suspect_bars(df, window=240, range_mult=10.0)
            if len(suspect):
                df = clip_suspect_bars(df, suspect)
                self.cleaned += len(suspect)
        win = M1Window(
            df["time_utc"].to_numpy(dtype="datetime64[ns]"),
            df["high"].to_numpy(float),
            df["low"].to_numpy(float),
            df["spread"].to_numpy(float) * self._point,
        )
        del df
        self._cache[year] = win
        while len(self._cache) > self._keep:
            self._cache.popitem(last=False)
        return win

    @staticmethod
    def _cut(win: M1Window, start, end):
        a = np.searchsorted(win.times, np.datetime64(start, "ns"), "left")
        b = np.searchsorted(win.times, np.datetime64(end, "ns"), "left")
        if b <= a:
            return None
        return M1Window(win.times[a:b], win.high[a:b], win.low[a:b], win.spread[a:b])

    def window(self, start, end):
        if self.empty:
            return None
        y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
        parts = []
        for y in range(y0, y1 + 1):
            whole = self._load(y)
            if whole is None:
                continue
            cut = self._cut(whole, start, end)
            if cut is not None:
                parts.append(cut)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return M1Window(*(np.concatenate([p[i] for p in parts]) for i in range(4)))


def build_fill(direction, entry_bar, entry_time, zone_edge, struct_level,
               spread_price, pip, sl_buffer_pips, risk_reward,
               zone_far_edge=None) -> Fill:
    """
    Turn a zone tap into a concrete order, costs included.

    `zone_far_edge` is the far side of the breaker block - the zone bottom
    for a long, the top for a short. When given, the stop is pushed beyond
    it, so it can never come to rest inside the zone the trade just entered
    from. A structural swing sometimes sits within the block, and a stop
    there would be taken out by ordinary movement inside the zone rather
    than by the structure actually failing.
    """
    buf = sl_buffer_pips * pip
    if direction == LONG:
        entry_price = zone_edge + spread_price      # fill at the ask
        ref = struct_level if zone_far_edge is None else min(struct_level, zone_far_edge)
        stop = ref - buf
        risk = entry_price - stop
        target = entry_price + risk_reward * risk
    else:
        entry_price = zone_edge                     # fill at the bid
        ref = struct_level if zone_far_edge is None else max(struct_level, zone_far_edge)
        stop = ref + buf                            # compared against ask
        risk = stop - entry_price
        target = entry_price - risk_reward * risk
    return Fill(direction, entry_bar, entry_time, entry_price, stop, target,
                risk, pip, spread_price)


def simulate(fill: Fill, m15, m1index, method: dict,
             commission_r: float = 0.0, ambiguous: str = "worst",
             max_bars_in_trade=None) -> Outcome:
    """
    Walk forward from the entry bar until the position is flat.

    `method` keys: partial_trigger_pips (or None), partial_fraction,
    move_to_breakeven, be_includes_costs.
    """
    out = Outcome()
    n = len(m15)
    d = fill.direction
    pip = fill.pip

    # The partial can be set either as a fixed pip distance or as a fraction
    # of R. Fixed pips does not survive a change of stop size: +10 pips is
    # 0.74R on a 13 pip stop and 0.23R on a 44 pip one, so the same config
    # means two different strategies. `partial_trigger_r` scales with the
    # stop and takes precedence when both are given.
    trig_r = method.get("partial_trigger_r")
    trig_pips = method.get("partial_trigger_pips")
    frac = float(method.get("partial_fraction") or 0.0)

    if trig_r is not None:
        trig_dist = float(trig_r) * fill.risk
    elif trig_pips is not None:
        trig_dist = float(trig_pips) * pip
    else:
        trig_dist = None

    use_partial = trig_dist is not None and frac > 0
    partial_level = fill.entry_price + d * trig_dist if use_partial else None
    be_level = fill.entry_price if method.get("move_to_breakeven") else None

    highs = m15["high"].to_numpy(float)
    lows = m15["low"].to_numpy(float)
    times = m15["time_utc"].to_numpy(dtype="datetime64[ns]")
    spreads = m15["_spread_price"].to_numpy(float)

    stop = fill.stop
    partial_done = False
    saw_m1 = saw_m15 = False

    last_allowed = n - 1 if max_bars_in_trade is None else min(
        n - 1, fill.entry_bar + int(max_bars_in_trade))

    for i in range(fill.entry_bar, last_allowed + 1):
        # exit-price space: shorts are closed at the ask
        if d == LONG:
            bar_hi, bar_lo = highs[i], lows[i]
        else:
            bar_hi, bar_lo = highs[i] + spreads[i], lows[i] + spreads[i]

        # Does anything happen in this bar at all?
        touched_stop = (bar_lo <= stop) if d == LONG else (bar_hi >= stop)
        touched_target = (bar_hi >= fill.target) if d == LONG else (bar_lo <= fill.target)
        touched_partial = use_partial and not partial_done and (
            (bar_hi >= partial_level) if d == LONG else (bar_lo <= partial_level)
        )
        if not (touched_stop or touched_target or touched_partial):
            continue

        bar_start = times[i]
        bar_end = bar_start + np.timedelta64(15, "m")
        win = m1index.window(bar_start, bar_end) if m1index is not None else None

        if win is not None:
            saw_m1 = True
            events = _walk_m1(fill, win, stop, fill.target,
                              partial_level if not partial_done else None,
                              ambiguous)
        else:
            saw_m15 = True
            events = _resolve_m15(fill, bar_hi, bar_lo, stop, fill.target,
                                  partial_level if not partial_done else None,
                                  ambiguous)

        for kind, price, ts in events:
            if kind == "partial":
                partial_done = True
                out.partial_time = ts if ts is not None else bar_start
                out.partial_price = price
                if be_level is not None:
                    stop = be_level
            else:
                out.exit_bar = i
                out.exit_time = ts if ts is not None else bar_start
                out.exit_price = price
                out.exit_reason = kind
                break
        if out.exit_reason != "open":
            break

    if out.exit_reason == "open" and n:
        last = last_allowed
        close = m15["close"].to_numpy(float)[last]
        out.exit_bar, out.exit_time = last, times[last]
        out.exit_price = close if d == LONG else close + spreads[last]
        # Flat on the clock rather than on a level: tell the two apart, a
        # time stop is a management choice while end_of_data is a limitation.
        out.exit_reason = ("time_stop" if max_bars_in_trade is not None
                           and last < n - 1 else "end_of_data")

    out.intrabar_source = (
        "M1" if saw_m1 and not saw_m15 else "M15" if saw_m15 and not saw_m1
        else "M1/M15" if saw_m1 else "none"
    )
    out.bars_held = max(0, out.exit_bar - fill.entry_bar)
    _score(fill, out, frac, commission_r)
    return out


def _walk_m1(fill, win, stop, target, partial_level, ambiguous):
    """Replay the M1 bars inside one M15 bar, in order."""
    d = fill.direction
    events = []
    for k in range(len(win.times)):
        if d == LONG:
            hi, lo = win.high[k], win.low[k]
        else:
            sp = win.spread[k]          # shorts close at the ask
            hi, lo = win.high[k] + sp, win.low[k] + sp
        ts = win.times[k]

        hit_stop = (lo <= stop) if d == LONG else (hi >= stop)
        hit_target = (hi >= target) if d == LONG else (lo <= target)
        hit_partial = partial_level is not None and (
            (hi >= partial_level) if d == LONG else (lo <= partial_level)
        )

        # Worse-first inside a single M1 bar, as documented.
        if hit_stop and (hit_target or hit_partial) and ambiguous == "worst":
            events.append(("stop", stop, ts))
            return events
        if hit_partial:
            events.append(("partial", partial_level, ts))
            partial_level = None
        if hit_stop:
            events.append(("stop", stop, ts))
            return events
        if hit_target:
            events.append(("target", target, ts))
            return events
    return events


def _resolve_m15(fill, bar_hi, bar_lo, stop, target, partial_level, ambiguous):
    """No M1 for this bar - apply the conservative rule."""
    d = fill.direction
    hit_stop = (bar_lo <= stop) if d == LONG else (bar_hi >= stop)
    hit_target = (bar_hi >= target) if d == LONG else (bar_lo <= target)
    hit_partial = partial_level is not None and (
        (bar_hi >= partial_level) if d == LONG else (bar_lo <= partial_level)
    )

    if hit_stop and ambiguous == "worst":
        return [("stop", stop, None)]          # stop beats partial and target
    if hit_partial and hit_target:
        return [("partial", partial_level, None), ("target", target, None)]
    if hit_partial:
        return [("partial", partial_level, None)]
    if hit_target:
        return [("target", target, None)]
    if hit_stop:
        return [("stop", stop, None)]
    return []


def _score(fill: Fill, out: Outcome, frac: float, commission_r: float) -> None:
    """Convert the exit into R and pips, net of commission."""
    d, pip, risk = fill.direction, fill.pip, fill.risk
    if risk <= 0:
        out.r_multiple, out.pips = 0.0, 0.0
        return

    took_partial = out.partial_time is not None
    closed_frac = frac if took_partial else 0.0

    def r_of(price):
        return d * (price - fill.entry_price) / risk

    r = 0.0
    if took_partial:
        r += closed_frac * r_of(out.partial_price)
    r += (1.0 - closed_frac) * r_of(out.exit_price)

    out.r_multiple = r - commission_r
    out.pips = r * risk / pip
