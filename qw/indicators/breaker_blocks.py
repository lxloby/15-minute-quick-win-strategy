"""
Python port of "Breaker Blocks with Signals [LuxAlgo]" (Pine v5).

The port is a single forward pass over bars that mirrors Pine's execution
model one statement at a time, which is the only way to be sure there is no
lookahead: on bar t the engine can touch bars <= t and nothing else.

Three details of the original are easy to miss and change the results a lot,
so they are reproduced deliberately and surfaced in config:

1. `ta.pivothigh(hi, len, 1)` has right=1. A pivot is therefore knowable one
   bar after it prints, never on the pivot bar itself.
2. The script holds exactly ONE `var block BB`. A new market-structure shift
   overwrites the previous breaker rather than adding to it, so at most one
   zone is ever live. `single_active_zone` reproduces that; turning it off
   keeps every zone, which is a deviation from TradingView.
3. `per = last_bar_index - bar_index <= 2000` restricts drawing to the last
   2000 bars of the chart. That is a rendering budget, not strategy logic, so
   `recent_bars_only: null` applies the indicator across all history.

Zone geometry, default inputs: scanning backwards from the broken swing high
D towards the preceding swing low C, the first bullish candle found supplies
the zone, and the zone is that candle's high/low (body only if
`use_only_candle_body`). Mirrored for bearish.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .swings import pivot_flags

BULL, BEAR = 1, -1
ZZ_SIZE = 50          # Pine's mCxSize
PIVOT_RIGHT = 1       # hard-coded in the Pine source: ta.pivothigh(hi, left, 1)


@dataclass
class Breaker:
    """One breaker block, with every bar index it transitions on."""

    direction: int           # BULL (+1) or BEAR (-1)
    top: float
    bottom: float
    source_bar: int          # the candle the zone is drawn from
    appeared_bar: int        # the MSS bar - the first bar this zone exists on
    swing_broken_price: float        # D - the swing level the close broke
    structural_bar: int              # C - swing that bounds the search
    structural_price: float = float("nan")   # C's own level (low for a bull zone)
    mitigated_bar: int | None = None   # close beyond the far edge
    broken_bar: int | None = None      # close through the mid-line ("cancel")
    superseded_bar: int | None = None  # replaced by a newer breaker
    signal_bars: list = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def dead_bar(self, on_mitigated: bool = True, on_broken: bool = True):
        """First bar on which this zone stops being tradeable, if any."""
        candidates = [self.superseded_bar]
        if on_mitigated:
            candidates.append(self.mitigated_bar)
        if on_broken:
            candidates.append(self.broken_bar)
        live = [c for c in candidates if c is not None]
        return min(live) if live else None

    def is_live_at(self, bar, on_mitigated=True, on_broken=True) -> bool:
        """
        Can this zone still be entered on `bar`?

        Entries are only possible strictly after the zone appears. A zone that
        dies on bar k is treated as usable up to and including k, because the
        tap happens intrabar while the kill condition is a close.
        """
        if bar <= self.appeared_bar:
            return False
        dead = self.dead_bar(on_mitigated, on_broken)
        return dead is None or bar <= dead


class _ZigZag:
    """Pine's `aZZ` - fixed-length deques of direction / bar index / price."""

    __slots__ = ("d", "x", "y")

    def __init__(self) -> None:
        self.d = [0] * ZZ_SIZE
        self.x = [0] * ZZ_SIZE
        self.y = [np.nan] * ZZ_SIZE

    def unshift(self, d: int, x: int, y: float) -> None:
        self.d.insert(0, d); self.d.pop()
        self.x.insert(0, x); self.x.pop()
        self.y.insert(0, y); self.y.pop()


def detect_breakers(
    open_,
    high,
    low,
    close,
    length: int = 5,
    use_only_candle_body: bool = False,
    use_2_candles: bool = False,
    till_first_break: bool = True,
    recent_bars_only=None,
    single_active_zone: bool = True,
    pivot_tie_mode: str = "left_eq",
):
    """Run the indicator over a whole series and return every breaker created."""
    open_ = np.asarray(open_, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    close = np.asarray(close, float)
    n_bars = close.size

    mx = np.maximum(close, open_)   # Pine `mx` - body top
    mn = np.minimum(close, open_)   # Pine `mn` - body bottom

    ph, pl = pivot_flags(high, low, length, PIVOT_RIGHT, pivot_tie_mode)

    zz = _ZigZag()
    mss_dir = 0
    breakers = []
    active = None
    last_bar_index = n_bars - 1

    for t in range(n_bars):
        # `per` - Pine only draws over the most recent slice of the chart.
        within_budget = (
            True if recent_bars_only is None
            else (last_bar_index - t) <= recent_bars_only
        )

        # -- draw(): zig-zag maintenance ------------------------------------
        x2 = t - PIVOT_RIGHT
        if not np.isnan(ph[t]):
            y2 = high[x2]
            if zz.d[0] < 1:
                zz.unshift(1, x2, y2)
            elif zz.d[0] == 1 and y2 > zz.y[0]:
                zz.x[0], zz.y[0] = x2, y2
        if not np.isnan(pl[t]):
            y2 = low[x2]
            if zz.d[0] > -1:
                zz.unshift(-1, x2, y2)
            elif zz.d[0] == -1 and y2 < zz.y[0]:
                zz.x[0], zz.y[0] = x2, y2

        iH = 2 if zz.d[2] == 1 else 1
        iL = 2 if zz.d[2] == -1 else 1

        # -- draw(): market-structure shift ---------------------------------
        new_block = None

        bull_mss = (
            within_budget and mss_dir < 1
            and zz.d[iH] == 1 and close[t] > zz.y[iH]
        )
        bear_mss = (
            within_budget and mss_dir > -1
            and zz.d[iL] == -1 and close[t] < zz.y[iL]
        )

        # Pine's `switch` takes the first branch that matches, never both.
        if bull_mss:
            Ey, Dx, Cx, Cy = zz.y[iH - 1], zz.x[iH], zz.x[iH + 1], zz.y[iH + 1]
            if Ey < Cy and Cx != Dx:
                new_block = _scan_zone(
                    BULL, Dx, Cx, open_, high, low, close, mx, mn, t,
                    use_only_candle_body, use_2_candles, zz.y[iH],
                )
            mss_dir = 1

        elif bear_mss:
            Ey, Dx, Cx, Cy = zz.y[iL - 1], zz.x[iL], zz.x[iL + 1], zz.y[iL + 1]
            if Ey > Cy and Cx != Dx:
                new_block = _scan_zone(
                    BEAR, Dx, Cx, open_, high, low, close, mx, mn, t,
                    use_only_candle_body, use_2_candles, zz.y[iL],
                )
            mss_dir = -1

        if new_block is not None:
            if active is not None and single_active_zone:
                active.superseded_bar = t
            breakers.append(new_block)
            active = new_block

        # -- post-draw(): state machine on the live block -------------------
        # Runs on every bar, including the creation bar, exactly as Pine does.
        tracked = [active] if single_active_zone else breakers
        for blk in tracked:
            if blk is None or blk.superseded_bar is not None:
                continue
            _step_state(blk, t, open_[t], high[t], low[t], close[t], till_first_break)

    return breakers


def _scan_zone(
    direction, Dx, Cx,
    open_, high, low, close, mx, mn,
    t, only_body, two_candles, swing_price,
):
    """
    Pine's `for i = n - Dx to n - Cx` loop, written in absolute bar indices:
    walk backwards from the broken swing D towards the earlier swing C and
    take the first candle running in the breaker's direction.
    """
    if not (0 <= Cx <= Dx < len(close)):
        return None

    for b in range(Dx, Cx - 1, -1):
        bullish_candle = close[b] > open_[b]
        if (direction == BULL) != bullish_candle:
            continue

        src_bar = b
        top = mx[b] if only_body else high[b]
        bottom = mn[b] if only_body else low[b]

        if two_candles and b - 1 >= 0:
            prev_matches = (close[b - 1] > open_[b - 1]) == (direction == BULL)
            if prev_matches:
                top2 = mx[b - 1] if only_body else high[b - 1]
                bot2 = mn[b - 1] if only_body else low[b - 1]
                if top2 > top or bot2 < bottom:
                    src_bar = b - 1
                top, bottom = max(top, top2), min(bottom, bot2)

        return Breaker(
            direction=direction,
            top=float(top),
            bottom=float(bottom),
            source_bar=int(src_bar),
            appeared_bar=int(t),
            swing_broken_price=float(swing_price),
            structural_bar=int(Cx),
            # C is a swing LOW for a bullish zone and a swing HIGH for a
            # bearish one - the level the indicator's own search is bounded by.
            structural_price=float(low[Cx] if direction == BULL else high[Cx]),
        )
    return None


def _step_state(blk, t, o, h, l, c, till_first_break):
    """One bar of the `switch BB.dir` block that follows `draw()` in Pine."""
    if blk.mitigated_bar is not None:
        return                      # `if not BB.Mitigated` gates everything below

    top, btm, avg = blk.top, blk.bottom, blk.mid

    if blk.direction == BULL:
        if c < btm:
            blk.mitigated_bar = t
        if t > blk.appeared_bar:            # Pine: `n > BB.BB_boxB.get_left()`
            if blk.broken_bar is None:
                if o > avg and o < top and c > top:
                    blk.signal_bars.append(t)       # 'signal UP'
                elif c < avg and c > btm:
                    blk.broken_bar = t              # 'cancel UP'
            elif not till_first_break and c > top:
                blk.broken_bar = None               # '+BB (R)' reset
    else:
        if c > top:
            blk.mitigated_bar = t
        if t > blk.appeared_bar:
            if blk.broken_bar is None:
                if o < avg and o > btm and c < btm:
                    blk.signal_bars.append(t)       # 'signal DN'
                elif c > avg and c < top:
                    blk.broken_bar = t              # 'cancel DN'
            elif not till_first_break and c < btm:
                blk.broken_bar = None               # '-BB (R)' reset
