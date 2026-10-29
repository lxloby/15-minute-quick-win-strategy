"""
Pivot detection matching Pine's `ta.pivothigh` / `ta.pivotlow`, plus the
confirmed-swing bookkeeping the entry rules need.

The thing that matters for honesty here is CONFIRMATION LAG. Pine's
`ta.pivothigh(high, left, right)` evaluated on bar t describes the bar at
t - right, and it cannot be known any earlier than t, because it needs the
`right` bars that follow the pivot. Everything in this module therefore
reports a pivot together with the bar it became knowable on, and nothing
downstream is allowed to look at a pivot before that bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIGH, LOW = 1, -1


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing point."""

    kind: int        # HIGH (1) or LOW (-1)
    bar: int         # index of the pivot bar itself
    price: float     # high[bar] for a pivot high, low[bar] for a pivot low
    confirmed_bar: int   # first bar on which this pivot could be known


def pivot_flags(
    high: np.ndarray,
    low: np.ndarray,
    left: int,
    right: int = 1,
    tie_mode: str = "left_eq",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised `ta.pivothigh` / `ta.pivotlow`.

    Returns two arrays indexed by CONFIRMATION bar t:
      ph[t] = high value of the pivot high at t-right, else NaN
      pl[t] = low  value of the pivot low  at t-right, else NaN

    `tie_mode`:
      'left_eq' - the centre may equal bars on its left but must beat the
                  right. This is what TradingView does, verified against the
                  live indicator on 2026-10-27, and is the default.
      'strict'  - the centre bar must beat every bar in both windows outright.
                  Discards ~11% of pivots; kept only for comparison.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = high.size
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n < left + right + 1:
        return ph, pl

    centres = np.arange(left, n - right)
    hc = high[centres]
    lc = low[centres]

    ph_ok = np.ones(centres.size, dtype=bool)
    pl_ok = np.ones(centres.size, dtype=bool)

    for k in range(1, left + 1):
        if tie_mode == "left_eq":
            ph_ok &= hc >= high[centres - k]
            pl_ok &= lc <= low[centres - k]
        else:
            ph_ok &= hc > high[centres - k]
            pl_ok &= lc < low[centres - k]
    for k in range(1, right + 1):
        ph_ok &= hc > high[centres + k]
        pl_ok &= lc < low[centres + k]

    ph[centres[ph_ok] + right] = hc[ph_ok]
    pl[centres[pl_ok] + right] = lc[pl_ok]
    return ph, pl


def confirmed_pivots(
    high: np.ndarray,
    low: np.ndarray,
    left: int,
    right: int = 1,
    tie_mode: str = "left_eq",
) -> list[Pivot]:
    """All pivots in chronological order of CONFIRMATION."""
    ph, pl = pivot_flags(high, low, left, right, tie_mode)
    out: list[Pivot] = []
    for t in range(len(ph)):
        if not np.isnan(ph[t]):
            out.append(Pivot(HIGH, t - right, float(ph[t]), t))
        if not np.isnan(pl[t]):
            out.append(Pivot(LOW, t - right, float(pl[t]), t))
    out.sort(key=lambda p: (p.confirmed_bar, p.kind))
    return out


class SwingTracker:
    """
    Walks pivots in confirmation order and answers, for any bar t, the two
    questions the entry rules ask:

      * is the most recent confirmed swing high lower than the one before it?
        (a bearish sequence of highs -> we hunt longs on a break of it)
      * is the most recent confirmed swing low higher than the one before it?
        (a bullish sequence of lows -> we hunt shorts on a break of it)

    Call `advance_to(t)` before reading, and it will only admit pivots whose
    `confirmed_bar <= t`.
    """

    def __init__(self, pivots: list[Pivot]):
        self._pivots = pivots
        self._i = 0
        self.highs: list[Pivot] = []
        self.lows: list[Pivot] = []

    def advance_to(self, bar: int) -> None:
        while self._i < len(self._pivots) and self._pivots[self._i].confirmed_bar <= bar:
            p = self._pivots[self._i]
            (self.highs if p.kind == HIGH else self.lows).append(p)
            self._i += 1

    # -- structure tests -----------------------------------------------------
    def lower_high(self) -> Pivot | None:
        """Most recent confirmed swing high, if it is below the previous one."""
        if len(self.highs) < 2:
            return None
        last, prev = self.highs[-1], self.highs[-2]
        return last if last.price < prev.price else None

    def higher_low(self) -> Pivot | None:
        """Most recent confirmed swing low, if it is above the previous one."""
        if len(self.lows) < 2:
            return None
        last, prev = self.lows[-1], self.lows[-2]
        return last if last.price > prev.price else None

    def last_low_before(self, bar: int) -> Pivot | None:
        """Structural low preceding a break - the stop reference for longs."""
        for p in reversed(self.lows):
            if p.bar < bar:
                return p
        return None

    def last_high_before(self, bar: int) -> Pivot | None:
        """Structural high preceding a break - the stop reference for shorts."""
        for p in reversed(self.highs):
            if p.bar < bar:
                return p
        return None
