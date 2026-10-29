"""
"15 Minute Quick Win" setup detection on M15.

A long needs four things, in this order and no other:

  1. a BREAK of a lower high - the most recent confirmed swing high sits below
     the one before it, and a candle CLOSES above that lower high;
  2. that same closing body above SMA44;
  3. a BULLISH breaker block from the LuxAlgo indicator appearing at or after
     the break bar;
  4. a retrace that taps the zone (handled by the engine, not here).

Shorts mirror it exactly.

Note that step 1 is the user's structure rule and is deliberately NOT the
same test the LuxAlgo script runs internally. The indicator's bullish shift
keys off lower LOWS (`Ey < Cy`) while this rule keys off lower HIGHS, so the
two are evaluated separately and a setup needs both to agree.

Everything here is a forward pass: bar t sees confirmed pivots whose
`confirmed_bar <= t`, the SMA value at t, and breakers whose `appeared_bar
<= t`. Nothing later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..indicators.breaker_blocks import BULL, BEAR
from ..indicators.swings import SwingTracker, confirmed_pivots

LONG, SHORT = 1, -1


@dataclass
class Setup:
    """A break plus a matching zone - everything but the fill."""

    direction: int
    break_bar: int           # the candle that closed through the structure
    break_level: float       # the lower high / higher low that was broken
    struct_bar: int          # swing that the stop hangs off
    struct_level: float
    indicator_struct_level: float   # the swing the LuxAlgo zone search bounds on
    zone_top: float
    zone_bottom: float
    zone_appeared_bar: int
    zone_source_bar: int
    zone_dead_bar: object    # bar the zone stops being tradeable, or None
    zone_id: int

    @property
    def entry_edge(self) -> float:
        """Limit price: zone top for longs, zone bottom for shorts."""
        return self.zone_top if self.direction == LONG else self.zone_bottom


def find_setups(df, sma, breakers, pivot_length: int,
                longs_enabled=True, shorts_enabled=True,
                invalidate_on_mitigated=True, invalidate_on_broken=True,
                pivot_tie_mode="left_eq", max_bars_break_to_zone=None,
                entry_mode="both"):
    """Pair structure breaks with the breaker zones that follow them."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(close)

    pivots = confirmed_pivots(high, low, pivot_length, 1, pivot_tie_mode)
    tracker = SwingTracker(pivots)

    by_appear = defaultdict(list)
    for i, blk in enumerate(breakers):
        by_appear[blk.appeared_bar].append((i, blk))

    indicator_only = entry_mode == "indicator_only"
    pending = {LONG: None, SHORT: None}
    used_levels = {LONG: set(), SHORT: set()}
    setups = []

    for t in range(n):
        tracker.advance_to(t)
        sma_t = sma[t]

        # A break that has waited too long for its breaker is stale: by then
        # the structure that justified it has usually been overtaken.
        if max_bars_break_to_zone is not None:
            for side in (LONG, SHORT):
                p = pending[side]
                if p is not None and t - p[0] > max_bars_break_to_zone:
                    pending[side] = None

        # -- 1+2. structure break, confirmed by the SMA -----------------
        # Skipped entirely under 'indicator_only': the zone carries its own
        # structure shift and the SMA is applied at the tap instead.
        if indicator_only:
            pass
        elif longs_enabled and not np.isnan(sma_t):
            lh = tracker.lower_high()
            if (lh is not None and lh.bar < t and lh.bar not in used_levels[LONG]
                    and close[t] > lh.price and close[t] > sma_t):
                struct = tracker.last_low_before(t)
                if struct is not None:
                    used_levels[LONG].add(lh.bar)
                    pending[LONG] = (t, lh.price, struct)

        if (not indicator_only) and shorts_enabled and not np.isnan(sma_t):
            hl = tracker.higher_low()
            if (hl is not None and hl.bar < t and hl.bar not in used_levels[SHORT]
                    and close[t] < hl.price and close[t] < sma_t):
                struct = tracker.last_high_before(t)
                if struct is not None:
                    used_levels[SHORT].add(hl.bar)
                    pending[SHORT] = (t, hl.price, struct)

        # -- 3. a zone of the right flavour appears ----------------------
        for zone_id, blk in by_appear.get(t, []):
            side = LONG if blk.direction == BULL else SHORT
            if side == LONG and not longs_enabled:
                continue
            if side == SHORT and not shorts_enabled:
                continue

            if indicator_only:
                # The zone IS the signal. Its stop still hangs off real
                # structure: the last confirmed swing on the correct side
                # before the zone appeared, which `sl_beyond_zone` then
                # pushes clear of the block if it sits inside it.
                struct = (tracker.last_low_before(t) if side == LONG
                          else tracker.last_high_before(t))
                if struct is None:
                    continue
                break_bar, break_level = blk.appeared_bar, blk.swing_broken_price
            else:
                p = pending[side]
                if p is None:
                    continue
                break_bar, break_level, struct = p
                if blk.appeared_bar < break_bar:
                    continue
                if (max_bars_break_to_zone is not None
                        and blk.appeared_bar - break_bar > max_bars_break_to_zone):
                    continue
            setups.append(Setup(
                direction=side,
                break_bar=break_bar,
                break_level=break_level,
                struct_bar=struct.bar,
                struct_level=struct.price,
                indicator_struct_level=blk.structural_price,
                zone_top=blk.top,
                zone_bottom=blk.bottom,
                zone_appeared_bar=blk.appeared_bar,
                zone_source_bar=blk.source_bar,
                zone_dead_bar=blk.dead_bar(invalidate_on_mitigated,
                                           invalidate_on_broken),
                zone_id=zone_id,
            ))
            if not indicator_only:
                pending[side] = None

    return setups
