"""
Breaker-block port checks.

The zig-zag needs real swing structure to work on, so most of these build a
price path through waypoints A(low) B(high) C(low) D(high) E(low) and then
drive a close back through D, which is exactly the shape the Pine script's
bullish branch looks for.
"""

import numpy as np
import pytest

from conftest import random_ohlc
from qw.indicators.breaker_blocks import (
    BEAR, BULL, Breaker, _step_state, detect_breakers,
)


def path(waypoints, per_leg=8, wick=0.05, seed=0):
    """
    Interpolate between waypoints and dress the result up as candles.

    A little jitter is essential: a perfectly interpolated path makes
    `open[i+1] == close[i]`, so neighbouring bars share an exact high or low
    and strict pivot detection - correctly - refuses to call either one a
    pivot. The noise is ~1% of a leg, so the structure is untouched.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        pts.extend(np.linspace(a, b, per_leg, endpoint=False))
    pts.append(waypoints[-1])
    close = np.array(pts, float) + rng.normal(0, 0.02, len(pts))
    open_ = np.concatenate([[close[0] - 0.1], close[:-1]])
    high = np.maximum(open_, close) + wick + rng.uniform(0, 0.02, close.size)
    low = np.minimum(open_, close) - wick - rng.uniform(0, 0.02, close.size)
    return open_, high, low, close


# A(low) B(high) C(low) D(high) E(low, below C) then a push back above D.
BULL_WAYPOINTS = [110, 100, 112, 102, 118, 98, 125]
BEAR_WAYPOINTS = [100, 112, 98, 110, 92, 120, 85]


def test_bullish_breaker_is_created_on_the_break_bar():
    o, h, l, c = path(BULL_WAYPOINTS)
    blocks = detect_breakers(o, h, l, c, length=5)
    bulls = [b for b in blocks if b.direction == BULL]
    assert bulls, "the A-B-C-D-E shape should produce a bullish breaker"

    b = bulls[0]
    assert b.appeared_bar >= b.source_bar, "a zone cannot predate its own candle"
    assert b.source_bar >= b.structural_bar, "zone candle sits between C and D"
    assert b.top == pytest.approx(h[b.source_bar])
    assert b.bottom == pytest.approx(l[b.source_bar])
    assert c[b.appeared_bar] > b.swing_broken_price, "it appears on a close through D"


def test_bearish_breaker_mirrors():
    o, h, l, c = path(BEAR_WAYPOINTS)
    blocks = detect_breakers(o, h, l, c, length=5)
    bears = [b for b in blocks if b.direction == BEAR]
    assert bears
    b = bears[0]
    assert c[b.appeared_bar] < b.swing_broken_price
    assert b.top == pytest.approx(h[b.source_bar])
    assert b.bottom == pytest.approx(l[b.source_bar])


def test_body_only_mode_shrinks_the_zone():
    o, h, l, c = path(BULL_WAYPOINTS)
    wide = [b for b in detect_breakers(o, h, l, c, length=5) if b.direction == BULL][0]
    body = [b for b in detect_breakers(o, h, l, c, length=5,
                                       use_only_candle_body=True)
            if b.direction == BULL][0]
    assert body.source_bar == wide.source_bar
    assert body.top <= wide.top and body.bottom >= wide.bottom
    assert (body.top - body.bottom) < (wide.top - wide.bottom)


def test_zone_candle_runs_in_the_breakers_direction():
    o, h, l, c = random_ohlc(4000, seed=7)
    for b in detect_breakers(o, h, l, c, length=5):
        is_green = c[b.source_bar] > o[b.source_bar]
        assert is_green == (b.direction == BULL)


def test_single_active_zone_supersedes_the_previous_one():
    o, h, l, c = random_ohlc(4000, seed=3)
    blocks = detect_breakers(o, h, l, c, length=5, single_active_zone=True)
    assert len(blocks) > 2
    for earlier, later in zip(blocks[:-1], blocks[1:]):
        assert earlier.superseded_bar == later.appeared_bar
    assert blocks[-1].superseded_bar is None

    kept = detect_breakers(o, h, l, c, length=5, single_active_zone=False)
    assert all(b.superseded_bar is None for b in kept)


def test_zones_are_never_live_on_their_own_appearance_bar():
    o, h, l, c = random_ohlc(3000, seed=11)
    for b in detect_breakers(o, h, l, c, length=5):
        assert not b.is_live_at(b.appeared_bar)
        assert b.is_live_at(b.appeared_bar + 1) or b.dead_bar() is not None


# ---------------------------------------------------------------------------
# The post-draw state machine, exercised directly
# ---------------------------------------------------------------------------
def make_block(direction=BULL, top=101.0, bottom=100.0, appeared=10):
    return Breaker(direction=direction, top=top, bottom=bottom,
                   source_bar=5, appeared_bar=appeared,
                   swing_broken_price=102.0, structural_bar=1)


def test_bullish_zone_is_mitigated_by_a_close_below_it():
    b = make_block()
    _step_state(b, 11, o=100.8, h=101.0, l=99.5, c=99.8, till_first_break=True)
    assert b.mitigated_bar == 11


def test_bullish_zone_is_cancelled_by_a_close_under_the_midline():
    b = make_block()
    _step_state(b, 11, o=100.9, h=101.0, l=100.1, c=100.2, till_first_break=True)
    assert b.broken_bar == 11 and b.mitigated_bar is None


def test_a_close_in_the_upper_half_leaves_the_zone_alone():
    b = make_block()
    _step_state(b, 11, o=100.6, h=101.0, l=100.4, c=100.9, till_first_break=True)
    assert b.broken_bar is None and b.mitigated_bar is None


def test_signal_up_fires_on_a_close_back_above_the_zone():
    b = make_block()
    _step_state(b, 11, o=100.7, h=101.4, l=100.6, c=101.3, till_first_break=True)
    assert b.signal_bars == [11]


def test_mitigation_freezes_all_later_state():
    b = make_block()
    _step_state(b, 11, o=100.8, h=101.0, l=99.5, c=99.8, till_first_break=True)
    _step_state(b, 12, o=100.0, h=101.5, l=99.9, c=101.4, till_first_break=True)
    assert b.signal_bars == [] and b.mitigated_bar == 11


def test_till_first_break_false_allows_a_reset():
    b = make_block()
    _step_state(b, 11, o=100.9, h=101.0, l=100.1, c=100.2, till_first_break=False)
    assert b.broken_bar == 11
    _step_state(b, 12, o=100.3, h=101.5, l=100.2, c=101.4, till_first_break=False)
    assert b.broken_bar is None, "a close back above the top revives the zone"


def test_bearish_state_machine_mirrors():
    b = make_block(direction=BEAR)
    _step_state(b, 11, o=100.2, h=101.5, l=100.1, c=101.4, till_first_break=True)
    assert b.mitigated_bar == 11

    b2 = make_block(direction=BEAR)
    _step_state(b2, 11, o=100.1, h=100.9, l=100.0, c=100.8, till_first_break=True)
    assert b2.broken_bar == 11
