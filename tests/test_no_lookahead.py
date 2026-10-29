"""
The lookahead guarantee, tested the only way that really settles it: run the
whole pipeline on the full series, run it again on a prefix, and demand that
everything the prefix could have known about is bit-identical.

A state change that happens at or after the cut must be absent from the
prefix run (not merely different) - that is the half of the test that catches
an indicator quietly peeking forward.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_m15, random_ohlc
from qw.indicators.breaker_blocks import detect_breakers
from qw.indicators.sma import sma
from qw.indicators.swings import confirmed_pivots, pivot_flags
from qw.strategy.signals import find_setups

CUTS = [500, 1200, 2500, 3300]
CREATION_FIELDS = ("direction", "top", "bottom", "source_bar",
                   "appeared_bar", "swing_broken_price", "structural_bar")
STATE_FIELDS = ("mitigated_bar", "broken_bar", "superseded_bar")


@pytest.fixture(scope="module")
def series():
    return random_ohlc(4000, seed=42)


def test_sma_prefix_is_unchanged(series):
    _, _, _, c = series
    full = sma(c, 44)
    for k in CUTS:
        np.testing.assert_allclose(sma(c[:k], 44), full[:k], equal_nan=True)


def test_pivot_flags_prefix_is_unchanged(series):
    _, h, l, _ = series
    ph, pl = pivot_flags(h, l, 5, 1)
    for k in CUTS:
        ph_k, pl_k = pivot_flags(h[:k], l[:k], 5, 1)
        np.testing.assert_allclose(ph_k, ph[:k], equal_nan=True)
        np.testing.assert_allclose(pl_k, pl[:k], equal_nan=True)


def test_confirmed_pivots_never_predate_their_confirmation(series):
    _, h, l, _ = series
    for p in confirmed_pivots(h, l, 5, 1):
        assert p.confirmed_bar > p.bar


def test_breakers_do_not_change_when_the_future_is_removed(series):
    o, h, l, c = series
    full = detect_breakers(o, h, l, c, length=5)

    for k in CUTS:
        prefix = detect_breakers(o[:k], h[:k], l[:k], c[:k], length=5)
        past = [b for b in full if b.appeared_bar < k]
        assert len(prefix) == len(past), (
            f"cut at {k}: prefix produced {len(prefix)} zones, "
            f"full history had {len(past)} up to that bar"
        )

        for a, b in zip(past, prefix):
            for f in CREATION_FIELDS:
                assert getattr(a, f) == getattr(b, f), f"{f} moved at cut {k}"
            for f in STATE_FIELDS:
                full_v, pre_v = getattr(a, f), getattr(b, f)
                if full_v is not None and full_v < k:
                    assert pre_v == full_v, (
                        f"{f} differs at cut {k}: {pre_v} vs {full_v}"
                    )
                else:
                    assert pre_v is None, (
                        f"{f} was set to {pre_v} at cut {k} but only happens "
                        f"at bar {full_v} - that is lookahead"
                    )
            assert [s for s in a.signal_bars if s < k] == b.signal_bars


def test_setups_do_not_change_when_the_future_is_removed(series):
    o, h, l, c = series
    df = make_m15(list(zip(o, h, l, c)))
    sma_full = sma(c, 44)
    full = find_setups(df, sma_full, detect_breakers(o, h, l, c, length=5),
                       pivot_length=5)

    for k in CUTS:
        dfk = df.iloc[:k].reset_index(drop=True)
        prefix = find_setups(dfk, sma(c[:k], 44),
                             detect_breakers(o[:k], h[:k], l[:k], c[:k], length=5),
                             pivot_length=5)
        past = [s for s in full if s.zone_appeared_bar < k]
        assert len(prefix) == len(past), f"setup count changed at cut {k}"

        for a, b in zip(past, prefix):
            assert (a.direction, a.break_bar, a.zone_appeared_bar,
                    a.zone_source_bar) == (b.direction, b.break_bar,
                                           b.zone_appeared_bar, b.zone_source_bar)
            assert a.break_level == pytest.approx(b.break_level)
            assert a.struct_level == pytest.approx(b.struct_level)
            assert a.zone_top == pytest.approx(b.zone_top)
            assert a.zone_bottom == pytest.approx(b.zone_bottom)


def test_a_setup_never_references_a_bar_after_its_own_signal(series):
    o, h, l, c = series
    df = make_m15(list(zip(o, h, l, c)))
    setups = find_setups(df, sma(c, 44), detect_breakers(o, h, l, c, length=5),
                         pivot_length=5)
    assert setups, "fixture should generate some setups"
    for s in setups:
        assert s.break_bar <= s.zone_appeared_bar
        assert s.zone_source_bar <= s.zone_appeared_bar
        assert s.struct_bar < s.break_bar
