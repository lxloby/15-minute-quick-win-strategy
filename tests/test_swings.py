import numpy as np

from qw.indicators.swings import HIGH, LOW, SwingTracker, confirmed_pivots, pivot_flags


def test_pivot_high_is_confirmed_one_bar_late():
    # Peak at index 5; left=5 needs bars 0-4 below it, right=1 needs bar 6 below.
    high = np.array([1, 2, 3, 4, 5, 9, 4, 3, 2, 1, 2, 3], float)
    low = high - 1.0
    ph, pl = pivot_flags(high, low, left=5, right=1)

    assert np.isnan(ph[5]), "a pivot cannot be known on its own bar"
    assert ph[6] == 9.0, "pivot high confirms exactly one bar later"
    assert np.count_nonzero(~np.isnan(ph)) == 1


def test_pivot_low_mirrors():
    low = np.array([9, 8, 7, 6, 5, 1, 6, 7, 8, 9, 8, 7], float)
    high = low + 1.0
    ph, pl = pivot_flags(high, low, left=5, right=1)
    assert pl[6] == 1.0
    assert np.isnan(pl[5])


def test_insufficient_left_bars_yields_nothing():
    high = np.array([1, 5, 1], float)
    ph, pl = pivot_flags(high, high - 1, left=5, right=1)
    assert np.all(np.isnan(ph))


def test_strict_mode_rejects_a_tie():
    # Two equal highs next to each other: neither beats the other outright.
    high = np.array([1, 2, 3, 4, 5, 9, 9, 3, 2, 1, 2, 3], float)
    ph, _ = pivot_flags(high, high - 1, left=5, right=1, tie_mode="strict")
    assert np.all(np.isnan(ph))

    ph2, _ = pivot_flags(high, high - 1, left=5, right=1, tie_mode="left_eq")
    assert ph2[7] == 9.0, "left_eq lets the later of two equal highs count"


def test_confirmed_pivots_are_ordered_by_confirmation():
    high = np.array([1, 2, 3, 4, 5, 9, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 2], float)
    low = high - 1.0
    pivots = confirmed_pivots(high, low, 5, 1)
    bars = [p.confirmed_bar for p in pivots]
    assert bars == sorted(bars)
    for p in pivots:
        assert p.confirmed_bar == p.bar + 1


def test_tracker_only_admits_pivots_it_could_know_about():
    high = np.array([1, 2, 3, 4, 5, 9, 4, 3, 2, 1, 2, 3], float)
    pivots = confirmed_pivots(high, high - 1.0, 5, 1)
    tr = SwingTracker(pivots)

    tr.advance_to(5)
    assert tr.highs == [], "pivot at bar 5 is not knowable at bar 5"
    tr.advance_to(6)
    assert len(tr.highs) == 1 and tr.highs[0].bar == 5


def test_lower_high_and_higher_low():
    # Two confirmed highs, the second lower than the first.
    base = [1, 1, 1, 1, 1]
    seq = base + [9, 1] + base + [7, 1] + base
    high = np.array(seq, float)
    pivots = confirmed_pivots(high, high - 1.0, 5, 1)
    tr = SwingTracker(pivots)
    tr.advance_to(len(high) - 1)

    highs = [p.price for p in tr.highs]
    assert highs == [9.0, 7.0]
    lh = tr.lower_high()
    assert lh is not None and lh.price == 7.0

    # Flip it: a higher second high is not a lower high.
    seq2 = base + [7, 1] + base + [9, 1] + base
    high2 = np.array(seq2, float)
    tr2 = SwingTracker(confirmed_pivots(high2, high2 - 1.0, 5, 1))
    tr2.advance_to(len(high2) - 1)
    assert tr2.lower_high() is None


def test_structural_reference_lookup():
    low = np.array([9, 8, 7, 6, 5, 1, 6, 7, 8, 9, 8, 7], float)
    high = low + 1.0
    tr = SwingTracker(confirmed_pivots(high, low, 5, 1))
    tr.advance_to(11)
    ref = tr.last_low_before(11)
    assert ref is not None and ref.bar == 5 and ref.price == 1.0
    assert tr.last_low_before(5) is None


def test_left_eq_is_the_default_and_finds_more_pivots():
    """
    Verified against the live LuxAlgo indicator on TradingView, 2026-10-27:
    Pine's ta.pivothigh lets the centre bar EQUAL bars on its left. Using a
    strict comparison discarded ~11% of pivots and, because the zig-zag is
    path-dependent and MSS.dir latches, suppressed breaker zones for days.
    """
    # default must be the verified mode, not strict
    high = np.array([1, 2, 3, 4, 5, 9, 9, 3, 2, 1, 2, 3], float)
    ph_default, _ = pivot_flags(high, high - 1.0, left=5, right=1)
    ph_left_eq, _ = pivot_flags(high, high - 1.0, left=5, right=1, tie_mode="left_eq")
    np.testing.assert_array_equal(
        np.isnan(ph_default), np.isnan(ph_left_eq),
        err_msg="pivot_flags default must be left_eq")
    assert ph_default[7] == 9.0


def test_left_eq_never_finds_fewer_pivots_than_strict():
    """left_eq is a superset: it only ever adds pivots, never removes them."""
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 4000))
    # round hard, so exact ties actually occur
    high = np.round(close + 0.2, 2)
    low = np.round(close - 0.2, 2)

    ph_s, pl_s = pivot_flags(high, low, 5, 1, "strict")
    ph_l, pl_l = pivot_flags(high, low, 5, 1, "left_eq")

    strict_h = set(np.flatnonzero(~np.isnan(ph_s)))
    left_h = set(np.flatnonzero(~np.isnan(ph_l)))
    strict_l = set(np.flatnonzero(~np.isnan(pl_s)))
    left_l = set(np.flatnonzero(~np.isnan(pl_l)))

    assert strict_h <= left_h, "left_eq dropped a pivot high that strict found"
    assert strict_l <= left_l, "left_eq dropped a pivot low that strict found"
    assert len(left_h) > len(strict_h), "ties should give left_eq extra highs"
    assert len(left_l) > len(strict_l), "ties should give left_eq extra lows"
