"""
Bad-tick detection.

The whole value of this module is telling a phantom print apart from a real
shock, so both cases are tested explicitly - using the two real CHFJPY bars
that motivated it.
"""

import numpy as np
import pandas as pd
import pytest

from qw.data.quality import clip_suspect_bars, find_suspect_bars


def calm(n=400, price=115.0, rng_size=0.05):
    """A quiet series with a steady small range."""
    t = pd.date_range("2015-01-01", periods=n, freq="15min")
    close = price + np.sin(np.arange(n) / 20.0) * 0.1
    return pd.DataFrame({
        "time_utc": t,
        "open": close, "close": close,
        "high": close + rng_size, "low": close - rng_size,
    })


def test_a_phantom_spike_is_flagged():
    # The real one: 2015-01-14 22:00 CHFJPY printed 139.043 between bars at ~115.
    df = calm()
    i = 200
    df.loc[i, "high"] = 139.043
    df.loc[i, "low"] = 113.820

    s = find_suspect_bars(df)
    assert len(s) == 1
    assert int(s.iloc[0]["bar"]) == i
    assert s.iloc[0]["range_x_median"] > 50
    assert s.iloc[0]["persisted_frac"] < 0.25


def test_a_real_shock_is_left_alone():
    # The genuine unpeg: a huge bar, but price STAYS at the new level.
    df = calm()
    i = 200
    df.loc[i, "high"] = 142.95
    df.loc[i, "close"] = 139.77
    df.loc[i:, "close"] = df.loc[i:, "close"] + 24.0
    df.loc[i:, "high"] = df.loc[i:, "close"] + 0.05
    df.loc[i:, "low"] = df.loc[i:, "close"] - 0.05
    df.loc[i, "low"] = 114.99
    df.loc[i, "high"] = 142.95

    s = find_suspect_bars(df)
    flagged = set(s["bar"].tolist()) if len(s) else set()
    assert i not in flagged, "a move that persists is not a bad tick"


def test_a_quiet_series_flags_nothing():
    assert find_suspect_bars(calm()).empty


def test_short_series_is_handled():
    assert find_suspect_bars(calm(n=10)).empty


def test_clipping_trims_only_the_phantom_extreme():
    df = calm()
    i = 200
    df.loc[i, "high"] = 139.043
    df.loc[i, "low"] = 113.820
    before = df.copy()

    s = find_suspect_bars(df)
    out = clip_suspect_bars(df, s)

    assert out.loc[i, "high"] < 116.0, "phantom high pulled back"
    assert out.loc[i, "low"] > 114.0, "phantom low pulled back"
    assert len(out) == len(before), "bar count must not change"
    # every other bar untouched
    others = [j for j in range(len(df)) if j != i]
    assert np.allclose(out.loc[others, "high"], before.loc[others, "high"])
    assert np.allclose(out.loc[others, "low"], before.loc[others, "low"])


def test_clipping_never_inverts_a_bar():
    df = calm()
    for i in (120, 200, 310):
        df.loc[i, "high"] = 139.0
        df.loc[i, "low"] = 100.0
    out = clip_suspect_bars(df, find_suspect_bars(df))
    assert (out["high"] >= out["low"]).all()
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()


def test_clipping_with_nothing_flagged_is_a_no_op():
    df = calm()
    out = clip_suspect_bars(df, pd.DataFrame())
    pd.testing.assert_frame_equal(df, out)
