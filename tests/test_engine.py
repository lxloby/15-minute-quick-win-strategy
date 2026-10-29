"""
End-to-end checks on the backtest loop: filters, the one-position rule, and
the shape of what lands in trades.csv.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from conftest import make_m15, random_ohlc
from qw.backtest import metrics
from qw.backtest.engine import run_symbol
from qw.config import load_config
from qw.data.filters import NewsFilter, session_mask, symbol_currencies


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["filters"]["session"]["enabled"] = False
    c["filters"]["news"]["enabled"] = False
    c["strategy"]["sl_distance_filter"]["enabled"] = False
    return c


@pytest.fixture(scope="module")
def synthetic():
    o, h, l, c = random_ohlc(6000, seed=5, start=1.1000, vol=0.0015)
    df = make_m15(list(zip(o, h, l, c)))
    df["spread"] = 10.0          # 10 points = 1.0 pip on a 5-digit symbol
    return df


def test_engine_produces_well_formed_trades(cfg, synthetic):
    trades, diag = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    assert diag["breakers"] > 0
    assert not trades.empty, "synthetic series should generate some trades"

    assert (trades["sl_pips"] > 0).all()
    assert trades["direction"].isin({"long", "short"}).all()
    assert trades["intrabar_source"].isin({"M15", "M1", "M1/M15", "none"}).all()
    assert (trades["entry_time"] <= trades["exit_time"]).all()
    assert (trades["entry_bar"] <= trades["exit_bar"]).all()


def test_entries_never_precede_the_zone(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    assert (trades["entry_time"] > trades["signal_time"]).all()
    assert (trades["signal_time"] >= trades["break_time"]).all()


def test_stop_and_target_sit_the_right_side_of_entry(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    longs = trades[trades["direction"] == "long"]
    shorts = trades[trades["direction"] == "short"]
    assert (longs["sl"] < longs["entry_price"]).all()
    assert (longs["tp"] > longs["entry_price"]).all()
    assert (shorts["sl"] > shorts["entry_price"]).all()
    assert (shorts["tp"] < shorts["entry_price"]).all()


def test_only_one_position_is_ever_open(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    ordered = trades.sort_values("entry_bar")
    prev_exit = -1
    for _, row in ordered.iterrows():
        assert row["entry_bar"] > prev_exit, "a trade opened before the last one closed"
        prev_exit = row["exit_bar"]


def test_each_zone_is_used_at_most_once(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    assert trades["zone_id"].is_unique


def test_baseline_outcomes_are_only_ever_minus_one_or_plus_two(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    closed = trades[trades["exit_reason"].isin({"stop", "target"})]
    assert not closed.empty
    r = closed["r_multiple"].round(6)
    assert set(r.unique()) <= {-1.0, 2.0}


def test_safe_method_changes_the_outcome_distribution(cfg, synthetic):
    base, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    safe, _ = run_symbol(cfg, "EURUSD", synthetic, None, "safe")
    assert len(safe) > 0
    assert (safe["partial_time"].notna()).any(), "some trades should bank a partial"
    assert set(safe["r_multiple"].round(6).unique()) != {-1.0, 2.0}


def test_sl_filter_only_keeps_trades_inside_the_band(cfg, synthetic):
    c = copy.deepcopy(cfg)
    c["strategy"]["sl_distance_filter"] = {"enabled": True, "min_pips": 10.0,
                                           "max_pips": 20.0}
    trades, diag = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    if not trades.empty:
        assert trades["sl_pips"].between(10.0, 20.0).all()
    assert diag["rejected_sl_size"] > 0


def test_session_filter_removes_out_of_window_entries(cfg, synthetic):
    c = copy.deepcopy(cfg)
    c["filters"]["session"] = {"enabled": True, "start_utc": "12:00",
                               "end_utc": "23:00", "manage_outside_window": True}
    trades, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    if not trades.empty:
        hours = pd.DatetimeIndex(trades["entry_time"]).hour
        assert hours.min() >= 12 and hours.max() < 23


def test_spread_is_actually_charged(cfg, synthetic):
    wide = synthetic.copy()
    wide["spread"] = 300.0          # 30 pips
    a, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    b, _ = run_symbol(cfg, "EURUSD", wide, None, "baseline")
    assert b["sl_pips"].mean() > a["sl_pips"].mean(), (
        "a wider spread must widen the risk distance on longs"
    )


# ---------------------------------------------------------------------------
# Filters in isolation
# ---------------------------------------------------------------------------
def test_session_mask_handles_a_plain_window():
    idx = pd.date_range("2024-01-02", periods=96, freq="15min")
    m = session_mask(idx, "12:00", "23:00")
    assert m[idx.hour == 12].all()
    assert not m[idx.hour == 23].any()
    assert not m[idx.hour == 11].any()


def test_session_mask_handles_a_window_over_midnight():
    idx = pd.date_range("2024-01-02", periods=96, freq="15min")
    m = session_mask(idx, "22:00", "06:00")
    assert m[idx.hour == 23].all() and m[idx.hour == 2].all()
    assert not m[idx.hour == 12].any()


def test_symbol_currencies():
    assert symbol_currencies("EURJPY") == ["EUR", "JPY"]
    assert symbol_currencies("CHFJPY") == ["CHF", "JPY"]


def test_news_filter_blocks_the_window_either_side():
    ev = pd.DataFrame({
        "utc_time": [pd.Timestamp("2024-01-05 13:30")],
        "currency": ["USD"], "impact": ["High"], "title": ["NFP"],
    })
    nf = NewsFilter(ev, 30, 30, ["High"], ["EUR", "USD", "JPY", "CHF"])
    times = pd.to_datetime([
        "2024-01-05 12:55",   # outside
        "2024-01-05 13:00",   # exactly 30 min before -> blocked
        "2024-01-05 13:30",   # on the event
        "2024-01-05 14:00",   # exactly 30 min after -> blocked
        "2024-01-05 14:05",   # outside
    ])
    assert list(nf.blocked_mask(times, "EURUSD")) == [False, True, True, True, False]
    # JPY/CHF pair is untouched by a USD event
    assert not nf.blocked_mask(times, "CHFJPY").any()


def test_metrics_round_trip(cfg, synthetic):
    trades, _ = run_symbol(cfg, "EURUSD", synthetic, None, "baseline")
    s = metrics.summarise(trades, 1.0)
    assert s["trades"] == len(trades)
    assert s["wins"] + s["losses"] <= s["trades"]
    assert s["max_dd_r"] >= 0 and s["max_dd_pct"] >= 0
    table = metrics.summary_table(trades, 1.0)
    assert {"method", "symbol", "direction"} <= set(table.columns)
    assert (table["symbol"] == "ALL").any()


def test_engine_keeps_every_stop_beyond_the_zone(cfg, synthetic):
    """With sl_beyond_zone on, no stop may sit inside the block."""
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["strategy"]["sl_beyond_zone"] = True
    trades, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    assert not trades.empty
    longs = trades[trades["direction"] == "long"]
    shorts = trades[trades["direction"] == "short"]
    assert (longs["sl"] <= longs["zone_bottom"] + 1e-12).all()
    assert (shorts["sl"] >= shorts["zone_top"] - 1e-12).all()


def test_the_rule_only_ever_widens_a_stop(cfg, synthetic):
    """
    Compared trade by trade, never in aggregate: widening a stop also moves
    the target, which changes the exit bar and therefore which setups the
    one-position-at-a-time rule allows afterwards. The two runs produce
    different trade populations, so only trades common to both are comparable.
    """
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["strategy"]["sl_beyond_zone"] = False
    off, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    c["strategy"]["sl_beyond_zone"] = True
    on, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")

    key = ["zone_id", "entry_bar", "direction"]
    both = off.merge(on, on=key, suffixes=("_off", "_on"))
    assert len(both) > 5, "expected a decent overlap between the two runs"

    longs = both[both["direction"] == "long"]
    shorts = both[both["direction"] == "short"]
    assert (longs["sl_on"] <= longs["sl_off"] + 1e-12).all()
    assert (shorts["sl_on"] >= shorts["sl_off"] - 1e-12).all()
    assert (both["sl_pips_on"] >= both["sl_pips_off"] - 1e-9).all()


# ---------------------------------------------------------------------------
# Day-trade constraints
# ---------------------------------------------------------------------------
def test_stale_break_never_pairs_with_a_late_breaker(cfg, synthetic):
    import copy as _copy
    import numpy as np
    from qw.backtest.engine import prepare_symbol

    # The break-to-zone cap only exists in 'both' mode; 'indicator_only'
    # has no separate break to wait from.
    loose = _copy.deepcopy(cfg)
    loose["strategy"]["entry_mode"] = "both"
    loose["strategy"]["max_bars_break_to_zone"] = None
    tight = _copy.deepcopy(cfg)
    tight["strategy"]["entry_mode"] = "both"
    tight["strategy"]["max_bars_break_to_zone"] = 20

    a = prepare_symbol(loose, "EURUSD", synthetic)["setups"]
    b = prepare_symbol(tight, "EURUSD", synthetic)["setups"]
    assert len(b) < len(a), "a cap must remove some setups"
    assert all(s.zone_appeared_bar - s.break_bar <= 20 for s in b)


def test_zone_expires_if_untouched(cfg, synthetic):
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["strategy"]["zone_expiry"] = "bars"
    c["strategy"]["max_bars_to_entry"] = 10
    trades, diag = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    assert diag["rejected_stale_zone"] > 0
    waited = ((pd.DatetimeIndex(trades["entry_time"])
               - pd.DatetimeIndex(trades["signal_time"]))
              .total_seconds() / 900).round()
    assert (waited <= 10).all()


def test_time_stop_caps_holding_period(cfg, synthetic):
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["management"]["max_bars_in_trade"] = 20
    trades, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    assert (trades["bars_held"] <= 20).all()
    assert (trades["exit_reason"] == "time_stop").any(), "some trades should time out"


def test_time_stop_exit_is_marked_apart_from_end_of_data(cfg, synthetic):
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["management"]["max_bars_in_trade"] = 20
    trades, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    timed = trades[trades["exit_reason"] == "time_stop"]
    assert not timed.empty
    # a time stop exits at a close, so it is neither the stop nor the target
    assert (timed["exit_price"] != timed["sl"]).all()
    assert (timed["exit_price"] != timed["tp"]).all()


# ---------------------------------------------------------------------------
# indicator_only mode, SMA at the tap, session-end expiry
# ---------------------------------------------------------------------------
def test_indicator_only_yields_more_setups_than_requiring_both(cfg, synthetic):
    import copy as _copy
    from qw.backtest.engine import prepare_symbol

    both = _copy.deepcopy(cfg); both["strategy"]["entry_mode"] = "both"
    only = _copy.deepcopy(cfg); only["strategy"]["entry_mode"] = "indicator_only"

    a = prepare_symbol(both, "EURUSD", synthetic)["setups"]
    b = prepare_symbol(only, "EURUSD", synthetic)["setups"]
    assert len(b) > len(a), "dropping the second structure test must widen the net"
    # every setup still points at a real zone
    assert all(s.zone_top > s.zone_bottom for s in b)
    assert all(s.struct_bar < s.zone_appeared_bar for s in b)


def test_sma_at_tap_filters_on_the_previous_bar(cfg, synthetic):
    import copy as _copy
    import numpy as np
    from qw.indicators.sma import sma as sma_fn

    c = _copy.deepcopy(cfg)
    c["strategy"]["sma_check"] = "tap"
    trades, diag = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    assert diag["rejected_sma"] > 0, "some taps should fail the SMA test"

    sma_vals = sma_fn(synthetic["close"].to_numpy(float), 44)
    for _, r in trades.iterrows():
        prev = sma_vals[int(r["entry_bar"]) - 1]
        edge = r["zone_top"] if r["direction"] == "long" else r["zone_bottom"]
        if r["direction"] == "long":
            assert edge > prev
        else:
            assert edge < prev


def test_session_end_expiry_kills_untapped_zones_the_same_day(cfg, synthetic):
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["strategy"]["zone_expiry"] = "session_end"
    c["filters"]["session"]["end_utc"] = "23:00"
    trades, diag = run_symbol(c, "EURUSD", synthetic, None, "baseline")
    assert diag["rejected_stale_zone"] > 0

    sig = pd.DatetimeIndex(trades["signal_time"])
    ent = pd.DatetimeIndex(trades["entry_time"])
    # entry must fall before the first 23:00 that follows the signal
    same = sig.normalize() + pd.Timedelta(hours=23)
    cutoff = np.where(sig.to_numpy() < same.to_numpy(), same.to_numpy(),
                      (same + pd.Timedelta(days=1)).to_numpy())
    assert (ent.to_numpy() < cutoff).all()


def test_zone_expiry_modes_differ(cfg, synthetic):
    import copy as _copy
    out = {}
    for mode in ("session_end", "bars", None):
        c = _copy.deepcopy(cfg)
        c["strategy"]["zone_expiry"] = mode
        c["strategy"]["max_bars_to_entry"] = 24
        t, _ = run_symbol(c, "EURUSD", synthetic, None, "baseline")
        out[str(mode)] = len(t)
    assert out["None"] >= out["bars"], "no expiry should allow the most entries"
    assert out["session_end"] > 0


def test_news_filter_reports_currencies_it_has_no_events_for():
    """Adding a pair without rebuilding the calendar must not pass silently."""
    ev = pd.DataFrame({
        "utc_time": [pd.Timestamp("2024-01-05 13:30")],
        "currency": ["USD"], "impact": ["High"], "title": ["NFP"],
    })
    nf = NewsFilter(ev, 30, 30, ["High"], ["EUR", "USD", "JPY", "CHF", "GBP"])
    gaps = nf.missing_currencies(["EURUSD", "GBPJPY", "USDCAD"])
    assert gaps["GBPJPY"] == ["GBP", "JPY"]
    assert gaps["USDCAD"] == ["CAD"]
    assert "EURUSD" in gaps and gaps["EURUSD"] == ["EUR"]


def test_no_gap_reported_when_the_calendar_covers_the_pair():
    ev = pd.DataFrame({
        "utc_time": [pd.Timestamp("2024-01-05 13:30"), pd.Timestamp("2024-01-06 09:00")],
        "currency": ["USD", "EUR"], "impact": ["High", "High"],
        "title": ["NFP", "CPI"],
    })
    nf = NewsFilter(ev, 30, 30, ["High"], ["EUR", "USD"])
    assert nf.missing_currencies(["EURUSD"]) == {}


# ---------------------------------------------------------------------------
# stop_reference
# ---------------------------------------------------------------------------
def _run_ref(cfg, synthetic, ref, buffer=0.0, beyond=True):
    import copy as _copy
    c = _copy.deepcopy(cfg)
    c["strategy"]["stop_reference"] = ref
    c["strategy"]["sl_buffer_pips"] = buffer
    c["strategy"]["sl_beyond_zone"] = beyond
    return run_symbol(c, "EURUSD", synthetic, None, "baseline")[0]


def test_zone_reference_puts_the_stop_at_the_zone_edge(cfg, synthetic):
    t = _run_ref(cfg, synthetic, "zone")
    longs = t[t["direction"] == "long"]
    shorts = t[t["direction"] == "short"]
    assert np.allclose(longs["sl"], longs["zone_bottom"])
    assert np.allclose(shorts["sl"], shorts["zone_top"])


def test_zone_reference_gives_tighter_stops_than_structure(cfg, synthetic):
    z = _run_ref(cfg, synthetic, "zone")
    s = _run_ref(cfg, synthetic, "structure")
    assert z["sl_pips"].median() < s["sl_pips"].median()


def test_buffer_widens_the_zone_reference(cfg, synthetic):
    a = _run_ref(cfg, synthetic, "zone", buffer=0.0)
    b = _run_ref(cfg, synthetic, "zone", buffer=5.0)
    key = ["zone_id", "entry_bar", "direction"]
    both = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(both) > 5
    assert (both["sl_pips_b"] > both["sl_pips_a"]).all()
    assert np.allclose(both["sl_pips_b"] - both["sl_pips_a"], 5.0)


def test_indicator_swing_reference_runs_and_is_recorded(cfg, synthetic):
    t = _run_ref(cfg, synthetic, "indicator_swing")
    assert not t.empty
    assert (t["stop_reference"] == "indicator_swing").all()
    longs = t[t["direction"] == "long"]
    assert (longs["sl"] <= longs["zone_bottom"] + 1e-12).all()


def test_stop_reference_is_recorded_on_every_trade(cfg, synthetic):
    for ref in ("structure", "zone", "indicator_swing"):
        t = _run_ref(cfg, synthetic, ref)
        assert (t["stop_reference"] == ref).all()


def test_every_reference_still_puts_the_stop_the_right_side_of_entry(cfg, synthetic):
    for ref in ("structure", "zone", "indicator_swing"):
        t = _run_ref(cfg, synthetic, ref)
        longs = t[t["direction"] == "long"]
        shorts = t[t["direction"] == "short"]
        assert (longs["sl"] < longs["entry_price"]).all()
        assert (shorts["sl"] > shorts["entry_price"]).all()
        assert (t["sl_pips"] > 0).all()
