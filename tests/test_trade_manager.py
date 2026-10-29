"""
Handcrafted price paths with outcomes worked out by hand, both directions,
both management methods.

Geometry used throughout: pip = 0.01, entry 100.00, stop 15 pips away, so
risk = 0.15 and the 2R target sits 30 pips away. The safe method's +10 pip
partial therefore lands at 100.10 (long) / 99.90 (short).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_m15
from qw.backtest.trade_manager import LONG, SHORT, M1Index, build_fill, simulate

PIP = 0.01
SAFE = {"partial_trigger_pips": 10.0, "partial_fraction": 0.5,
        "move_to_breakeven": True, "be_includes_costs": True}
BASELINE = {"partial_trigger_pips": None, "partial_fraction": 0.0,
            "move_to_breakeven": False}

LEAD = [(100.5, 100.6, 100.4, 100.5),
        (100.4, 100.5, 100.3, 100.4),
        (100.0, 100.05, 99.95, 100.0)]      # bar 2 = entry bar, nothing triggers


def long_fill(spread=0.0):
    return build_fill(LONG, 2, pd.Timestamp("2024-01-02 12:30"),
                      zone_edge=100.00, struct_level=99.85,
                      spread_price=spread, pip=PIP,
                      sl_buffer_pips=0.0, risk_reward=2.0)


def short_fill(spread=0.0):
    return build_fill(SHORT, 2, pd.Timestamp("2024-01-02 12:30"),
                      zone_edge=100.00, struct_level=100.15,
                      spread_price=spread, pip=PIP,
                      sl_buffer_pips=0.0, risk_reward=2.0)


def run(fill, bars, method, m1=None):
    df = make_m15(LEAD + bars)
    return simulate(fill, df, M1Index(m1), method)


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------
def test_long_pays_the_spread_at_entry():
    f = long_fill(spread=0.0002)
    assert f.entry_price == pytest.approx(100.0002)
    assert f.stop == pytest.approx(99.85)
    assert f.risk == pytest.approx(0.1502)
    assert f.target == pytest.approx(100.0002 + 2 * 0.1502)


def test_short_pays_the_spread_on_the_stop_side():
    f = short_fill(spread=0.0002)
    assert f.entry_price == pytest.approx(100.00), "short fills at the bid"
    assert f.risk == pytest.approx(0.15)
    assert f.target == pytest.approx(99.70)


def test_sl_buffer_widens_the_stop():
    f = build_fill(LONG, 2, None, 100.0, 99.85, 0.0, PIP,
                   sl_buffer_pips=2.0, risk_reward=2.0)
    assert f.stop == pytest.approx(99.83)
    assert f.risk == pytest.approx(0.17)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def test_baseline_long_target():
    out = run(long_fill(), [(100.0, 100.35, 99.98, 100.3)], BASELINE)
    assert out.exit_reason == "target"
    assert out.r_multiple == pytest.approx(2.0)
    assert out.pips == pytest.approx(30.0)
    assert out.partial_time is None


def test_baseline_long_stop():
    out = run(long_fill(), [(100.0, 100.02, 99.80, 99.85)], BASELINE)
    assert out.exit_reason == "stop"
    assert out.r_multiple == pytest.approx(-1.0)
    assert out.pips == pytest.approx(-15.0)


def test_baseline_short_target():
    out = run(short_fill(), [(100.0, 100.02, 99.65, 99.70)], BASELINE)
    assert out.exit_reason == "target"
    assert out.r_multiple == pytest.approx(2.0)


def test_baseline_short_stop():
    out = run(short_fill(), [(100.0, 100.20, 99.98, 100.15)], BASELINE)
    assert out.exit_reason == "stop"
    assert out.r_multiple == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Safe method
# ---------------------------------------------------------------------------
def test_safe_long_partial_then_breakeven():
    out = run(long_fill(), [
        (100.0, 100.12, 99.98, 100.05),   # +10 pips -> half off, stop to BE
        (100.05, 100.06, 99.99, 100.0),   # drifts back through entry
    ], SAFE)
    assert out.partial_time is not None
    assert out.exit_reason == "stop"
    assert out.exit_price == pytest.approx(100.0), "stop sits at breakeven"
    # half at +10 pips on 15 pips of risk, half at zero
    assert out.r_multiple == pytest.approx(0.5 * (10 / 15))


def test_safe_long_partial_then_target():
    out = run(long_fill(), [
        (100.0, 100.12, 99.98, 100.05),
        (100.05, 100.35, 100.02, 100.30),
    ], SAFE)
    assert out.exit_reason == "target"
    assert out.r_multiple == pytest.approx(0.5 * (10 / 15) + 0.5 * 2.0)


def test_safe_short_partial_then_breakeven():
    out = run(short_fill(), [
        (100.0, 100.02, 99.88, 99.95),
        (99.95, 100.01, 99.94, 100.0),
    ], SAFE)
    assert out.partial_time is not None
    assert out.exit_reason == "stop"
    assert out.r_multiple == pytest.approx(0.5 * (10 / 15))


def test_safe_stop_before_partial_is_a_full_loss():
    out = run(long_fill(), [(100.0, 100.05, 99.80, 99.85)], SAFE)
    assert out.partial_time is None
    assert out.r_multiple == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------
def test_without_m1_an_ambiguous_bar_assumes_the_stop():
    # The bar spans both 99.85 and 100.30. OHLC cannot say which came first.
    out = run(long_fill(), [(100.0, 100.35, 99.80, 100.0)], BASELINE)
    assert out.exit_reason == "stop"
    assert out.r_multiple == pytest.approx(-1.0)
    assert out.intrabar_source == "M15"


def test_m1_resolves_an_ambiguous_bar_in_the_trades_favour():
    bar_start = pd.Timestamp("2024-01-02 12:45")      # bar index 3
    m1 = pd.DataFrame({
        "time_utc": [bar_start + pd.Timedelta(minutes=k) for k in range(15)],
        "open": 100.0, "close": 100.0,
        # target first, stop only later in the bar
        "high": [100.35] + [100.0] * 14,
        "low": [100.0] * 7 + [99.80] + [100.0] * 7,
        "_spread_price": 0.0,
    })
    out = run(long_fill(), [(100.0, 100.35, 99.80, 100.0)], BASELINE, m1=m1)
    assert out.exit_reason == "target"
    assert out.r_multiple == pytest.approx(2.0)
    assert out.intrabar_source == "M1"


def test_a_single_m1_bar_holding_both_still_assumes_the_stop():
    bar_start = pd.Timestamp("2024-01-02 12:45")
    m1 = pd.DataFrame({
        "time_utc": [bar_start + pd.Timedelta(minutes=k) for k in range(15)],
        "open": 100.0, "close": 100.0,
        "high": [100.35] + [100.0] * 14,
        "low": [99.80] + [100.0] * 14,       # same minute holds both
        "_spread_price": 0.0,
    })
    out = run(long_fill(), [(100.0, 100.35, 99.80, 100.0)], BASELINE, m1=m1)
    assert out.exit_reason == "stop"
    assert out.r_multiple == pytest.approx(-1.0)


def test_position_still_open_at_the_end_is_marked():
    out = run(long_fill(), [(100.0, 100.02, 99.99, 100.0)], BASELINE)
    assert out.exit_reason == "end_of_data"


def test_intrabar_source_flags_a_mixed_trade():
    # M1 covers bar 3 but not bar 4, and the trade resolves on bar 4.
    bar_start = pd.Timestamp("2024-01-02 12:45")
    m1 = pd.DataFrame({
        "time_utc": [bar_start + pd.Timedelta(minutes=k) for k in range(15)],
        "open": 100.0, "close": 100.0,
        "high": [100.12] + [100.0] * 14,
        "low": 100.0,
        "_spread_price": 0.0,
    })
    out = run(long_fill(), [
        (100.0, 100.12, 99.98, 100.05),     # partial, resolved on M1
        (100.05, 100.06, 99.99, 100.0),     # BE stop, resolved on M15
    ], SAFE, m1=m1)
    assert out.intrabar_source == "M1/M15"


# ---------------------------------------------------------------------------
# Stop must sit beyond the breaker block
# ---------------------------------------------------------------------------
def test_long_stop_is_pushed_below_the_zone_when_structure_sits_inside_it():
    # Zone 99.90-100.00, structural low 99.95 - inside the zone.
    f = build_fill(LONG, 2, None, zone_edge=100.00, struct_level=99.95,
                   spread_price=0.0, pip=PIP, sl_buffer_pips=0.0,
                   risk_reward=2.0, zone_far_edge=99.90)
    assert f.stop == pytest.approx(99.90), "stop must clear the whole block"
    assert f.risk == pytest.approx(0.10)


def test_short_stop_is_pushed_above_the_zone_when_structure_sits_inside_it():
    f = build_fill(SHORT, 2, None, zone_edge=100.00, struct_level=100.05,
                   spread_price=0.0, pip=PIP, sl_buffer_pips=0.0,
                   risk_reward=2.0, zone_far_edge=100.10)
    assert f.stop == pytest.approx(100.10)
    assert f.risk == pytest.approx(0.10)


def test_structure_beyond_the_zone_still_wins():
    # Structural low 99.70 is already below the zone bottom 99.90 - keep it.
    f = build_fill(LONG, 2, None, 100.00, 99.70, 0.0, PIP, 0.0, 2.0,
                   zone_far_edge=99.90)
    assert f.stop == pytest.approx(99.70)

    f2 = build_fill(SHORT, 2, None, 100.00, 100.30, 0.0, PIP, 0.0, 2.0,
                    zone_far_edge=100.10)
    assert f2.stop == pytest.approx(100.30)


def test_buffer_applies_on_top_of_the_zone_edge():
    f = build_fill(LONG, 2, None, 100.00, 99.95, 0.0, PIP, 2.0, 2.0,
                   zone_far_edge=99.90)
    assert f.stop == pytest.approx(99.88), "zone edge minus the 2 pip buffer"


def test_risk_is_never_smaller_than_the_zone_height():
    for struct in (99.99, 99.95, 99.91):
        f = build_fill(LONG, 2, None, 100.00, struct, 0.0, PIP, 0.0, 2.0,
                       zone_far_edge=99.90)
        assert f.risk > 0.10 - 1e-9, "entry at top, stop at or below bottom"


def test_without_the_rule_the_stop_can_land_inside_the_zone():
    """The old behaviour, kept reachable via config for comparison."""
    f = build_fill(LONG, 2, None, 100.00, 99.95, 0.0, PIP, 0.0, 2.0,
                   zone_far_edge=None)
    assert f.stop == pytest.approx(99.95)


# ---------------------------------------------------------------------------
# R-based partial trigger
# ---------------------------------------------------------------------------
SAFE_R = {"partial_trigger_r": 1.0, "partial_fraction": 0.5,
          "move_to_breakeven": True, "be_includes_costs": True}


def test_partial_trigger_r_scales_with_the_stop():
    """+1R must mean 15 pips on a 15 pip stop, not a fixed distance."""
    out = run(long_fill(), [
        (100.0, 100.16, 99.98, 100.10),   # +15 pips = +1R -> partial
        (100.10, 100.12, 99.99, 100.0),   # back to breakeven
    ], SAFE_R)
    assert out.partial_time is not None
    assert out.partial_price == pytest.approx(100.15), "entry 100.00 + 1R (0.15)"
    assert out.r_multiple == pytest.approx(0.5 * 1.0)


def test_partial_trigger_r_on_a_wider_stop_moves_the_level():
    wide = build_fill(LONG, 2, pd.Timestamp("2024-01-02 12:30"),
                      zone_edge=100.00, struct_level=99.70,   # 30 pip stop
                      spread_price=0.0, pip=PIP,
                      sl_buffer_pips=0.0, risk_reward=2.0)
    assert wide.risk == pytest.approx(0.30)
    out = run(wide, [
        (100.0, 100.31, 99.98, 100.30),
        (100.30, 100.32, 99.99, 100.0),
    ], SAFE_R)
    assert out.partial_price == pytest.approx(100.30), "+1R is now 30 pips away"


def test_r_trigger_takes_precedence_over_pips():
    method = dict(SAFE_R, partial_trigger_pips=5.0)
    out = run(long_fill(), [
        (100.0, 100.16, 99.98, 100.10),
        (100.10, 100.12, 99.99, 100.0),
    ], method)
    assert out.partial_price == pytest.approx(100.15), "R wins, not the 5 pips"


def test_no_partial_when_neither_trigger_is_set():
    method = {"partial_trigger_r": None, "partial_trigger_pips": None,
              "partial_fraction": 0.5, "move_to_breakeven": True}
    out = run(long_fill(), [(100.0, 100.35, 99.98, 100.30)], method)
    assert out.partial_time is None
    assert out.r_multiple == pytest.approx(2.0)


def test_r_partial_then_target():
    out = run(long_fill(), [
        (100.0, 100.16, 99.98, 100.10),    # +1R partial
        (100.10, 100.35, 100.02, 100.30),  # runs to 2R
    ], SAFE_R)
    assert out.exit_reason == "target"
    assert out.r_multiple == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)
