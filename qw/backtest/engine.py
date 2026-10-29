"""
The backtest loop.

Strictly chronological: bar t is evaluated with nothing but bars <= t. A
setup becomes tappable the bar AFTER its zone appears, and a symbol holds at
most one position, so while a trade is live no new taps are considered. A
trade that exits on bar t does not allow a re-entry until t+1, which avoids
having to invent an intrabar ordering between an exit and the next fill.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import pip_size
from ..data.filters import NewsFilter, session_mask
from ..indicators.breaker_blocks import detect_breakers
from ..indicators.sma import sma as sma_fn
from ..strategy.signals import LONG, SHORT, find_setups
from .trade_manager import M1Index, build_fill, simulate

TRADE_COLUMNS = [
    "symbol", "direction", "method", "signal_time", "break_time",
    "entry_time", "entry_price", "sl", "tp", "sl_pips",
    "partial_time", "exit_time", "exit_price", "exit_reason",
    "r_multiple", "pips", "pnl", "intrabar_source",
    "zone_top", "zone_bottom", "zone_id", "stop_reference",
    "entry_bar", "exit_bar", "bars_held",
]


def add_spread_price(df: pd.DataFrame, point: float) -> pd.DataFrame:
    """Convert MT5's integer `spread` (points) into a price distance."""
    out = df.copy()
    spread = out["spread"].to_numpy(float) if "spread" in out else np.zeros(len(out))
    out["_spread_price"] = spread * point
    return out


def point_size(cfg, symbol: str, meta: dict | None = None) -> float:
    """
    Price value of one MT5 'point'. Taken from the symbol's digits when the
    fetch recorded them, else the 5-/3-digit convention of pip/10.
    """
    pip = pip_size(cfg, symbol)
    if meta and symbol in meta and meta[symbol].get("point"):
        return float(meta[symbol]["point"])
    return pip / 10.0


def commission_in_r(cfg, symbol: str, sl_pips: float, meta: dict | None = None) -> float:
    """
    Commission expressed as a fraction of the risk, which is the only form
    that survives into R-space.

        lots        = risk_cash / (sl_pips * pip_value_per_lot)
        commission  = lots * per_lot_round_turn
        in R        = commission / risk_cash
                    = per_lot_round_turn / (sl_pips * pip_value_per_lot)

    Equity drops out, so no account size is needed.
    """
    per_lot = float(cfg.dotted("costs.commission_per_lot_round_turn", 0.0) or 0.0)
    if per_lot == 0.0 or sl_pips <= 0:
        return 0.0
    pip_value = None
    if meta and symbol in meta:
        pip_value = meta[symbol].get("pip_value_per_lot")
    if not pip_value:
        pip_value = 10.0        # USD-quoted default; documented approximation
    return per_lot / (sl_pips * float(pip_value))


def _zone_expiry_bars(times, mode, fcfg_end="23:00", max_wait=None):
    """
    Last bar on which a zone that appeared at bar i may still be tapped.

    'session_end' is the day-trader's rule: a setup that has not been taken
    by the end of the session it appeared in is stale. The cutoff is the next
    `end_utc` at or after the zone's own bar, so a zone appearing at 14:00
    dies at 23:00 the same day, and one appearing at 23:30 dies at 23:00 the
    following day.
    """
    n = len(times)
    if mode == "bars" or mode is None:
        if mode is None or max_wait is None:
            return np.full(n, n - 1, dtype=int)
        return np.minimum(np.arange(n) + int(max_wait), n - 1)

    if mode != "session_end":
        raise ValueError(f"unknown zone_expiry {mode!r}")

    t = pd.DatetimeIndex(times)
    hh, mm = (int(x) for x in str(fcfg_end).split(":"))
    same_day = t.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    cutoff = np.where(t.to_numpy() < same_day.to_numpy(),
                      same_day.to_numpy(),
                      (same_day + pd.Timedelta(days=1)).to_numpy())
    # last bar strictly before the cutoff
    return np.searchsorted(t.to_numpy(), cutoff, side="left") - 1


def prepare_symbol(cfg, symbol: str, m15: pd.DataFrame, meta: dict | None = None):
    """
    Everything that depends only on price: SMA, breaker zones, setups.

    Split out because none of it varies with the management method or the SL
    filter, and recomputing it for each run costs ~30s per symbol over 18
    years of M15 for no change in the answer.
    """
    point = point_size(cfg, symbol, meta)
    m15 = add_spread_price(m15.reset_index(drop=True), point)
    bcfg = cfg.dotted("indicators.breaker")
    scfg = cfg.dotted("strategy")

    close = m15["close"].to_numpy(float)
    sma_vals = sma_fn(close, cfg.dotted("indicators.sma.length", 44))

    breakers = detect_breakers(
        m15["open"].to_numpy(float), m15["high"].to_numpy(float),
        m15["low"].to_numpy(float), close,
        length=bcfg["length"],
        use_only_candle_body=bcfg["use_only_candle_body"],
        use_2_candles=bcfg["use_2_candles"],
        till_first_break=bcfg["till_first_break"],
        recent_bars_only=bcfg.get("recent_bars_only"),
        single_active_zone=bcfg.get("single_active_zone", True),
        pivot_tie_mode=bcfg.get("pivot_tie_mode", "left_eq"),
    )
    setups = find_setups(
        m15, sma_vals, breakers,
        pivot_length=scfg["swing_pivot_length"],
        longs_enabled=scfg.get("longs_enabled", True),
        shorts_enabled=scfg.get("shorts_enabled", True),
        invalidate_on_mitigated=scfg.get("invalidate_on_mitigated", True),
        invalidate_on_broken=scfg.get("invalidate_on_broken", True),
        pivot_tie_mode=bcfg.get("pivot_tie_mode", "left_eq"),
        max_bars_break_to_zone=scfg.get("max_bars_break_to_zone"),
        entry_mode=scfg.get("entry_mode", "both"),
    )
    return {"m15": m15, "sma": sma_vals, "breakers": breakers,
            "setups": setups, "point": point}


def run_symbol(cfg, symbol: str, m15: pd.DataFrame, m1, method_name: str,
               news: NewsFilter | None = None, meta: dict | None = None,
               apply_sl_filter: bool | None = None, prepared: dict | None = None):
    """Run one symbol under one management method. Returns (trades_df, diag)."""
    pip = pip_size(cfg, symbol)
    prep = prepared or prepare_symbol(cfg, symbol, m15, meta)
    m15, sma_vals = prep["m15"], prep["sma"]
    breakers, setups, point = prep["breakers"], prep["setups"], prep["point"]

    # `m1` is either a DataFrame (tests, short series) or an already-built
    # index such as YearlyM1Index, which reads off disk a year at a time.
    if m1 is None or isinstance(m1, pd.DataFrame):
        m1index = M1Index(add_spread_price(m1.reset_index(drop=True), point)
                          if m1 is not None else None)
    else:
        m1index = m1

    bcfg = cfg.dotted("indicators.breaker")
    scfg = cfg.dotted("strategy")
    mcfg = dict(cfg.dotted(f"management.{method_name}"))
    max_wait = scfg.get("max_bars_to_entry")
    sma_at_tap = scfg.get("sma_check", "break") == "tap"
    stop_ref = scfg.get("stop_reference", "structure")

    high = m15["high"].to_numpy(float)
    low = m15["low"].to_numpy(float)
    times = pd.DatetimeIndex(m15["time_utc"])
    n = len(m15)

    expiry_bar = _zone_expiry_bars(times, scfg.get("zone_expiry"),
                                   fcfg_end=cfg.dotted("filters.session.end_utc"),
                                   max_wait=max_wait)

    # -- filters ---------------------------------------------------------
    fcfg = cfg.dotted("filters")
    if fcfg["session"]["enabled"]:
        in_session = session_mask(times, fcfg["session"]["start_utc"],
                                  fcfg["session"]["end_utc"])
    else:
        in_session = np.ones(n, dtype=bool)

    if news is not None:
        news_blocked = news.blocked_mask(times, symbol)
    else:
        news_blocked = np.zeros(n, dtype=bool)

    use_sl_filter = (
        scfg["sl_distance_filter"]["enabled"] if apply_sl_filter is None
        else apply_sl_filter
    )
    sl_min = float(scfg["sl_distance_filter"]["min_pips"])
    sl_max = float(scfg["sl_distance_filter"]["max_pips"])

    spread_price = m15["_spread_price"].to_numpy(float)

    by_appear = defaultdict(list)
    for s in setups:
        by_appear[s.zone_appeared_bar].append(s)

    diag = {
        "breakers": len(breakers),
        "setups": len(setups),
        "rejected_session": 0,
        "rejected_news": 0,
        "rejected_sl_size": 0,
        "rejected_busy": 0,
        "rejected_stale_zone": 0,
        "rejected_sma": 0,
        "sl_pips_all": [],
    }

    active, trades = [], []
    open_until = -1

    for t in range(n):
        active.extend(by_appear.get(t, []))
        if active:
            active = [s for s in active
                      if s.zone_dead_bar is None or t <= s.zone_dead_bar]
        if t <= open_until or not active:
            continue

        for s in list(active):
            if t <= s.zone_appeared_bar:
                continue
            # A zone that has sat untouched past its expiry is no longer the
            # retrace this setup was waiting for.
            if t > expiry_bar[s.zone_appeared_bar]:
                diag["rejected_stale_zone"] += 1
                active.remove(s)
                continue
            edge = s.entry_edge
            tapped = (low[t] <= edge) if s.direction == LONG else (high[t] >= edge)
            if not tapped:
                continue

            # SMA judged at the tap, on the last COMPLETED bar. Using sma[t]
            # would need close[t], which is not known when the limit fills.
            if sma_at_tap:
                prev = sma_vals[t - 1] if t > 0 else np.nan
                if np.isnan(prev):
                    continue
                ok = (edge > prev) if s.direction == LONG else (edge < prev)
                if not ok:
                    diag["rejected_sma"] += 1
                    continue

            # Far side of the zone: below it for longs, above it for shorts.
            far_edge = s.zone_bottom if s.direction == LONG else s.zone_top

            # Which level the stop hangs off, before the zone clamp.
            if stop_ref == "zone":
                ref_level = far_edge
            elif stop_ref == "indicator_swing":
                ref_level = s.indicator_struct_level
                if ref_level != ref_level:          # NaN guard
                    ref_level = s.struct_level
            else:
                ref_level = s.struct_level

            fill = build_fill(
                s.direction, t, times[t], edge, ref_level,
                spread_price[t], pip,
                float(scfg.get("sl_buffer_pips", 0.0)),
                float(scfg.get("risk_reward", 2.0)),
                zone_far_edge=far_edge if scfg.get("sl_beyond_zone", True) else None,
            )
            sl_pips = fill.risk / pip
            diag["sl_pips_all"].append(sl_pips)

            if fill.risk <= 0:
                active.remove(s)
                continue
            if use_sl_filter and not (sl_min <= sl_pips <= sl_max):
                diag["rejected_sl_size"] += 1
                active.remove(s)
                continue
            if not in_session[t]:
                diag["rejected_session"] += 1
                continue                       # window may reopen later
            if news_blocked[t]:
                diag["rejected_news"] += 1
                continue

            out = simulate(
                fill, m15, m1index, mcfg,
                commission_r=commission_in_r(cfg, symbol, sl_pips, meta),
                ambiguous=cfg.dotted("management.ambiguous_bar_resolution", "worst"),
                max_bars_in_trade=cfg.dotted("management.max_bars_in_trade"),
            )
            risk_cash = cfg.dotted("reporting.risk_per_trade_pct", 1.0) / 100.0

            trades.append({
                "symbol": symbol,
                "direction": "long" if s.direction == LONG else "short",
                "method": method_name,
                "signal_time": times[s.zone_appeared_bar],
                "break_time": times[s.break_bar],
                "entry_time": fill.entry_time,
                "entry_price": fill.entry_price,
                "sl": fill.stop,
                "tp": fill.target,
                "sl_pips": sl_pips,
                "partial_time": out.partial_time,
                "exit_time": out.exit_time,
                "exit_price": out.exit_price,
                "exit_reason": out.exit_reason,
                "r_multiple": out.r_multiple,
                "pips": out.pips,
                "pnl": out.r_multiple * risk_cash,
                "intrabar_source": out.intrabar_source,
                "zone_top": s.zone_top,
                "zone_bottom": s.zone_bottom,
                "zone_id": s.zone_id,
                "stop_reference": stop_ref,
                "entry_bar": t,
                "exit_bar": out.exit_bar,
                "bars_held": out.bars_held,
            })
            open_until = out.exit_bar
            active.remove(s)
            break

    df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return df, diag
