"""
Sanity-check plots: N bullish and N bearish breaker blocks per pair, drawn
with candles, SMA44, confirmed swings and the zone itself, so each one can be
opened side by side with the same date on TradingView.

Run this BEFORE the backtest. If the zones do not line up with the indicator
on TradingView, nothing downstream is worth reading.

Usage:
    python scripts/preview_breakers.py                       # random sample
    python scripts/preview_breakers.py --count 3 --pad 60
    python scripts/preview_breakers.py --latest --symbols EURUSD EURJPY

--latest is for comparing against TradingView. The Pine source carries

    per = last_bar_index - bar_index <= 2000

so TradingView only ever draws breaker blocks inside the last 2000 bars of
the chart - about a month on M15. A randomly sampled zone from two years ago
shows nothing on TradingView and cannot be checked. --latest takes the most
recent zones instead, and runs the detector with the SAME 2000-bar budget so
the comparison is like for like.

That budget is not merely cosmetic. `per` gates the market-structure-shift
branch itself, so `MSS.dir` enters the visible window at 0 on TradingView
while the backtest (recent_bars_only: null) carries decades of state into it.
--latest therefore runs the detector both ways and reports any zone that
differs, since only zones that agree can be used to validate the port.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qw.config import load_config                                    # noqa: E402
from qw.data.mt5_loader import load_cached                           # noqa: E402
from qw.indicators.breaker_blocks import BULL, detect_breakers       # noqa: E402
from qw.indicators.sma import sma as sma_fn                          # noqa: E402
from qw.indicators.swings import confirmed_pivots                    # noqa: E402
from qw.reports.plots import plot_window                             # noqa: E402


def _detect(df, bcfg, recent_bars_only, single_active_zone):
    """Run the port over a frame with explicit budget / single-zone settings."""
    return detect_breakers(
        df["open"].to_numpy(float), df["high"].to_numpy(float),
        df["low"].to_numpy(float), df["close"].to_numpy(float),
        length=bcfg["length"],
        use_only_candle_body=bcfg["use_only_candle_body"],
        use_2_candles=bcfg["use_2_candles"],
        till_first_break=bcfg["till_first_break"],
        recent_bars_only=recent_bars_only,
        single_active_zone=single_active_zone,
        pivot_tie_mode=bcfg.get("pivot_tie_mode", "left_eq"),
    )


def _fate(blk):
    """How the zone ended, earliest event first."""
    events = [(blk.mitigated_bar, "mitigated"), (blk.broken_bar, "cancelled"),
              (blk.superseded_bar, "superseded")]
    live = [(b, name) for b, name in events if b is not None]
    if not live:
        return "still live", None
    bar, name = min(live)
    return name, bar


def _identity(blk):
    """What has to match for two runs to have found the same zone."""
    return (blk.direction, blk.appeared_bar, blk.source_bar,
            round(blk.top, 6), round(blk.bottom, 6))


def max_simultaneous(blocks, lo_bar, hi_bar):
    """
    Peak number of zones alive at once, ignoring supersession.

    The port assumes the Pine script keeps exactly ONE breaker alive (a single
    `var block BB` that each new shift overwrites). To test that assumption we
    have to ask what would happen if it did NOT: run with supersession off and
    count how many zones are simultaneously un-mitigated and un-cancelled. If
    TradingView ever shows more than one box at a time, the assumption is wrong
    and everything downstream of it is wrong too.
    """
    events = []
    for b in blocks:
        end = [x for x in (b.mitigated_bar, b.broken_bar) if x is not None]
        end = min(end) if end else hi_bar
        if end < lo_bar or b.appeared_bar > hi_bar:
            continue
        events.append((max(b.appeared_bar, lo_bar), 1))
        events.append((min(end, hi_bar) + 1, -1))
    if not events:
        return 0, None
    events.sort()
    cur = best = 0
    at = None
    for bar, delta in events:
        cur += delta
        if cur > best:
            best, at = cur, bar
    return best, at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--count", type=int, default=3, help="per direction, per pair")
    ap.add_argument("--pad", type=int, default=70, help="bars of context each side")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--since", default=None,
                    help="only sample zones after this date (default: the last "
                         "2 years, which TradingView is certain to still hold)")
    ap.add_argument("--latest", action="store_true",
                    help="take the most recent zones instead of a random sample, "
                         "restricted to the last --tv-window bars so every one is "
                         "actually visible on TradingView")
    ap.add_argument("--tv-window", type=int, default=2000,
                    help="Pine's `per` budget (default 2000, as in the source)")
    ap.add_argument("--symbols", nargs="+", default=None,
                    help="override data.symbols for this run")
    args = ap.parse_args()
    if args.latest:
        return run_latest(args)

    cfg = load_config(args.config)
    bcfg = cfg.dotted("indicators.breaker")
    tf = cfg.dotted("data.signal_timeframe")
    out_dir = cfg.path("out_dir") / "preview_breakers"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(
        args.seed if args.seed is not None else cfg.dotted("reporting.random_seed", 42))

    written, missing = [], []
    for symbol in (args.symbols or cfg.dotted("data.symbols")):
        df = load_cached(cfg.path("cache_dir"), symbol, tf)
        if df is None or df.empty:
            missing.append(symbol)
            print(f"{symbol}: no cached {tf} data - run scripts/fetch_data.py first")
            continue

        sma_vals = sma_fn(df["close"].to_numpy(float),
                          cfg.dotted("indicators.sma.length", 44))
        blocks = detect_breakers(
            df["open"].to_numpy(float), df["high"].to_numpy(float),
            df["low"].to_numpy(float), df["close"].to_numpy(float),
            length=bcfg["length"],
            use_only_candle_body=bcfg["use_only_candle_body"],
            use_2_candles=bcfg["use_2_candles"],
            till_first_break=bcfg["till_first_break"],
            recent_bars_only=bcfg.get("recent_bars_only"),
            single_active_zone=bcfg.get("single_active_zone", True),
            pivot_tie_mode=bcfg.get("pivot_tie_mode", "left_eq"),
        )
        print(f"\n{symbol}: {len(df):,} {tf} bars "
              f"({df['time_utc'].iloc[0]:%Y-%m-%d} -> {df['time_utc'].iloc[-1]:%Y-%m-%d}), "
              f"{len(blocks):,} breakers")

        # Computed once per symbol: a pivot pass over 600k bars for every
        # chart is wasteful, and with a large download running alongside it
        # can exhaust memory outright.
        pivots = confirmed_pivots(
            df["high"].to_numpy(float), df["low"].to_numpy(float),
            cfg.dotted("strategy.swing_pivot_length"), 1,
            bcfg.get("pivot_tie_mode", "left_eq"))

        since = (pd.Timestamp(args.since) if args.since
                 else pd.Timestamp(df["time_utc"].iloc[-1]) - pd.DateOffset(years=2))
        first_bar = int(np.searchsorted(
            pd.DatetimeIndex(df["time_utc"]).to_numpy(), np.datetime64(since, "ns")))
        print(f"  sampling zones after {since:%Y-%m-%d} (bar {first_bar:,})")

        for direction, tag in ((BULL, "bullish"), (-BULL, "bearish")):
            pool = [b for b in blocks if b.direction == direction
                    and b.appeared_bar > max(args.pad, first_bar)
                    and b.appeared_bar < len(df) - args.pad]
            if not pool:
                print(f"  {tag}: none available")
                continue
            pick = rng.choice(len(pool), size=min(args.count, len(pool)), replace=False)
            for k, i in enumerate(sorted(pick), 1):
                blk = pool[int(i)]
                ts = pd.Timestamp(df["time_utc"].iloc[blk.appeared_bar])
                srv = pd.Timestamp(df["time_server"].iloc[blk.appeared_bar])
                src = pd.Timestamp(df["time_server"].iloc[blk.source_bar])
                fate = ("mitigated" if blk.mitigated_bar else
                        "cancelled" if blk.broken_bar else "superseded/live")
                name = f"{symbol}_{tag}_{k}_{ts:%Y%m%d_%H%M}.png"
                # Both clocks in the title: TradingView will be showing one or
                # the other depending on the chart's timezone setting.
                offset = round((srv - ts).total_seconds() / 3600)
                title = (f"{symbol} {tf} - {tag} breaker | appeared "
                         f"{ts:%Y-%m-%d %H:%M} UTC = {srv:%H:%M} broker (GMT+{offset})"
                         f"  |  zone candle {src:%d %b %H:%M} broker"
                         f"  |  zone {blk.bottom:.5g}-{blk.top:.5g}  |  {fate}")
                p = plot_window(
                    df, blk.source_bar - args.pad, blk.appeared_bar + args.pad,
                    sma_vals, breakers=[blk], pivots=pivots,
                    title=title, out_path=out_dir / name,
                    sma_label=f"SMA{cfg.dotted('indicators.sma.length', 44)}",
                )
                written.append(p)
                print(f"  {tag} {k}: {ts:%Y-%m-%d %H:%M} UTC  -> {name}")

    print(f"\n{len(written)} chart(s) in {out_dir}")
    if missing:
        print(f"no data for: {', '.join(missing)}")
        return 1
    return 0


def run_latest(args) -> int:
    """The most recent zones, inside TradingView's drawing window."""
    cfg = load_config(args.config)
    bcfg = cfg.dotted("indicators.breaker")
    tf = cfg.dotted("data.signal_timeframe")
    sma_len = cfg.dotted("indicators.sma.length", 44)
    out_dir = cfg.path("out_dir") / "preview_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or cfg.dotted("data.symbols")
    W = args.tv_window

    table, charts, notes = [], [], []
    for symbol in symbols:
        df = load_cached(cfg.path("cache_dir"), symbol, tf)
        if df is None or df.empty:
            notes.append(f"{symbol}: no cached {tf} data")
            continue
        n = len(df)
        lo_bar = max(0, n - W)
        t0, t1 = df["time_utc"].iloc[lo_bar], df["time_utc"].iloc[-1]
        print(f"\n{symbol}: {n:,} {tf} bars, last to {df['time_utc'].iloc[-1]:%Y-%m-%d %H:%M} UTC")
        print(f"  TradingView window = last {W:,} bars = "
              f"{t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} UTC")

        close = df["close"].to_numpy(float)
        sma_vals = sma_fn(close, sma_len)
        pivots = confirmed_pivots(
            df["high"].to_numpy(float), df["low"].to_numpy(float),
            cfg.dotted("strategy.swing_pivot_length"), 1,
            bcfg.get("pivot_tie_mode", "left_eq"))

        # TradingView-faithful: same `per` budget the Pine source uses.
        tv = _detect(df, bcfg, W, True)
        tv_win = [b for b in tv if b.appeared_bar >= lo_bar]
        # How the backtest sees the same window (config default, usually null).
        bt = _detect(df, bcfg, bcfg.get("recent_bars_only"), True)
        bt_win = [b for b in bt if b.appeared_bar >= lo_bar]

        same = {_identity(b) for b in tv_win} == {_identity(b) for b in bt_win}
        print(f"  zones in window: {len(tv_win)} with per={W} (TradingView), "
              f"{len(bt_win)} with recent_bars_only="
              f"{bcfg.get('recent_bars_only')} (backtest)")
        if same:
            print("  -> identical, so this check validates the backtest config too")
        else:
            only_tv = {_identity(b) for b in tv_win} - {_identity(b) for b in bt_win}
            only_bt = {_identity(b) for b in bt_win} - {_identity(b) for b in tv_win}
            msg = (f"{symbol}: the two budgets DISAGREE in this window "
                   f"({len(only_tv)} only with per={W}, {len(only_bt)} only without). "
                   f"Compare against TradingView using the per={W} set.")
            print(f"  -> {msg}")
            notes.append(msg)

        # How many zones would be alive at once if nothing superseded them?
        multi = _detect(df, bcfg, W, False)
        peak, at_bar = max_simultaneous(multi, lo_bar, n - 1)
        when = (f" (first at {pd.Timestamp(df['time_utc'].iloc[at_bar]):%Y-%m-%d %H:%M} UTC)"
                if at_bar is not None else "")
        print(f"  max zones alive simultaneously, supersession OFF: {peak}{when}")
        notes.append(f"{symbol}: peak simultaneous zones = {peak} "
                     f"(port assumes 1){when}")

        chosen = sorted(tv_win, key=lambda b: b.appeared_bar)[-args.count:][::-1]
        for k, blk in enumerate(chosen, 1):
            ts = pd.Timestamp(df["time_utc"].iloc[blk.appeared_bar])
            srv = pd.Timestamp(df["time_server"].iloc[blk.appeared_bar])
            src_utc = pd.Timestamp(df["time_utc"].iloc[blk.source_bar])
            src_srv = pd.Timestamp(df["time_server"].iloc[blk.source_bar])
            offset = round((srv - ts).total_seconds() / 3600)
            fate, fate_bar = _fate(blk)
            fate_t = (f"{pd.Timestamp(df['time_utc'].iloc[fate_bar]):%d %b %H:%M}"
                      if fate_bar is not None else "-")
            tag = "bullish" if blk.direction == BULL else "bearish"

            name = f"{symbol}_{k}_{tag}_{ts:%Y%m%d_%H%M}.png"
            # Two lines: one long title overruns the figure and silently
            # truncates whatever is on the right, which was the zone's fate.
            title = (
                f"{symbol} {tf}  -  #{k} most recent {tag} breaker  |  "
                f"appeared {ts:%Y-%m-%d %H:%M} UTC  =  "
                f"{srv:%Y-%m-%d %H:%M} broker (GMT+{offset})\n"
                f"zone {blk.bottom:.5f} - {blk.top:.5f}   |   "
                f"drawn from candle {src_utc:%d %b %H:%M} UTC = {src_srv:%H:%M} broker"
                f"   |   ended: {fate}"
                + (f" at {fate_t} UTC" if fate_bar is not None else "")
            )
            plot_window(df, blk.source_bar - args.pad, blk.appeared_bar + args.pad,
                        sma_vals, breakers=[blk], pivots=pivots, title=title,
                        out_path=out_dir / name, sma_label=f"SMA{sma_len}")
            charts.append(out_dir / name)
            table.append({
                "pair": symbol, "#": k, "direction": tag,
                "appeared_utc": f"{ts:%Y-%m-%d %H:%M}",
                "appeared_broker": f"{srv:%Y-%m-%d %H:%M}",
                "gmt": f"+{offset}",
                "zone_top": f"{blk.top:.5f}", "zone_bottom": f"{blk.bottom:.5f}",
                "zone_candle_utc": f"{src_utc:%d %b %H:%M}",
                "zone_candle_broker": f"{src_srv:%d %b %H:%M}",
                "ended": fate, "ended_utc": fate_t,
                "bars_from_end": n - 1 - blk.appeared_bar,
            })
            print(f"    #{k} {tag:7s} {ts:%Y-%m-%d %H:%M} UTC -> {name}")

    if table:
        tbl = pd.DataFrame(table)
        tbl.to_csv(out_dir / "latest_zones.csv", index=False)
        print("\n" + "=" * 120)
        print("ZONES TO CHECK AGAINST TRADINGVIEW")
        print("=" * 120)
        print(tbl.to_string(index=False))
        print(f"\n{len(charts)} chart(s) + latest_zones.csv in {out_dir}")

    if notes:
        print("\nNOTES")
        for m in notes:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
