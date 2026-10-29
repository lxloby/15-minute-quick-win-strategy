"""
Run the full backtest and write every output in section 6 of the spec.

Produces, under out/:
    trades.csv                     every trade, both methods
    summary.csv                    per pair / direction / method
    summary_m1_period.csv          same, restricted to the M1-covered window
    breakdown_year.csv
    breakdown_quarter.csv
    breakdown_hour_sgt.csv
    sl_distribution.csv / .png     with the SL filter OFF
    equity_safe_vs_baseline.png
    equity_long_vs_short.png
    trade_charts/                  sampled trades, both directions
    diagnostics.json               counts and rejection reasons

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --no-news      # only with explicit consent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qw.backtest import metrics                                      # noqa: E402
from qw.backtest.engine import point_size, prepare_symbol, run_symbol  # noqa: E402
from qw.backtest.trade_manager import YearlyM1Index                  # noqa: E402
from qw.config import load_config                                    # noqa: E402
from qw.data.filters import NewsFilter                               # noqa: E402
from qw.data.mt5_loader import load_cached                           # noqa: E402
from qw.data.quality import clip_suspect_bars, find_suspect_bars     # noqa: E402
from qw.indicators.breaker_blocks import detect_breakers             # noqa: E402
from qw.indicators.sma import sma as sma_fn                          # noqa: E402
from qw.reports.plots import plot_equity, plot_sl_histogram, plot_window  # noqa: E402


def clip(df, start, end):
    """Restrict a bar frame to the configured backtest window."""
    if df is None or df.empty:
        return df
    t = pd.DatetimeIndex(df["time_utc"])
    mask = np.ones(len(df), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    return df[mask].reset_index(drop=True)


def window(cfg, symbol: str | None = None):
    """
    Backtest window, with per-symbol overrides applied.

    A pair whose early history is dominated by a regime the strategy is not
    being tested against can be held to a later start without shortening the
    others - see data.symbol_windows in config.yaml.
    """
    start = cfg.dotted("data.start_date")
    end = cfg.dotted("data.end_date")
    if symbol:
        over = (cfg.dotted("data.symbol_windows") or {}).get(symbol) or {}
        start = over.get("start_date", start)
        end = over.get("end_date", end)
    start = pd.Timestamp(start) if start else None
    # An end date means "through the end of that day".
    end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) if end else None
    return start, end


def load_pair(cfg, symbol):
    """
    M15 as a frame; M1 as a lazy year-by-year index.

    M1 is ~7M rows per symbol. Holding it as a DataFrame costs ~292MB and the
    same again as arrays, which does not fit beside everything else on a 6GB
    machine - YearlyM1Index keeps one year (~18MB) in play instead.
    """
    start, end = window(cfg, symbol)
    # Only the columns the engine reads. time_server and the volume columns
    # are dead weight here, and on a memory-tight machine that matters.
    m15 = load_cached(cfg.path("cache_dir"), symbol,
                      cfg.dotted("data.signal_timeframe"),
                      columns=["time_utc", "open", "high", "low", "close", "spread"])
    m15 = clip(m15, start, end)

    cleaned = 0
    if m15 is not None and cfg.dotted("data.clean_suspect_bars", False):
        q = cfg.dotted("data.suspect_bar") or {}
        suspect = find_suspect_bars(m15, window=q.get("window", 96),
                                    range_mult=q.get("range_mult", 8.0),
                                    persist_tol=q.get("persist_tol", 0.25))
        cleaned = len(suspect)
        if cleaned:
            m15 = clip_suspect_bars(m15, suspect)

    meta_path = cfg.path("cache_dir") / "symbols_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    m1 = YearlyM1Index(cfg.path("cache_dir"), symbol,
                       cfg.dotted("data.intrabar_timeframe"),
                       point_size(cfg, symbol, meta),
                       clean=cfg.dotted("data.clean_suspect_bars", False))
    return m15, m1, cleaned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-news", action="store_true",
                    help="run WITHOUT the news filter (results are not comparable)")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--direction", choices=["both", "long", "short"], default="both",
                    help="restrict to one side. Note this also changes WHICH trades "
                         "are taken: with one side off, the other is no longer "
                         "blocked by the one-position-per-symbol rule.")
    ap.add_argument("--start-date", default=None,
                    help="override data.start_date for this run")
    ap.add_argument("--stop-reference", choices=["structure","zone","indicator_swing"],
                    default=None, help="override strategy.stop_reference")
    ap.add_argument("--sl-buffer-pips", type=float, default=None,
                    help="override strategy.sl_buffer_pips")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="override management.methods, e.g. --methods baseline")
    ap.add_argument("--no-charts", action="store_true",
                    help="skip PNG generation - much lighter on memory")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave the feed untouched (the bad-tick cleaner is the "
                         "one step that inspects future bars)")
    ap.add_argument("--end-date", default=None,
                    help="override data.end_date for this run")
    ap.add_argument("--max-bars-in-trade", default=None,
                    help="override management.max_bars_in_trade ('none' to disable)")
    ap.add_argument("--out-dir", default=None,
                    help="write outputs here instead of paths.out_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.start_date:
        cfg["data"]["start_date"] = args.start_date
        # a per-symbol window would otherwise silently win
        cfg["data"]["symbol_windows"] = {}
    if args.stop_reference:
        cfg["strategy"]["stop_reference"] = args.stop_reference
    if args.sl_buffer_pips is not None:
        cfg["strategy"]["sl_buffer_pips"] = args.sl_buffer_pips
    if args.methods:
        cfg["management"]["methods"] = args.methods
    if args.no_clean:
        cfg["data"]["clean_suspect_bars"] = False
    if args.end_date:
        cfg["data"]["end_date"] = args.end_date
        cfg["data"]["symbol_windows"] = {}
    if args.max_bars_in_trade is not None:
        v = args.max_bars_in_trade
        cfg["management"]["max_bars_in_trade"] = (
            None if str(v).lower() in ("none", "null", "0") else int(v))
    if args.direction != "both":
        cfg["strategy"]["longs_enabled"] = args.direction == "long"
        cfg["strategy"]["shorts_enabled"] = args.direction == "short"
    out = Path(args.out_dir) if args.out_dir else cfg.path("out_dir")
    out.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or cfg.dotted("data.symbols")
    risk_pct = cfg.dotted("reporting.risk_per_trade_pct", 1.0)

    meta_path = cfg.path("cache_dir") / "symbols_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # -- news filter: refuse to run silently without it -------------------
    news = None
    if cfg.dotted("filters.news.enabled") and not args.no_news:
        ncfg = cfg.dotted("filters.news")
        try:
            news = NewsFilter.from_csv(
                ROOT / ncfg["csv"], ncfg["minutes_before"], ncfg["minutes_after"],
                ncfg["impacts"], ncfg["currencies"])
        except FileNotFoundError as exc:
            print(f"\nSTOPPING: {exc}\n", file=sys.stderr)
            print("Build it with:  python scripts/build_news.py --start <YYYY-MM-DD>",
                  file=sys.stderr)
            print("or re-run with --no-news if you accept an unfiltered backtest.",
                  file=sys.stderr)
            return 2
        lo, hi = news.coverage()
        print(f"News calendar: {len(news.events):,} high-impact events, {lo} -> {hi}")

        gaps = news.missing_currencies(symbols)
        if gaps:
            print("\nSTOPPING: the news calendar has no events for:", file=sys.stderr)
            for sym, ccys in gaps.items():
                print(f"  {sym}: {', '.join(ccys)}", file=sys.stderr)
            print("\nThose trades would be filtered on nothing while the run still\n"
                  "reported the news filter as enabled. Rebuild the calendar with\n"
                  "the currencies included:\n"
                  "  python scripts/convert_calendar.py --currencies "
                  + " ".join(sorted({c for v in gaps.values() for c in v}
                                    | set(cfg.dotted("filters.news.currencies"))))
                  + "\nor re-run with --no-news to accept an unfiltered backtest.",
                  file=sys.stderr)
            return 2
        end_cfg = cfg.dotted("data.end_date")
        if end_cfg and pd.Timestamp(end_cfg) > hi:
            print(f"  !! backtest end_date {end_cfg} is past the calendar's last "
                  f"event ({hi:%Y-%m-%d}); trades after that are UNFILTERED")
    elif args.no_news:
        print("!! News filter DISABLED by --no-news. Results include news entries.")

    # -- run ---------------------------------------------------------------
    all_trades, diags, m1_windows, sl_all = [], {}, {}, []
    for symbol in symbols:
        m15, m1, cleaned = load_pair(cfg, symbol)
        if m15 is None or m15.empty:
            print(f"{symbol}: no cached M15 - run scripts/fetch_data.py")
            continue
        if not m1.empty:
            m1_windows[symbol] = m1.year_range

        print(f"\n{symbol}: {len(m15):,} M15 bars "
              f"({m15['time_utc'].iloc[0]:%Y-%m-%d} -> {m15['time_utc'].iloc[-1]:%Y-%m-%d}), "
              f"M1 years {m1.year_range[0]}-{m1.year_range[1]}"
              + (f"  [{cleaned} suspect M15 bars clipped]" if cleaned else ""),
              flush=True)

        # Indicators depend only on price, so they are computed once and
        # shared by every method and by the filter-off diagnostic run.
        prep = prepare_symbol(cfg, symbol, m15, meta)
        print(f"  {len(prep['breakers']):,} breakers -> "
              f"{len(prep['setups']):,} setups", flush=True)

        for method in cfg.dotted("management.methods"):
            trades, diag = run_symbol(cfg, symbol, m15, m1, method,
                                      news=news, meta=meta, prepared=prep)
            all_trades.append(trades)
            diags[f"{symbol}:{method}"] = {k: v for k, v in diag.items()
                                           if k != "sl_pips_all"}
            print(f"  {method:<9} {len(trades):>4} trades   "
                  f"total {trades['r_multiple'].sum():+7.2f}R" if len(trades)
                  else f"  {method:<9}    0 trades")

        # SL distribution with the filter OFF, for the spec's "run it once without"
        _, diag_off = run_symbol(cfg, symbol, m15, m1,
                                 cfg.dotted("management.methods")[0],
                                 news=news, meta=meta, apply_sl_filter=False,
                                 prepared=prep)
        sl_all.extend(diag_off["sl_pips_all"])
        diags[f"{symbol}:filter_off"] = {k: v for k, v in diag_off.items()
                                         if k != "sl_pips_all"}

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades.empty:
        print("\nNo trades produced. Check diagnostics.json for rejection counts.")
        (out / "diagnostics.json").write_text(json.dumps(diags, indent=2, default=str))
        return 1

    trades = trades.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    trades.to_csv(out / "trades.csv", index=False)
    print(f"\nWrote {len(trades):,} trades -> {out / 'trades.csv'}")

    # -- summaries ---------------------------------------------------------
    summary = metrics.summary_table(trades, risk_pct)
    summary.to_csv(out / "summary.csv", index=False)

    # "M1-covered" is decided by how each trade actually resolved, not by a
    # date window - a trade can straddle the edge of M1 coverage.
    m1_trades = trades[trades["intrabar_source"] == "M1"]
    metrics.summary_table(m1_trades, risk_pct).to_csv(
        out / "summary_m1_period.csv", index=False)
    print(f"Resolved fully on M1: {len(m1_trades):,} of {len(trades):,} trades "
          f"({len(m1_trades) / max(len(trades), 1) * 100:.1f}%); "
          f"by source {trades['intrabar_source'].value_counts().to_dict()}")

    for key, fname in (("year", "breakdown_year.csv"),
                       ("quarter", "breakdown_quarter.csv"),
                       ("hour", "breakdown_hour_sgt.csv")):
        metrics.breakdown(trades, key, risk_pct,
                          cfg.dotted("reporting.display_timezone")).to_csv(
            out / fname, index=False)

    # -- SL distribution with the filter off -------------------------------
    flt = cfg.dotted("strategy.sl_distance_filter")
    metrics.sl_distribution(sl_all).to_csv(out / "sl_distribution.csv", index=False)
    plot_sl_histogram(sl_all, out / "sl_distribution.png",
                      flt["min_pips"], flt["max_pips"])

    # -- equity curves ------------------------------------------------------
    plot_equity(
        {m: metrics.equity_curve(trades[trades["method"] == m])
         for m in trades["method"].unique()},
        "Safe vs Baseline - cumulative R, all pairs",
        out / "equity_safe_vs_baseline.png")

    primary = cfg.dotted("management.methods")[0]
    prim = trades[trades["method"] == primary]
    plot_equity(
        {f"{d} ({primary})": metrics.equity_curve(prim[prim["direction"] == d])
         for d in ("long", "short")},
        f"Long vs Short - cumulative R, {primary} method",
        out / "equity_long_vs_short.png")

    # -- sampled trade charts ------------------------------------------------
    rng = np.random.default_rng(cfg.dotted("reporting.random_seed", 42))
    n_charts = 0 if args.no_charts else cfg.dotted("reporting.sample_charts_per_direction", 10)
    chart_dir = out / "trade_charts"
    bcfg = cfg.dotted("indicators.breaker")
    cache = {}
    made = 0
    for direction in ("long", "short"):
        pool = prim[prim["direction"] == direction]
        if pool.empty:
            continue
        pick = rng.choice(len(pool), size=min(n_charts, len(pool)), replace=False)
        for row in pool.iloc[sorted(pick)].to_dict("records"):
            sym = row["symbol"]
            if sym not in cache:
                m15, _, _ = load_pair(cfg, sym)
                blocks = detect_breakers(
                    m15["open"].to_numpy(float), m15["high"].to_numpy(float),
                    m15["low"].to_numpy(float), m15["close"].to_numpy(float),
                    length=bcfg["length"],
                    use_only_candle_body=bcfg["use_only_candle_body"],
                    use_2_candles=bcfg["use_2_candles"],
                    till_first_break=bcfg["till_first_break"],
                    recent_bars_only=bcfg.get("recent_bars_only"),
                    single_active_zone=bcfg.get("single_active_zone", True),
                    pivot_tie_mode=bcfg.get("pivot_tie_mode", "left_eq"))
                cache[sym] = (m15, sma_fn(m15["close"].to_numpy(float),
                                          cfg.dotted("indicators.sma.length", 44)),
                              blocks)
            m15, sma_vals, blocks = cache[sym]
            zb = [b for b in blocks if abs(b.top - row["zone_top"]) < 1e-9
                  and abs(b.bottom - row["zone_bottom"]) < 1e-9]
            lo = int(row["entry_bar"]) - 80
            hi = int(row["exit_bar"]) + 30
            name = (f"{sym}_{direction}_{pd.Timestamp(row['entry_time']):%Y%m%d_%H%M}"
                    f"_{row['exit_reason']}_{row['r_multiple']:+.2f}R.png")
            plot_window(m15, lo, hi, sma_vals, breakers=zb,
                        pivot_length=cfg.dotted("strategy.swing_pivot_length"),
                        trade=row,
                        title=(f"{sym} {direction} {pd.Timestamp(row['entry_time']):%Y-%m-%d %H:%M} UTC"
                               f"  |  {row['exit_reason']}  {row['r_multiple']:+.2f}R"
                               f"  |  SL {row['sl_pips']:.1f} pips"
                               f"  |  intrabar {row['intrabar_source']}"),
                        out_path=chart_dir / name)
            made += 1

    (out / "diagnostics.json").write_text(json.dumps(diags, indent=2, default=str))

    print(f"\nSummary (all pairs, both methods):")
    head = summary[(summary["symbol"] == "ALL") & (summary["direction"] == "combined")]
    cols = ["method", "trades", "win_rate", "expectancy_r", "total_r",
            "profit_factor", "max_dd_r", "max_dd_pct", "longest_losing_streak",
            "avg_trades_per_week"]
    print(head[cols].to_string(index=False))
    print(f"\n{made} trade charts -> {chart_dir}")
    print(f"All outputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
