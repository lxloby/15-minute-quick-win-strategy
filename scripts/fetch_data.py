"""
Pull M15 and M1 history for every configured symbol and cache it as parquet.

Also reports, per symbol and timeframe, the start date actually obtained and
whether the terminal's *Max bars in chart* setting truncated it, and runs the
empirical server-time checks so the UTC conversion is never taken on trust.

Usage:
    python scripts/fetch_data.py                 # use cache where it exists
    python scripts/fetch_data.py --refresh       # re-pull everything
    python scripts/fetch_data.py --symbols EURUSD --timeframes M15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qw.config import load_config                                    # noqa: E402
from qw.data import mt5_loader as ml                                 # noqa: E402
from qw.data.timezones import verify_offset                          # noqa: E402

# Timeframes big enough that one DataFrame per symbol will not fit in memory.
PER_YEAR_TFS = {"M1", "M5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--timeframes", nargs="+", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--from-year", type=int, default=None,
                    help="skip years before this one")
    ap.add_argument("--until-year", type=int, default=None,
                    help="stop the yearly walk here (M1 is ~370k bars/year/symbol, "
                         "so pulling past the backtest window is expensive)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    symbols = args.symbols or cfg.dotted("data.symbols")
    timeframes = args.timeframes or [cfg.dotted("data.signal_timeframe"),
                                     cfg.dotted("data.intrabar_timeframe")]

    info = ml.connect(cfg.dotted("mt5.terminal_paths"))
    print("Connected to MT5")
    for k in ("path", "company", "login", "server", "currency", "maxbars"):
        print(f"  {k:<9}: {info[k]}")
    maxbars = info.get("maxbars")
    if maxbars and maxbars < 1_000_000:
        print(f"\n  !! Max bars in chart = {maxbars:,}. This CAPS how much history\n"
              f"     the terminal will ever serve. Tools > Options > Charts >\n"
              f"     'Max bars in chart' -> Unlimited, then restart the terminal.")

    meta, reports, problems = {}, [], []
    try:
        for want in symbols:
            sym = ml.resolve_symbol(want)
            if sym is None:
                problems.append(f"{want}: not found on this broker")
                print(f"\n{want}: NOT FOUND")
                continue
            si = __import__("MetaTrader5").symbol_info(sym)
            meta[want] = {
                "resolved": sym,
                "digits": getattr(si, "digits", None),
                "point": getattr(si, "point", None),
                "pip_value_per_lot": getattr(si, "trade_tick_value", None),
            }
            print(f"\n{want} -> {sym}  (digits={meta[want]['digits']}, "
                  f"point={meta[want]['point']})")

            for tf in timeframes:
                try:
                    if tf in PER_YEAR_TFS:
                        # Written year by year; never held whole in memory.
                        ml.fetch_history_per_year(
                            sym, tf, ml.year_dir(cfg.path("cache_dir"), sym, tf),
                            probe_from_year=cfg.dotted("mt5.probe_from_year", 2003),
                            retries=cfg.dotted("mt5.retries", 3),
                            sleep=cfg.dotted("mt5.retry_sleep", 3),
                            progress=lambda m: print(m, flush=True),
                            until_year=args.until_year,
                            from_year=args.from_year,
                        )
                        df = ml.load_cached(cfg.path("cache_dir"), sym, tf,
                                            columns=["time_utc"])
                    else:
                        df = ml.get_bars(cfg, sym, tf, refresh=args.refresh,
                                         progress=lambda m: print(m, flush=True),
                                         until_year=args.until_year)
                except Exception as exc:                      # noqa: BLE001
                    problems.append(f"{want} {tf}: {exc}")
                    print(f"  {tf}: FAILED - {exc}")
                    continue

                rep = ml.history_report(df, want, tf, maxbars)
                reports.append(rep)
                flag = "  <-- LIKELY TRUNCATED" if rep.get("likely_truncated") else ""
                print(f"  {tf:<4} {rep['bars']:>9,} bars  "
                      f"{rep['start_utc']:%Y-%m-%d} -> {rep['end_utc']:%Y-%m-%d} UTC{flag}")

                if tf == cfg.dotted("data.signal_timeframe") and "open" in df.columns:
                    v = verify_offset(df, want)
                    status = "ok" if v["ok"] else "PROBLEM"
                    print(f"       server-time check: {status}  "
                          f"offsets {v['offsets_seen']}  "
                          f"week opens server {v['week_open_server']!r} "
                          f"({v['week_open_server_share']:.0%}) "
                          f"-> UTC {v['week_open_utc']}  "
                          f"vol peak {v.get('volume_peak_hour', {}).get('server')}")
                    for p in v["problems"]:
                        print(f"         ! {p}")
                        problems.append(f"{want} {tf}: {p}")
    finally:
        ml.disconnect()

    (cfg.path("cache_dir") / "symbols_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("HISTORY OBTAINED")
    print(pd.DataFrame(reports).to_string(index=False) if reports else "  nothing")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
