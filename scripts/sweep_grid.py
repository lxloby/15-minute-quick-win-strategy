"""
Stop placement x partial-TP trigger, as a grid, in ONE process.

The question this answers: for each way of placing the stop, what partial
rule (if any) is worth running?

It has to be a grid rather than two separate sweeps, because a fixed-pip
partial means a different thing at every stop size - +10 pips is 0.74R on a
13 pip stop and 0.23R on a 44 pip one. Expressing the trigger as a fraction
of R makes the two axes independent, so the comparison is meaningful.

Zones and setups do not depend on either axis, so they are computed once and
reused across every cell. That keeps peak memory at roughly one backtest.

Usage:
    python scripts/sweep_grid.py --symbol EURJPY
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qw.backtest.engine import point_size, prepare_symbol, run_symbol   # noqa: E402
from qw.backtest.trade_manager import YearlyM1Index                     # noqa: E402
from qw.config import load_config                                       # noqa: E402
from qw.data.filters import NewsFilter                                  # noqa: E402
from qw.data.mt5_loader import load_cached                              # noqa: E402
from qw.data.quality import clip_suspect_bars, find_suspect_bars        # noqa: E402

STOPS = [
    ("A zone",      "zone",            0.0),
    ("D zone+5p",   "zone",            5.0),
    ("C indicator", "indicator_swing", 0.0),
    ("B structure", "structure",       0.0),
]
# None = baseline (no partial, full size to 2R)
PARTIALS = [None, 0.25, 0.50, 0.75, 1.00, 1.50]


def load_everything(cfg, sym):
    start = cfg.dotted("data.start_date")
    end = cfg.dotted("data.end_date")
    over = (cfg.dotted("data.symbol_windows") or {}).get(sym) or {}
    start = pd.Timestamp(over.get("start_date", start)) if start else None
    end = pd.Timestamp(over.get("end_date", end)) if end else None
    if end is not None:
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    m15 = load_cached(cfg.path("cache_dir"), sym, cfg.dotted("data.signal_timeframe"),
                      columns=["time_utc", "open", "high", "low", "close", "spread"])
    t = pd.DatetimeIndex(m15["time_utc"])
    mask = np.ones(len(m15), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    m15 = m15[mask].reset_index(drop=True)
    del mask, t
    gc.collect()

    if cfg.dotted("data.clean_suspect_bars", False):
        q = cfg.dotted("data.suspect_bar") or {}
        sus = find_suspect_bars(m15, window=q.get("window", 96),
                                range_mult=q.get("range_mult", 8.0),
                                persist_tol=q.get("persist_tol", 0.25))
        if len(sus):
            m15 = clip_suspect_bars(m15, sus)
        del sus
        gc.collect()

    meta_path = cfg.path("cache_dir") / "symbols_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    m1 = YearlyM1Index(cfg.path("cache_dir"), sym,
                       cfg.dotted("data.intrabar_timeframe"),
                       point_size(cfg, sym, meta),
                       clean=cfg.dotted("data.clean_suspect_bars", False))
    news = None
    if cfg.dotted("filters.news.enabled"):
        n = cfg.dotted("filters.news")
        news = NewsFilter.from_csv(ROOT / n["csv"], n["minutes_before"],
                                   n["minutes_after"], n["impacts"], n["currencies"])
    return m15, m1, news, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURJPY")
    ap.add_argument("--out-dir", default="out_grid")
    args = ap.parse_args()

    cfg = load_config()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sym = args.symbol

    m15, m1, news, meta = load_everything(cfg, sym)
    prep = prepare_symbol(cfg, sym, m15, meta)
    print(f"{sym}: {len(m15):,} M15 bars, {len(prep['setups']):,} setups\n", flush=True)

    # A dedicated method slot we rewrite per cell.
    cfg["management"]["grid"] = {}
    rows = []

    for stop_label, ref, buf in STOPS:
        cfg["strategy"]["stop_reference"] = ref
        cfg["strategy"]["sl_buffer_pips"] = buf
        for trig in PARTIALS:
            cfg["management"]["grid"] = {
                "partial_trigger_r": trig,
                "partial_fraction": 0.5 if trig is not None else 0.0,
                "move_to_breakeven": trig is not None,
                "be_includes_costs": True,
            }
            trades, _ = run_symbol(cfg, sym, m15, m1, "grid",
                                   news=news, meta=meta, prepared=prep)
            r = trades["r_multiple"].to_numpy(float)
            n = len(r)
            if n == 0:
                continue
            se = r.std(ddof=1) / sqrt(n)
            eq = np.cumsum(r)
            peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
            rows.append({
                "stop": stop_label,
                "partial": "none" if trig is None else f"{trig:.2f}R",
                "med_SL": round(float(trades["sl_pips"].median()), 1),
                "trades": n,
                "win_rate": round(float((r > 0).mean() * 100), 2),
                "total_R": round(float(r.sum()), 1),
                "PF": round(float(r[r > 0].sum() / -r[r < 0].sum()), 3)
                if (r < 0).any() else np.inf,
                "max_dd_R": round(float((peak - eq).max()), 1),
                "t": round(float(r.mean() / se), 2),
            })
            print(f"  {stop_label:12s} partial {rows[-1]['partial']:>5s}  "
                  f"n={n:4d}  win={rows[-1]['win_rate']:5.2f}%  "
                  f"{r.sum():+7.1f}R  PF={rows[-1]['PF']:.3f}  "
                  f"maxDD={rows[-1]['max_dd_R']:5.1f}R", flush=True)
            del trades, r
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out / "grid.csv", index=False)

    print("\n" + "=" * 86)
    print(f"{sym}  --  total R by stop placement (rows) and partial trigger (cols)")
    print(df.pivot(index="stop", columns="partial", values="total_R").to_string())
    print(f"\nprofit factor")
    print(df.pivot(index="stop", columns="partial", values="PF").to_string())
    print(f"\nmax drawdown (R)")
    print(df.pivot(index="stop", columns="partial", values="max_dd_R").to_string())
    print(f"\n-> {out / 'grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
