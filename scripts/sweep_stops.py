"""
Compare stop-loss placement rules in ONE process.

Running the variants as separate invocations of run_backtest.py re-imports
pandas, reloads the M15 history and recomputes every breaker and setup each
time - four times the memory churn for work whose answer never changes. On a
6GB machine that is what made the comparison fail outright.

Nothing about the zones or the setups depends on where the stop goes, so this
loads once, prepares once, and then runs each stop rule over the same
prepared setups. Peak memory is roughly one backtest instead of four.

Usage:
    python scripts/sweep_stops.py --symbol EURJPY
    python scripts/sweep_stops.py --symbol EURJPY --method baseline
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

# (label, stop_reference, sl_buffer_pips)
VARIANTS = [
    ("A  zone edge",        "zone",            0.0),
    ("D  zone edge +5p",    "zone",            5.0),
    ("C  indicator swing",  "indicator_swing", 0.0),
    ("B  structure (old)",  "structure",       0.0),
]
BREAK_EVEN = 1.0 / 3.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURJPY")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--out-dir", default="out_stop_sweep")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sym = args.symbol

    # -- load once ------------------------------------------------------
    start = cfg.dotted("data.start_date")
    end = cfg.dotted("data.end_date")
    over = (cfg.dotted("data.symbol_windows") or {}).get(sym) or {}
    start = pd.Timestamp(over.get("start_date", start)) if start else None
    end = pd.Timestamp(over.get("end_date", end)) if end else None
    if end is not None:
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    m15 = load_cached(cfg.path("cache_dir"), sym, cfg.dotted("data.signal_timeframe"),
                      columns=["time_utc", "open", "high", "low", "close", "spread"])
    mask = np.ones(len(m15), dtype=bool)
    t = pd.DatetimeIndex(m15["time_utc"])
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
        print(f"{sym}: {len(m15):,} M15 bars, {len(sus)} suspect bars clipped",
              flush=True)
        del sus
        gc.collect()

    meta_path = cfg.path("cache_dir") / "symbols_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    m1 = YearlyM1Index(cfg.path("cache_dir"), sym,
                       cfg.dotted("data.intrabar_timeframe"),
                       point_size(cfg, sym, meta),
                       clean=cfg.dotted("data.clean_suspect_bars", False))

    news = None
    if cfg.dotted("filters.news.enabled") and not args.no_news:
        n = cfg.dotted("filters.news")
        news = NewsFilter.from_csv(ROOT / n["csv"], n["minutes_before"],
                                   n["minutes_after"], n["impacts"], n["currencies"])

    # -- prepare once: zones and setups do not depend on the stop -------
    prep = prepare_symbol(cfg, sym, m15, meta)
    print(f"  {len(prep['breakers']):,} breakers -> {len(prep['setups']):,} setups\n",
          flush=True)

    # -- run each variant over the same setups --------------------------
    rows, frames = [], {}
    for label, ref, buf in VARIANTS:
        cfg["strategy"]["stop_reference"] = ref
        cfg["strategy"]["sl_buffer_pips"] = buf
        trades, diag = run_symbol(cfg, sym, m15, m1, args.method,
                                  news=news, meta=meta, prepared=prep)
        r = trades["r_multiple"].to_numpy(float)
        n = len(r)
        if n == 0:
            print(f"  {label}: no trades"); continue
        p = (r > 0).mean()
        se = r.std(ddof=1) / sqrt(n)
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
        rows.append({
            "option": label, "trades": n,
            "med_SL_pips": round(float(trades["sl_pips"].median()), 1),
            "win_rate": round(p * 100, 2),
            "break_even": 33.33,
            "total_R": round(float(r.sum()), 1),
            "expectancy_R": round(float(r.mean()), 4),
            "profit_factor": round(float(r[r > 0].sum() / -r[r < 0].sum()), 3)
            if (r < 0).any() else np.inf,
            "max_dd_R": round(float((peak - eq).max()), 1),
            "t": round(float(r.mean() / se), 2),
            "in_10_20_pct": round(float(trades["sl_pips"].between(10, 20).mean() * 100), 1),
        })
        trades.to_csv(out / f"trades_{ref}_{buf:g}.csv", index=False)
        frames[label] = trades.set_index(["zone_id", "entry_bar", "direction"])["r_multiple"]
        print(f"  {label:22s} n={n:4d}  medSL={rows[-1]['med_SL_pips']:5.1f}p  "
              f"win={p*100:5.2f}%  {r.sum():+7.1f}R  PF={rows[-1]['profit_factor']:.3f}",
              flush=True)
        del trades, r
        gc.collect()

    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)

    # -- matched subset: the same trades under every rule ---------------
    common = None
    for label, s in frames.items():
        s = s[~s.index.duplicated()]
        common = s.to_frame(label) if common is None else common.join(
            s.to_frame(label), how="inner")
    if common is not None and len(common):
        matched = pd.DataFrame([{
            "option": c, "trades": len(common),
            "win_rate": round(float((common[c] > 0).mean() * 100), 2),
            "total_R": round(float(common[c].sum()), 1),
        } for c in common.columns])
        matched.to_csv(out / "comparison_matched.csv", index=False)
    else:
        matched = pd.DataFrame()

    print("\n" + "=" * 78)
    print(f"{sym}  {args.method}  -- stop placement comparison")
    print(table.to_string(index=False))
    if len(matched):
        print(f"\nMatched: the {len(common)} trades taken under ALL FOUR rules")
        print(matched.to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
