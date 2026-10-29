"""
Convert the HuggingFace Forex Factory dump into data/news_calendar.csv.

Source
------
https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar
saved locally as forex_factory_cache.csv. Columns:

    DateTime,Currency,Impact,Event,Actual,Forecast,Previous,Detail

`DateTime` is ISO-8601 with an explicit offset, always +03:30 or +04:30
(Tehran standard / daylight - the dataset author's Forex Factory timezone
setting). An explicit offset is unambiguous, so converting to UTC is exact
wherever the time itself is real.

The catch: lost times
---------------------
About 31% of high-impact rows - and 47% of the EUR/USD/JPY/CHF rows inside
the 2022-10 -> 2025-04 window - are stamped at local **00:00**. That is not
a midnight release, it is a scrape that lost the time field. The affected
events are precisely the ones that matter: Non-Farm Employment Change (161
of 220 rows), Unemployment Claims, Retail Sales, CPI.

Spot-checks on rows that *do* carry a time come out right:
    FOMC Federal Funds Rate 2024-01-31 -> 19:00 UTC   correct (14:00 ET)
    ADP Non-Farm             2024-01-04 -> 13:15 UTC   correct (08:15 ET)
A lost-time row comes out wrong on both counts:
    Non-Farm Employment      real 2024-01-05 13:30 UTC
                             dataset 2024-01-04 20:30 UTC  (wrong day too)

Because local 00:00 under +03:30/+04:30 lands at 19:30-20:30 UTC the
previous day, keeping those rows would block a window in the middle of the
12:00-23:00 UTC session while leaving the real release unblocked - worse
than not filtering at all.

`--lost-time` picks what happens to them:
    drop       (default) discard; the filter under-blocks, and the report
               says exactly which events are missing
    block-day  expand each one to cover the whole UTC day for that currency;
               nothing slips through, but it removes a lot of session time

Usage:
    python scripts/convert_calendar.py
    python scripts/convert_calendar.py --lost-time block-day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMPACT_MAP = {
    "High Impact Expected": "High",
    "Medium Impact Expected": "Medium",
    "Low Impact Expected": "Low",
    "Non-Economic": "Holiday",
}
LOST_TIME_LOCAL = "00:00"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "forex_factory_cache.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "news_calendar.csv"))
    ap.add_argument("--lost-time", choices=["drop", "block-day"], default="drop")
    ap.add_argument("--currencies", nargs="+",
                    default=["EUR", "USD", "JPY", "CHF"])
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 2

    df = pd.read_csv(src, usecols=["DateTime", "Currency", "Impact", "Event"])
    print(f"read {len(df):,} rows from {src.name}")

    df["impact"] = df["Impact"].map(IMPACT_MAP).fillna(df["Impact"])
    df["currency"] = df["Currency"].astype(str).str.strip().str.upper()
    df["title"] = df["Event"].astype(str).str.strip()
    df["local_hm"] = df["DateTime"].str.slice(11, 16)
    df["utc_time"] = (pd.to_datetime(df["DateTime"], format="ISO8601", utc=True)
                        .dt.tz_localize(None))

    print(f"  coverage: {df['utc_time'].min()} -> {df['utc_time'].max()} UTC")

    keep = df[df["impact"] == "High"].copy()
    relevant = keep[keep["currency"].isin(args.currencies)]
    lost = relevant[relevant["local_hm"] == LOST_TIME_LOCAL]
    good = relevant[relevant["local_hm"] != LOST_TIME_LOCAL]

    print(f"\nhigh-impact rows            : {len(keep):,}")
    print(f"  for {'/'.join(args.currencies)}: {len(relevant):,}")
    print(f"  with a usable time        : {len(good):,}")
    print(f"  time lost (local 00:00)   : {len(lost):,} "
          f"({len(lost) / max(len(relevant), 1) * 100:.1f}%)")
    if len(lost):
        print("  most affected events:")
        for ev, n in lost["title"].value_counts().head(8).items():
            print(f"    {n:>5}  {ev}")

    out_rows = [good[["utc_time", "currency", "impact", "title"]]]

    if args.lost_time == "block-day" and len(lost):
        # One synthetic event every hour across the UTC day the release fell
        # on, so the +/- window stitches into continuous cover.
        day = lost["utc_time"].dt.normalize()
        expanded = []
        for offset in range(0, 24):
            chunk = lost.copy()
            chunk["utc_time"] = day + pd.Timedelta(hours=offset)
            chunk["title"] = chunk["title"] + " [time unknown - whole day blocked]"
            expanded.append(chunk[["utc_time", "currency", "impact", "title"]])
        out_rows.extend(expanded)
        print(f"\n  lost-time policy: block-day -> expanded to "
              f"{len(lost) * 24:,} hourly markers")
    else:
        print(f"\n  lost-time policy: drop -> {len(lost):,} events are NOT filtered")

    out = (pd.concat(out_rows, ignore_index=True)
             .drop_duplicates(subset=["utc_time", "currency", "title"])
             .sort_values("utc_time")
             .reset_index(drop=True))

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nwrote {len(out):,} rows -> {dest}")
    print(f"  usable coverage: {out['utc_time'].min()} -> {out['utc_time'].max()} UTC")
    print("  by currency:")
    for ccy, n in out["currency"].value_counts().items():
        print(f"    {ccy}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
