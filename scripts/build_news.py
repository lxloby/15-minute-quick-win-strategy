"""
Build data/news_calendar.csv from the Forex Factory calendar.

How it works
------------
Each weekly page (https://www.forexfactory.com/calendar?week=mon1.2024)
embeds its data in a JavaScript assignment:

    window.calendarComponentStates[1] = { days: [ ... ], ... }

The top level is a JS object literal with unquoted keys, but the `days`
array itself is valid JSON, so it is located by index and handed to
`json.JSONDecoder().raw_decode`, which copes with the nesting that a regex
would not.

Every event carries `dateline`, a Unix timestamp. That is an absolute
instant, so no timezone guessing is involved anywhere. Spot-checked against
Non-Farm Payrolls for 2024-01-05, which comes out at 13:30 UTC - 08:30
Eastern, as published.

Output columns: utc_time,currency,impact,title

Usage
-----
    python scripts/build_news.py --start 2022-09-01 --end 2026-10-25
    python scripts/build_news.py --start 2022-09-01            # end = today

Raw weekly payloads are cached under data/news_cache/ so a re-run or an
extended date range only fetches the weeks it is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "https://www.forexfactory.com/calendar?week={slug}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
STATE_RE = re.compile(r"window\.calendarComponentStates\[\d+\]\s*=\s*(\{.*?\});\s*\n", re.S)

IMPACT_MAP = {
    "High Impact Expected": "High",
    "Medium Impact Expected": "Medium",
    "Low Impact Expected": "Low",
    "Non-Economic": "Holiday",
}


def week_slug(d: date) -> str:
    return f"{d.strftime('%b').lower()}{d.day}.{d.year}"


def mondays(start: date, end: date):
    cur = start - timedelta(days=start.weekday())
    while cur <= end:
        yield cur
        cur += timedelta(days=7)


def fetch_week(slug: str, cache_dir: Path, session: requests.Session,
               pause: float, retries: int = 3):
    cached = cache_dir / f"{slug}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    url = BASE.format(slug=slug)
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                m = STATE_RE.search(r.text)
                if not m:
                    last = "calendar state blob not found in page"
                else:
                    blob = m.group(1)
                    i = blob.index("days:")
                    j = blob.index("[", i)
                    days, _ = json.JSONDecoder().raw_decode(blob, j)
                    cached.write_text(json.dumps(days), encoding="utf-8")
                    time.sleep(pause)
                    return days
            else:
                last = f"HTTP {r.status_code}"
        except Exception as exc:                       # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(pause * (attempt + 2))
    raise RuntimeError(f"could not fetch {url}: {last}")


def rows_from_days(days):
    for day in days:
        for ev in day.get("events", []):
            dateline = ev.get("dateline")
            if not dateline:
                continue
            yield {
                "utc_time": datetime.fromtimestamp(dateline, tz=timezone.utc)
                                    .replace(tzinfo=None),
                "currency": (ev.get("currency") or "").upper(),
                "impact": IMPACT_MAP.get(ev.get("impactTitle"), ev.get("impactTitle") or ""),
                "title": ev.get("name") or "",
            }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--out", default=str(ROOT / "data" / "news_calendar.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "news_cache"))
    ap.add_argument("--pause", type=float, default=1.5,
                    help="seconds between requests - be polite")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    weeks = list(mondays(start, end))
    print(f"Forex Factory calendar: {len(weeks)} weeks, {start} -> {end}")

    session = requests.Session()
    rows, failed = [], []
    for k, monday in enumerate(weeks, 1):
        slug = week_slug(monday)
        try:
            days = fetch_week(slug, cache_dir, session, args.pause)
            got = list(rows_from_days(days))
            rows.extend(got)
            if k % 10 == 0 or k == len(weeks):
                print(f"  [{k:>4}/{len(weeks)}] {slug:<12} +{len(got):>3} events "
                      f"(total {len(rows):,})")
        except Exception as exc:                        # noqa: BLE001
            failed.append((slug, str(exc)))
            print(f"  [{k:>4}/{len(weeks)}] {slug:<12} FAILED: {exc}")

    if not rows:
        print("\nNo events retrieved at all - refusing to write an empty calendar.",
              file=sys.stderr)
        return 2

    df = (pd.DataFrame(rows)
            .drop_duplicates(subset=["utc_time", "currency", "title"])
            .sort_values("utc_time")
            .reset_index(drop=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    high = df[df["impact"] == "High"]
    print(f"\nWrote {len(df):,} events -> {out}")
    print(f"  coverage : {df['utc_time'].min()} -> {df['utc_time'].max()}")
    print(f"  high impact: {len(high):,}")
    print("  by currency (high impact):")
    for ccy, cnt in high["currency"].value_counts().head(8).items():
        print(f"    {ccy}: {cnt:,}")
    if failed:
        print(f"\n  {len(failed)} week(s) FAILED - the calendar has holes:")
        for slug, err in failed[:10]:
            print(f"    {slug}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
