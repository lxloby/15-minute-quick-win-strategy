"""
Broker server time -> UTC.

MT5 hands back a Unix epoch that is really the SERVER's wall clock pretending
to be UTC, so it has to be shifted before any session filter can mean
anything. Axi runs GMT+2 in winter and GMT+3 in summer.

Rather than hardcode two numbers and two switch dates, this module leans on
the invariant that actually drives MT5 broker clocks: the server day is cut at
the New York close, so

    server wall clock == New York wall clock + 7 hours

all year round (NY is UTC-5 -> server UTC+2; NY is UTC-4 -> server UTC+3).
Expressing it that way lets the IANA database supply the transition dates,
and `verify_offset` then checks the result against the feed itself rather
than trusting the assumption.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
SERVER_MINUS_NY = timedelta(hours=7)


def server_to_utc(server_times) -> pd.DatetimeIndex:
    """Convert naive broker-server timestamps to tz-aware UTC."""
    s = pd.DatetimeIndex(pd.to_datetime(server_times))
    if s.tz is not None:
        s = s.tz_convert(None)

    ny_wall = s - SERVER_MINUS_NY
    # DST transitions land at 02:00 NY on a Sunday, when the FX market is shut,
    # so no real bar can be ambiguous or non-existent. Resolve defensively.
    localised = pd.DatetimeIndex(ny_wall).tz_localize(
        NY_TZ, ambiguous=True, nonexistent="shift_forward"
    )
    return localised.tz_convert("UTC")


def offset_hours(server_times) -> np.ndarray:
    """Server-minus-UTC offset in hours for each timestamp (expect 2 or 3)."""
    s = pd.DatetimeIndex(pd.to_datetime(server_times))
    if s.tz is not None:
        s = s.tz_convert(None)
    utc = server_to_utc(s)
    return ((s - utc.tz_localize(None)) / pd.Timedelta(hours=1)).to_numpy()


# ---------------------------------------------------------------------------
# Empirical verification
# ---------------------------------------------------------------------------
def verify_offset(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Check the GMT+2/+3 model against the data instead of assuming it.

    The trading week is found from the data itself - a gap of more than 12
    hours between consecutive bars is a weekend - rather than from calendar
    weeks. (Grouping by ISO week does NOT work: Sunday is ISO weekday 7, so
    the "last bar of the week" comes out as the Sunday 23:45 UTC bar of the
    session that has just opened, not Friday's close.)

    Three independent probes:

    1. Offsets produced must be exactly {2, 3}.
    2. Week boundaries must be STABLE IN SERVER TIME and shifting in UTC.
       That asymmetry is the whole signature of a NY-aligned broker clock:
       the week opens Monday 00:00 server all year, which is Sunday 21:00 UTC
       in summer and Sunday 22:00 UTC in winter. If the model were wrong the
       server-side times would wobble instead.
    3. Tick volume peaks at the London/NY overlap, so for the same reason the
       peak hour must sit still in server time. A broker following EU rather
       than US DST dates would show it moving.

    Returns a dict of findings; `ok` is False if anything disagrees.
    """
    out: dict = {"symbol": symbol}
    if len(df) < 100:
        return {**out, "ok": False, "problems": ["not enough data"]}

    server = pd.DatetimeIndex(df["time_server"])
    utc = pd.DatetimeIndex(df["time_utc"])
    offsets = np.round(offset_hours(server)).astype(int)
    out["offsets_seen"] = sorted(set(offsets.tolist()))

    # -- 1. weekend boundaries, located empirically ----------------------
    gap_h = (utc[1:] - utc[:-1]).total_seconds() / 3600.0
    brk = np.where(gap_h > 12)[0]
    out["weekends_found"] = int(len(brk))

    def modal(idx, fmt="%a %H:%M"):
        if not len(idx):
            return None, 0.0
        vc = pd.Series(pd.DatetimeIndex(idx).strftime(fmt)).value_counts()
        return vc.index[0], float(vc.iloc[0] / vc.sum())

    out["week_open_server"], open_share = modal(server[brk + 1])
    out["week_close_server"], close_share = modal(server[brk])
    out["week_open_server_share"] = round(open_share, 3)
    out["week_close_server_share"] = round(close_share, 3)

    # The same boundary in UTC, split by season.
    open_utc = utc[brk + 1]
    season = offsets[brk + 1]
    out["week_open_utc"] = {
        f"+{o}": modal(open_utc[season == o])[0]
        for o in sorted(set(season.tolist()))
    }

    # -- 2. activity profile ---------------------------------------------
    if "tick_volume" in df.columns:
        vol = df["tick_volume"].to_numpy(dtype=float)
        is_summer = offsets == 3
        prof = {}
        for label, idx in (("server", server), ("utc", utc)):
            # Median, not mean: the weekly-open bar carries a huge tick burst
            # and would otherwise drag the peak onto the rollover hour.
            byhour = pd.DataFrame({"h": idx.hour, "v": vol})
            prof[label] = {
                "summer": int(byhour[is_summer].groupby("h")["v"].median().idxmax())
                if is_summer.any() else None,
                "winter": int(byhour[~is_summer].groupby("h")["v"].median().idxmax())
                if (~is_summer).any() else None,
            }
        out["volume_peak_hour"] = prof
        srv = prof["server"]
        out["server_peak_stable"] = (
            srv["summer"] is None or srv["winter"] is None
            or srv["summer"] == srv["winter"]
        )

    # -- verdict -----------------------------------------------------------
    problems = []
    if set(out["offsets_seen"]) - {2, 3}:
        problems.append(f"unexpected offsets {out['offsets_seen']} (expected 2 and/or 3)")

    if not (out["week_open_server"] or "").startswith("Mon 00:00"):
        problems.append(
            f"week opens at server {out['week_open_server']!r}, expected 'Mon 00:00' "
            f"- the broker may not be NY-aligned"
        )
    elif open_share < 0.7:
        # Over 20+ years a broker changes its session schedule occasionally,
        # so this is deliberately loose - it is looking for a broken model,
        # not for a perfectly uniform history.
        problems.append(f"week open only {open_share:.0%} consistent in server time")

    if not (out["week_close_server"] or "").startswith("Fri "):
        problems.append(f"week closes at server {out['week_close_server']!r}, expected a Friday")

    expected_open = {3: "Sun 21:00", 2: "Sun 22:00"}
    for off, seen in out["week_open_utc"].items():
        want = expected_open.get(int(off.lstrip("+")))
        if want and seen != want:
            problems.append(
                f"with offset {off} the week opens at {seen} UTC, expected {want}"
            )

    if out.get("server_peak_stable") is False:
        problems.append(
            "tick-volume peak moves in server time between summer and winter - "
            "the broker may not follow US DST dates"
        )

    out["ok"] = not problems
    out["problems"] = problems
    return out
