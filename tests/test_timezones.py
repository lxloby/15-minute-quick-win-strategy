"""
Broker server time -> UTC.

The important test here is the last one: a verification routine that cannot
fail is worthless, so the suite feeds it a deliberately wrong conversion and
requires it to object.
"""

import numpy as np
import pandas as pd
import pytest

from qw.data.timezones import offset_hours, server_to_utc, verify_offset


def test_winter_is_plus_two_and_summer_plus_three():
    t = pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"])
    u = server_to_utc(t)
    assert u[0] == pd.Timestamp("2024-01-15 10:00", tz="UTC")
    assert u[1] == pd.Timestamp("2024-07-15 09:00", tz="UTC")


def test_transitions_follow_us_dst_not_eu():
    # US DST 2024: starts 10 Mar, ends 3 Nov. EU: 31 Mar and 27 Oct.
    # The gap weeks are what tell the two apart.
    t = pd.to_datetime(["2024-03-08 12:00", "2024-03-12 12:00",   # US switched
                        "2024-10-29 12:00", "2024-11-05 12:00"])  # EU switched, US not
    assert list(np.round(offset_hours(t)).astype(int)) == [2, 3, 3, 2]


def test_offsets_are_only_ever_two_or_three():
    t = pd.date_range("2023-01-01", "2025-12-31", freq="6h")
    assert set(np.round(offset_hours(t)).astype(int).tolist()) == {2, 3}


def synthetic_broker_weeks(start="2023-09-01", weeks=70):
    """
    A broker feed shaped like the real one: bars from Monday 00:00 to Friday
    23:45 server time, every week, spanning a DST change in both directions.
    """
    rows = []
    monday = pd.Timestamp(start) - pd.Timedelta(days=pd.Timestamp(start).weekday())
    for w in range(weeks):
        wk = monday + pd.Timedelta(weeks=w)
        rows.append(pd.date_range(wk, wk + pd.Timedelta(days=4, hours=23, minutes=45),
                                  freq="15min"))
    server = pd.DatetimeIndex(np.concatenate([r.to_numpy() for r in rows]))
    n = len(server)
    rng = np.random.default_rng(0)
    # Volume peaking at server 16:00 - the London/NY overlap on a NY-aligned clock.
    hour = server.hour.to_numpy()
    vol = 200 + 400 * np.exp(-((hour - 16) ** 2) / 8.0) + rng.normal(0, 5, n)
    return pd.DataFrame({
        "time_server": server,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "tick_volume": vol, "spread": 10,
    })


def test_verify_passes_on_a_correctly_converted_feed():
    df = synthetic_broker_weeks()
    df["time_utc"] = server_to_utc(df["time_server"]).tz_localize(None)

    v = verify_offset(df, "SYNTH")
    assert v["ok"], v["problems"]
    assert v["offsets_seen"] == [2, 3]
    assert v["week_open_server"].startswith("Mon 00:00")
    assert v["week_close_server"].startswith("Fri 23:45")
    assert v["week_open_utc"] == {"+2": "Sun 22:00", "+3": "Sun 21:00"}
    assert v["server_peak_stable"] is True
    assert v["volume_peak_hour"]["server"]["summer"] == 16
    assert v["volume_peak_hour"]["server"]["winter"] == 16


def test_verify_rejects_a_fixed_offset_conversion():
    """A broker clock treated as a constant +2 must be caught."""
    df = synthetic_broker_weeks()
    df["time_utc"] = df["time_server"] - pd.Timedelta(hours=2)

    v = verify_offset(df, "WRONG")
    assert not v["ok"], "a constant-offset conversion should not pass"
    assert any("week opens at" in p for p in v["problems"]), v["problems"]


def test_verify_rejects_a_feed_that_is_not_ny_aligned():
    """Shift the whole feed so the week no longer opens at Monday 00:00."""
    df = synthetic_broker_weeks()
    df["time_server"] = df["time_server"] + pd.Timedelta(hours=5)
    df["time_utc"] = server_to_utc(df["time_server"]).tz_localize(None)

    v = verify_offset(df, "SHIFTED")
    assert not v["ok"]
    assert any("week opens at server" in p for p in v["problems"]), v["problems"]


def test_verify_needs_enough_data():
    v = verify_offset(pd.DataFrame({"time_server": [], "time_utc": []}), "EMPTY")
    assert not v["ok"]
