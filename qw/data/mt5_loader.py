"""
Pull OHLC history from a local MT5 terminal and cache it as parquet.

Two MT5 quirks shape everything here:

* A cold cache answers `(-2, 'Invalid params')` instead of "please wait", so
  every request needs a primer call and retries while history streams in.
* `copy_rates_from_pos` with more bars than the terminal holds also returns
  -2 rather than giving you what it has, so history is pulled in dated
  chunks with `copy_rates_range` instead.

The terminal's *Max bars in chart* setting (Tools > Options > Charts) is a
hard ceiling on what can ever come back. `history_report` surfaces it so a
truncated pull is never mistaken for a short-lived broker.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .timezones import server_to_utc

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - import guarded for non-Windows dev
    mt5 = None

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}
COLUMNS = ["time_utc", "time_server", "open", "high", "low", "close",
           "tick_volume", "spread", "real_volume"]


class MT5Error(RuntimeError):
    pass


def _tf(name: str):
    if name not in TIMEFRAMES:
        raise MT5Error(f"unsupported timeframe {name!r}")
    return getattr(mt5, TIMEFRAMES[name])


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def connect(terminal_paths) -> dict:
    """Attach to the first terminal that initialises. Returns terminal info."""
    if mt5 is None:
        raise MT5Error("MetaTrader5 package not installed (Windows only).")

    errors = []
    for path in terminal_paths:
        if not Path(path).exists():
            errors.append(f"{path}: not installed")
            continue
        if mt5.initialize(path=path):
            ti, ac = mt5.terminal_info(), mt5.account_info()
            if ac is None:
                mt5.shutdown()
                errors.append(f"{path}: terminal running but NO ACCOUNT logged in")
                continue
            return {
                "path": path,
                "company": ti.company,
                "name": ti.name,
                "maxbars": ti.maxbars,
                "login": ac.login,
                "server": ac.server,
                "currency": ac.currency,
            }
        errors.append(f"{path}: initialize failed {mt5.last_error()}")
        mt5.shutdown()

    raise MT5Error(
        "Could not reach a logged-in MT5 terminal. Tried:\n  "
        + "\n  ".join(errors)
        + "\nOpen the Axi terminal, log in, and leave it running."
    )


def disconnect() -> None:
    if mt5 is not None:
        mt5.shutdown()


def resolve_symbol(want: str, wait: float = 20.0):
    """
    Axi may decorate names (.a, -ECN). Find the real one, select it, and wait
    for the terminal to actually subscribe.

    A symbol that has never been looked at is not merely empty - the terminal
    answers `(-1, 'Terminal: Call failed')` to every history request until it
    has subscribed and started streaming. Selecting it is asynchronous, so
    polling for a first tick before asking for bars turns a hard failure into
    a short wait.
    """
    syms = mt5.symbols_get()
    if not syms:
        return None
    names = [s.name for s in syms]

    def norm(s: str) -> str:
        return s.upper().replace(".", "").replace("-", "").replace("_", "")

    target = norm(want)
    for pool in (
        [n for n in names if n == want],
        [n for n in names if norm(n) == target],
        [n for n in names if norm(n).startswith(target)],
    ):
        if pool:
            chosen = sorted(pool, key=len)[0]
            mt5.symbol_select(chosen, True)
            deadline = time.time() + wait
            while time.time() < deadline:
                tick = mt5.symbol_info_tick(chosen)
                if tick is not None and tick.time:
                    break
                time.sleep(1.0)
            return chosen
    return None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def earliest_bar(symbol: str, timeframe: str, probe_from_year: int = 2003,
                 retries: int = 3, sleep: float = 3.0):
    """
    First bar the terminal will serve, found by asking for data from a date
    earlier than any plausible history and taking what comes back.

    A cold symbol needs the terminal to build its timeseries first, which it
    only starts on being asked. The first few attempts therefore fail by
    design; backing off gives the download time to arrive instead of
    reporting "no history" for a symbol that has plenty.
    """
    tf = _tf(timeframe)
    attempts = max(retries, 6)
    for attempt in range(attempts):
        # Nudge the terminal into building the series, then ask properly.
        mt5.copy_rates_from_pos(symbol, tf, 0, 10)
        rates = mt5.copy_rates_from(
            symbol, tf, datetime(probe_from_year, 1, 1, tzinfo=timezone.utc), 2
        )
        if rates is not None and len(rates):
            return pd.to_datetime(rates[0]["time"], unit="s")
        time.sleep(min(sleep * (attempt + 1), 20.0))
    return None


def _chunk(symbol: str, tf, start: datetime, end: datetime,
           retries: int, sleep: float):
    for attempt in range(retries):
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is not None and len(rates):
            return rates
        time.sleep(sleep)
    return None


def fetch_history(symbol: str, timeframe: str, probe_from_year: int = 2003,
                  retries: int = 3, sleep: float = 3.0,
                  progress=None, until_year: int | None = None) -> pd.DataFrame:
    """
    Pull everything the terminal has, in yearly chunks, oldest-first.

    `until_year` stops the walk early, which matters for M1: the terminal
    serves ~370k bars per symbol per year, so pulling past the end of the
    backtest window costs a lot of download for data nothing will read.
    """
    tf = _tf(timeframe)

    # Primer - wakes the terminal's download for this symbol/timeframe.
    for _ in range(retries):
        if mt5.copy_rates_from_pos(symbol, tf, 0, 10) is not None:
            break
        time.sleep(sleep)

    first = earliest_bar(symbol, timeframe, probe_from_year, retries, sleep)
    if first is None:
        raise MT5Error(
            f"{symbol} {timeframe}: terminal returned no history after retrying "
            f"({mt5.last_error()}).\n"
            f"  This usually means the terminal has not downloaded it yet rather "
            f"than that it does not exist.\n"
            f"  Open a {symbol} {timeframe} chart in MT5, scroll back once to "
            f"force the download, then re-run."
        )

    frames = []
    year = int(first.year)
    last_year = until_year or datetime.now(timezone.utc).year
    while year <= last_year:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        rates = _chunk(symbol, tf, start, end, retries, sleep)
        if rates is not None and len(rates) > 1:   # 1 bar == placeholder, no real history
            frames.append(pd.DataFrame(rates))
            if progress:
                progress(f"    {symbol} {timeframe} {year}: {len(rates):,} bars")
        year += 1

    if not frames:
        raise MT5Error(f"{symbol} {timeframe}: every yearly chunk came back empty")

    df = pd.concat(frames, ignore_index=True)
    df["time_server"] = pd.to_datetime(df["time"], unit="s")
    df = (df.drop(columns=["time"])
            .drop_duplicates(subset="time_server")
            .sort_values("time_server")
            .reset_index(drop=True))
    # Stored tz-NAIVE but always meaning UTC. Keeping one convention
    # everywhere avoids tz-aware/naive mismatches in numpy conversions.
    df["time_utc"] = server_to_utc(df["time_server"]).tz_localize(None)
    for col in ("tick_volume", "spread", "real_volume"):
        if col not in df.columns:
            df[col] = 0
    return df[COLUMNS]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    return Path(cache_dir) / f"{symbol}_{timeframe}.parquet"


def year_dir(cache_dir: Path, symbol: str, timeframe: str) -> Path:
    """Year-partitioned cache, used for M1 - see `fetch_history_per_year`."""
    return Path(cache_dir) / f"{symbol}_{timeframe}"


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["time_utc"], utc=True)
    df["time_utc"] = t.dt.tz_localize(None)        # naive, meaning UTC
    if "time_server" in df.columns:
        df["time_server"] = pd.to_datetime(df["time_server"])
    return df


def load_cached(cache_dir: Path, symbol: str, timeframe: str,
                columns=None, start=None, end=None):
    """
    Read a cached series, from either the single-file or the year-partitioned
    layout.

    `columns` matters for M1: a full 18-year pull is ~6.7M rows, and dropping
    the columns the backtest never reads (time_server, volumes) roughly halves
    the memory it occupies. `start`/`end` skip whole year files.
    """
    single = cache_path(cache_dir, symbol, timeframe)
    if single.exists():
        return _finish(pd.read_parquet(single, columns=columns))

    d = year_dir(cache_dir, symbol, timeframe)
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    if start is not None or end is not None:
        lo = pd.Timestamp(start).year if start is not None else -1
        hi = pd.Timestamp(end).year if end is not None else 9999
        files = [f for f in files if lo <= int(f.stem) <= hi]
    if not files:
        return None
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    del frames
    return _finish(df)


def fetch_history_per_year(symbol: str, timeframe: str, out_dir: Path,
                           probe_from_year: int = 2003, retries: int = 3,
                           sleep: float = 3.0, progress=None,
                           until_year: int | None = None,
                           from_year: int | None = None) -> dict:
    """
    Pull year by year and write each year straight to its own parquet.

    M1 is ~372k bars per symbol per year, so twenty-odd years will not survive
    a single `pd.concat` - that briefly holds both the parts and the whole,
    and the previous attempt died with a MemoryError followed by a segfault.
    Writing as we go keeps peak memory at one year.
    """
    tf = _tf(timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(retries):
        if mt5.copy_rates_from_pos(symbol, tf, 0, 10) is not None:
            break
        time.sleep(sleep)

    first = earliest_bar(symbol, timeframe, probe_from_year, retries, sleep)
    if first is None:
        raise MT5Error(f"no {timeframe} history at all for {symbol}: {mt5.last_error()}")

    year = max(int(first.year), from_year or 0)
    last_year = until_year or datetime.now(timezone.utc).year
    written, total = [], 0

    while year <= last_year:
        dest = out_dir / f"{year}.parquet"
        if dest.exists():
            if progress:
                progress(f"    {symbol} {timeframe} {year}: cached")
            written.append(year)
            year += 1
            continue

        rates = _chunk(symbol, tf,
                       datetime(year, 1, 1, tzinfo=timezone.utc),
                       datetime(year + 1, 1, 1, tzinfo=timezone.utc),
                       retries, sleep)
        if rates is not None and len(rates) > 1:
            df = pd.DataFrame(rates)
            df["time_server"] = pd.to_datetime(df["time"], unit="s")
            df = (df.drop(columns=["time"])
                    .drop_duplicates(subset="time_server")
                    .sort_values("time_server")
                    .reset_index(drop=True))
            df["time_utc"] = server_to_utc(df["time_server"]).tz_localize(None)
            for col in ("tick_volume", "spread", "real_volume"):
                if col not in df.columns:
                    df[col] = 0
            df[COLUMNS].to_parquet(dest, index=False)
            total += len(df)
            written.append(year)
            if progress:
                progress(f"    {symbol} {timeframe} {year}: {len(df):,} bars")
            del df
        year += 1

    if not written:
        raise MT5Error(f"{symbol} {timeframe}: every yearly chunk came back empty")
    return {"years": written, "bars": total, "dir": str(out_dir)}


def get_bars(cfg, symbol: str, timeframe: str, refresh: bool = False,
             progress=None, until_year: int | None = None) -> pd.DataFrame:
    """Cached history, pulling from MT5 only when the parquet is missing."""
    cache_dir = cfg.path("cache_dir")
    if not refresh:
        cached = load_cached(cache_dir, symbol, timeframe)
        if cached is not None and not cached.empty:
            return cached

    df = fetch_history(
        symbol, timeframe,
        probe_from_year=cfg.dotted("mt5.probe_from_year", 2003),
        retries=cfg.dotted("mt5.retries", 3),
        sleep=cfg.dotted("mt5.retry_sleep", 3),
        progress=progress,
        until_year=until_year,
    )
    cache_path(cache_dir, symbol, timeframe).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path(cache_dir, symbol, timeframe), index=False)
    return df


def history_report(df: pd.DataFrame, symbol: str, timeframe: str,
                   maxbars: int | None = None) -> dict:
    """Facts about a pull, including whether Max bars in chart truncated it."""
    rep = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": len(df),
        "start_utc": df["time_utc"].iloc[0] if len(df) else None,
        "end_utc": df["time_utc"].iloc[-1] if len(df) else None,
    }
    if maxbars:
        # Within 1% of the ceiling is a cap, not a coincidence.
        rep["maxbars"] = maxbars
        rep["likely_truncated"] = len(df) >= maxbars * 0.99
    return rep
