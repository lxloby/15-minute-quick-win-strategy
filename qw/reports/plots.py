"""
Chart helpers - candles, SMA44, confirmed swings and breaker zones.

The point of these is comparison against TradingView, so the drawing rules
mirror what the indicator actually does: a zone box runs from the candle it
was cut from to the bar where it died (mitigated, cancelled or superseded),
and a swing marker is placed on the pivot bar, not on the bar that confirmed
it. If a chart here disagrees with TradingView, the port is wrong - that is
the whole idea.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ..indicators.breaker_blocks import BULL
from ..indicators.swings import HIGH, confirmed_pivots

BULL_FILL = "#0cb51a"
BEAR_FILL = "#ff1100"
UP_C, DOWN_C = "#26a69a", "#ef5350"


def _candles(ax, sub: pd.DataFrame) -> None:
    """Draw candles at integer positions - see `plot_window` for why."""
    x = np.arange(len(sub))
    up = sub["close"].to_numpy() >= sub["open"].to_numpy()
    colors = np.where(up, UP_C, DOWN_C)
    ax.vlines(x, sub["low"], sub["high"], color=colors, linewidth=0.8, zorder=2)
    lo = np.minimum(sub["open"], sub["close"])
    ax.bar(x, height=np.abs(sub["close"] - sub["open"]).clip(lower=1e-9),
           bottom=lo, width=0.7, color=colors,
           edgecolor=colors, linewidth=0.4, zorder=3)


def _time_axis(ax, times: pd.DatetimeIndex, n_ticks: int = 9) -> None:
    """Label an integer x-axis with the timestamps the positions stand for."""
    n = len(times)
    step = max(1, n // n_ticks)
    pos = np.arange(0, n, step)
    ax.set_xticks(pos)
    ax.set_xticklabels([times[p].strftime("%d %b\n%H:%M") for p in pos], fontsize=8)
    ax.set_xlim(-1, n)


def plot_window(df: pd.DataFrame, lo_bar: int, hi_bar: int, sma_vals,
                breakers=None, pivot_length: int | None = None,
                trade: dict | None = None, title: str = "",
                out_path: str | Path | None = None, sma_label: str = "SMA44",
                pivots=None):
    """
    Draw bars [lo_bar, hi_bar] with every overlay that matters.

    `trade` may carry entry_price / sl / tp / entry_time / exit_time and is
    drawn as horizontal levels plus entry and exit markers.

    Pass `pivots` (from `confirmed_pivots`) when drawing several charts off
    the same series - recomputing them per chart means a full pass over
    600k+ bars every time, which is both slow and needlessly memory-hungry.
    """
    lo_bar = max(0, lo_bar)
    hi_bar = min(len(df) - 1, hi_bar)
    sub = df.iloc[lo_bar:hi_bar + 1]
    if sub.empty:
        raise ValueError("empty plot window")

    times = pd.DatetimeIndex(sub["time_utc"])
    all_times = pd.DatetimeIndex(df["time_utc"])

    def pos(bar: int) -> float:
        """Window-relative x position for an absolute bar index."""
        return float(np.clip(bar - lo_bar, -1, len(sub)))

    def pos_of_time(ts) -> float:
        return pos(int(np.searchsorted(all_times.to_numpy(),
                                       np.datetime64(pd.Timestamp(ts), "ns"))))

    fig, ax = plt.subplots(figsize=(15, 7.5))
    _candles(ax, sub)

    x = np.arange(len(sub))
    sma_slice = np.asarray(sma_vals)[lo_bar:hi_bar + 1]
    ax.plot(x, sma_slice, color="#f0b90b", linewidth=1.4, label=sma_label, zorder=4)

    # -- swings -----------------------------------------------------------
    if pivots is None and pivot_length:
        pivots = confirmed_pivots(df["high"].to_numpy(float),
                                  df["low"].to_numpy(float), pivot_length, 1)
    if pivots:
        for p in pivots:
            if lo_bar <= p.bar <= hi_bar:
                ax.plot(pos(p.bar), p.price,
                        marker="v" if p.kind == HIGH else "^",
                        color="#9aa0a6", markersize=7, zorder=5)

    # -- breaker zones -----------------------------------------------------
    for blk in (breakers or []):
        end = blk.dead_bar()
        end = len(df) - 1 if end is None else end
        if blk.source_bar > hi_bar or end < lo_bar:
            continue
        a, b = pos(blk.source_bar), pos(min(end, len(df) - 1))
        colour = BULL_FILL if blk.direction == BULL else BEAR_FILL
        ax.add_patch(Rectangle((a, blk.bottom), max(b - a, 1.0),
                               blk.top - blk.bottom,
                               facecolor=colour, alpha=0.18,
                               edgecolor=colour, linewidth=1.0, zorder=1))
        ax.hlines(blk.mid, a, b, color=colour, linestyles=":", linewidth=0.9, zorder=2)
        if lo_bar <= blk.appeared_bar <= hi_bar:
            ap = pos(blk.appeared_bar)
            ax.axvline(ap, color=colour, linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            ax.annotate("+BB" if blk.direction == BULL else "-BB",
                        (ap, blk.top if blk.direction == BULL else blk.bottom),
                        textcoords="offset points", xytext=(4, 6 if blk.direction == BULL else -12),
                        color=colour, fontsize=9, fontweight="bold")

    # -- the trade ---------------------------------------------------------
    if trade:
        for key, colour, style in (("entry_price", "#2157f3", "-"),
                                   ("sl", "#d32f2f", "--"),
                                   ("tp", "#2e7d32", "--")):
            v = trade.get(key)
            if v is not None and not pd.isna(v):
                ax.axhline(v, color=colour, linestyle=style, linewidth=1.1, alpha=0.9)
                ax.annotate(f"{key.replace('_price','').upper()} {v:.5g}",
                            (len(sub) - 1, v), textcoords="offset points", xytext=(6, 2),
                            color=colour, fontsize=8)
        for key, marker, colour in (("entry_time", "o", "#2157f3"),
                                    ("partial_time", "s", "#f0b90b"),
                                    ("exit_time", "X", "#111111")):
            ts = trade.get(key)
            if ts is None or pd.isna(ts):
                continue
            px = pos_of_time(ts)
            ax.axvline(px, color=colour, linewidth=0.9, alpha=0.5)
            price = trade.get("entry_price" if key == "entry_time" else "exit_price")
            if price is not None and not pd.isna(price):
                ax.plot(px, price, marker=marker, color=colour, markersize=9, zorder=7)

    # Integer x-axis, so weekend gaps collapse exactly as they do on
    # TradingView. Plotting against real datetimes leaves a dead band every
    # weekend and draws the SMA as a straight line across it.
    _time_axis(ax, times)
    ax.set_title(title, fontsize=11)
    # Say it explicitly: the axis is UTC, while the broker chart these are
    # compared against runs on GMT+2/+3, so the two differ by 2-3 hours.
    ax.set_xlabel("time (UTC)  -  broker/MT5 time is this +2 (winter) / +3 (summer)",
                  fontsize=9)
    ax.set_ylabel("price")
    ax.grid(alpha=0.15, zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return str(out_path)
    return fig


def plot_equity(curves: dict, title: str, out_path: str | Path):
    """Overlay several R-based equity curves on one axis."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, df in curves.items():
        if df is None or df.empty:
            continue
        ax.plot(pd.DatetimeIndex(df["entry_time"]), df["cum_r"],
                linewidth=1.5, label=f"{label}  ({df['cum_r'].iloc[-1]:+.1f}R)")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("cumulative R")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def plot_sl_histogram(sl_pips, out_path, min_pips=10, max_pips=20):
    """Stop-size distribution with the configured filter band marked."""
    s = pd.Series(list(sl_pips), dtype=float).dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    if not s.empty:
        upper = float(np.nanpercentile(s, 99))
        ax.hist(s.clip(upper=upper), bins=60, color="#4a7fb5", edgecolor="white")
        ax.axvspan(min_pips, max_pips, color="#2e7d32", alpha=0.15,
                   label=f"filter {min_pips}-{max_pips} pips")
        inside = ((s >= min_pips) & (s <= max_pips)).mean() * 100
        ax.set_title(f"SL distance distribution - {inside:.1f}% inside the filter "
                     f"(n={len(s):,}, 99th pct clipped at {upper:.0f})")
    ax.set_xlabel("SL distance (pips)")
    ax.set_ylabel("candidate entries")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
