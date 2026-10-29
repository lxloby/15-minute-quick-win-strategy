# 15 Minute Quick Win — backtest

Backtest of the "15 Minute Quick Win" M15 strategy on Axi MT5 data for
EURJPY, EURUSD and CHFJPY, with a faithful Python port of
**Breaker Blocks with Signals [LuxAlgo]**.

---

## Status

| Piece | State |
|---|---|
| Project structure, config | done |
| MT5 loader + parquet cache | done |
| SMA44, pivots, swing structure | done |
| LuxAlgo breaker port | done |
| Strategy, trade manager, engine | done |
| Metrics, plots, reports | done |
| Unit tests (58, incl. no-lookahead) | done, passing |
| M15/M1 history pull | cap lifted, pulling |
| News calendar | built, **with a known gap — read below** |
| Visual sanity check vs TradingView | waiting on data |
| Full backtest + RESULTS.md | waiting on the above |

---

## Layout

```
config.yaml              every parameter in the spec
data/                    parquet cache, news_calendar.csv
out/                     trades.csv, summaries, charts
qw/
  config.py              loader; resolves "inherit" fields
  data/
    mt5_loader.py        MT5 pull + parquet cache
    timezones.py         broker server time -> UTC, with verification
    filters.py           session + news
  indicators/
    sma.py               SMA44
    swings.py            ta.pivothigh/low + swing-structure tracking
    breaker_blocks.py    the LuxAlgo port
  strategy/signals.py    break -> SMA -> zone pairing
  backtest/
    trade_manager.py     fills, partials, BE, intrabar sequencing
    engine.py            the bar loop
    metrics.py           R-based statistics
  reports/plots.py       candles, zones, equity curves
scripts/
  fetch_data.py          pull + cache + report history depth
  convert_calendar.py    HuggingFace FF dump -> news_calendar.csv
  build_news.py          live Forex Factory scraper (fallback)
  preview_breakers.py    3 bullish + 3 bearish zones per pair
  run_backtest.py        everything in section 6
tests/                   58 tests
```

Run order:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py              # pull + cache, report history depth
python scripts/convert_calendar.py        # forex_factory_cache.csv -> news_calendar.csv
python scripts/preview_breakers.py        # eyeball against TradingView FIRST
python scripts/run_backtest.py            # everything in section 6
```

`scripts/build_news.py` is the live Forex Factory scraper, kept as a
fallback; see "News data" for why it is not the default.

---

## The entry rule

What the backtest actually trades, in order. Everything here is a key in
`config.yaml`.

**1. A breaker block appears** — a bullish zone for a long, bearish for a
short, from the LuxAlgo port. Under `entry_mode: indicator_only` (the
default) the indicator's own market-structure shift *is* the structure test.
Under `entry_mode: both` it must additionally agree with a break of a lower
high / higher low, which is stricter.

**2. Price retraces and taps the zone's near edge** — the top for a long,
the bottom for a short — and fills there as a limit order.

**3. SMA44 must agree at that moment**, and the test is symmetric:

* **Long** — the entry price must be **above** SMA44
* **Short** — the entry price must be **below** SMA44

(44-period simple moving average of closes. One line in `engine.py`:
`ok = (edge > prev) if direction == LONG else (edge < prev)`.)

`sma_check: tap` judges this on the **last completed bar**, because the tap
bar's own close does not exist yet when a limit order fills — using it would
be lookahead. `sma_check: break` is the alternative, applying the test at the
structure break instead (the original spec's wording).

This filter does real work. On EURJPY over 2007–2025 it rejects **3,768 of
6,007 setups** — second only to the session window.

**4. Stop** goes at `stop_reference` (`structure`, `zone` or
`indicator_swing`), pushed beyond the far side of the block when
`sl_beyond_zone` is on, plus `sl_buffer_pips`.

**5. Target** at `risk_reward` × the stop distance (2R by default).

**6. The setup expires** at the end of the session it appeared in
(`zone_expiry: session_end`) if never tapped.

Entries are additionally gated by the session window (12:00–23:00 UTC) and
the news filter. Relevant settings:

| | Key | Default |
|---|---|---|
| SMA length | `indicators.sma.length` | **44** |
| SMA source | `indicators.sma.source` | `close` |
| Where SMA is tested | `strategy.sma_check` | `tap` |
| Entry rule | `strategy.entry_mode` | `indicator_only` |
| Risk:reward | `strategy.risk_reward` | `2.0` |
| Stop reference | `strategy.stop_reference` | `structure` |

## History depth

The Axi terminal originally reported `maxbars = 100000`, which capped M15
history at 2022-10-18 (100,000 M15 bars ≈ 2.85 years — the start date
matched the cap exactly, so it was the setting, not the broker).

*Max bars in chart* is now **Unlimited**. The setting alone was not enough:
the API reported the new value while still refusing anything before
2022-10-18. **Restarting the terminal** is what actually released it.

`scripts/fetch_data.py` prints the start date obtained per symbol and
timeframe and flags any series that comes back within 1% of `maxbars`.

## News data — built, but it has a hole

`data/news_calendar.csv` is built by `scripts/convert_calendar.py` from
`forex_factory_cache.csv`, the HuggingFace dump
[Ehsanrs2/Forex_Factory_Calendar](https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar).
It covers 2007-01-01 → **2025-04-07**, which is why `data.end_date` is set
to 2025-04-07: the backtest stops where the filter stops.

`scripts/build_news.py` (live Forex Factory scraper) is kept as a fallback.
It parses correctly — verified NFP 2024-01-05 → 13:30 UTC — but Forex
Factory starts returning HTTP 403 to scripted requests after a handful, so
it is not usable in bulk from here.

### The hole

Source timestamps carry explicit +03:30/+04:30 offsets (the dataset
author's Tehran timezone), so UTC conversion is exact **where the time is
real**. It often is not: **38% of the high-impact EUR/USD/JPY/CHF rows are
stamped at local 00:00**, which is a scrape that lost the time field rather
than a midnight release.

Spot-checks:

| Event | Dataset | Reality | |
|---|---|---|---|
| FOMC Federal Funds Rate 2024-01-31 | 19:00 UTC | 19:00 UTC | correct |
| ADP Non-Farm 2024-01-04 | 13:15 UTC | 13:15 UTC | correct |
| Non-Farm Employment 2024-01-05 | 2024-01-**04** 20:30 UTC | 13:30 UTC | **wrong day and time** |

Local 00:00 under those offsets lands at 19:30–20:30 UTC *the previous day* —
right inside the 12:00–23:00 UTC session. Keeping those rows would block the
wrong window while leaving the real release open, so they are **dropped** by
default. The filter therefore **under-blocks**: 3,260 events are missing,
including 161 Non-Farm Payrolls, 461 Unemployment Claims, 211 CPI y/y.

`--lost-time block-day` is the conservative alternative: it expands each
lost-time row to cover that whole UTC day for that currency, so nothing
slips through, at the cost of a lot of session time.

What actually goes into the filter: **5,266** high-impact events,
2007-01-03 → 2025-04-04 (USD 3,124 · EUR 1,255 · JPY 505 · CHF 382).

---

## Verification against TradingView (2026-10-27)

The port was checked against the real LuxAlgo indicator for the first time on
2026-10-27, on EURUSD and EURJPY M15. **It found a bug.**

### Why the check needed its own tooling

The Pine source carries `per = last_bar_index - bar_index <= 2000`, so
TradingView only ever *draws* breaker blocks inside the last 2000 bars —
about a month on M15. The original `preview_breakers.py` sampled randomly
from the last two years, so most of what it produced showed nothing on
TradingView and could not be compared at all.

`--latest` fixes that: it takes the most recent zones and runs the detector
with the same 2000-bar budget. It also runs it *both* ways and reports any
disagreement, because `per` gates the market-structure branch itself —
`MSS.dir` enters the window at 0 on TradingView while the backtest
(`recent_bars_only: null`) carries decades of state into it. On the checked
window the two agreed, so the comparison validates the backtest config too.

```bash
python scripts/preview_breakers.py --latest --symbols EURUSD EURJPY
```

### The bug

EURUSD produced **no zone at all between 21 and 27 October** while
TradingView drew one on the 26th. The cause, traced bar by bar:

* a bearish shift fired on 26 Oct 10:30 whose shape test failed, so it drew
  nothing — but still latched `MSS.dir = -1`, exactly as Pine does;
* the bearish condition then held for seven straight hours but was blocked
  by that latch, and no bullish shift ever fired to reset it;
* the bullish shift never fired because the zig-zag had not advanced — and
  it had not advanced because a pivot at 15:15 was **rejected on a tie**.

One discarded pivot suppressed every zone for six days. The zig-zag is
path-dependent and the latch has no timeout, so a single missed pivot
propagates indefinitely.

### Evidence

| | TradingView | port, `strict` | port, `left_eq` |
|---|---|---|---|
| EURUSD `-BB` | 26 Oct ~20:15 UTC, ~1.1176–1.1179 | **nothing** | **26 Oct 20:15, 1.11757–1.11794** |
| EURJPY `-BB` | 27 Oct ~08:45–09:00 UTC, ~175.940–176.021 | 27 Oct 09:00, 175.941–176.026 | same |
| EURJPY swing Ey | ~176.230 | 06:15 = 176.238 | same |
| EURJPY swing Cy | ~176.186 | 03:30 = 176.187 | same |

Two independent confirmations, no counterexample: where no tie occurs both
modes agree and both match TradingView; where a tie occurs only `left_eq`
matches. Across a 2000-bar window `left_eq` is a strict **superset** — it
added one zone on EURJPY and removed none.

Feed differences were ruled out first. The TradingView charts were FXCM
(EURUSD) and ICE (EURJPY), this project uses Axi, and they agree closely:
EURUSD SMA44 1.11627 against 1.11626 on the same bar.

### Cost of the bug

| Pair | Pivots (`strict` → `left_eq`) | Zones, full history |
|---|---|---|
| EURUSD | 128,853 → **144,117** (+11.8%) | 8,362 → **8,958** |
| EURJPY | 130,313 → **142,872** (+9.6%) | 7,472 → **7,790** |

**Every result in `RESULTS.md` predates this fix and needs re-running.** The
missing ~7% of zones are not spread evenly — the latch concentrates them
into multi-day dead stretches.

### Still open

* The single-zone assumption (`single_active_zone: true`) is supported by
  the Pine source — one `var block BB` whose boxes are *moved*, never
  recreated — but has not been confirmed by eye. With supersession off, up
  to **3** zones would be live at once in the checked window, so a chart
  showing two boxes at any moment would disprove it.
* `left_eq` was verified on the pivot-**high** side. The same code path
  handles lows, but no bullish zone has been matched explicitly.

## Regimes and black swans, per pair

Measured from the cached M15 itself, not recalled from memory. Dates are
UTC; `out/volatility_by_year.csv` has the full table.

### Annualised daily volatility (%)

| Era | EURUSD | EURJPY | CHFJPY | |
|---|---|---|---|---|
| 2002–2007 | 7–10 | 6–10 | 7–10 | pre-crisis |
| **2008–2009** | **11–13** | **15–19** | **15** | GFC |
| 2010–2013 | 7–11 | 11–14 | 10–14 | EZ debt crisis, Abenomics |
| 2014 | 5.7 | 6.7 | 6.5 | the quiet before 2015 |
| **2015** | **11.1** | 9.6 | **18.5** | CHF unpeg, ECB QE |
| 2016–2021 | 5–8 | 5–10 | 5–10 | |
| 2022 | 9.0 | 10.6 | 9.4 | Ukraine, rate shock, EURUSD parity |
| 2023–2026 | 5–7 | 7–9 | 6–9 | |

### EURUSD — largest daily moves

| Date | Move | What |
|---|---|---|
| 2008-10-29 | +3.58% | GFC, peak panic |
| 2009-03-18 | +3.11% | Fed announces QE1 |
| 2015-12-03 | +2.89% | ECB underdelivers; violent short squeeze |
| 2025-04-10 | +2.80% | US tariff turmoil |
| 2016-06-23 21:00 | 4.53% in one M15 bar | **Brexit referendum result** |
| 2016-11-08 22:00 | 3.56% in one M15 bar | US election night |

### EURJPY — largest daily moves

| Date | Move | What |
|---|---|---|
| 2008-10-28 | +7.17% (10.8% range) | GFC carry-trade unwind |
| 2013-04-04 | +4.95% | BoJ launches QQE under Kuroda |
| 2016-06-24 | −4.64% | Brexit; the 23rd 21:00 bar alone ranged **10.29%** |
| 2010-05-06 | −3.95% | US "flash crash" |
| 2019-01-02 22:00 | 4.50% in one M15 bar | **JPY flash crash** (thin New Year liquidity) |

### CHFJPY — why it is held to 2023+

| Date | Move | What |
|---|---|---|
| **2015-01-15** | **+17.87%, 31.5% daily range** | **SNB abandons the EUR floor** |
| 2011-09-06 | −7.99% | SNB *introduces* the floor |
| 2009-01-26 | −7.82% | GFC aftermath |
| 2016-06-23 21:00 | 8.72% in one M15 bar | Brexit |

The unpeg is not a tradeable regime — it is a central bank removing a peg,
with the 09:30 bar running 115.0 → 142.9 and the next ranging 131.4–151.2
while spreads tripled. Any backtest spanning it is really reporting one day.
`data.symbol_windows` therefore starts CHFJPY at **2023-01-01**. The cost is
sample size: ~2.3 years against ~18 for the other two, so CHFJPY numbers
should be read as indicative only.

## Data quality: phantom rollover bars

The Axi feed carries bars whose high or low never traded. The clearest case:

```
2015-01-14 22:00  CHFJPY  open 115.156  high 139.043  low 113.820  close 115.078
```

Every neighbouring bar sits at ~115.0, and the 139.0 print is **eleven hours
before** the SNB actually unpegged. It is a bad print, not a move.

This is systematic, not random:

| | |
|---|---|
| Suspect M15 bars found | 3,094 across the three pairs (~0.18%) |
| At 21:00 or 22:00 UTC | **96.5%** — the daily rollover |
| Inside the 12:00–23:00 trading window | **98.6%** |
| Concentrated in | **2014–2018** (500–650/yr), near-zero after 2019 |

M1 carries the identical artifact at the same hours. It matters because the
backtest reads highs and lows to decide whether a stop was hit, so a phantom
range closes trades that were never touched — and on a tight stop that is a
guaranteed fake loss, inside the session, in a quarter of the sample.

`data.clean_suspect_bars: true` clips these back to their neighbours. The bar
is kept and only the phantom extreme is trimmed, so bar counts and indices
never move. A genuine shock is left alone: the test requires both an extreme
range **and** that the move did not persist into the next bars, so the real
unpeg bar (2015-01-15 09:30) is correctly untouched while the bad tick the
day before is caught. `out/suspect_bars.csv` lists every one.

Set it to `false` to run on the untouched feed; the run reports the count
either way.

## Assumptions

Everything below is a judgement call that could reasonably have gone the
other way. Each is a single switch in `config.yaml`.

### Indicator port

1. **One live zone at a time.** The Pine script holds a single
   `var block BB`, so a new market-structure shift *overwrites* the previous
   breaker. Reproduced as `single_active_zone: true`, which means a zone can
   die by being superseded as well as by mitigation or cancellation. Setting
   it to `false` keeps every zone alive and departs from TradingView.
2. **`per` is ignored.** Pine limits drawing to the last 2,000 bars
   (`per = last_bar_index - bar_index <= 2000`). That is a rendering budget,
   not strategy logic, so `recent_bars_only: null` applies the indicator over
   all history. Set it to `2000` to reproduce the chart exactly.
3. **Pivot ties — `left_eq`, verified, was wrong until 2026-10-27.**
   Pine's `ta.pivothigh` lets the centre bar **equal** bars on its left and
   only requires it to beat the bar on its right. The port originally used a
   strict comparison on both sides, on the assumption that exact ties are
   rare on FX M15. That assumption was wrong — see "Verification against
   TradingView" below. `pivot_tie_mode: left_eq` is now the default;
   `strict` is retained only for comparison.
4. **Pivot lag is real.** `ta.pivothigh(hi, len, 1)` has `right = 1`, so a
   pivot is knowable exactly one bar after it prints — never on its own bar.
5. **Zone geometry.** Default inputs (`use_only_candle_body: false`) make the
   zone the full high–low of the first candle running in the breaker's
   direction, found by scanning back from the broken swing D towards the
   preceding swing C.
6. **Invalidation** = the indicator's own states: *mitigated* (close beyond
   the far edge) or *broken* / "cancel" (close through the mid-line). Both
   are togglable.

### Structure rule vs the indicator

The spec's entry rule 1 is a break of a **lower high** (longs). The LuxAlgo
script's own bullish shift keys off **lower lows** (`Ey < Cy`). These are
different tests, so they are evaluated separately: `strategy/signals.py`
applies the spec's rule and the indicator applies its own. A setup needs
both to agree. `swing_pivot_length` defaults to the indicator's `length` (5).

### Costs and fills

7. **Bars are bid.** A long fills at the ask (`zone_top + spread`) and exits
   on the bid; a short fills at the bid (`zone_bottom`) and exits on the ask.
   Each trade pays the spread exactly once, taken from the `spread` column of
   the entry bar.
8. **SL distance** is the *risk* distance entry → stop, which therefore
   includes the spread — not the raw structural distance. This is the number
   that sets position size. The 10–20 pip filter is **off**
   (`sl_distance_filter.enabled: false`): it was rejecting more than half of
   all surviving setups. The run still reports the full distribution.
8b. **The stop must clear the whole breaker block** (`sl_beyond_zone: true`):
   below the zone bottom for longs, above the zone top for shorts. A
   structural swing sometimes falls *inside* the block, and a stop there sits
   between the entry and the far edge, where ordinary movement within the
   zone takes it out rather than the structure actually failing. Risk is
   therefore never less than the zone height plus spread, which also means
   the zone's width now sets a floor on position sizing.
9. **Commission** enters as a fraction of risk:
   `per_lot_round_turn / (sl_pips × pip_value_per_lot)`. Account size cancels
   out. Default 0; set `costs.commission_per_lot_round_turn` for a Raw
   account. `pip_value_per_lot` falls back to 10.0 (USD-quoted) when MT5 does
   not supply it, which is approximate for JPY crosses.
10. **Fills are assumed to happen.** A tap of the zone (`low ≤ zone top` for
    longs) is treated as a filled limit order at the zone edge. No slippage,
    no partial fills, no requotes.

### Trade management

11. **Worse-case on ambiguity.** If stop and target both fall inside the same
    bar, or inside the same M1 bar, the stop is taken. Same for a stop that
    shares a bar with the +10 pip partial: the stop wins, and the trade is a
    full −1R.
12. **`intrabar_source`** records how each trade was resolved — `M1` where
    minute data covered every decisive bar, `M15` where it did not, `M1/M15`
    where the trade straddled the boundary. Report the two populations apart.
13. **Breakeven** is the entry price in exit-price space, so a BE stop nets
    zero before commission — the spread is already inside the entry.
14. **One position per symbol.** A trade that exits on bar *t* does not allow
    a new entry until *t+1*, which avoids inventing an intrabar ordering
    between an exit and the next fill.
15. **No time stop.** A position runs until stop, target, or the end of data
    (`exit_reason = end_of_data`).

### Time and filters

16. **Server time → UTC** uses the invariant that MT5 broker clocks cut the
    day at the New York close, so server wall clock = New York wall clock
    + 7 hours all year (NY at UTC−5 → server UTC+2; NY at UTC−4 → server
    UTC+3). The IANA database then supplies the transition dates rather than
    anything being hardcoded. `verify_offset` checks this against the feed:
    offsets seen must be exactly {2, 3}, weekly closes must land on 21:00 or
    22:00 UTC, and the tick-volume peak must be stable in *server* time
    (it would drift if the broker followed EU rather than US DST dates).
17. **Session** 12:00–23:00 UTC = 20:00–07:00 SGT. Entries only; open trades
    are managed outside the window.
17b. **Everything is UTC**, including every chart axis and `trades.csv`.
    Broker/MT5 time is UTC+2 in winter and UTC+3 in summer, so a TradingView
    chart set to the broker's timezone reads 2–3 hours ahead of these charts.
    The preview chart titles print both. The only exception is the
    hour-of-entry breakdown, which is SGT because the session was specified
    that way.
18. **News** blocks entries ±30 min around High-impact events for either
    currency in the pair.

---

## No lookahead

The guarantee is tested, not asserted. `tests/test_no_lookahead.py` runs the
whole pipeline on the full series and again on prefixes cut at four points,
and requires that:

* every breaker the prefix could know about is identical in direction, zone
  bounds, source bar and appearance bar;
* any state change that happens **at or after** the cut is *absent* from the
  prefix run, not merely different — this is the half that catches an
  indicator peeking forward;
* setups, SMA values and pivot flags match element for element.

Plus structural invariants: a pivot always confirms strictly after its own
bar, a zone is never tradeable on its appearance bar, and no setup references
a bar later than its own signal.

```
58 passed
```

---

## Known gaps

* No walk-forward or parameter robustness analysis — single pass, fixed
  parameters, as specified.
* `pip_value_per_lot` for JPY crosses is approximated when MT5 does not
  report it; only matters once commission is non-zero.
* M1 coverage is far shorter than M15, so most of the history resolves on
  M15 with the conservative rule. That biases results **down**, and the
  `summary_m1_period.csv` split is there to show by how much.
* Sample sizes per pair/direction/year will be small. Treat the breakdowns as
  descriptive, not predictive.
