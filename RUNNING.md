# How to run this

Everything you'd normally want to change lives in **one file: `config.yaml`**.
The Python files don't need editing.

---

## One-time setup

Open the folder in VS Code: **File → Open Folder →**
`C:\Users\charl\Downloads\TV_claude\quickwin15`

Open a terminal inside VS Code with **Ctrl + `** (backtick). It starts in the
project folder, which is what every command below assumes.

```bash
pip install -r requirements.txt
```

MT5 must be installed, logged in, and **running** for `fetch_data.py` only.
Everything else reads the local cache and works with MT5 closed.

---

## The commands, in order

```bash
# 1. Pull price history from MT5 (slow, only needed when you want fresh data)
python scripts/fetch_data.py --timeframes M15
python scripts/fetch_data.py --timeframes M1 --from-year 2007 --until-year 2025

# 2. Build the news calendar from forex_factory_cache.csv (seconds)
python scripts/convert_calendar.py

# 3. Eyeball breaker zones against TradingView (optional but recommended)
python scripts/preview_breakers.py

# 4. Run the backtest
python scripts/run_backtest.py
```

Steps 1 and 2 are already done — the cache and `data/news_calendar.csv` exist.
**Day to day you only need step 4.**

Results land in `out/`: `trades.csv`, `summary.csv`, the breakdowns, the
equity curves, and `trade_charts/`.

---

## Changing settings WITHOUT editing the file

These flags override `config.yaml` for one run, which is the safest way to
experiment — nothing is left changed afterwards.

```bash
# one pair only
python scripts/run_backtest.py --symbols EURJPY

# longs only (or short / both)
python scripts/run_backtest.py --symbols EURJPY --direction long

# a different date range
python scripts/run_backtest.py --symbols EURJPY --start-date 2022-01-01

# keep runs apart instead of overwriting out/
python scripts/run_backtest.py --symbols EURJPY --out-dir out_eurjpy

# no bad-tick cleaning (the only step that inspects future bars)
python scripts/run_backtest.py --symbols EURJPY --no-clean

# force flat after N M15 bars (96 = 24h), or 'none' to disable
python scripts/run_backtest.py --max-bars-in-trade 96
```

Combine freely:

```bash
python scripts/run_backtest.py --symbols EURJPY --direction long --start-date 2022-01-01 --out-dir out_test
```

**Always use `--out-dir` when comparing runs**, otherwise each run overwrites
the last one's `out/`.

---

## Where each setting lives in `config.yaml`

| Want to change | Setting | Line | Now |
|---|---|---|---|
| **Which pairs** | `data.symbols` | 26 | `[EURJPY, EURUSD, CHFJPY]` |
| **Backtest start** | `data.start_date` | 34 | `2007-01-03` |
| **Backtest end** | `data.end_date` | 35 | `2025-04-07` |
| **One pair's own dates** | `data.symbol_windows` | 42 | CHFJPY from 2023 |
| **Bad-tick cleaning on/off** | `data.clean_suspect_bars` | 53 | `true` |
| Commission per lot | `costs.commission_per_lot_round_turn` | 85 | `0.0` |
| **SMA length** | `indicators.sma.length` | 92 | `44` |
| **Breaker pivot length** | `indicators.breaker.length` | 97 | `5` |
| **Pivot tie handling** | `indicators.breaker.pivot_tie_mode` | 121 | `left_eq` — verified against TradingView, don't change |
| Zone = body not wick | `indicators.breaker.use_only_candle_body` | 98 | `false` |
| One live zone at a time | `indicators.breaker.single_active_zone` | 107 | `true` |
| **Longs / shorts on-off** | `strategy.longs_enabled` / `shorts_enabled` | 108–109 | `true` |
| **Entry rule** | `strategy.entry_mode` | 141 | `indicator_only` |
| **Where SMA is checked** | `strategy.sma_check` | 147 | `tap` |
| **When a zone expires** | `strategy.zone_expiry` | 153 | `session_end` |
| Stop buffer (pips) | `strategy.sl_buffer_pips` | 166 | `0.0` |
| **Risk:reward** | `strategy.risk_reward` | 167 | `2.0` |
| **10–20 pip SL filter** | `strategy.sl_distance_filter.enabled` | 172 | `false` |
| Stop must clear the zone | `strategy.sl_beyond_zone` | 181 | `true` |
| Break → breaker max wait | `strategy.max_bars_break_to_zone` | 193 | `44` |
| Zone → entry max wait | `strategy.max_bars_to_entry` | 196 | `24` |
| **Session window (UTC)** | `filters.session.start_utc` / `end_utc` | 177–178 | `12:00` / `23:00` |
| **News filter on/off** | `filters.news.enabled` | 217 | `true` |
| News blackout minutes | `filters.news.minutes_before` / `_after` | 186–187 | `30` / `30` |
| **Partial TP trigger** | `management.safe.partial_trigger_pips` | 236 | `10.0` |
| **Partial size** | `management.safe.partial_fraction` | 237 | `0.5` |
| Time stop (bars) | `management.max_bars_in_trade` | 255 | `null` |
| **Risk per trade %** | `reporting.risk_per_trade_pct` | 261 | `1.0` |
| Charts per direction | `reporting.sample_charts_per_direction` | 263 | `10` |

Line numbers drift if you add lines — search for the setting name instead.

**YAML rules that bite:** indentation is meaningful (spaces, never tabs),
"off" is written `false` not `False`, and "no limit" is `null` not `none`.

---

## Checking the port against TradingView

```bash
python scripts/preview_breakers.py --latest --symbols EURUSD EURJPY
```

Use `--latest`, never the plain random sample, for anything you intend to
compare. TradingView only draws breaker blocks inside the last **2000 bars**
(`per` in the Pine source) - roughly a month on M15 - so a randomly sampled
zone from last year shows nothing on its chart. `--latest` takes the most
recent zones and applies the same 2000-bar budget, prints a table of zone
times in BOTH UTC and broker time, and reports how many zones would be alive
at once if supersession were off.

This check found a real bug on 2026-10-27 (see README). Run it after any
change to the indicator.

## Adding a new pair

Changing `data.symbols` is **not enough** — there are four steps, and one of
them used to fail silently.

**1. Add it to `data.symbols`** (line 26):
```yaml
  symbols: [EURJPY, GBPJPY, CHFJPY]
```

**2. Check `data.pip_size` has it** (line ~59). JPY crosses are `0.01`,
everything else `0.0001`. The majors and JPY crosses are already listed. A
symbol that is missing stops the run with a clear error rather than guessing.

**3. Make sure the news calendar covers both its currencies.** This is the
one that matters: `convert_calendar.py` filters to a currency list when it
writes `news_calendar.csv`, so a calendar built for EUR/USD/JPY/CHF contains
**zero GBP rows**. Adding GBP to `filters.news.currencies` does nothing until
the file is rebuilt:

```bash
python scripts/convert_calendar.py --currencies EUR USD JPY CHF GBP AUD CAD NZD
```

This has already been run, so the calendar now holds 11,586 events across all
eight majors. The backtest also **stops with an error** if a pair's currency
has no events, so this can no longer pass unnoticed.

**4. Pull its price data** (MT5 must be open and logged in):
```bash
python scripts/fetch_data.py --symbols GBPJPY --timeframes M15
python scripts/fetch_data.py --symbols GBPJPY --timeframes M1 --from-year 2007 --until-year 2025
```
M15 takes a few minutes, M1 considerably longer — it is ~370,000 bars per
year. Both are cached, so it is a one-off.

Then run it:
```bash
python scripts/run_backtest.py --symbols GBPJPY --out-dir out_gbpjpy
```

Confirmed available on Axi, all 3-digit (pip = 0.01) unless noted: GBPJPY,
AUDJPY, CADJPY, NZDJPY, USDJPY, and GBPUSD / GBPAUD (5-digit, pip = 0.0001).

---

## Common things you might want

**Try a different risk:reward** — line 133:
```yaml
  risk_reward: 3.0
```

**Put the 10–20 pip SL filter back on** — line 138:
```yaml
  sl_distance_filter:
    enabled: true
```

**Trade all day instead of 12:00–23:00 UTC** — line 176:
```yaml
  session:
    enabled: false
```

**Change the partial from 10 pips / 50% to 15 pips / 33%** — lines 197–198:
```yaml
  safe:
    partial_trigger_pips: 15.0
    partial_fraction: 0.33
```

**Go back to requiring your lower-high / higher-low rule as well** — line 118:
```yaml
  entry_mode: both
```

---

## Checking you haven't broken anything

```bash
python -m pytest tests/ -q
```

88 tests. Run this after editing `config.yaml` — several tests read the real
config, so a YAML typo shows up here immediately rather than halfway through
a backtest.

---

## Running from VS Code without typing commands

`.vscode/launch.json` has ready-made configurations. Press **F5**, or open the
**Run and Debug** panel (Ctrl+Shift+D) and pick one from the dropdown:

* *Backtest — all pairs*
* *Backtest — EURJPY only*
* *Backtest — EURJPY longs 2022+*
* *Preview breaker charts*
* *Fetch M15 data*
* *Run tests*

F5 also lets you set breakpoints by clicking in the gutter left of a line
number, which is the easiest way to inspect what the engine is doing on a
particular trade.
