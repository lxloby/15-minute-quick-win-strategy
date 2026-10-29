# 15 Minute Quick Win — results

**Bottom line: no edge was found. Across 2,601 trades the strategy returns
−135R, and every pair now finishes negative or flat. The single
configuration that still looks profitable does not survive one pip of
slippage.**

Numbers produced 2026-10-27 with `pivot_tie_mode: left_eq`, after the port
was verified against the live LuxAlgo indicator on TradingView. See
"The bug that moved these numbers" below for what changed and why it matters.

---

## What was run

Entry rule as finally specified (`entry_mode: indicator_only`):

1. A LuxAlgo breaker block appears — its own market structure shift is the
   structure test.
2. Price retraces and taps the zone's near edge (top for longs, bottom for
   shorts) and fills there as a limit order.
3. At the tap, the entry price is on the right side of **SMA44**, judged on
   the last completed bar so nothing peeks at the tap bar's close.
4. Stop beyond the far side of the block, target at **2R**.
5. Setup expires at **23:00 UTC** the day the zone appears if never tapped.
6. Entries only 12:00–23:00 UTC, never within ±30 min of high-impact news.

Costs: spread charged once per trade from the feed (longs fill at the ask,
shorts exit at the ask). Commission 0. **Fills are assumed to happen at the
zone edge with no slippage** — optimistic, and section "Slippage" shows how
much that assumption is worth.

| Pair | Period | M15 bars |
|---|---|---|
| EURUSD | 2007-01-03 → 2025-04-07 | 452,314 |
| EURJPY | 2007-01-03 → 2025-04-07 | 452,404 |
| CHFJPY | **2023-01-01** → 2025-04-07 | 56,359 |

CHFJPY is held to 2023+ deliberately — its earlier history is dominated by
the SNB era (the 2015-01-15 unpeg moved it **+17.9% in a day**). The window
starts 2007 because that is where the news calendar starts.

Every trade resolved on **M1** (100% `intrabar_source = M1`), so no trade
needed the conservative same-bar assumption.

---

## Headline — Baseline (no partials, full size to 2R or stop)

| Pair | Years | Trades | Win rate | Total | PF | Max DD | t |
|---|---|---|---|---|---|---|---|
| EURJPY | 18.3 | 1,246 | 33.31% | **−1.0R** | 0.999 | 39.4% | −0.02 |
| EURUSD | 18.2 | 1,172 | 31.06% | **−80.0R** | 0.901 | 68.1% | −1.68 |
| CHFJPY | 2.2 | 183 | 23.50% | **−54.0R** | 0.614 | 47.5% | **−3.13** |
| **Pooled** | — | **2,601** | **31.60%** | **−135.0R** | — | — | **−1.90** |

Break-even at 1:2 RR is **33.33%**. Pooled came in **1.73 points below** it.

EURJPY lands on 33.31% against a break-even of 33.33% — a coin flip to two
decimal places, t = −0.02.

### Safe method (50% off at +10 pips, stop to breakeven)

| Pair | Trades | Win rate | Total | PF | Max DD |
|---|---|---|---|---|---|
| EURJPY | 1,582 | 80.40% | −12.3R | 0.960 | 24.8% |
| EURUSD | 1,462 | 77.15% | −15.0R | 0.955 | 40.9% |
| CHFJPY | 234 | 67.52% | −47.1R | 0.369 | 39.1% |

Those 67–80% win rates are the most misleading numbers in this project. A
trade that banks +0.08R and then stops at breakeven counts as a "win". All
three are losing.

---

## Is any of it real?

**Pooled: 2,601 trades, expectancy −0.0519R, 95% CI −0.1055R to +0.0017R.**

That sample can detect an edge of **2.6 percentage points** above break-even
at 80% power. The observed figure is **1.73pp below**. This is not "too small
a sample to tell" — it is a properly powered test that found nothing.

Only CHFJPY is individually significant, and it is significantly **negative**
(t = −3.13).

---

## What drove the losses: the short side

| Pair | Longs | Shorts |
|---|---|---|
| CHFJPY | −4.0R (31.82%) | **−50.0R (15.79%)** |
| EURUSD | −14.0R (32.50%) | **−66.0R (29.72%)** |
| EURJPY | −3.0R (33.19%) | +2.0R (33.45%) |

On the two pairs that lost materially, **the shorts did nearly all of it**.
CHFJPY rose **+22.0%** over its window; the short rule fires when price
closes below SMA44, which in a sustained uptrend means selling pullbacks.
EURJPY, which had no single dominant trend across 18 years, is flat both ways.

This held across three different entry definitions, so it is a property of
the rule rather than of one market.

---

## Stop placement

Four ways to place the stop, EURJPY baseline, same 6,000+ setups:

| Option | Median SL | Trades | Win rate | Total | PF |
|---|---|---|---|---|---|
| A zone edge | 13.5p | 1,593 | 31.45% | **−90.0R** | 0.918 |
| D zone +5p | 18.5p | 1,572 | 31.30% | **−96.0R** | 0.911 |
| C indicator swing | 25.5p | 1,495 | 32.31% | **−46.0R** | 0.955 |
| B structure | 43.7p | 1,246 | 33.31% | **−1.0R** | 0.999 |

On the **1,235 trades taken under all four rules** — same entries, only the
stop moved — the ordering is identical: A −89R, D −101R, C −26R, B −11R.

**Without a partial, wider stops are better.** At a fixed 1:2 RR, tightening
the stop tightens the target in proportion, so you gain nothing in
reward-for-risk and lose the room price needs. A 13.5-pip stop sits beneath a
zone that is itself ~10 pips tall, so movement *inside the block you just
bought from* takes you out.

**This also means the original 10–20 pip SL rule is actively harmful here.**
Option D puts 53.8% of trades in that band and loses 96R. The spec's
tight-stop intent does not survive contact with LuxAlgo zones.

---

## Partial take-profit: 0.25R is best everywhere

Total R by stop placement (rows) and partial trigger (columns), EURJPY:

| Stop | **0.25R** | 0.50R | 0.75R | 1.00R | 1.50R | none |
|---|---|---|---|---|---|---|
| **A zone** | **+58.3** | +36.2 | +9.6 | +6.0 | −87.7 | −90.0 |
| **D zone+5p** | **+32.7** | −6.2 | −22.1 | −65.0 | −93.0 | −96.0 |
| **C indicator** | **+38.5** | 0.0 | −46.1 | −36.5 | −57.0 | −46.0 |
| **B structure** | **+14.5** | −41.8 | −56.8 | −30.0 | 0.0 | −1.0 |

**The 0.25R column wins in every row.** Profit factor and drawdown agree —
A+0.25R has the best PF (1.193) and much the lowest drawdown of any A cell
(15.1R versus 113R).

Note this **reverses the stop-placement ordering above**. Without a partial,
zone-edge is worst (−90R); with a 0.25R partial it is best (+58.3R). The two
choices interact, and neither can be judged while holding the other fixed.

**Why a fixed-pip partial fails.** The spec's +10 pips means 0.74R on a
13.5-pip stop and 0.23R on a 43.7-pip one — two entirely different
strategies from one config line. `partial_trigger_r` scales with the stop and
makes the comparison meaningful.

---

## Slippage: what kills the one positive result

A zone + 0.25R reaches **t = +2.49**, the only statistically significant
positive result in this project. But its partial sits a median of **3.4 pips**
from entry against a **1.3 pip** spread.

Modelling slippage on stop exits only (stops are market orders and slip on
whatever size is still open; targets and partials are limits and do not slip):

| Cell | 0.0p | 0.5p | **1.0p** | 2.0p | t(0p) | t(1p) |
|---|---|---|---|---|---|---|
| A zone + 0.25R | **+58.3** | +22.7 | **−12.8** | −83.8 | +2.49 | −0.52 |
| C indicator + 0.25R | +38.5 | +18.9 | −0.6 | −39.8 | +1.79 | −0.03 |
| B structure + 0.25R | +14.5 | +2.7 | −9.1 | −32.8 | +0.66 | −0.41 |
| B structure + none | −1.0 | −12.3 | −23.6 | −46.2 | −0.02 | −0.47 |

**Every cell is negative by one pip.** The arithmetic: on a 13.5-pip stop,
one pip of slippage is **7.4% of R**; on a 43.7-pip stop it is 2.3%. Tight
stops are three times more exposed, and this strategy takes ~1,600 of them.

Taken at face value and ignoring slippage, the best configuration is
**+58.3R over 18.3 years** — about **+3.2% a year** at 1% risk, against a
~15% drawdown. At 0.5 pips of slippage that falls to +1.2% a year. Past about
0.8 pips it is a loss.

The model also **flatters** these cells in one respect: limit orders do not
slip, but they can fail to fill. A 3.4-pip partial that price touches and
reverses from is recorded as filled here; in reality it might not be.

---

## The bug that moved these numbers

The first version of this port required a pivot's centre bar to be strictly
higher than all five bars on its left. **Pine's `ta.pivothigh` allows ties on
the left** and only requires the centre to beat the bar on its right.

Verified against the live indicator on TradingView on 2026-10-27. The strict
comparison discarded **~11% of all pivots** (EURUSD 128,853 → 144,117). Worse
than 11% sounds, because the zig-zag is path-dependent and `MSS.dir` latches
with no timeout: one missed pivot suppresses every zone for days. EURUSD drew
no zone at all between 21 and 27 October 2026 while TradingView drew one on
the 26th.

What it did to the conclusions:

| | Old (strict) | New (left_eq) |
|---|---|---|
| EURJPY baseline | **+29.0R**, 34.12%, t +0.58 | **−1.0R**, 33.31%, t −0.02 |
| EURUSD baseline | −84.0R | −80.0R |
| CHFJPY baseline | −49.0R | −54.0R |
| A zone + 0.25R | +62.9R, t +2.70 | +58.3R, t +2.49 |

**EURJPY's +29R was an artifact of the bug.** It was always thin — t = +0.58,
ten extra wins out of 1,234 — and 7% more zones erased it. The stop-placement
and partial-trigger conclusions survived intact.

That is the strongest argument in this document for checking an indicator
port against the real thing before trusting anything built on it.

---

## Caveats

**These numbers are optimistic.** Every fill is assumed to happen at the zone
edge with no slippage, requotes or partial fills. A resting limit at a level
everyone can see is exactly where fills are worst.

**Hourly and yearly breakdowns are not tradeable signals.** With 24 hourly
buckets and an overall edge of zero, some will look excellent by chance.
Choosing them after seeing the data is curve-fitting.

**Selection effects.** The three headline runs were specified in advance. The
longs-only and 2022+ slices were chosen after seeing results, and they
behaved accordingly: EURJPY longs-only over 2022–2025 gave +11R, but on
2007–2021 — the years not used to pick it — it gave **−15R**, with the win
rate dropping from 35.71% to 32.49%. Combined over 745 trades: **−4R**, one
trade below a coin flip.

**Data quality.** 3,094 phantom rollover bars were found and clipped — 96.5%
at 21:00/22:00 UTC, concentrated in 2014–2018, inside the trading window. A
genuine shock (the 2015 unpeg) is correctly left alone. The detector itself
inspects neighbouring bars, so it is the one step in the pipeline that is not
causal; on EURJPY it was worth +15R under the old settings.

**News coverage has a hole.** 38% of high-impact rows in the source dataset
lost their timestamps and were dropped, so the filter under-blocks — notably
missing most Non-Farm Payrolls prints.

**Still unverified.** `single_active_zone` matches the Pine source by reading
but has not been confirmed by eye on TradingView; with supersession off, up
to three zones would be live at once. And `left_eq` was confirmed on the
pivot-**high** side only — the same code path handles lows, but no bullish
zone has been matched explicitly.

---

## Verdict

The strategy as specified does not have an edge on these three pairs over
this period.

* 2,601 trades, pooled expectancy **−0.05R**, win rate **1.73pp below
  break-even**, t = −1.90.
* Enough power to detect a 2.6pp edge; none present.
* Every pair now negative or flat. The one previously positive pair was a
  bug artifact.
* The single significant positive configuration (zone edge + 0.25R partial,
  t = +2.49) **turns negative at one pip of slippage**, and its edge depends
  on taking profit 3.4 pips from entry.

What is solid and worth carrying forward:

1. **The short side fails badly in a sustained trend** — consistent across
   pairs and across three entry definitions.
2. **A partial must be a fraction of R, not a fixed pip distance**, or it
   means something different on every pair and every stop setting.
3. **0.25R is the best partial trigger** at every stop placement tested.
4. **Tight stops are an execution-cost trap.** They look best on paper and
   are the most fragile in practice.

If this is to be pursued, the **entry** needs to change — not the
management, not the parameters.
