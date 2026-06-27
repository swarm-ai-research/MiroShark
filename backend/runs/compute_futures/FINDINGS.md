# H100 compute-futures hedging — findings (bd k9w)

**Question.** Does the GTB compute-futures market (bd ja2) let an actor hedge
H100 price risk, and do its invariants hold at population scale? Framed around
the distinctive compute story: an **AI lab locking the cost of an H100 training
run** by going long a dated forward before it buys the compute.

**Method.** `scripts/compute_futures_experiment.py`, N=100 seeds. Settlement
spot is a deterministic, per-seed **log-normal** H100 rental price (median
$3.50/GPU-hr, log-vol 0.5) — right-skewed with occasional spikes, mimicking
observed GPU-rental volatility. Reproducible (no RNG; the spot is a hash of the
seed). Re-run: `uv run python -m scripts.compute_futures_experiment --n-seeds 100`.

Numbers below are from the committed N=100 run (`TABLE.md`, `aggregate.csv`,
`hedge_bands.svg`). Treat single-seed anecdotes as noise (CLAUDE.md); all claims
here are N=100 aggregates.

---

## Headline

**The mechanism delivers a textbook linear hedge on H100 cost.** Sweeping the
hedge ratio h (fraction of the Q=100 GPU-hr exposure locked forward at a fairly
priced forward), the seed-to-seed spread of net training-run cost collapses
monotonically, reaching **100% variance reduction at full hedge**:

| hedge ratio | net cost p10–p90 | std | variance reduction |
|---|---|---|---|
| 0.00 (unhedged) | $244 – $492 | $99.1 | 0.0% |
| 0.25 | $273 – $459 | $74.4 | 43.8% |
| 0.50 | $302 – $426 | $49.6 | 75.0% |
| 0.75 | $331 – $393 | $24.8 | 93.7% |
| 1.00 (full) | $360 – $360 | $0.0 | 100.0% |

The variance reductions track the analytic perfect-hedge curve 1−(1−h)² almost
exactly (43.75 / 75 / 93.75 / 100%) — the forward P&L offsets the spot exposure
linearly, as it should. See `hedge_bands.svg`: the p10–p90 band narrows to a
point at h=1.

The H100 spot it hedges against is genuinely volatile: mean **$3.60**, median
$3.47, p10–p90 **[$2.44, $4.92]**, full range [$1.64, $6.64] — i.e. the
unhedged lab faces a ~2× swing in run cost across seeds, which the full hedge
removes entirely.

---

## Honest caveats — read these before quoting the 100%

1. **Mean cost is flat by design, not by luck.** The forward is struck at the
   *fair* price (= expected spot), so all five rows have mean cost $360. A hedge
   reduces *variance*, not *expected* cost — that is the whole point, and the
   experiment is set up to isolate it. If you instead strike the forward *below*
   expected spot (e.g. at the median, which is cheaper than the mean under the
   right-skewed price), hedging *also* lowers expected cost — but that is a
   pricing win from the skew, **not** a property of hedging. Don't conflate them.

2. **Check A is the mechanism in isolation, by construction.** It mints one
   matched forward through the real order book and settles it through the real
   end-of-epoch engine, then reads the lab's coin delta. This proves the
   settlement *math* is an exact linear hedge and is coin-conserving. It
   deliberately does **not** model thin-market execution — no slippage, no
   partial fills, no trouble finding a counterparty at the fair price. The clean
   1−(1−h)² is the ceiling, not what a real desk would realize.

3. **Naive scripted policies do not capture this hedge (check C).** With the
   scripted `FuturesHedgerPolicy`, the hedged arm ends with *more* variance than
   the unhedged arm, not less. This is **not** evidence against hedging — the
   comparison is degenerate: the scripted unhedged producer is near-inert
   (final-coin std $0.12, it barely transacts), so any futures activity reads as
   pure added variance. The takeaway matches the wood study (bd-c7i): the
   mechanism makes a perfect hedge *available*, but discovering a *matched*
   hedge against a real exposure needs better policies than the current scripts.
   The gap between A (100%) and C (negative) is exactly the
   policy/execution gap, and it is the honest open problem here.

---

## Population robustness (check B, N=100)

A full compute-futures lineup (ZI compute desk + futures maker/taker/hedger)
run over 10 epochs × 100 seeds holds every invariant at scale:

- **Active market:** mean **96.6** contracts matched / seed, **81.4** settled / seed.
- **No negative balances:** min worker balance over all seeds = **0.00** (the
  settlement clamp holds; no account is driven below zero).
- **Zero-sum settlement:** long P&L + short P&L = 0 on every settled contract
  (structural — the env transfers one side's loss to the other), max residual 0.

---

## Bottom line

The bd ja2 compute-futures market is a *correct* hedging instrument for H100
price risk: at the mechanism level it removes up to 100% of cost variance
(N=100, scaling as 1−(1−h)²) while conserving coin and keeping balances
non-negative at population scale. What it does **not** yet show is that an
*agent* can realize this in a thin market — the scripted policies under-hedge,
and check A's clean payoff excludes execution friction. A follow-up should add a
hedger policy that sizes a matched forward against its realized spot exposure
and measure variance reduction *through the order book* (with slippage), to
close the A↔C gap.
