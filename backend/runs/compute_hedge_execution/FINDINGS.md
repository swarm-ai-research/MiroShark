# H100 hedging through the order book — findings (bd 5ij)

**Question.** bd k9w left an A↔C gap. Check A proved the compute-futures
*mechanism* is a perfect linear hedge on H100 cost (up to 100% variance
reduction, N=100) — but in isolation, minting one matched forward with no
execution friction. Check C showed the *naive* scripted hedger realized none of
it. So: **how much of the 100% ceiling survives when you hedge through a real,
finite-depth forward book — paying the spread and possibly failing to fill?**

**Method.** `scripts/compute_hedge_execution_experiment.py`, N=100 seeds. A
producer with a known exposure (sell Q=100 H100-hr at the per-seed settlement
spot — k9w's exact right-skewed log-normal price, median $3.50/GPU-hr) hedges by
crossing a resting forward **bid ladder** (5 levels, best bid at F·0.98, each
deeper level 2% lower). Every fill goes through the real `_match_futures_orders`
(fills at the resting bid) and the real `end_epoch` settlement. We sweep the
book's total depth as a multiple of the exposure Q and measure realized
hedged-revenue variance reduction, coverage, and slippage. The producer runs the
`MatchedHedgerPolicy` rule (hit the best bid, sized to exposure); the unhedged
control has the *same* genuine P·Q exposure (std $99), fixing k9w check C's
near-inert control.

Numbers below are the committed N=100 run (`TABLE.md`, `aggregate.csv`,
`execution_bands.svg`). Re-run:
`uv run python -m scripts.compute_hedge_execution_experiment --n-seeds 100`.

---

## Headline — the 100% ceiling survives execution, once the book is deep enough

| fwd depth (×Q) | coverage | variance reduction | slippage (% of F·Q) |
|---|---|---|---|
| 0.25 | 25% | 43.8% | 1.50% |
| 0.50 | 50% | 75.0% | 3.00% |
| **1.0** | **100%** | **100.0%** | 6.00% |
| **2.0** | **100%** | **100.0%** | 3.60% |
| **4.0** | **100%** | **100.0%** | 2.40% |

Two distinct execution costs, and they behave differently:

1. **Coverage gates variance reduction.** When the book is shallower than the
   exposure (depth < Q), the producer simply cannot hedge all of Q; the residual
   rides the spot and variance reduction caps at exactly **1 − (1 − coverage)²**
   (25% → 43.8%, 50% → 75%) — the same curve k9w found for the hedge ratio,
   because coverage *is* the realized hedge ratio. This is the dominant cost of a
   thin book: not a bad price, but unhedged residual risk.

2. **Once depth ≥ Q, variance reduction is a full 100%** — the mechanism's
   ceiling is realized through real matching and settlement. The remaining cost
   is **slippage**: the producer fills below the fair forward F because it
   crosses the spread and walks the ladder. Here it is a modest **2.4–6% of
   notional**, i.e. it lowers the *mean* locked cost, it does not reintroduce
   variance.

---

## The non-monotone slippage is the interesting part

Slippage is **worst exactly at depth = 1×Q (6%)**, not at the shallowest book,
and *improves* with surplus depth (3.6% at 2×, 2.4% at 4×). Reason: at 1×Q the
book exactly covers the exposure, so the producer must consume **every** level
including the deepest, worst-priced one (avg fill = F·0.94). With a deeper book
the same Q only skims the top, better-priced levels (avg fill F·0.976 at 4×).
Caveat the other way: the shallow cells (0.25×, 0.5×) show *small* slippage %
only because slippage is measured against the full F·Q notional — per hedged
unit they pay the same top-of-book spread, they just hedge very little, so almost
all their "cost" is the residual variance in column 3, not slippage.

**Practical reading.** To hedge an H100 cost exposure you want forward depth at
least equal to the exposure (below that, residual risk dominates), and ideally a
*multiple* of it so you skim the top of the book rather than eat your own impact.
At ~2× depth you get complete variance elimination for ~3.6% slippage.

---

## This closes the A↔C gap

k9w check C (naive hedger realizes no variance reduction) was a **policy**
failure, not a market impossibility. With a hedger that (a) sizes the forward to
its exposure and (b) executes by crossing the resting book — the new
`MatchedHedgerPolicy`, unit-tested in `test_unit_gtb_compute.py` — the
mechanism's full variance reduction **is** realized through genuine execution,
and the unhedged control (same P·Q exposure, std $99) is no longer degenerate.
The honest residual is a few percent of slippage, fully explained by spread and
book depth.

---

## Caveats — what this still does not model

- **Static, passive book.** A single producer crosses a dealer's pre-rested
  ladder that does not reprice or thin under one-sided flow. Real books widen and
  pull away when they see persistent selling, so the slippage here is a **floor**,
  not a worst case.
- **The dealer warehouses the whole hedge** (it takes the long side of every
  fill and holds to settlement). Endogenous dealer capacity/pricing — and
  multiple producers competing for the same depth — are the next layer.
- **Spot exposure is notional P·Q** (clean, as in k9w check A). This isolates
  *hedge* execution; spot-market thinness (the separate bd ja2 caveat) is out of
  scope here.
- **Single settlement spot per seed** — no intra-horizon price path, so no
  margin-call / early-unwind dynamics (the env's first-cut settlement has no
  daily mark-to-market). A path-dependent study would need that.

## Bottom line

Through a real, adequately deep forward book, the compute-futures hedge delivers
the same 100% H100-cost variance reduction the k9w mechanism promised, for a
2.4–6% slippage premium. The binding constraint is **book depth relative to
exposure**: too thin and you carry residual risk (variance reduction ∝
1−(1−coverage)²); deep enough and the only cost is a few percent of spread.
