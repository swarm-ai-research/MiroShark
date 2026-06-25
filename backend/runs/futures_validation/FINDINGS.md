# Commodity futures first cut — validation

**Issue:** bd-c7i (caps the bd-af2 first cut) · **N = 100** ·
`scripts/futures_validation_experiment.py`. Three checks, each reported
faithfully. See `hedge.svg`, `hedge.csv`.

The first cut is: a standalone futures order book (bd-oo7) minting
cash-settled `FuturesContract`s, settled at expiry against spot (bd-dog),
with open-interest/basis/volume metrics (bd-2qe) and maker/taker/hedger
policies (this issue).

## A. The mechanism delivers a textbook hedge — 100% variance reduction

A producer will sell `Q = 4` units of wood at the settlement-epoch spot
price `S`, which varies seed-to-seed (mean 0.99, std 0.35). We mint a
matched `Q`-unit short forward at a price `F = 1.0` locked early, **through
the real order book**, and settle it **through the real `end_epoch`
machinery**.

| revenue | mean | std |
|---|---|---|
| unhedged (`S·Q`) | 3.95 | **1.387** |
| hedged (`S·Q` + futures P&L) | 4.80 | **0.000** |

**Variance reduction: 100%.** The hedged revenue is *exactly constant*
across every seed, because the settlement machinery delivers

    hedged = S·Q + [margin + (F − S)·Q] = F·Q + margin   (independent of S)

So when the producer's short position matches its spot exposure, the
futures contract perfectly offsets price risk — the defining property of a
hedge. This isolates the *mechanism* (real matching + real settlement)
from execution noise, and confirms it pays out correctly.

## B. Robust at population scale — invariants hold across 100 seeds

A futures-active lineup (4 ZI spot traders + 2 makers + 2 takers + 3
hedgers), 10 epochs:

- **Market is active:** 163 contracts matched/seed, 135 settled/seed.
- **No negative balances:** min worker balance over all 100 seeds = 0.000
  (the settlement clamp binds exactly at 0 under the worst adverse move —
  never goes negative).
- **Zero-sum settlement:** max zero-sum error = 0.0 — every coin the long
  gains is a coin the short loses; margins fully released. Coin is
  conserved across the futures lifecycle (also unit-proven in
  `test_unit_gtb_futures`).

## C. Naive policies under-hedge — honest negative

Running hedged vs unhedged spot-selling producers under the *simple*
bd-c7i policies (hedged shorts futures every 4th step; unhedged never):

| naive arm | final coin mean | std |
|---|---|---|
| hedged | 9.68 | 2.03 |
| unhedged | 11.81 | 1.56 |

**Naive "variance reduction": −70%** — the naive hedger does *worse* on
variance, not better. This is expected and reported faithfully: the naive
hedger spot-sells most steps but shorts futures only periodically, so its
futures position is far smaller than its spot exposure. An *unmatched*
hedge adds price-risk noise instead of cancelling it (and costs margin +
energy, lowering the mean too). The mechanism is sound (part A); naive
agents simply don't discover the matched position that uses it well.

## Verdict

The commodity-futures first cut **works**: forwards are discovered through
a real order book, contracts cash-settle correctly at expiry, coin is
conserved and balances never go negative at scale, and — given a matched
position — the market delivers a perfect hedge. What the first cut does
*not* provide is agents smart enough to size that position; that is a
policy problem, not a mechanism problem.

## Caveats / scope

- Part A is deterministic by construction (matched Q, no execution noise) —
  that is the point (isolate the mechanism), not a flaw. The 100% figure
  means "exactly offsetting," not "noisily large."
- The GTB wealth model is coin + houses, so a producer only has spot
  exposure if it *spot-sells* its output; a house-builder has none to
  hedge. The validation uses spot-selling producers for this reason.
- Cash-settled, settle-at-expiry, single forward per (resource, date). No
  physical delivery, no daily mark-to-market / margin calls — those are the
  deferred bd-af2 children.

## Follow-ups (deferred bd-af2 children)

- **Position-sizing hedger policy** that shorts futures to match its
  realized/expected spot sales (would turn part C positive).
- **Physical delivery** settlement arm (short delivers the good).
- **Daily mark-to-market + margin calls / liquidation** (the realistic
  risk-management layer; the current clamp is its first-cut stand-in).
- **Basis convergence study** — needs a spot market active enough each
  epoch to print a continuous spot reference (this run's spot is thin).
