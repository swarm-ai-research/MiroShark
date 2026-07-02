# Compute marketplace: bisection negotiation vs listed-price

Hand-curated interpretation of `scripts.compute_market_experiment`
(N=100 seeds × 400 buyer/seller encounters/seed, deterministic/reproducible).
Auto-generated numbers live in `TABLE.md` + `aggregate.csv`; the chart with
p10–p90 whiskers is `surplus_bands.svg`.

## What this tests

The GTB world already trades `COMPUTE` through a continuous double auction (the
spot book) and a dated-forward futures book. The arkhai `simple-compute-market`
reference implementation is a **third** microstructure for the same commodity:
sellers publish GPU-capacity listings to a discovery registry, and each buyer
negotiates a rate **bilaterally** over a bounded-round, buyer-driven message
thread before settling through escrow. That port lives in
`worlds/gather_trade_build/compute_market.py`.

This experiment isolates the one thing the microstructure choice actually
decides — **how a buyer and a seller with private valuations converge on a
price** — by pitting the ported mechanism against the simplest alternative on the
same drawn population:

- **BISECTION** — `compute_market.negotiate`: buyer-driven, ≤5 rounds, each side
  bisects toward the peer's quote; the deal lands inside the bargaining range
  `[floor, reservation]` whenever one exists.
- **LISTED-PRICE** — take-it-or-leave-it at the seller's advertised ask
  (`floor × 1.30`): the buyer accepts iff `ask ≤ reservation`, no counter.
  (Upstream's `listed_price` buyer policy with the bisection middleware removed.)

Per encounter, `seller_floor` and `buyer_reservation` are independent per-seed
log-normal draws around a base H100 rate (`BASE=3.50`, `SIGMA=0.45`), so the
population spans both feasible (`floor ≤ reservation`) and infeasible pairs.

## Findings (N=100, p10–p90 in brackets)

| metric | bisection | listed-price | verdict |
| --- | --- | --- | --- |
| fill rate | **0.504** [0.463, 0.537] | 0.343 [0.305, 0.375] | bands disjoint → real |
| welfare capture (realized ÷ feasible GFT) | **1.000** [1.000, 1.000] | 0.920 [0.900, 0.939] | bands disjoint → real |
| mean realized rate | 3.720 [3.599, 3.844] | 3.442 [3.304, 3.581] | **confounded — do not read as a price effect** |
| buyer surplus share | 0.641 [0.632, 0.651] | 0.589 [0.564, 0.616] | **confounded (see below)** |

### 1. Bisection clears materially more feasible trades, and captures ~all the surplus (real)

Fill rate 0.50 vs 0.34 and welfare capture 1.00 vs 0.92 — and in both cases the
p10–p90 bands are **disjoint** across all 100 seeds (bisection fill p10 = 0.463
sits above listed-price fill p90 = 0.375; bisection welfare p10 = 1.000 above
listed-price welfare p90 = 0.939). This is the finding that survives the
methodology bar: a take-it-or-leave-it ask at a 30% markup **prices out ~8% of
the feasible gains-from-trade** — every pair with `floor ≤ reservation < 1.3·floor`
is a trade both sides would have made and the listed price kills. Bilateral
negotiation recovers essentially all of it because the deal price is free to land
anywhere in `[floor, reservation]`.

### 2. The rate and surplus-share gaps are SELECTION artifacts, NOT price effects (honest negative)

It is tempting to read "bisection realized rate 3.72 > listed-price 3.44" as
"negotiation raises prices" and "buyer share 0.64 > 0.59" as "negotiation helps
buyers." **Both are wrong** — they are composition effects. Listed-price only
clears the *cheap* subset (low `floor` → low `ask` ≤ `reservation`), so its
conditional mean rate is mechanically lower; bisection additionally clears the
higher-floor feasible trades that listed-price rejects, pulling its conditional
mean up. The two columns are averages over **different transacted subsets**, so
no directional price claim can be made from them. Reported here only to flag the
trap; the defensible claims are fill rate and welfare capture, which are measured
against the *same* feasible population.

## Caveats / scope

- **Conditional on the 30% advertised markup.** The listed-price disadvantage is
  driven entirely by `ask = floor × MARKUP` exceeding some buyers' reservations.
  As `MARKUP → 1` the ask approaches the floor and listed-price converges to
  bisection on fill; a hidden-reserve listing (`advertised_rate=None`, ask =
  floor) would already accept every feasible bid. The result says "a marked-up
  posted price leaves feasible trades on the table," not "posted prices are bad."
- **`max_rounds=5`, `gap_ratio=1.5`.** Bisection converges well inside 5 rounds
  for feasible pairs, so the fill number is not round-limited; it misses only a
  thin band of near-degenerate feasible pairs (tiny surplus), which is why
  welfare capture rounds to 1.000 even though fill (0.50) is below the raw
  feasibility rate (~0.51).
- This is a **single-encounter** bargaining model — no inventory depletion,
  competition between listings, or repeated play. It measures the price-formation
  primitive, not a full market-day. The end-to-end `ComputeMarketplace`
  (discover → negotiate → settle, with escrow and listing depletion) is exercised
  by the unit tests, not this sweep.

## Reproduce

```bash
cd backend
uv run python -m scripts.compute_market_experiment --n-seeds 100
```

Deterministic (SHA-256-seeded draws, no `random`): re-running produces a
byte-identical `aggregate.csv`.
