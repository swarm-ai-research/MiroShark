# Dealer funding-ruin under asymmetric margining — FINDINGS (bd b2s, ← umm/1z3)

**N=200 seeds/cell** · buffer ∈ {0.05…1.6}× hedge notional × β ∈ {1.0, 0.8}
Table: `TABLE.md` · Chart: `ruin_curve.svg` · Repro: `uv run python -m scripts.compute_funding_ruin_experiment --n-seeds 200`

Uses the bd-1z3 daily-MTM kernel to make Pirrong's (Trafigura) central claim concrete: the binding constraint on a dealer is **variation-margin cash flow**, not basis variance — a book hedged in *value* can fail on *cash*. Setup: the dealer's SKU short is **OTC** (settles at expiry, no daily margin); its **basket hedge (β·Q) is daily-MTM'd** through the real futures engine. Each seed is a basket price path over T=6 epochs; adverse intra-path moves drain the dealer's finite coin buffer while the offsetting SKU gain is locked until expiry.

---

## Headline (β=1 — economically flat at expiry, ZERO terminal price risk)

| buffer ÷ notional | ruin freq | net P&L std | net P&L p10 |
|---|---|---|---|
| 0.05× | **38%** | 3.24 | −2.05 |
| 0.10× | 24% | 2.64 | −0.23 |
| 0.20× | 8% | 1.12 | −0.00 |
| 0.40× | **0%** | **0.00** | 0.00 |
| 0.80× | 0% | 0.00 | 0.00 |
| 1.60× | 0% | 0.00 | 0.00 |

At β=1 the dealer is **perfectly hedged** — short SKU + long basket, net zero at expiry. With an adequate buffer (≥0.4× notional) its P&L is *exactly* flat (std 0.00) and it is never ruined. **Yet with a thin buffer it is ruined 38% of the time** and its P&L acquires std 3.24 and a −2.05 p10 tail — **pure funding risk injected into a position with zero price risk.** This is the cleanest possible statement of Pirrong's point.

The mechanism: the dealer is long the basket, so a *downswing* forces daily variation-margin payments; the compensating SKU gain (it's short SKU, SKU also fell) is OTC and locked until expiry. The dealer is *right* — at β=1 the hedge fully reverses by expiry — but it is **liquidated before it can be proven right**. Solvent-in-value, insolvent-in-cash.

**A critical buffer exists (~0.4× hedge notional here).** Ruin is not gradual — it collapses to zero once the buffer can absorb the worst intra-path drawdown. Below it, ruin rises steeply as the buffer thins.

---

## Funding risk stacks on top of basis risk (β=0.8)

At β=0.8 the position carries genuine terminal basis (√(1−r_i)) — net P&L std ≈ 5.0 and p10 ≈ −7.9 **even at a full buffer**. A thin buffer then adds **21%** ruin *on top* (0.05× buffer). Funding ruin and basis risk are **additive, independent failure modes** — a good basis index does nothing to prevent a funding-driven liquidation, and a fat buffer does nothing about basis.

---

## Synthesis — the compute exchange needs TWO things, not one

The phase-1..4 studies said exchange viability is gated by **basis hedgeability** (can the dealer lay off SKU risk on the basket?) and, in the one-sided corner, by short-side participation. This adds an orthogonal gate: **dealer funding capacity**.

> A compute dealer must be *both* able to hedge the basis (phases 1–4) *and* able to fund the variation-margin cash flows of that hedge (this study). Either can kill the exchange independently. And they compound: an OTC compute forward (reserved SKU capacity, bilateral, unmargined) hedged with an exchange-traded, daily-margined basket future is *exactly* the asymmetric-margining structure that turns a hedged dealer into a fragile one — the Pirrong "borrow-short / illiquid-offset" fragility, now grounded in a real MTM engine.

This is the strongest form of the through-line from the commodity-trading lessons: **low basis hedgeability doesn't just widen the dealer's spread — combined with thin funding it converts the dealer from a safe self-liquidating intermediary into one that can be liquidated while economically hedged.**

---

## Limitations
- The OTC SKU leg is modelled analytically (settles at expiry, no margin) while the basket leg runs through the real MTM kernel — the deliberate *asymmetry*. A symmetric world (both legs margined) nets the variation margin and removes the funding stress; the asymmetry is the point, and it's the realistic case for OTC-reservation-vs-exchange-hedge.
- Basket path is a driftless GBM random walk (per-epoch logvol 0.16); fat-tailed / jump paths (the other Pirrong caveat) would raise ruin at every buffer.
- Single dealer, single hedge; no dynamic buffer management or margin-financing (a real dealer draws a credit line — which is exactly the funding-liquidity line Pirrong says dries up in stress).
- Initial margin (0.2× notional) is a fixed buffer already; the swept buffer is *additional* free coin.

## Follow-ups (bd)
- Fat-tailed / jump basket paths (compounds with the tail-basis follow-up from the commodity-trading lessons).
- Symmetric vs asymmetric margining as an explicit factor (quantify how much of the ruin is the asymmetry).
- Credit-line / margin-financing model — does access to funding (vs raw coin) move the critical buffer, and does it dry up in stress (Pirrong's "wrong-way" funding liquidity)?
