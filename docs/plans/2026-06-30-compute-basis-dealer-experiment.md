# Compute basis-risk & dealer intermediation — experiment design

**Status:** design only (no code yet) · **Date:** 2026-06-30 · **Author:** design doc for review
**Depends on / extends:** bd ja2 (compute kernel resource), bd k9w (`runs/compute_futures`), bd 5ij (`runs/compute_hedge_execution`)

---

## 0. TL;DR

The compute-market thesis this experiment tests: **principals (inference platforms, app layer / neoclouds) never trade a generic compute exchange directly — they trade bespoke SKU-specific forwards with OTC dealers, and dealers hedge the *fungible component* of that exposure on a generic basket exchange, warehousing the residual basis.** Whether a liquid central exchange forms at all is a function of **one number**: how much of a SKU forward's variance a dealer can lay off on the basket.

We model that number directly as a factor loading `β`, sweep it, and look for a **phase transition** in exchange participation. The repo already contains the `β = 1` endpoint (perfect hedge, 100% variance reduction — bd k9w/5ij). This experiment sweeps `β` *down* from there and asks where dealer intermediation breaks.

**This is a design doc. No code is written yet.** §7 enumerates exactly what would need to change, grounded in current file/line references, so implementation is a decision away — not part of this deliverable.

---

## 1. The one number the experiment exists to measure

Everything reduces to the **dealer basis R²** — the fraction of a SKU-specific forward's price variance a dealer can hedge by trading the generic basket index `I`:

```
r_i = corr²(ΔF_i^SKU, ΔI^basket)
```

- **r_i → 1:** dealers lay off ~all risk on the exchange, warehouse a thin residual → intermediation is cheap → a liquid central exchange **forms**. The thesis endgame realizes.
- **r_i → 0:** dealers cannot hedge; they warehouse the *whole* SKU risk; bid/ask blows out; principals stay in pure bilateral OTC → the basket exchange **never gets liquid**.

So the experiment is **not** "does hedging reduce variance" (bd k9w already proved 100% reduction in the fungible case). It is: **above what critical basis correlation does a central exchange endogenously attract volume, and below what value does it collapse to bilateral?** That threshold — with p10–p90 bands across seeds — is the deliverable.

---

## 2. Falsifiable prediction (stated before running, per CLAUDE.md)

> **H1 (phase transition):** Basket-exchange volume share is **sigmoidal in `β`**, with a critical point in the range **β ≈ 0.5–0.7** (r_i ≈ 0.25–0.5). Below it, exchange volume collapses toward zero (principals revert to bilateral) and dealer effective spread diverges. Above it, volume saturates and the bd k9w/5ij variance-reduction results are recovered.

> **H0 (thesis too strong):** If exchange volume share is roughly **linear or flat** in `β` — i.e. dealers form a liquid market even at poor basis — then "you need a hub price before you get a market" is false in this model, and the nonfungibility argument does not by itself prevent a central exchange.

Either outcome is publishable. Report whichever the data shows, loudly — including if the transition is smooth rather than sharp (that itself bounds how "phase-transition-like" real compute markets would be).

---

## 3. The synthetic factor mechanism (the knob)

Chosen over emergent-from-gridworld because the knob **is** the independent variable — a clean, fully-controlled sweep. Each SKU spot price is a factor model on the basket index plus idiosyncratic noise:

```
S_i(t) = β · I(t) + √(1 − β²) · ε_i(t)          ε_i ⟂ I,  Var(ε_i) = Var(I)
⇒ corr(S_i, I) = β        ⇒ r_i = β²
```

`I(t)` is the existing per-seed log-normal compute spot (median $3.50/GPU-hr, log-vol 0.5 — from `compute_futures_experiment.py:81–120`). Normalising `Var(ε_i) = Var(I)` keeps `Var(S_i)` constant across `β`, so variance reduction is attributable to `β` alone, not to changing the size of the thing being hedged. This mirrors the discipline in bd k9w where forwards are struck at fair value to isolate variance reduction from a pricing edge.

**Primary sweep:** `β ∈ {0.20, 0.45, 0.63, 0.80, 0.95}` → `r_i ∈ {0.04, 0.20, 0.40, 0.64, 0.90}`, chosen to bracket the predicted 0.25–0.5 critical band with points on both sides.

---

## 4. The analytic backbone (what the sim should reproduce)

A principal-long wants SKU `i`. It buys a bespoke SKU forward from a dealer → **principal is fully hedged by construction** (its cost is locked at the dealer's forward price). The risk is now the *dealer's*. The dealer is short SKU `i`, exposed to `S_i` rising, and hedges by going long the basket `I` at optimal ratio:

```
h*  = Cov(S_i, I) / Var(I) = β · σ_i/σ_I
Residual (unhedgeable) dealer variance  =  σ_i² · (1 − β²)  =  σ_i² · (1 − r_i)
```

This residual is the load-bearing quantity. A **risk-averse dealer** must quote a spread that compensates for warehousing it:

```
spread_i  ∝  γ · σ_i² · (1 − r_i) · |inventory_i|          (γ = dealer risk aversion)
```

A principal participates on-exchange (via the dealer) only if the dealer's spread is cheaper than bearing the unhedged risk itself. That inequality flips at a critical `r_i` → **the S-curve is endogenous**, driven entirely by the `(1 − r_i)` residual term. As `β → 1`, residual → 0, spread → the pure execution slippage bd 5ij measured (2.4–6% of notional), and we recover the existing results.

**Two frictions now stack** — this is the new physics vs. bd 5ij:
1. **Coverage/depth** (bd 5ij): variance reduction gated by `1 − (1 − coverage)²`.
2. **Basis** (this experiment): variance reduction ceiled at `r_i = β²`.

Combined, a dealer hedging fraction `h` of exposure through a basket with correlation `β` achieves roughly `1 − (1 − h·β²)²`-shaped reduction on the warehoused book (exact form to be confirmed empirically — do **not** hard-code it as ground truth; measure it).

---

## 5. Agent roles

| Role | Maps to (thesis) | Existing policy to base on | New behavior needed |
|---|---|---|---|
| **Principal-long** | Fireworks/Modal/Baseten, Cursor, Perplexity | `FuturesTakerPolicy` (`agents.py:298`-ish taker) | Buys **SKU** forward from dealer only; never touches basket book |
| **Principal-short** | CoreWeave/Nebius/Lambda, indie DCs | `MatchedHedgerPolicy` (`agents.py:858–955`) | Sells SKU forward; subject to `short_side_reluctance` haircut |
| **Dealer / OTC desk** | MM / compute dealer — **the key actor** | `FuturesMakerPolicy` (`agents.py:705–753`) | Quotes bilateral SKU forwards **and** hedges warehoused inventory on the basket book; **risk-aware spread** off `(1−r_i)` residual |
| **Basket exchange** | the H200/H100-basket future | existing `(resource, epoch)` book (`env.py:714–801`) | Only **dealers** post/cross here |

**The one hard structural constraint from the thesis** (encode as a rule, verify with an assertion in the harness): *principals never trade the basket exchange; only dealers do.* This is the single most important thing to get right — if principals can hit the basket directly, the model is no longer testing dealer intermediation and the result is meaningless. The current model has no such routing distinction (all agents share one `(resource, epoch)` book — scout finding §4), so this separation is a genuine addition.

---

## 6. Factor grid & metrics

### Factors
| Factor | Values | Purpose |
|---|---|---|
| `basis_correlation` (β) — **primary** | 0.20, 0.45, 0.63, 0.80, 0.95 | The phase-transition variable |
| `sku_dispersion` | 1, 4, 12 distinct SKUs | Fragmentation: more SKUs → thinner per-SKU liquidity, tests whether dispersion alone starves the book |
| `short_side_reluctance` | 0.0, 0.5 | Neocloud "short-compute *and* short-a-tech-transition" haircut on natural sellers' willingness to sell forward |

Full grid = 5 × 3 × 2 = 30 cells. If runtime is a constraint, run the **β sweep at `sku_dispersion=1, reluctance=0` first** (5 cells, the headline S-curve), then the interaction slices.

### Metrics (per seed → aggregate mean/median/p10/p90 via the sweep harness)
1. **Exchange volume share** = basket-exchange notional ÷ total notional. **The order parameter.** Plot vs β → look for the S-curve.
2. **Dealer effective bid/ask** vs β — predicted to diverge as β → 0.
3. **Residual warehoused risk per dealer** (unhedged basis inventory variance) — the capacity cap; should track `σ_i²(1−β²)`.
4. **Realized principal hedging cost** vs the **unhedged counterfactual** — does dealer intermediation actually lower principal cost, and below what β does it stop helping?
5. **Dealer P&L variance / ruin frequency** — fallback order parameter if we keep spread fixed instead of risk-aware (see §8).

---

## 7. What would need to change in the code (grounded; NOT built here)

Enumerated so the implement/no-implement decision is concrete. All references from the code-map scout.

1. **SKU dimension on the forward book.** Matching key `_futures_key(resource, epoch)` → `(resource, sku, epoch)` (`env.py:714`). SKU forwards live in per-SKU books; the **basket** contract keeps the current `(resource, epoch)` aggregate key. Two book types, not one.
2. **A basket index contract + synthetic factor spot.** SKU settlement spot `S_i = β·I + √(1−β²)·ε_i` layered on the existing per-seed log-normal `I` (`compute_futures_experiment.py:81–120`). The basket settles on `I`; each SKU settles on its `S_i`.
3. **Risk-aware dealer policy.** Extend `FuturesMakerPolicy` (`agents.py:705–753`, currently *static* 10% spread — scout §4 caveat) to (a) quote SKU forwards, (b) hedge warehoused inventory on the basket book, (c) widen spread with `γ·σ_i²(1−r_i)·|inventory|`. Without (c) there is no endogenous S-curve — see §8.
4. **Routing constraint.** Principals restricted to SKU books; dealers to both. Assert `principal_id ∉ basket_book.participants` each step.
5. **Harness cell.** New `scripts/compute_basis_experiment.py` modeled on `compute_hedge_execution_experiment.py` (depth-sweep structure is the closest analog); reuse the SVG-with-p10–p90-bands renderer from `scripts/audit_sweep_experiment.py`.

Estimated scope: ~2 kernel touches (`env.py` key, settlement), 1 new policy, 1 new harness script + renderer reuse. The kernel touches are the sensitive ones (vendored upstream — CLAUDE.md invariant); document any divergence like the existing `env.py:_run_audits` note.

---

## 8. Key design decision flagged for implementation time

**Endogenous volume (recommended) vs. fixed-spread proxy.**

- **Option A — risk-aware dealer (recommended):** dealer prices spread off residual variance (§4). Exchange volume share is *emergent*; the S-curve falls out of agent behavior. Richer, directly tests the thesis, needs new dealer logic (§7.3).
- **Option B — fixed spread, measure dealer risk:** keep `FuturesMakerPolicy`'s static spread; read "market forms or not" off **dealer P&L variance / ruin frequency** (metric 5) instead of emergent volume. Simpler, no new pricing logic, but the phase transition is inferred from dealer solvency rather than shown as volume migration.

Recommend **A** for the headline result, with **B** as a cheap sanity pre-check (does dealer risk blow up at low β at all?) before investing in the risk-aware policy.

---

## 9. Methodology guardrails (CLAUDE.md — non-negotiable)

- **N = 100 seeds/cell** for anything in FINDINGS; N = 50 minimum for a directional read. Single seeds are anecdote — welfare/Gini swing 50%+ across seeds here.
- **Aggregate mean/median/p10/p90**, never point estimates. Render the S-curve **with p10–p90 confidence bands** (SVG renderer in `audit_sweep_experiment.py`, no matplotlib).
- **Hand-curated FINDINGS.md** separate from auto-generated TABLE.md, opened the moment there's data, written **per-cell** (catches aggregation off-by-ones no test catches — bd-an2 precedent).
- **Persona one-sidedness trap:** if principals/dealers are LLM-driven, a persona-aligned lineup makes the market one-directional and "volume" becomes a sentiment poll, not price discovery. Prefer scripted (ZI-style) principals/dealers here; if LLM, use `BALANCED_LLM_LINEUP` and check `confidence_source`. (Scripted is cleaner for a mechanism sweep and avoids this entirely.)

---

## 10. Open risks / what could invalidate the result

- **Index Schelling-point failure (external validity):** the model *assumes* a basket index exists and is trusted. Real-world, manufacturing that hub price is the hard part (cf. failed bandwidth-commodity markets, early 2000s). The experiment tests "given a hub, does intermediation form" — not "does a hub emerge." State this boundary explicitly in FINDINGS.
- **Short-side reluctance may dominate:** if neoclouds won't sell forward (short-compute-*and*-short-a-tech-transition), the exchange starves for sellers even at β = 1. The `short_side_reluctance` factor probes this; watch for volume collapse that is *seller-driven*, not basis-driven — don't misattribute.
- **Basis non-stationarity:** the synthetic model holds β constant over the run. Real basis widens exactly when you need the hedge (correlation → 1 in a crunch, or breaks in a supply shock). A constant-β result is an upper bound on hedgeability; note it.
- **Static basket book depth (inherited from bd 5ij):** the maker's book doesn't reprice as the dealer hedges into it (`compute_hedge_execution/FINDINGS.md:90`). At scale, dealer hedging would move the index — a second-order feedback this design does not capture. Flag as future work.
- **Result may be smooth, not sharp.** If the transition is gradual, resist calling it a "phase transition." Report the slope and the half-max β; let the shape speak.

---

## 11. Next step

Deliverable for this turn is the design only. To proceed: open a bd issue (`feature`) tracking implementation of §7, decide Option A vs B (§8), and start with the 5-cell β-only headline sweep at N=50 before committing to the full 30-cell × N=100 run.
