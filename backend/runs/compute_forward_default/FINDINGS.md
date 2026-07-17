# Forward-contract default → retreat to spot — FINDINGS (bd 79l, ← umm)

**N=400 seeds, 300 bootstrap resamples** · κ/σ ∈ {0.1…4, ∞} × spot log-vol ∈ {0.3, 0.5, 0.8}
Table: `TABLE.md` · Chart: `default_curve.svg` · Repro: `uv run python -m scripts.compute_forward_default_experiment --n-seeds 400`

Tests Pirrong's (Trafigura) counterparty-performance risk: a party reneges on a forward when performing costs more than the reneging penalty κ, so the forward's protection is **capped at ±κ** — truncated *exactly in the tail* where a hedge matters most. In cotton 2010–11 this drove traders to "deal only on a spot basis." Model: realised per-unit hedge payoff `h(κ)=clip(S−F, −κ, +κ)`; effectiveness `E(κ)=1−Var((S−F)−h)/Var(S−F)` = the forward-market participation proxy (participation is monotone in E under heterogeneous hedgers).

---

## Result: forwards need enforceability of ~1σ; below that, retreat to spot

Hedge effectiveness E(κ), spot log-vol 0.5:

| κ/σ | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 | ∞ |
|---|---|---|---|---|---|---|---|
| **E** | 0.15 | 0.34 | 0.58 | 0.84 | 0.98 | 1.00 | 1.00 |

- **κ ≳ 1σ** recovers ~85% of the hedge — a forward is worth writing.
- **κ ≈ 0.5σ** loses ~40%; **κ ≈ 0.1σ** is worth only 15% — the forward is near-worthless, so hedgers **retreat to spot / bilateral**.
- Full enforcement (κ=∞) is the perfect hedge (E=1); this is the limit the phase-1..4 studies implicitly assumed.

The collapse is steep and convex in the weak-enforcement region: participation ∝ E falls from 100% to ~15% as enforcement drops from ∞ to 0.1σ, with the knee around **κ ≈ 0.5–1σ**.

## Fatter-tailed prices need stronger enforcement

At a fixed κ/σ in the binding region, higher spot log-vol → **lower** effectiveness (default truncation bites the fatter/right-skewed tail harder):

| κ/σ | σ=0.3 | σ=0.5 | σ=0.8 |
|---|---|---|---|
| 1.0 | 0.85 | 0.84 | 0.81 |
| 2.0 | 0.99 | 0.98 | 0.96 |

GPU-rental prices are right-skewed (capacity crunches spike the tail — the same log-normal the whole study uses). So **volatile compute SKUs are the *most* fragile to counterparty default** and need the *strongest* contract enforcement to sustain a forward market — precisely the SKUs where hedging demand is highest.

---

## Synthesis — a THIRD orthogonal gate on the compute exchange

This collapse occurs with **perfect basis** (a direct forward on the SKU) and **infinite funding** — so contract enforceability is independent of the other two gates:

| gate | question | study |
|---|---|---|
| **Basis hedgeability** | can the dealer lay off SKU risk on the basket? | phases 1–4 (umm/z45/0h4/keq) |
| **Dealer funding** | can the dealer fund variation-margin cash flows? | bd b2s |
| **Contract enforceability** | will counterparties actually perform in the tail? | **this (bd 79l)** |

**All three must hold for a liquid compute forward/exchange to exist.** Each fails independently, and each maps to a real feature of compute markets: nonfungibility/index quality (basis), forward-dated reservations funded short (funding), and — the new one — the fact that compute counterparties (startups, neoclouds) are often thin-balance-sheet entities with weak enforceability, so a compute forward struck today may simply not be honored when spot spikes. That is a first-order reason compute reservations remain **bilateral, collateralized, or prepaid** (cf. Pirrong's structured trade-finance/prepay bundles) rather than clean exchange forwards.

---

## Limitations
- Symmetric enforcement κ (both sides equally likely to renege). Real asymmetry — a well-capitalized dealer vs a thin-balance-sheet startup — would make the *principal's* leg the fragile one; a two-sided-κ extension would show who defaults on whom.
- Participation is read as ∝ E (uniform heterogeneous thresholds); a specific risk-aversion/cost distribution would sharpen the participation curve but not the effectiveness result.
- κ is exogenous; in reality it's endogenous to reputation, collateral posted, and legal jurisdiction (Pirrong's "weak rule of law" territories) — collateral/margin *raises* effective κ, linking this back to the bd-1z3 margin machinery.
- Driftless log-normal spot; explicit jumps (the tail-basis follow-up) would deepen the fat-tail effect.

## Follow-ups (bd)
- Asymmetric κ (dealer vs principal) — who defaults, and does collateral posting (bd-1z3 margin) restore the forward market by raising effective enforcement?
- Endogenous κ from posted collateral — unifies this gate with the funding/margin machinery.
