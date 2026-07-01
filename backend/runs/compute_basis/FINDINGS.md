# Compute basis-risk & dealer intermediation — FINDINGS (bd umm, Option A headline)

**N=100 seeds/cell** · β ∈ {0.20, 0.45, 0.63, 0.80, 0.95, 1.0} × reluctance ∈ {0.0, 0.5, 1.0}
Design: `docs/plans/2026-06-30-compute-basis-dealer-experiment.md` · Auto table: `TABLE.md` · Chart: `basis_scurve.svg`
Repro: `uv run python -m scripts.compute_basis_experiment --n-seeds 100`

This is the **Option A** headline (design §8): a *risk-averse* dealer prices its clearing spread off the residual basis it must warehouse, so exchange participation is *emergent*, not imposed. Standalone Monte-Carlo of the participation game (the kernel has no SKU/basket split yet — design §7); execution slippage (bd 5ij) is separable and not stacked here.

---

## Headline (the sweep the issue asked for: reluctance = 0)

**The predicted basis-gated collapse does not happen at balanced flow. Netting clears the market; basis quality is *priced*, not *gated*.**

At reluctance = 0 (principal-longs and principal-shorts both present, ~balanced), the dealer nets matched longs against shorts on the same SKU and only warehouses the small *net imbalance*. Consequently:

| β | r_i | volume share (p10–p90) | clearing spread | willing-clear |
|---|---|---|---|---|
| 0.20 | 0.04 | **0.97** (0.20–1.00) | 6.71% | 0.80 |
| 0.45 | 0.20 | 1.00 (0.23–1.00) | 6.37% | 0.82 |
| 0.63 | 0.40 | 1.00 (0.26–1.00) | 5.65% | 0.84 |
| 0.80 | 0.64 | 1.00 (0.50–1.00) | 4.69% | 0.90 |
| 0.95 | 0.90 | 1.00 (1.00–1.00) | 1.82% | 1.00 |
| 1.00 | 1.00 | 1.00 (1.00–1.00) | 0.00% | 1.00 |

The **median** market is full (≈1.00) across the entire basis range down to β=0.20. What basis quality moves is (a) the **dealer's clearing spread**, which rises monotonically 0% → 6.7% as β falls, and (b) the **p10 tail** — the worst-case seed's volume share falls to 0.20 at β=0.20. So poor basis makes intermediation *more expensive* and *more fragile in the tail*, but does not kill the market when the dealer can net two-sided flow.

**Prediction outcome (faithful, per CLAUDE.md).** The design's **H1 — "exchange volume share is sigmoidal in β with a critical point β≈0.5–0.7" — is NOT supported at reluctance = 0.** Volume share is flat-high with a mild p10 tail; there is no sigmoid and no critical β in the balanced regime. H1 implicitly assumed the design §4 *one-sided* dealer (warehouses gross exposure); the balanced two-sided market behaves differently because netting, not basis-hedging, does the work.

---

## Where the phase transition actually lives: one-sided flow (reluctance → 1)

`reluctance` = the probability a short principal (neocloud) withdraws from selling forward — the "short-compute-**and**-short-a-tech-transition" reluctance. reluctance = 1.0 is exactly the design §4 one-sided model (no shorts → dealer warehouses gross long demand → nothing to net).

**Willing-clear fraction** (fraction of *willing* principals the dealer profitably intermediates — strips the mechanical reluctance cap, isolating the pure basis effect):

| β → | 0.20 | 0.45 | 0.63 | 0.80 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|
| reluctance 0.0 | 0.80 | 0.82 | 0.84 | 0.90 | 1.00 | 1.00 |
| reluctance 0.5 | 0.64 | 0.67 | 0.72 | 0.83 | 0.98 | 1.00 |
| **reluctance 1.0** | **0.12** | **0.13** | **0.15** | **0.28** | **0.80** | **1.00** |

The one-sided row is a **sharp sigmoid**: flat ≈0.13 for β ≤ 0.63, then a steep rise through β=0.80 (0.28) → 0.95 (0.80) → 1.0 (1.00). **Critical β ≈ 0.85–0.95 (r_i ≈ 0.75–0.90)** — *higher* than the design's predicted 0.5–0.7, i.e. one-sided single-SKU intermediation needs *near-perfect* basis to form. The transition's sharpness increases with reluctance (compare the three rows).

So the refined, supported claim is:

> **H1′ (supported):** A basis-gated phase transition in exchange participation appears **in the one-sided limit** (reluctance → 1). It is sharp, with a high critical basis (β ≈ 0.9). As flow becomes two-sided, the transition softens and the critical basis falls, until at balanced flow the market forms across essentially the whole basis range.

---

## Synthesis — the thesis is refined, not confirmed

The market-structure thesis says a liquid compute exchange forms iff dealers can lay off SKU risk on a generic basket (high basis R²). This experiment says that is **the wrong gating variable for a two-sided market**:

1. **Netting dominates basis-hedging.** When both principal-longs and principal-shorts are present, the dealer's value-add is *matching* them; the fungible basket only has to absorb the residual *imbalance*. Basis quality then sets the *price* of intermediation (the spread), not its *existence*.
2. **Basis quality becomes load-bearing exactly when flow is one-sided** — i.e. when the short side (neoclouds) structurally withdraws from selling forward. That is precisely the reluctance the original market-structure note flagged. **Short-side two-sidedness, not index quality, is the primary gate on exchange viability.**
3. Corollary for the index-builders (Silicon Data / Ornn / Compute Desk): a high-R² basket index matters most in the *one-sided* corner. If the market is naturally balanced, dealers form it even with a mediocre index; if it is chronically long-only, even a good index leaves the exchange thin until basis is near-perfect.

**Dealer economics (all cells):** the risk-averse dealer earns a **non-negative expected spread** everywhere (P&L mean 0.7–3.2, > 0), with P&L σ tracking the residual it warehouses. At β=1 the residual, spread, and P&L are all exactly 0 — recovering the bd k9w/5ij perfect-hedge endpoint this sweep descends from. Unit-pinned in `tests/test_unit_gtb_compute_basis.py`.

---

## Limitations (what would reverse or refine this)

- **Parameter-dependent critical point.** Absolute spreads and the exact critical β depend on the dealer/principal risk-aversion ratio (γ_D, γ_P), principal count N_P, and size dispersion (baseline: γ_D=γ_P=1, N_P=20). The **qualitative** result (netting dominates balanced flow; the transition requires one-sidedness) is robust; the **location** of the critical β is not universal. A γ-ratio sensitivity sweep is the obvious next cell and is *not* run here.
- **Single SKU (sku_dispersion = 1).** Netting here is within one SKU (longs vs shorts, same S_i). Multi-SKU flow would add cross-SKU diversification of the dealer's book, easing capacity further and pushing the transition even deeper into the one-sided/high-reluctance corner. Not tested in the headline.
- **Standalone Monte-Carlo, not the gridworld kernel.** No execution slippage (bd 5ij, separable and already characterised), no order-book depth/repricing (bd 4tw), no intra-horizon price path or margin calls (bd 1z3). This deliberately isolates the participation/basis mechanism.
- **A trusted basket index is assumed to exist.** The Schelling-point / index-formation problem (the hard real-world part) is out of scope — this measures "given a hub, does intermediation form", not "does a hub emerge".
- **Reluctance is exogenous.** Short withdrawal is a fixed probability; in reality it is endogenous to the forward price the dealer offers.

## Recommended follow-ups (bd)
- γ-ratio and N_P sensitivity on the critical β (turns the "location is parameter-dependent" caveat into a mapped surface).
- The full 30-cell grid with `sku_dispersion ∈ {1,4,12}` — does cross-SKU diversification move the transition as predicted?
- Kernel implementation (design §7: SKU key + basket contract + routing) to re-run the *balanced* headline through the real order book and layer bd 5ij slippage on top.
