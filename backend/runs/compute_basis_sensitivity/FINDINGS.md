# Compute basis — critical-β sensitivity (γ-ratio × N_P) — FINDINGS (bd keq, ← umm)

**N=100 seeds/cell** · one-sided (reluctance=1), single SKU · γ_P/γ_D ∈ {1…12} × N_P ∈ {10,20,40}
Table: `TABLE.md` · Chart: `critbeta_sensitivity.svg` · Repro: `uv run python -m scripts.compute_basis_sensitivity_experiment --n-seeds 100`

Phase 4 (capstone). Every prior phase carried the caveat that the *location* of the one-sided phase transition (critical basis β*) is parameter-dependent. This maps it and validates a closed form, turning the caveat into a law.

---

## Result: the critical basis follows β* ≈ √(1 − 4·(γ_P/γ_D)/N_P), confirmed

Simulated β* (where one-sided willing-clear crosses ½) vs the analytic prediction:

| N_P | γ_P/γ_D = 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| **20** emp | 0.90 | 0.78 | 0.65 | 0.48 | — | — |
| 20 analytic | 0.89 | 0.77 | 0.63 | 0.45 | 0.00 | 0.00 |
| **40** emp | 0.95 | 0.89 | 0.84 | 0.78 | 0.63 | 0.45 |
| 40 analytic | 0.95 | 0.89 | 0.84 | 0.77 | 0.63 | 0.45 |

Empirical tracks analytic within ~0.03 across the surface (unit-pinned to 0.06 tolerance). "—" / β*=0 means the transition falls below the β-grid floor — the market forms at *every* basis tested (no transition).

**Two comparative statics:**
1. **β* rises with N_P** (market size). At γ_P/γ_D=2: β* = 0.45 (N_P=10) → 0.77 (20) → 0.89 (40). More one-sided principals pile more *perfectly-correlated* risk on the dealer (single SKU, no netting, no diversification), so it demands better basis. Concentration is the enemy.
2. **β* falls with γ_P/γ_D** (dealer risk tolerance). At N_P=20: β* = 0.90 → 0.48 as the ratio goes 1 → 4. A dealer much less risk-averse than its principals warehouses more, forming the market at worse basis.

So the one-sided critical basis is not a universal number — it can sit anywhere in [0,1], set by `4·(γ_P/γ_D)/N_P`. A large, concentrated, one-sided market with a cautious dealer needs near-perfect basis (β*→1); a small market with a risk-tolerant dealer forms at almost any basis (β*→0).

This is the same mechanism phase 2 exploited from the other side: SKU dispersion lowers the *effective* N_P per SKU (spreads the concentrated block), which lowers β* — consistent with this law.

---

## Capstone: the four-phase picture

The dealer's warehoused residual variance ∝ (effective concentrated one-sided imbalance)² · (1−r_i), and *everything* in this study is a lever on one of those two terms:

| phase | lever | moves | net effect on exchange viability |
|---|---|---|---|
| 1 (umm) | two-sidedness (reluctance) | shrinks the *imbalance* by netting | **robust** — balanced flow forms the market at any β |
| 2 (z45) | SKU count K (independent idio) | splits the imbalance into K diversifiable blocks | real but **fragile** |
| 3 (0h4) | cross-SKU correlation ρ_sku | re-concentrates the blocks via a common factor | **erases** the phase-2 gain by ρ≈0.5 |
| 4 (keq) | γ_P/γ_D and N_P | set the dealer's tolerance vs the block size | **locates** the transition: β*≈√(1−4ρ_γ/N_P) |

**The through-line:** a liquid compute exchange forms when the dealer's *concentrated, un-diversifiable, one-sided* residual is small enough to warehouse cheaply. The only robust way to shrink it is **two-sided flow** (netting). Diversification across SKUs helps in theory but dies under realistic SKU correlation. And however you set the dealer/market parameters, they merely *slide* the one-sided critical basis along √(1−4ρ_γ/N_P) — they don't remove the fundamental dependence on two-sidedness. Basis-index quality (the thing the whole "standardize compute" industry is racing to build) is decisive **only** in the one-sided, high-correlation, concentrated corner — which is precisely the corner a chronically long compute market (reluctant neocloud short side) lands in.

---

## Limitations
- The analytic constant "4" assumes a ~50/50 long/short population (so willing longs ≈ N_P/2) and the ½-crossing definition; a different split or threshold rescales it. The *shape* √(1−c·ρ_γ/N_P) is what's validated, with c≈4 here.
- Equal-size derivation vs dispersed sizes in the sim — agreement within 0.06 shows dispersion doesn't break it, but heavy-tailed sizes would.
- One-sided single-SKU by construction (the sharp-transition regime). Balanced flow has no sharp β* to locate (phase 1).
