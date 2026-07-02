# Compute basis — SKU dispersion vs netting — FINDINGS (bd z45, ← umm)

**N=100 seeds/cell** · β ∈ {0.20…1.0} × K ∈ {1, 4, 12} × reluctance ∈ {0.0, 1.0}
Design factor `sku_dispersion`. Table: `TABLE.md` · Chart: `sku_scurve.svg` (one-sided panel)
Repro: `uv run python -m scripts.compute_basis_sku_experiment --n-seeds 100`

Phase 2 of bd umm. The headline (`runs/compute_basis/FINDINGS.md`) found the basis-gated phase transition only appears in the **one-sided** limit (reluctance→1). This tests the pre-registered prediction: with one-sided demand spread over K SKUs whose **idiosyncratic components are independent**, the dealer warehouses K independent residuals (Var = Σ_k D_k²·V·(1−r)), so diversification (Var ~ 1/K) should let the market form even when the dealer cannot net.

Order parameter: **willing-clear fraction** (fraction of *willing* demand the dealer intermediates — isolates the basis/diversification effect from the mechanical reluctance cap). Cross-script consistency: K=1 reproduces the headline single-SKU numbers bit-for-bit (pinned in `tests/test_unit_gtb_compute_basis_sku.py`).

---

## Result 1 (predicted, confirmed): SKU dispersion is a *substitute* for netting

In the one-sided limit (reluctance = 1), more SKUs pulls the phase transition sharply **left** — the market forms at progressively *worse* basis:

**Willing-clear fraction, reluctance = 1.0:**

| β → | 0.20 | 0.45 | 0.63 | 0.80 | 0.95 | crit. β (½-max) |
|---|---|---|---|---|---|---|
| K=1  | 0.12 | 0.13 | 0.15 | 0.28 | 0.80 | ≈ **0.90** |
| K=4  | 0.21 | 0.27 | 0.36 | 0.65 | 1.00 | ≈ **0.73** |
| K=12 | 0.28 | 0.43 | 0.63 | 0.93 | 1.00 | ≈ **0.58** |

Critical basis falls **0.90 → 0.73 → 0.58** as K goes 1 → 4 → 12 — a ~0.3 leftward shift in β. At K=12, a one-sided market intermediates 63% of willing demand at β=0.63 (r_i=0.40), where a single-SKU market manages only 15%. Dealer P&L rises with K in this regime (0.78 → 1.76 at β=0.2), i.e. the dealer does *more* profitable business precisely because diversification lets it warehouse more one-sided flow.

**So there are two independent routes to an intermediable compute exchange: net two-sided flow, *or* diversify one-sided flow across many independent SKUs.** They are substitutes — either shrinks the residual the dealer must warehouse.

---

## Result 2 (the flip side, not predicted): dispersion *fragments* netting

When flow *is* two-sided (reluctance = 0), more SKUs **hurts** at low basis — the opposite sign. Willing-clear at β=0.20 falls **0.80 → 0.70 → 0.57** (K=1→4→12), volume share falls **0.97 → 0.83 → 0.59**, and the clearing spread *rises* **6.7% → 10.3% → 12.9%**.

Mechanism: netting needs long/short counterparties *within the same SKU*. Splitting 20 principals across 12 SKUs leaves ~1–2 per SKU, so most SKUs can't net internally and the dealer is left holding un-nettable singletons. Fragmenting the pool destroys the very netting that made the balanced market cheap. (The effect is confined to low β — by β≥0.63 the residual is small enough that fragmentation no longer bites, and K=12 slightly *beats* K=1.)

---

## Synthesis — netting and diversification trade off

| flow | few SKUs | many SKUs |
|---|---|---|
| **two-sided** (balanced) | ✅ deep per-SKU netting | ⚠️ fragmented pools, thin netting |
| **one-sided** (reluctant shorts) | ❌ concentrated residual, no market | ✅ diversified residual, market forms |

The dealer's warehoused risk is governed by *both* how nettable the flow is (two-sidedness) *and* how diversifiable it is (SKU count with independent idio). SKU proliferation is not monotonically good or bad for liquidity: it **helps a one-sided market and hurts a two-sided one**. The real question for a compute exchange is which regime it is in — and the earlier finding says the short side's structural reluctance is what puts it in the one-sided regime in the first place.

Corollary for index-builders: a many-SKU market that is *also* one-sided is the case where a good basket index pays off most (diversification already helps; a high-R² index compounds it). A balanced market barely needs the index at all.

---

## Limitations (what would reverse or soften this)

- **Independent idio is the optimistic ceiling.** The K SKUs here have *exactly independent* idiosyncratic components (orthonormalised in-sample). Real SKUs co-move heavily — H200-us-east and H200-us-west share GPU, vendor, and demand-cycle risk. Positive cross-SKU idio correlation ρ_sku would cut the diversification benefit (residual Var falls only to ~[1/K + (1−1/K)·ρ_sku] of the concentrated value, not 1/K). **The Result-1 leftward shift is an upper bound; a cross-SKU-correlation knob is the priority follow-up.**
- Same standalone-MC scope as the headline: no execution slippage (bd 5ij), no order-book depth (bd 4tw), no price path/margin (bd 1z3); assumes a trusted basket index exists.
- SKU assignment is uniform and independent of side; clustered assignment (e.g. all shorts on one SKU) would change per-SKU netting.
- Result-2 fragmentation is sensitive to N_P / K (principals-per-SKU); at large N_P the balanced-flow penalty shrinks.

## Follow-ups (bd)
- **Cross-SKU idio correlation** ρ_sku ∈ [0,1] — turns Result 1 from a ceiling into a realistic surface (highest priority).
- γ-ratio / N_P sensitivity on the critical β (shared with the headline's open follow-up).
- Kernel §7 implementation to re-run through the real order book with bd 5ij slippage.
