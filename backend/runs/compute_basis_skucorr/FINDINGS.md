# Compute basis — cross-SKU idio correlation — FINDINGS (bd 0h4, ← z45 ← umm)

**N=100 seeds/cell** · β ∈ {0.20…1.0} × ρ_sku ∈ {0.0, 0.5, 0.9, 1.0} × K ∈ {4, 12}, reluctance = 1.0
Table: `TABLE.md` · Chart: `skucorr_scurve.svg` (K=12) · Repro: `uv run python -m scripts.compute_basis_skucorr_experiment --n-seeds 100`

Phase 3. Phase 2 (bd z45) showed SKU dispersion substitutes for netting — but assumed **perfectly independent** SKU idio, the optimistic ceiling. This measures how much of that benefit survives a cross-SKU idio correlation `ρ_sku`. Construction: one common idio factor `c` (⟂ basket) + K independent own factors `o_k`, with `eN_k = √ρ·c + √(1−ρ)·o_k`, so corr(eN_j,eN_k)=ρ and the dealer's warehoused residual variance uses `R_eff = ρ·(Σ_k D_k)² + (1−ρ)·Σ_k D_k²` — interpolating independent (ρ=0) and fully-correlated ≈ single-SKU (ρ=1). Limits are unit-pinned exact (`tests/test_unit_gtb_compute_basis_skucorr.py`).

---

## Headline: the diversification benefit is real but *fragile* — moderate correlation erases most of it

Critical basis β (willing-clear crosses ½), one-sided flow:

| ρ_sku → | 0.0 (independent) | 0.5 | 0.9 | 1.0 (≈ single SKU) |
|---|---|---|---|---|
| **K=12** | **0.51** | 0.82 | 0.86 | 0.86 |
| **K=4**  | 0.71 | 0.83 | 0.86 | 0.86 |

For K=12 the critical basis jumps 0.51 → 0.82 as ρ goes 0 → 0.5. The full diversification gain (single-SKU 0.86 → independent-12 0.51, i.e. a 0.35 leftward shift) is **~89% erased by ρ_sku = 0.5** (only 0.86−0.82 = 0.04 of the 0.35 survives). By ρ=0.9 the K=12 market is indistinguishable from a single SKU.

Per-cell willing-clear (K=12) shows the same collapse — e.g. at β=0.63: **0.63** (ρ=0) → 0.24 (ρ=0.5) → 0.16 (ρ=0.9) → 0.15 (ρ=1). K=4 and K=12 converge for ρ ≥ 0.9: once idio is mostly common, SKU count stops mattering.

**Why it collapses convexly:** the common factor carries fraction ρ of each SKU's idio variance and is *un-diversifiable* — the dealer holds `√ρ·c·Σ_k D_k`, whose risk scales with *total* one-sided imbalance exactly like a single SKU. Only the `(1−ρ)` own-factor part diversifies. So the effective book size is `√(ρ·(ΣD)² + (1−ρ)ΣD_k²)`; even a modest ρ makes the un-diversifiable `(ΣD)²` term dominate because for one-sided flow `(ΣD)² ≫ ΣD_k²` (by ~K).

---

## Synthesis across phases 1–3: three levers on the dealer's residual, only one is robust

| lever | mechanism | effect | robustness |
|---|---|---|---|
| **Two-sidedness** (phase 1, reluctance) | net longs vs shorts within a SKU | forms the market across all β when balanced | **robust** — the primary gate |
| **SKU dispersion** (phase 2, K) | hold many *independent* idio residuals | pulls critical β left (0.90→0.58 at K=12) | **fragile** — needs independence |
| **Cross-SKU correlation** (phase 3, ρ_sku) | shared idio factor is un-diversifiable | erases the dispersion gain by ρ≈0.5 | — |

The practical reading for compute markets: **real SKUs are highly correlated** (an H200 in us-east and one in us-west share GPU generation, vendor supply, and the AI-demand cycle — plausibly ρ_sku ≫ 0.5). So the diversification route to an intermediable exchange is largely **illusory in practice**; a chronically one-sided compute market will not be rescued by SKU proliferation. **Netting two-sided flow is the only robust route** — which puts the spotlight back on short-side (neocloud) participation as the true determinant of exchange viability (phase 1). And it sharpens the index-builders' target: a high-R² *basket* index matters most exactly where diversification fails — one-sided, high-ρ_sku markets — because there the dealer's only lever is basis quality itself.

---

## Limitations
- **Single common factor** (uniform ρ_sku across all SKU pairs). Real correlation is block-structured — SKUs cluster by GPU generation / region, so within-cluster ρ is high and cross-cluster lower. A block model would let *some* diversification survive across clusters; the uniform model is a conservative lower bound on diversification (upper bound on collapse).
- ρ_sku is applied only to the idio component; the basket loading β is held common across SKUs (all SKUs have identical basis). Heterogeneous per-SKU β is unmodelled.
- Same standalone-MC scope as phases 1–2 (no execution slippage / order-book depth / price path; trusted index assumed).

## Follow-ups (bd)
- **Block-correlation structure** (clusters of SKUs) — does cross-cluster diversification survive where within-cluster doesn't?
- γ-ratio / N_P sensitivity on the critical β (still open from the headline — turns "critical β is parameter-dependent" into a mapped surface).
- Kernel §7 implementation to re-run through the real order book + bd 5ij slippage.
