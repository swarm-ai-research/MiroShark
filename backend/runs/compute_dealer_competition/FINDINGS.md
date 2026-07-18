# Competing dealers vs basis — FINDINGS (bd hka, ← 6qu/4tw)

**N=50 seeds/cell** · n_dealers ∈ {1, 2, 4} × risk_factor ∈ {0, 0.05, 0.1, 0.2} × reluctance ∈ {0, 1}
Table: `TABLE.md` · Chart: `competition_curve.svg` (balanced) · Repro: `uv run python -m scripts.compute_dealer_competition_experiment --n-seeds 50`

The through-the-book participation study (bd 6qu) found a *single* sequential dealer breaks the Monte-Carlo's "balanced flow is basis-invariant" result — real-time netting is imperfect, the dealer's inventory random-walks, its spread spikes, and volume falls with the basis proxy. This asks whether **dealer competition** restores the MC upper bound. `n_dealers` RiskAwareDealerPolicy makers quote the same SKU; principals cross the best quote across them; each widens its *own* spread on its *own* inventory, so flow load-balances to the least-filled dealer.

---

## Result: competition restores balanced-flow basis-invariance — but not the one-sided gate

**Volume share (median):**

| | rf=0 | rf=0.05 | rf=0.1 | rf=0.2 |
|---|---|---|---|---|
| **balanced, 1 dealer** | 1.00 | 0.92 | 0.75 | 0.58 |
| balanced, 2 dealers | 1.00 | 1.00 | 0.92 | 0.79 |
| **balanced, 4 dealers** | 1.00 | 1.00 | 1.00 | **0.92** |
| **one-sided, 1 dealer** | 0.54 | 0.33 | 0.25 | 0.17 |
| one-sided, 2 dealers | 0.54 | 0.50 | 0.42 | 0.33 |
| **one-sided, 4 dealers** | 0.54 | 0.50 | 0.50 | **0.42** |

1. **Balanced flow: competition restores the MC upper bound.** With 4 dealers, balanced volume stays ≈1.0 across the whole basis range (0.92 at rf=0.2, vs 0.58 for a single dealer). Load-balancing shrinks each dealer's transient inventory ≈1/n, so the aggregate best quote stays tight and low-reservation principals are no longer priced out. **The 6qu single-dealer basis-sensitivity of balanced flow was a curable microstructure artifact** — competition erases most of it.
2. **One-sided flow: the basis gate survives competition.** Four dealers help (rf=0.2: 0.17 → 0.42) but one-sided volume *still falls with basis* (0.54 → 0.42). Competition can **divide** one-sided inventory across dealers but cannot **net** it — there is no opposing flow to cancel against. So the fundamental one-sided basis dependence is robust to competition, exactly as phase-1..4 implied.

---

## Synthesis — competition substitutes for netting only when flow is two-sided

This sharpens the whole arc's central result. There are two ways to keep a dealer's warehoused residual small: **net** two-sided flow, or **load-balance** it across competing dealers. Both work for *balanced* flow — and with either, basis quality barely matters. But **one-sided flow defeats both**: there is nothing to net (phase 1) and nothing to cancel by spreading across dealers (this study) — dividing a one-sided block by `n` still leaves each dealer holding one-sided basis risk that grows with the flow. So:

> **Basis-index quality is decisive precisely in the one-sided regime — the one corner that neither netting nor competition can rescue.** In balanced markets, microstructure (more dealers) makes basis nearly irrelevant; in one-sided markets (a chronically-long compute market with a reluctant neocloud short side), a high-R² basket index is the *only* lever, and its quality sets the ceiling on how much of the market can form.

This closes the participation thread: the MC's "netting beats basis" was right in spirit, the single-dealer book overstated basis's role via a microstructure friction, and competition confirms the friction is curable for balanced flow while leaving the genuine one-sided gate intact — reinforcing, from a third angle, that **short-side two-sidedness is the master variable**.

---

## Limitations
- Homogeneous dealers (same risk_factor, differing only by a seed offset in quote timing); heterogeneous dealer risk appetites would let the most risk-tolerant dealer absorb more, changing the load-balancing.
- Dealers compete only on price via the shared best quote; no inventory-driven quote-pulling or adverse-selection avoidance (bd 4tw's fuller "endogenous repricing" — this covers the competing-makers half).
- Same finite trading window (T=48) and one-lot quotes as bd 6qu; competition also mechanically adds quote capacity per unit time, which contributes to the balanced-flow recovery alongside the load-balancing effect.

## Follow-ups (bd)
- Heterogeneous dealer risk appetites / capital — does a single deep dealer beat many thin ones?
- Merge with bd 4tw's dealer quote-pulling under persistent one-sided flow (this did competing makers; 4tw adds endogenous depth withdrawal).
