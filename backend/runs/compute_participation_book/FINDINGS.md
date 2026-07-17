# Emergent participation through the real order book — FINDINGS (bd 6qu, ← umm)

**N=50 seeds/cell** · risk_factor ∈ {0, 0.02, 0.05, 0.1, 0.2} × reluctance ∈ {0, 1}
Table: `TABLE.md` · Chart: `participation_book.svg` · Repro: `uv run python -m scripts.compute_participation_book_experiment --n-seeds 50`

The phase-1..4 studies measured dealer-intermediation participation in a standalone Monte-Carlo. This drives a `RiskAwareDealerPolicy` (bd rzt) plus a heterogeneous two-sided principal population **through the real vendored order book** — the dealer quotes a SKU market whose half-spread widens with the inventory it warehouses (`spread = base + risk_factor·|net|`, `risk_factor` ∝ the basis residual 1−r_i), principals cross only when the spread is within their reservation, and the dealer hedges its net onto the basket. Volume is *emergent* from quoting, matching, and inventory feedback — not imposed.

---

## Result 1: the phase-1 structure reproduces emergently

Volume share (fraction of principal demand intermediated):

| risk_factor → | 0.0 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| **balanced** (reluctance 0) | 1.00 | 1.00 | 0.92 | 0.75 | 0.58 |
| **one-sided** (reluctance 1) | 0.54 | 0.50 | 0.33 | 0.25 | 0.17 |

Three phase-1 signatures survive real matching:
1. **At perfect basis (risk_factor=0), balanced flow is fully intermediated** (1.00); one-sided sits at ~0.54 (≈ the willing-longs fraction, shorts having withdrawn) — the same rf→0 anchor the Monte-Carlo had.
2. **One-sided always clears less than balanced** at every basis — netting genuinely helps (the dealer offsets longs against shorts, warehousing only the imbalance; hedge coverage onto the basket runs 0.88–1.00).
3. **The spread is emergent and inventory-driven**: the mean half-spread principals face rises 0.04 → 0.33 as risk_factor climbs, purely from the dealer widening on accumulated net inventory — not a set parameter.

## Result 2 (honest refinement): through a real sequential book, basis bites *even balanced flow*

The Monte-Carlo said balanced flow is basis-*invariant* — perfect netting makes the residual (and thus the spread) ~0 regardless of basis. Through the real book that idealization **breaks**: balanced volume falls 1.00 → 0.58 as risk_factor rises.

Why: netting in the MC is *instantaneous and complete*; through the book it is **sequential and imperfect**. Principals arrive and cross one lot at a time, so the dealer's inventory random-walks (mean |net| ≈ 2.5 even under balanced flow) rather than staying pinned at zero. During those transient imbalance windows the spread spikes and prices out low-reservation principals before an offsetting counterparty arrives. So **market microstructure adds a basis-sensitivity that the idealized clearing model abstracted away** — a friction that makes basis quality matter more, in reality, than the pure-netting result implied.

The ordering is preserved (one-sided still collapses harder — at rf=0.2, 0.17 vs 0.58), but the gap between the regimes *narrows* through the book: real-time netting captures only part of the benefit idealized netting promised.

---

## Synthesis

This closes the loop from the Monte-Carlo phases to the kernel: the dealer-intermediation participation curve **is reproducible through actual order matching** with an inventory-pricing dealer — validating that the phase-1 mechanism is real, not an artifact of the MC abstraction. And it sharpens the conclusion: **the "netting makes basis irrelevant for balanced flow" result is an upper bound**; a real sequential market realizes only part of it, so basis-index quality is more load-bearing than the idealized model suggested — consistent with the three-gate framing (basis quality never stops mattering; microstructure only raises its importance).

---

## Limitations
- **Finite trading window** (T=48 steps): a priced-out principal *can* cross later if an offsetting counterparty narrows the spread, but only within the window. A longer window would recover some balanced-flow volume at high risk_factor; the reported decline is partly a real-time-netting friction and partly window length. The qualitative ordering (one-sided < balanced, volume falls with basis) is robust to both.
- `risk_factor` is a spread-per-unit-inventory proxy for 1−r_i, not a settled-basis measurement; it couples the dealer's risk aversion and the basis into one knob (as the phase-1 clearing spread did).
- One dealer, one SKU + basket, one-lot quotes; a deeper book / multiple dealers would tighten spreads (competition) and raise volume.
- Homogeneous principal size (1 lot); heterogeneous sizes would weight the volume share.

## Follow-ups (bd)
- Window-length sensitivity to separate the real-time-netting friction from finite-T (does balanced volume recover toward the MC as T→∞?).
- Multiple competing dealers — does competition restore the MC's balanced-flow basis-invariance?
