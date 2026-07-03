# Basis-hedge through the real order book — FINDINGS (bd er8, ← umm §7 increment 2)

**N=100 seeds/cell** · β ∈ {0.20…1.0} · Table: `TABLE.md` · Chart: `kernel_residual.svg`
Repro: `uv run python -m scripts.compute_basis_kernel_experiment --n-seeds 100`

Phases 1–4 modelled the dealer's basis residual analytically (standalone Monte-Carlo). This runs the *same* mechanic through the **vendored futures engine** — a SKU forward on the new SKU-keyed book (bd bfl / increment 1) plus a basket hedge on the generic book — and checks the realised residual against theory.

## Result: the analytic residual reproduces exactly through real matching + settlement

| β | r_i | residual realized | residual analytic | rel err |
|---|---|---|---|---|
| 0.20 | 0.04 | 9.714 | 9.714 | **0.0%** |
| 0.45 | 0.20 | 8.853 | 8.853 | **0.0%** |
| 0.63 | 0.40 | 7.699 | 7.699 | **0.0%** |
| 0.80 | 0.64 | 5.948 | 5.948 | **0.0%** |
| 0.95 | 0.90 | 3.096 | 3.096 | **0.0%** |
| 1.00 | 1.00 | 0.000 | 0.000 | **0.0%** |

The std of the dealer's realised net P&L — from real order matching and real cash settlement of two contracts per seed (the SKU forward + the basket hedge) — equals the phase-1 analytic **Q·√((1−r_i)·V)** to 0.0% across the sweep, and is exactly 0 at β=1 (perfect hedge). corr(S,I)=β holds exactly (orthonormal factor innovations reused).

**Interpretation.** The basis-hedge mechanic that drives every phase-1..4 conclusion is not an artefact of the Monte-Carlo abstraction — it survives the real engine intact. The dealer really can short a SKU forward, hold a β·Q basket forward, and be left holding exactly the un-hedgeable residual `Q·√(1−r_i)·σ·ε`. This validates the SKU-keyed book + per-SKU settlement (increment 1) end-to-end and grounds the phase-1..4 residual term in real matching/settlement.

## Scope / what's still abstracted
- **No execution slippage here.** Both forwards are struck at the fair mid (crossing orders at F). Real slippage — walking a finite-depth book — is the separable bd 5ij layer; stacking it on this SKU/basket setup is increment 5.
- **Two-agent forwards** (one principal, one dealer, one basket counterparty) — this validates the *per-contract* residual, not the multi-principal participation game (that is the phase-1 headline; running the full clearing through the book is a later increment with the risk-aware dealer policy, increment 3).
- Synthetic factor spots are injected at settlement (deterministic per seed), exactly as k9w/5ij inject the settlement spot — the spot market is not itself simulated.

## Next (bd)
- Increment 3: risk-aware dealer *policy* (quote SKU forwards, auto-hedge the basket, widen spread off residual) so participation is emergent through the book.
- Increment 5: layer bd 5ij finite-depth slippage onto the SKU + basket books.
