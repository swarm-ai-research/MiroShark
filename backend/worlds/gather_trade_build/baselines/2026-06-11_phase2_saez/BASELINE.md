# GTB Phase 2 Acceptance — preferences, welfare, Saez planner

30-seed sweep of `scenarios/ai_economist_saez.yaml`: labor-responsive
(rational) workers with isoelastic preferences, the Saez top-rate
planner on the `eq_times_prod` wealth-Gini objective, coherent ledger,
lump-sum redistribution.

```
python -m worlds.gather_trade_build.sweep \
    worlds/gather_trade_build/scenarios/ai_economist_saez.yaml \
    --seeds 30 --processes 4
```

## Acceptance criteria (from the fidelity plan, Phase 2)

| Criterion | Result |
|---|---|
| Interior Laffer peak exists | ✅ unit-tested (`test_laffer_curve_has_interior_peak`): revenue ≈ 0 at a 0% flat rate, positive at 50%, and collapses at 95% — impossible pre-Phase-2, when effort never responded to taxes |
| Equality enters welfare with real weight | ✅ `welfare_eq_prod = mean production × (1 − wealth Gini)` and `welfare_utilitarian` (mean CRRA utility net of labor) replace the unit-mismatched legacy scalar for planner objectives |
| Wealth Gini measured | ✅ `gini_wealth` (coin + house replacement value) reported alongside the noisy per-epoch income Gini; planner configurable to react to either |
| Saez planner | ✅ top-rate inverse-elasticity rule `τ* = 1/(1 + a·e)` with online EMA elasticity estimation, per-update rate cap, monotonicity floor — unit-tested for direction, cap, and bounds |

## Headline numbers (mean ± 95% CI across 30 seeds)

| Metric | Epoch 0 | Epoch 9 | Epoch 19 |
|---|---|---|---|
| gini_wealth | 0.051 ± 0.017 | 0.402 ± 0.018 | 0.388 ± 0.017 |
| welfare_eq_prod | 0.34 | 12.70 | 19.11 |
| welfare_utilitarian | 4.24 | 31.10 | 62.15 |
| total_production | — | — | 432.5 ± 39.6 |
| total_redistributed | 0.5 | 71.1 | 118.4 |

Coin conserved in all 30 runs.

Reading notes:

- **Wealth Gini stabilizes (~0.39–0.40) instead of climbing** — compare
  the pre-Phase-1 legacy baseline, where income Gini rose monotonically
  to 0.57 with nothing pushing back. Redistribution recycles revenue
  while rich rational workers voluntarily reduce labor (income effect),
  both compressing the wealth distribution.
- **Production is lower than the legacy baseline (433 vs 684)** — also
  by design: leisure is now a choice. The legacy number assumed
  tax-invariant, satiation-free labor supply.
- The income/coin gap stays centered on 0 (±shift-timing noise),
  carried over from Phase 1.

The planner-side fix in `runner.py` matters for interpretation: the
headless runner previously fed the planner post-reset (all-zero) stats,
so every planner type was reacting to an empty epoch. It now uses the
pre-reset snapshot, like the world service. (Relevant to bead
miroshark-gtb-coq: re-run any long-horizon planner-reactivity smokes
after this fix.)
