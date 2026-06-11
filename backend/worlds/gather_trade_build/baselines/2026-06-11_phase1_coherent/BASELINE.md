# GTB Phase 1 Acceptance — coherent ledger

30-seed sweep of `scenarios/ai_economist_coherent.yaml` (same world as
the pre-Phase-1 baseline, with `ledger_mode: coin`,
`redistribution: lump_sum`, `debt_enabled: true`):

```
python -m worlds.gather_trade_build.sweep \
    worlds/gather_trade_build/scenarios/ai_economist_coherent.yaml \
    --seeds 30 --processes 4
```

## Acceptance criteria (from the fidelity plan, Phase 1)

| Criterion | Result |
|---|---|
| Coin conservation in every epoch of every run | ✅ all 30 runs |
| income == net coin flow | ✅ income_coin_gap = 0.000 at epoch 0; −2.7 ± 23.4 at epoch 19 (residual is the gaming agents' shift-income *timing*, which nets out across epochs — vs +85.3 structural gap in the legacy baseline) |
| tax revenue == coin actually moved | ✅ total_redistributed == taxes + fines, every epoch (271.8 vs 265.3 + fines at epoch 19) |

## Comparison vs `2026-06-11_pre_phase1`

| Metric (mean ± CI95) | legacy, ep0 | coherent, ep0 | legacy, ep19 | coherent, ep19 |
|---|---|---|---|---|
| total_production | 90.5 ± 3.1 | 5.3 ± 1.8 | 683.8 ± 64.9 | 686.1 ± 58.7 |
| gini_coefficient (income) | 0.259 | 0.517 | 0.569 | 0.591 |
| total_tax_revenue | 9.4 | 0.5 | 274.1 | 265.3 |
| income_coin_gap | 85.3 | 0.000 | 28.1 | −2.7 |

Reading notes:

- **Epoch-0 production collapses from 90.5 to 5.3** — by design. The
  legacy number was ~94% phantom (gather income with no coin). The
  coherent number is real coin income (early house payouts + trades).
- **Income Gini did not fall under redistribution.** Two reasons, both
  expected: (1) lump-sum transfers are not booked as taxable income, so
  the *income* Gini doesn't see them — this is precisely why Phase 2.3
  adds a wealth Gini, which is the right lens for redistribution;
  (2) in coin mode measured income is house-dominated, and houses are
  concentrated. Do not read the income-Gini column as "redistribution
  failed"; re-evaluate after Phase 2.3.
- total_tax_shortfall is 0 throughout: with redistribution recycling
  coin back to workers, everyone can cover their tax bill, so the debt
  mechanism (verified in unit tests) never binds in this scenario.
