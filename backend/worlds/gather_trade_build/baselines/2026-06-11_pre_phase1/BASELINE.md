# GTB Baseline Characterization — pre-Phase-1

30-seed sweep of `scenarios/ai_economist_full.yaml` (20 epochs × 15
steps, 14 workers), run with the Phase 0 harness:

```
python -m worlds.gather_trade_build.sweep \
    worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
    --seeds 30 --processes 4
```

`summary.csv` holds per-epoch mean/std/95%-CI/min/max for every metric;
`manifest.json` pins the exact resolved scenario config, its hash, and
the git SHA. Regenerate the (uncommitted) per-seed CSV with the command
above — same seeds, same numbers, the kernel is deterministic per seed.

This is the "before" snapshot the fidelity phases are judged against
(see `docs/plans/2026-06-11-gtb-economic-sim-fidelity.md`).

## Headline numbers (mean ± 95% CI across 30 seeds)

| Metric | Epoch 0 | Epoch 9 | Epoch 19 |
|---|---|---|---|
| total_production | 90.5 ± 3.1 | 413.4 ± 35.5 | 683.8 ± 64.9 |
| gini_coefficient | 0.259 ± 0.019 | 0.542 ± 0.021 | 0.569 ± 0.020 |
| welfare (legacy scalar) | 6.3 ± 0.2 | 29.3 ± 2.5 | 48.6 ± 4.6 |
| total_tax_revenue | 9.4 ± 0.4 | 111.3 ± 12.9 | 274.1 ± 31.3 |
| total_houses_built | 0.7 ± 0.2 | 25.9 ± 2.5 | 43.9 ± 4.1 |
| total_audits | 11.7 ± 0.4 | 13.9 ± 0.1 | 14.0 ± 0.1 |
| total_catches | 1.40 ± 0.20 | 0.73 ± 0.26 | 0.10 ± 0.11 |
| undetected_evasion_rate | 0.27 ± 0.10 | 0.00 | 0.00 |
| bunching_intensity | 0.052 ± 0.022 | 0.012 ± 0.010 | 0.031 ± 0.014 |
| collusion_events_detected | 37.1 ± 2.2 | 31.6 ± 2.8 | 34.3 ± 2.8 |
| governance_backfire_events | 10.3 ± 0.4 | 13.2 ± 0.2 | 13.9 ± 0.2 |
| income_coin_gap | 85.3 ± 2.6 | 33.1 ± 5.0 | 28.1 ± 18.9 |

Coin conservation (the hard ledger invariant) held in all 30 runs.

## What the baseline confirms

These quantify the fidelity gaps the plan diagnosed from the code:

1. **Income ≠ coin (Phase 1.1).** At epoch 0 the income/coin gap is
   85.3 of 90.5 total production — ~94% of booked income has no coin
   behind it (gather income mints nothing). The gap shrinks only
   because house income (which does mint coin) comes to dominate.
2. **House-income compounding with no redistribution (Phase 1.3/1.4).**
   Gini more than doubles (0.26 → 0.57) as house owners compound;
   taxes are burned, so nothing pushes back.
3. **Collusion detector is mostly noise (Phase 4.3).** ~34 detections
   per epoch in a world where the colluding coalition has 3 members
   (3 possible cartel pairs). Nearly every worker ends up audit-boosted:
   total_audits saturates at ~14 = n_workers, and 13-14 false-positive
   audits hit honest agents every epoch.
4. **Enforcement looks "perfect" for the wrong reason (Phase 4.2).**
   Catches fall to ~0.1/epoch and undetected evasion to 0 by mid-run —
   not because policy works, but because evasive agents get caught
   early (selection ⇒ certain conviction), frozen, and effectively
   exit the strategic game.
5. **No bunching (Phase 6.1).** Bunching intensity ≈ 0 throughout:
   bracket thresholds (10/25/50) are static while per-worker income
   grows past them, so the gaming agents' shift window rarely binds.
6. **Audit count is policy-insensitive.** total_audits pins at
   n_workers regardless of `audit_probability` because collusion
   boosts dominate — the an2 audit-rate sweep must control for this.
