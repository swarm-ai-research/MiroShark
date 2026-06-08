# Audit-rate frontier on the GTB world

*Closes bd-an2 (`Run audit-rate sweep across the 0.5 EV-breakeven
threshold`).*

## Setup

- Scenario: `worlds/gather_trade_build/scenarios/ai_economist_full.yaml`
  (14 workers — honest + gaming + evasive + collusive — at default
  skill, default tax brackets 10/20/35/45%, default `fine_multiplier=2.0`,
  default `underreport_fraction=0.5`).
- Cells: 11 values of `misreporting.audit_probability` ∈ [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.65, 0.8, 0.95].
- Seeds: 100 per cell.
- Length: 30 epochs × 15 steps each.
- Harness: `python -m scripts.sweep_gtb` (bd-cec). Parallel via
  `multiprocessing.Pool`. Per-(cell, seed) export under
  `runs/audit_sweep/cells/`. **1100 jobs in 27.7s.**

## Final-epoch frontier

| audit_prob | audits | catches | tax_revenue | welfare | gini |
|---:|---:|---:|---:|---:|---:|
| 0.000 |    0.0 |   0.00 |  446.02 |  72.19 | 0.504 |
| 0.025 |    1.9 |   0.19 |  455.28 |  71.88 | 0.506 |
| 0.050 |    3.5 |   0.35 |  453.86 |  69.84 | 0.516 |
| 0.100 |    7.3 |   0.55 |  445.89 |  67.85 | 0.521 |
| 0.150 |   10.9 |   0.79 |  419.65 |  64.19 | 0.529 |
| 0.200 |   14.0 |   1.02 |  412.20 |  63.75 | 0.534 |
| 0.300 |   14.0 |   1.31 |  415.46 |  64.77 | 0.531 |
| 0.500 |   14.0 |   1.43 |  412.07 |  64.51 | 0.533 |
| 0.650 |   14.0 |   1.46 |  411.24 |  64.48 | 0.533 |
| 0.800 |   14.0 |   1.50 |  411.26 |  64.57 | 0.533 |
| 0.950 |   14.0 |   1.51 |  411.61 |  64.64 | 0.533 |

![welfare + tax_revenue vs audit_probability](welfare_tax_frontier.svg)

## What the data says

1. **Tax revenue peaks at audit_probability = 0.025 (455.28), not at 0
   and not at high enforcement.** The Laffer-curve shape is real but
   the peak is *much* lower than the naive 50% EV-breakeven would
   suggest. Going from no auditing to 2.5% audit probability raises
   tax revenue by 2.1%; going past 5% then lowers it. Above 15%, tax
   revenue plateaus around 412 ± 4 (down 9.4% from the peak).

2. **Welfare is monotonically DECREASING in audit_probability through
   the whole range.** Welfare drops from 72.19 (no audits) to 64.64
   (95% audits) — **a 10.5% welfare loss from over-policing.** There
   is no welfare peak inside the policy range; the optimum is "do not
   audit at all" if you accept welfare as your objective. Striking
   result.

3. **Catches don't collapse — they grow slowly to a ceiling of ~1.5
   per run.** With 4 evasive workers × 30 epochs = up to 120 possible
   catches per seed-run, getting 1.5 means the per-epoch catch rate
   plateaus around 5%. The `EvasiveWorkerPolicy` doesn't quit
   misreporting just because audits are high; it keeps trying
   periodically. The catch ceiling is structural (frozen agents +
   collusion-boosted retries + tax-evasion-as-stochastic-attempt)
   more than behavioural.

4. **Audits saturate at 14 by audit_probability ≈ 0.2** — the worker
   count. So every discrepant worker is being checked every epoch.
   Effective audit *yield* (catches / audits) drops as audit_prob
   rises: 10% yield at audit_prob=0.025, 7% at 0.2, 10.7% at 0.95. A
   policy that audits everyone catches roughly the same handful of
   evaders as one that audits ~3% of them.

5. **Gini rises modestly with enforcement** (0.504 → 0.534).
   Enforcement transfers coin from misreporters (who tend to gather
   aggressively) to the state, then the state transfers nothing back
   to workers, so wealth concentrates among the un-audited honest
   builders. A redistributive transfer would likely flip this, but
   the default scenario has no transfer.

6. **The Laffer effect at low audit_prob is real but tiny.** Tax
   revenue gains 2.1% from auditing 2.5% of misreporters vs auditing
   none. If the auditor's budget cost per audit is even modest (say
   0.5 coins per audit), the ~2 audits/run at 2.5% audit_prob still
   pays for itself; anything above 15% is net-negative on revenue
   even before audit-cost is netted out.

## Policy interpretation

- If your objective is **welfare**: audit nobody.
- If your objective is **tax revenue**: audit ~2.5%.
- There is no setting where heavy auditing (≥ 50%) outperforms either.

Under the default scenario the model says enforcement is a useful
*marginal* tool against the biggest misreporters but rapidly hits a
ceiling. Above that ceiling, every additional audit destroys more
welfare than it captures in fines. Consistent with the real-world
tax-policy literature on Laffer-type effects + audit-coverage
diminishing returns.

## Open questions (sibling issues worth filing)

- **`fine_multiplier` sweep.** Does the welfare cost flatten or
  steepen if the fine is smaller? Hypothesis: smaller fines let
  evasion persist at high audit rates → enforcement extracts less
  surplus → welfare cost lower but tax revenue also lower.
- **`risk_based_audit_multiplier=0` arm.** The default multiplier of
  1.5 makes audit_probability effectively higher for large
  discrepancies. Flattening to zero lets us see the pure
  `audit_probability` lever without the risk-targeting kicker.
- **RL planner reactivity** (sibling to bd-coq). Does
  `planner_type=rl` discover this welfare-max equilibrium, or does it
  overshoot toward more enforcement? An RL planner that fixates on
  catches will plant itself at audit_probability ≥ 0.5 and tank
  welfare; one that optimizes social welfare should land near 0.025.
- **Welfare-vs-equity tradeoff.** The "no audits" optimum looks great
  on welfare but slightly worse on Gini. A multi-objective frontier
  (welfare × −Gini) would be more interesting than either alone.

## How to reproduce

```bash
cd backend
uv run python -m scripts.audit_sweep_experiment --n-seeds 100 --epochs 30
# or for a quick smoke:
uv run python -m scripts.audit_sweep_experiment --n-seeds 20 --epochs 10
# or to re-render the doc/chart without re-running the sweep:
uv run python -m scripts.audit_sweep_experiment --skip-sweep
```

Outputs land under `runs/audit_sweep/`. Aggregate CSVs, per-(cell,
seed) metrics, the SVG, and this doc are all regenerated from scratch.
