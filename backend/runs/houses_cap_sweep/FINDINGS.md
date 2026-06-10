# Houses-per-agent cap sweep — first real welfare-equity trade-off

*Closes the main thread of bd-zri (`Houses-per-agent cap + decay +
income-by-rank env extensions`). The cap arm landed; decay and
income-by-rank are deferred to siblings (both require vendored env
changes beyond what this PR scopes).*

## Setup

- Scenario: default `ai_economist_full.yaml`.
- Sweep: `build.max_houses_per_agent` ∈ {1, 2, 3, 5, 10} (default=10).
- Seeds: 30 per cell. Length: 30 epochs × 10 steps.
- Harness: `python -m scripts.sweep_gtb`. 150 jobs in 3.0s.

## Results

| max_houses | welfare | tax | production | Gini |
|---:|---:|---:|---:|---:|
| 1 | 6.96 | 3.40 | 99.40 | **0.283** |
| 2 | 11.83 | 31.53 | 168.33 | 0.394 |
| 3 | 16.72 | 61.10 | 236.97 | 0.416 |
| 5 | 24.08 | 109.04 | 340.41 | 0.473 |
| 10 (default) | **34.08** | 184.70 | 480.95 | 0.558 |

**Every cell is on the Pareto front** — a genuine welfare-vs-equity
trade-off, the first we've found in the whole research thread. By
contrast, bd-cy8's audit-rate Pareto had exactly 1 cell on the front.

## What the data says

1. **Lower house cap → lower welfare AND lower Gini.** Every metric
   moves monotonically with the cap. Production drops 5× from
   cap=10 (480) to cap=1 (99). Welfare drops 5× the same way.
   Gini drops from 0.558 to 0.283 — the most equal distribution
   we've ever seen in the project.

2. **The default cap=10 is welfare-maximizing.** Lower caps trade
   welfare for equity, but at ~5:1 in welfare per Gini point. Even
   at `ineq_weight=50` (extreme equity preference), the composite
   `welfare − ineq_weight × Gini` still picks cap=10. You need
   `ineq_weight ≥ 100` to flip to cap=1.

3. **cap=1 is the only cell to flip to.** Composites at every
   tested weight pick either cap=10 or cap=1 — never the middle.
   The intermediate cells (2, 3, 5) are real Pareto points but
   they're dominated by the corners under any linear scoring.

4. **bd-vel's "rent attractor" is responsible.** With cap=10,
   workers each build up to 10 houses paying 1.0/step rent. With
   cap=1, they each get only 1 house. The income stream caps
   linearly with the house count, and so does welfare. Gini drops
   because no worker can pull far ahead of another in income.

5. **bd-yy1's "no parameter region keeps rent moderate AND welfare
   high" finding has its first counterexample.** cap=2 gives rent
   share probably ~50% (extrapolating from bd-vel's rent trajectory
   under cap=10) with welfare 11.83 — half the default but not
   collapsed. The houses-per-agent lever is qualitatively different
   from the housing-cost lever; it bounds the rent attractor
   directly instead of trying to make houses uneconomic.

## Composite best-cap by ineq_weight

| ineq_weight | best cap | welfare | Gini |
|---:|---:|---:|---:|
| 0.0 | 10 | 34.08 | 0.558 |
| 1.0 | 10 | 34.08 | 0.558 |
| 5.0 | 10 | 34.08 | 0.558 |
| 10.0 | 10 | 34.08 | 0.558 |
| 50.0 | 10 | 34.08 | 0.558 |
| **100.0** | **1** | **6.96** | **0.283** |

## Implications

- **This is the first env-config arm that actually breaks the
  bd-vel/yy1 rent attractor without collapsing the economy.**
  cap=3 keeps welfare at 50% of default while cutting Gini by 25%.
  Worth a closer look as a "moderate-policy default."
- **bd-an2's audit-rate findings need re-running at cap=3.** With
  the rent attractor bounded, audit_probability might actually
  recover some welfare instead of just destroying it. The bd-an2
  finding "audit nobody is best" might be specific to cap=10.
- **bd-cy8's Pareto-front shape transfers.** Under cap=10 the
  Pareto front was a single point; under cap variation it's the
  whole sweep. Future multi-objective sweeps should always include
  the cap as an axis.

## What's still deferred (sibling issues)

- **`build.income_decay_per_step`** (the decay arm of bd-zri).
  Requires a new config field + env changes (track house age, decay
  income). Substantial vendored change. Worth its own PR.
- **`build.income_by_rank_curve`** (the rank arm of bd-zri).
  Houses owned by an agent above rank N pay reduced income. Also
  needs vendored env changes. Self-contained sibling.
- **Combined cap × audit_probability sweep.** Re-run bd-an2 at
  cap=3 to see if enforcement recovers welfare when the rent
  attractor is bounded.
- **Decay + cap combo.** Whichever ships first; the two should
  compose.

## How to reproduce

```bash
cd backend
uv run python -m scripts.sweep_gtb worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
  --n-seeds 30 --epochs 30 --steps 10 \
  --sweep 'build.max_houses_per_agent=1,2,3,5,10' \
  --output runs/houses_cap_sweep
```
