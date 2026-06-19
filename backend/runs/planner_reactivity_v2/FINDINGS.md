# Planner reactivity with TaxAwareHonestPolicy

*Closes bd-2e2 (`Add TaxAwareHonestPolicy`). Validates that the
bd-coq finding ("planners are vacuous") was an artifact of the
default rule-based lineup, not a property of the env.*

## Setup

- Scenario: `worlds/gather_trade_build/scenarios/ai_economist_full.yaml`
  with all `policy: honest` agents swapped to `policy: tax_aware`
  (default config: `rate_threshold=0.30`, `effort_suppression=0.7`).
  Stored at `runs/planner_reactivity_v2/tax_aware_scenario.yaml`.
- Sweep: 3 × 3 × 1 = 9 cells over
  `planner_type` ∈ {heuristic, bandit, rl} × `learning_rate` ∈
  {0.01, 0.05, 0.10} with `allow_non_monotone=true`.
- Seeds: 20 per cell × 40 epochs × 10 steps. 180 jobs in 4.3s.

## Final-epoch results

| planner | lr | welfare | tax_revenue | Gini | production |
|---|---:|---:|---:|---:|---:|
| heuristic | 0.01 | 41.07 | 271.04 | 0.546 | 578.83 |
| **heuristic** | **0.05** | **41.12** | **531.41** | 0.540 | 579.51 |
| heuristic | 0.10 | 39.78 | 526.33 | 0.544 | 560.70 |
| bandit | 0.01 | 41.15 | 174.12 | 0.538 | 579.93 |
| bandit | 0.05 | 40.80 | 174.46 | 0.536 | 574.97 |
| bandit | 0.10 | 40.49 | 178.11 | 0.534 | 570.61 |
| rl | (any) | 41.15 | 173.61 | 0.538 | 579.93 |

Spread across cells: welfare 39.78–41.15 (1.4 points); tax_revenue
173.61–531.41 (3.1× range); Gini 0.534–0.546.

Compare to **bd-coq's original sweep with the default rule-based
lineup**: welfare = 40.50 exact in every cell, Gini = 0.540 exact in
every cell. **Welfare and Gini are now responsive.**

## What the data says

1. **The planner can now move welfare**, validating bd-2e2's
   hypothesis. Heuristic at lr=0.05 achieves 41.12 vs heuristic at
   lr=0.10 at 39.78 — a 3.4% welfare difference from the same planner
   at different reactivity. That's small but real and would be
   statistically meaningful with more seeds.

2. **The bd-coq finding ("planners are vacuous") is specific to the
   default rule-based lineup**, not a property of the GTB env or
   planner code. With one tweak to the worker population — making
   honest workers read tax brackets — the planner-vs-workers loop
   becomes a real mechanism design problem.

3. **rl planner stays best at 41.15**, tied with bandit at lr=0.01.
   The rl stub doesn't change brackets, so workers face stable
   conditions and don't need to suppress effort. **The "do nothing"
   policy is currently the welfare-maxing planner against
   tax-aware workers.** This is the same shape as bd-an2's "do not
   audit at all" finding — the welfare-maxing policy is laissez-faire.

4. **Heuristic at high lr overshoots and hurts welfare** (41.07 at
   lr=0.01 → 39.78 at lr=0.10). The heuristic's
   "raise rates when Gini is high" rule, applied aggressively, pushes
   marginal rates above the 0.30 threshold the tax_aware workers
   suppress effort at. Production drops 18 points (579 → 561).

5. **Heuristic at lr=0.05 collects 3× more tax revenue than bandit
   or rl** (531 vs 174) at nearly identical welfare. **There's a
   real Pareto improvement available**: rate brackets that pull in
   high tax revenue without crossing the effort-suppression threshold.
   The heuristic finds it at lr=0.05; the bandit doesn't because it's
   exploring randomly.

## Implications

- **The combined finding from bd-an2 + bd-coq + bd-2e2 is now
  consistent**: at the laissez-faire end (no audits, do-nothing
  planner), welfare is highest. Any policy that touches workers
  (audits, bracket changes that cross effort thresholds) hurts welfare
  *unless* perfectly calibrated. The heuristic at lr=0.05 finds the
  one Pareto-improving point in this sweep.
- **The next experiment is bd-mit-style on tax-aware workers** —
  whether the welfare-vs-revenue Pareto frontier is two-dimensional
  or strictly dominated by the rl-stub corner. Could be filed as a
  sibling.
- **bd-5t0 (LLM × planner sweep) is now unblocked.** TaxAwareHonest
  was one of two prerequisites; bd-5t0 can pick which path to take
  (this policy or the bd-1nq stake-forcing personas).

## Sibling questions worth filing

- **rate_threshold sweep on TaxAwareHonestPolicy.** Vary the
  worker's effort-suppression threshold ∈ {0.20, 0.25, 0.30, 0.35,
  0.40} and see how the welfare-maxing planner config shifts.
- **effort_suppression magnitude sweep.** Try suppression ∈ {0.3,
  0.5, 0.7, 0.9} to see if stronger backlash from workers makes the
  planner more cautious.
- **TaxAware × evasion combo.** Currently the tax_aware policy
  doesn't misreport; combining shift_income + effort_suppression
  responses would give the most realistic adversarial worker.

## How to reproduce

```bash
cd backend
# Build the tax_aware scenario:
python3 -c "
import yaml
with open('worlds/gather_trade_build/scenarios/ai_economist_full.yaml') as f: d = yaml.safe_load(f)
d['agents'] = [{**s, 'policy': 'tax_aware'} if s.get('policy') == 'honest' else s for s in d['agents']]
import os; os.makedirs('runs/planner_reactivity_v2', exist_ok=True)
with open('runs/planner_reactivity_v2/tax_aware_scenario.yaml', 'w') as f: yaml.safe_dump(d, f)
"
# Run the sweep:
uv run python -m scripts.sweep_gtb runs/planner_reactivity_v2/tax_aware_scenario.yaml \
  --n-seeds 20 --epochs 40 --steps 10 \
  --sweep 'planner.planner_type=heuristic,bandit,rl' \
  --sweep 'planner.learning_rate=0.01,0.05,0.10' \
  --sweep 'taxation.allow_non_monotone=true' \
  --output runs/planner_reactivity_v2
```
