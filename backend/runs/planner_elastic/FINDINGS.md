# Elastic-workforce planner benchmark — does planner choice separate as labor gets more tax-elastic?

**Issue:** bd-anv (follow-up to **bd-cjx**) · **Scenario:** `ai_economist_saez.yaml` ·
**N = 100 seeds × 30 epochs**, paired seeds · 2D sweep
`planner_type × utility.labor_coeff` (4 × 4 = 16 cells).

```bash
uv run python -m scripts.sweep_gtb \
  worlds/gather_trade_build/scenarios/ai_economist_saez.yaml \
  --n-seeds 100 --epochs 30 \
  --sweep 'planner.planner_type=heuristic,bandit,saez,rl' \
  --sweep 'utility.labor_coeff=0.15,0.3,0.5,0.8' \
  --output runs/planner_elastic
uv run python -m scripts.planner_elastic_experiment --run-dir runs/planner_elastic
```

**Why `labor_coeff`:** `RationalWorkerPolicy` works one more unit iff
`(1 - marginal_rate)·crra_marginal(coin, eta) ≥ labor_coeff`. Raising
`labor_coeff` is the canonical AI-Economist labor-elasticity knob — it makes
labor supply quit sooner as the net-of-tax return falls. The saez scenario
sets it globally (`utility.labor_coeff`) and the rational agents inherit it,
so a single sweep axis moves the whole workforce's elasticity. See `elastic.svg`.

## Welfare — mean [p10–p90], N=100

| labor_coeff | heuristic | bandit | saez | rl | planner spread |
|---|---|---|---|---|---|
| 0.15 | 13.00 [9.7–17.1] | 13.01 | 13.01 | 12.65 [8.7–16.3] | 0.37 |
| 0.30 | 10.22 [6.9–13.2] | 10.18 | 10.21 | 9.85 | 0.38 |
| 0.50 | 6.11 [3.3–8.4] | 6.11 | 6.11 | 6.11 | **0.00** |
| 0.80 | 6.11 [3.3–8.4] | 6.11 | 6.11 | 6.11 | **0.00** |

The "planner spread" (max − min mean across the four planners) is at most
**0.37** — a small fraction of one cell's ~7-wide p10–p90 band — and collapses
to exactly 0 once `labor_coeff ≥ 0.5`. The Gini spread is ≤ 0.01 everywhere.

## Headline

**No. Making the workforce more labor-elastic does not let smarter planners
separate.** At every elasticity level the four planners are statistically
indistinguishable on welfare and Gini (bands fully overlap; bit-identical at
`labor_coeff ≥ 0.5`). This **generalizes bd-cjx**: planner choice is
welfare/equity-vacuous, and stays vacuous across the entire elasticity range
tested — the hypothesized "the lever just needs a workforce where it bites"
escape hatch is closed.

## What elasticity *does* do (uniformly across planners)

Elasticity is a powerful lever on **outcomes** — it just acts on the workforce,
not through the planner:

| metric (any planner) | 0.15 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|
| welfare | 13.0 | 10.2 | 6.1 | 6.1 |
| production | 186 | 147 | 91 | 91 |
| Gini (wealth) | 0.54 | 0.61 | **0.78** | 0.78 |

- Welfare and production roughly **halve** from the inelastic to the elastic
  regime, then **saturate** at `labor_coeff ≥ 0.5` (production floors at ~91 —
  a hard labor-supply floor set by the CRRA income effect, reached identically
  under all four planners).
- **Inequality rises with elasticity** (Gini 0.54 → 0.78): as marginal workers
  drop out under labor disutility, output concentrates among the low-disutility
  / high-skill minority. No planner counteracts this — it's pure worker
  selection, invisible to the tax schedule.

## RL still separates only on revenue — at every elasticity

| tax revenue, mean | 0.15 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|
| heuristic/bandit/saez | ~33 | ~25 | ~16 | ~16 |
| **rl** | **114** | **98** | **57** | **57** |

RL extracts 2.5–3.5× the revenue of the principled planners across the whole
range (bands barely overlap), while delivering *slightly worse* welfare and
Gini. The bd-cjx characterization — "RL is a high-tax, high-churn regime with
no welfare or equity payoff" — is robust to workforce elasticity.

## Mechanism

The planners can't separate because `heuristic`, `bandit`, and `saez` converge
to near-identical schedules (bandit ≡ saez bit-identical; see bd-cjx), so no
amount of worker rate-sensitivity differentiates them — identical rates →
identical effort. Raising `labor_coeff` lowers everyone's labor supply toward
the same CRRA floor, so it compresses the planners *together* rather than
fanning them apart. RL sets genuinely higher rates, but these workers absorb
them as extra revenue without a proportional effort cut, so RL moves revenue
and nothing else — at every elasticity.

## Caveats / scope

- One scenario, one objective (`eq_times_prod`), one elasticity knob. A
  workforce made elastic a *different* way (e.g. `tax_aware` effort-suppression,
  or higher `eta`) could behave differently, though the convergent-schedule
  mechanism would still bound separation.
- `labor_coeff ≥ 0.5` is a degenerate plateau (production floored); the
  informative elasticity range is 0.15–0.5.
- Saez's online elasticity estimate evidently does **not** translate the
  `labor_coeff` change into a meaningfully different top rate (its welfare and
  revenue track heuristic at every level) — worth a dedicated look.

## Follow-ups worth filing

- **Force schedule divergence.** The real blocker is convergent planner
  schedules. Re-run with planners seeded to *different* starting schedules, or
  add a planner that sets deliberately distinct rates, to test whether *any*
  schedule difference is welfare-visible under elastic labor.
- **Saez elasticity-tracking audit.** Confirm whether `_saez_update`'s online
  elasticity estimate actually responds to `labor_coeff` — if not, that's a
  planner bug, not an economics result.
- **Inequality-by-elasticity** is a clean standalone result (Gini 0.54→0.78);
  worth its own write-up decoupled from the planner question.
