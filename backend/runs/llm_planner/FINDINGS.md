# LLM workforce × planner reactivity

*Closes bd-5t0 (`LLM × planner sweep — combines bd-coq with LLM
workforce`). Was blocked on bd-2e2 OR bd-1nq; both landed first.*

## Setup

- Workforce: `BALANCED_LLM_LINEUP` (Ada / Bo / Cy / Dax, 4 LLM
  personas via `gemini-2.5-flash-lite`).
- Planner types: heuristic, bandit, rl(stub). All at
  `learning_rate=0.05`, `update_interval=1`, allow_non_monotone.
- Seeds: 5 per cell. Length: 8 epochs × 4 steps.
- Wall clock: ~17 min Gemini calls (4 workers × 32 ticks × 3 cells ×
  5 seeds = 480 ticks × ~2s).

## Results

| planner | welfare | tax_revenue | Gini |
|---|---:|---:|---:|
| **heuristic** | **0.46** | 0.06 | 0.187 |
| bandit | 0.33 | 0.19 | 0.706 |
| rl (stub) | 0.12 | 0.10 | 0.600 |

## What the data says

1. **Welfare DOES vary across planner types under an LLM
   workforce.** Heuristic gets 0.46, rl gets 0.12 — a 3.8× spread.
   bd-coq's "planners are vacuous" finding does not survive the
   transition from rule-based to LLM workforce, just as bd-2e2
   showed for the TaxAwareHonest rule-based policy.

2. **The rl stub is the WORST planner here** — first time it has
   lost a comparison. With a workforce that actively responds to
   bracket changes (LLMs read `tax_schedule` from obs), keeping the
   defaults locked is worse than perturbing them. This inverts the
   bd-coq finding completely.

3. **Heuristic produces dramatically lower Gini** (0.187 vs 0.706
   for bandit). The heuristic's "raise rates when Gini is high"
   rule actually works against the LLM workforce — they pull back
   effort under high marginal rates, reducing absolute production
   gaps. The bandit's random perturbations don't have any direction,
   so equality improves only when by luck.

4. **All three welfares are very small** (0.12–0.46) because of
   the small workforce (4 workers vs 14 default) and short
   horizon (32 ticks vs 450 in bd-an2). Don't compare absolutes
   to other sweeps; the relative ordering is the signal.

5. **LLM workers respond more strongly to brackets than bd-2e2's
   TaxAwareHonestPolicy did.** bd-2e2 saw welfare spread 1.4 points
   across 9 cells (39.78 → 41.15). This 3.8× ratio (0.12 → 0.46)
   is much larger relative to the absolute numbers. Plausibly the
   LLM agents are more aggressive at suppressing effort than the
   rule-based policy.

## Combined picture across the planner reactivity thread

| workforce | welfare spread across planners | rl rank |
|---|---|---|
| Default rule-based (bd-coq) | 0 | tied for best (do-nothing wins) |
| TaxAwareHonest (bd-2e2) | 1.4 points (39.78-41.15) | best (do-nothing still wins) |
| BALANCED_LLM_LINEUP (bd-5t0) | 0.34 abs, 3.8× ratio | **worst** |

The progression makes sense: as workers become more responsive to
brackets, doing nothing becomes a worse policy, and the planner that
matches the actual welfare gradient (heuristic's "raise on
inequality") starts to dominate.

## Implications

- **bd-an2's audit-rate findings need re-running with the LLM
  workforce too.** If audit_probability responds to workforce type
  the way planner type does, the "audit nobody is best" finding
  could invert.
- **Heuristic > bandit > rl on LLM workforce** is the first
  evidence that AI Economist-style mechanism design has something
  to teach when the workforce is also LLM-driven. Worth scaling up.
- **The default rule-based scenario is a stress test for
  "interventions don't work."** Most policy levers vacuous because
  workers don't respond. Swap in LLM workers and the levers come
  alive.

## Caveats

- 5 seeds is small. The 0.46 vs 0.12 spread is large enough to be
  meaningful, but the absolute numbers are noisy. Should be 20+
  seeds for any policy claim.
- gemini-2.5-flash-lite at temp=0.4. Different model or temperature
  could produce different equilibria.
- Short horizon (8 epochs). Planner reactivity over longer horizons
  (bd-coq used 40) might produce different rankings.

## Sibling questions worth filing

- **bd-5t0 at larger scale.** 20 seeds × 20 epochs × 4 steps.
  ~3 hours of Gemini calls. Definitively answers whether heuristic
  beats rl under LLM workforce.
- **Combined audit_probability × planner_type sweep.** Re-run
  bd-an2 with `BALANCED_LLM_LINEUP` — does the audit Pareto shift?
- **Per-persona effort suppression elasticity.** Which of Ada/Bo/Cy/Dax
  most strongly suppresses effort under high marginal rates?
  Diagnostic data exists in the event log; just needs aggregation.

## How to reproduce

```bash
cd backend
uv run python -m scripts.llm_planner_experiment --n-seeds 5 --epochs 8 --steps 4
```
