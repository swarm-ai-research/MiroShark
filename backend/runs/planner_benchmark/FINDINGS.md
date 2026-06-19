# Planner benchmark — Saez vs RL vs heuristic vs bandit

**Issue:** bd-cjx · **Scenario:** `ai_economist_saez.yaml` · **N = 100 seeds ×
30 epochs × 15 steps** · paired seeds across all four cells.

```bash
uv run python -m scripts.sweep_gtb \
  worlds/gather_trade_build/scenarios/ai_economist_saez.yaml \
  --n-seeds 100 --epochs 30 \
  --sweep 'planner.planner_type=heuristic,bandit,saez,rl' \
  --output runs/planner_benchmark
uv run python -m scripts.planner_benchmark_experiment --run-dir runs/planner_benchmark
```

Workforce: 14 heterogeneous, tax-responsive workers (7 rational / 2 gaming /
2 evasive / 3 collusive). Objective held fixed at `eq_times_prod` on the
wealth Gini. See `benchmark.svg` for welfare/Gini/revenue with p10–p90 bands.

## Final-epoch results (mean [p10–p90], N=100)

| planner | welfare | Gini (wealth) | tax revenue | production |
|---|---|---|---|---|
| heuristic | 12.996 [9.71–17.09] | 0.542 [0.46–0.63] | 33.5 [20.6–46.2] | 185.7 |
| bandit | 13.013 [9.71–17.09] | 0.543 [0.46–0.63] | 33.1 [20.6–47.3] | 186.0 |
| saez | 13.013 [9.71–17.09] | 0.543 [0.46–0.63] | 32.9 [20.8–45.8] | 186.0 |
| **rl** | **12.646** [8.75–16.30] | **0.553** [0.47–0.65] | **114.3** [45.6–186.3] | 180.9 |

## Headline

**No planner significantly beats another on welfare or equity.** All four
welfare bands overlap almost completely (p10–p90 ≈ 9.7–17.1 for the principled
three, 8.7–16.3 for RL); the spread in means (12.6–13.0) is a small fraction
of one seed's p10–p90 width. The same is true for Gini. **Under this
workforce, the choice of planner does not move real outcomes.** This extends
the project's recurring "vacuous lever" result (cf. `fine_multiplier_sweep`,
`risk_based_zero_sweep`) to the planner-type axis itself.

## Per-finding

1. **heuristic ≈ bandit ≈ Saez are statistically indistinguishable** on every
   real-outcome metric. `bandit` and `saez` are bit-identical to 6 sig figs
   (welfare 13.013, Gini 0.543, production 185.985); `heuristic` differs by a
   hair (welfare 12.996, production 185.735) — far inside seed noise.
   Sophistication (Saez's inverse-elasticity rule, bandit's exploration) buys
   nothing over the rule-based heuristic here.

2. **RL is distinguishable in exactly one dimension: revenue.** RL collects
   **114.3** vs ~33 for the others — a 3.5× difference whose bands barely
   touch (RL p10 = 45.6 ≈ others' p90 ≈ 46). This is the *only* robust,
   non-overlapping separation in the whole experiment.

3. **RL's extra extraction buys no welfare or equity.** Its welfare is the
   *lowest* of the four (12.65) and its Gini the *highest/worst* (0.553),
   both within noise. RL explores into a high-rate regime these workers
   tolerate without cutting production much (180.9 vs 186), so revenue
   balloons while welfare flatlines. It is a high-tax, high-churn regime with
   no distributional payoff — strictly weak-dominated by leaving the schedule
   roughly static.

4. **Trajectory:** all four start identical (epoch 0: welfare 0.122, revenue
   0.522). heuristic/Saez welfare climbs to ~12.4 (epoch 15) → ~13.0 (epoch
   29) with revenue ~29 → ~33. RL's revenue decouples from epoch 1 (98 by
   epoch 15 → 114) while its welfare tracks just *below* the others
   throughout. The divergence is monotone and early, not a late artifact.

## ⚠️ Methodology note — the smoke lied, loudly

The de-risking smoke (**3 seeds**, 30 epochs) showed RL **winning** on welfare
(12.24 vs 11.53 for the others) and looked like the headline. At **N=100 the
sign flips**: RL is the *worst* welfare planner (12.65 vs 13.0). The 3-seed
"RL advantage" was entirely seed noise. This is exactly the failure mode
CLAUDE.md warns about — a single- or few-seed result reversed by a proper
sweep. Anyone citing the smoke would have published the opposite of the truth.

## Caveats / scope

- **One scenario, one objective, one workforce.** Findings are specific to
  `ai_economist_saez` + `eq_times_prod` + this 14-worker mix. The planners may
  separate under a more labor-*elastic* workforce, where rate differences would
  actually move effort.
- **Different action spaces, not a controlled comparison.** Saez moves only
  the top rate (capped 0.05/update); heuristic and RL move all brackets. We
  compare *outcomes*, not mechanisms.
- The `bandit ≡ saez` bit-identity suggests both effectively leave the
  schedule near its initial state (bandit's ε-greedy no-ops; Saez's capped
  top-rate move), so both ≈ the static schedule. Hypothesis, not verified.
- RL is stochastic (samples rates from per-bracket Gaussians); the wide
  revenue band (45.6–186.3) is intrinsic, not measurement error.

## Follow-ups worth filing

- **Elastic-workforce arm.** Re-run with a more tax-elastic worker mix (e.g.
  heavier `tax_aware` / higher-`eta` rational share). If planners still don't
  separate, "planner choice is welfare-vacuous" generalizes; if they do, this
  benchmark just needs a workforce where the lever bites.
- **Objective sensitivity.** Repeat under `objective=welfare` and
  `utilitarian` — RL's revenue-maximizing behavior might score differently.
- **RL convergence horizon.** RL revenue is still climbing at epoch 29; a
  50–100 epoch arm would show whether it converges or runs away.
- **cap interaction.** Pair with the `houses_cap_sweep` finding — the cap=10
  rent attractor may be what pins production flat regardless of planner.
