# Welfare-weighted Saez — the first planner to beat the heuristic

**Issue:** bd-5gz (caps the bd-cjx → bd-kk5 → bd-kwc arc) · **Scenario:**
`ai_economist_saez.yaml` · **N = 100 × 30 epochs** · 2D sweep
`planner_type × utility.labor_coeff` (4 × 4). See `elastic.svg`.

## What changed

bd-kwc made Saez optimize the populated bracket, but it still used the
**revenue-maximizing** rate `tau* = 1/(1 + a·e)` (the Saez `g = 0` limit) and
was welfare-dominated by the heuristic — it over-taxed for revenue the
`eq_times_prod` objective doesn't reward. bd-5gz replaces it with the
**welfare-weighted** rate:

    tau* = (1 - g) / (1 - g + a·e)

where `g` is the marginal social welfare weight on the top tail — computed in
`env._aggregate_stats` from the world's own CRRA utility as the top earners'
mean `u'(c) = c^(-eta)` normalized by the population mean. `g = 0` recovers the
old revenue-max rule; `g > 0` lowers the optimal rate (society values top
earners' marginal consumption, so taxing them to the revenue peak is no longer
optimal). Realized `g ≈ 0.45–0.62` here, so Saez now sets materially lower
rates (bracket-1 rate ≈ 0.47 vs 0.70 under revenue-max at `lc=0.15`).

## Result — Saez now wins (weakly but significantly) at moderate elasticity

Saez − heuristic (means, N=100):

| metric | lc=0.15 | lc=0.3 | lc=0.5 | lc=0.8 |
|---|---|---|---|---|
| welfare | +0.03 | **+0.27** | 0.00 | 0.00 |
| production | +0.45 | **+3.64** | 0.00 | 0.00 |
| tax revenue | −1.1 | −0.1 | +1.7 | +1.7 |
| Gini | 0.00 | −0.01 | 0.00 | 0.00 |

Compared with revenue-max Saez (bd-kwc), the `lc=0.3` cell **flips sign**:
welfare −0.64 → **+0.27**, production −8.95 → **+3.64**. Lower rates on the
now-elastic workforce suppress less effort, so output and welfare rise while
revenue stays near the heuristic's (no more over-extraction).

## Paired-seed test (the honest magnitude) — `lc=0.3`

The marginal p10–p90 bands overlap completely (saez [7.6, 13.9] vs heuristic
~same), which would read as "no difference." The **paired** per-seed test tells
the real story:

```
paired saez − heuristic welfare, lc=0.3, N=100 (same seeds):
  mean +0.265   se 0.074   t = 3.58   (p < 0.001)
  saez wins 18/100,  ties 80/100,  losses 2/100
  median 0.000   (80% of seeds identical)   p90 +1.10
```

So the edge is **real but small and concentrated**: on ~80% of seeds the two
planners are identical, but on the ~18% where the elastic margin bites Saez
gains (~+1.5 each), and it loses on only 2%. It never systematically hurts.
This is the first planner in the whole arc with a *positive, significant*
welfare edge over the naive heuristic — and a clean illustration of why paired
analysis beats comparing marginal confidence bands (t=3.58 hides inside fully
overlapping bands).

## Where it does and doesn't help

- **`lc=0.15` (inelastic):** +0.03 — negligible. Labor doesn't respond, so the
  rate barely matters; the lever has nothing to grab.
- **`lc=0.3` (elastic, pre-collapse):** the win. Labor responds, so Saez's
  lighter touch preserves effort the heuristic taxes away.
- **`lc ≥ 0.5` (collapsed):** 0. Output is pinned at the CRRA labor floor (91 /
  6.11); no planner separates.

The edge lives in the narrow elasticity band where labor is responsive but not
yet collapsed — consistent with bd-anv's picture of the economy.

## Net across the arc (bd-cjx → bd-5gz)

1. bd-cjx: no planner separates on welfare at the default (inelastic) workforce.
2. bd-anv: more elasticity collapses output uniformly, not selectively.
3. bd-kk5: Saez's elasticity estimate was dead (top tail from an empty bracket).
4. bd-kwc: fixed targeting → Saez separates on *revenue*, still welfare-dominated.
5. **bd-5gz: welfare-weighted Saez finally beats the heuristic** — small,
   significant, and only where labor is elastic enough to reward a lighter
   top-rate. The principled planner pays off, but the margin is modest and
   confined.

## Caveats / follow-ups

- The win is one cell (`lc=0.3`) and modest (80% ties). True, but significant
  (t=3.58) and directionally backed by production (+3.64). Not a single-seed
  artifact — it survives a paired N=100 test.
- `g` is the **utilitarian/CRRA** welfare weight; the benchmark objective is
  `eq_times_prod`. They align well enough that lighter top rates help, but an
  `eq_times_prod`-exact weight is a further refinement.
- **Sweep elasticity finer around 0.2–0.4** to map the width of the band where
  Saez beats the heuristic.
