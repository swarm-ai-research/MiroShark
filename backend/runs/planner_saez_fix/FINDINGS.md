# Saez now optimizes the populated bracket — and is still welfare-dominated

**Issue:** bd-kwc (follows bd-kk5, bd-anv, bd-cjx) · **Scenario:**
`ai_economist_saez.yaml` · **N = 100 × 30 epochs** · 2D sweep
`planner_type × utility.labor_coeff` (4 × 4). See `elastic.svg`.

## What changed

Two fixes, in sequence, made the Saez planner actually function:

1. **bd-kk5 (env):** the top tail was computed from the configured top
   *bracket* threshold (50), which this economy never reaches → `zm = 0`
   every epoch → frozen elasticity, constant `tau* = 0.5`. Fixed by falling
   back to the observed top quintile.
2. **bd-kwc (planner):** even with a live elasticity estimate, `_saez_update`
   still applied the optimized rate to `brackets[-1]` — the empty top bracket.
   Now it targets the **highest bracket whose threshold ≤ z\*** (the bracket
   the top tail actually occupies — index 1, the `[10, 25)` band here),
   carrying the rate up through the empty higher brackets for monotonicity.

Post-fix, Saez behaves correctly: it targets the populated bracket and sets a
**lower rate as labor gets more elastic** (≈0.70 on the inelastic `lc=0.15`
workforce, ≈0.37 by `lc=0.3`) — textbook inverse-elasticity.

## Result — Saez now has a real lever, but it's the *wrong* one

| metric, saez − heuristic | lc=0.15 | lc=0.3 | lc=0.5 | lc=0.8 |
|---|---|---|---|---|
| welfare | +0.02 | **−0.64** | 0.00 | 0.00 |
| tax revenue | **+20.2** | **+14.4** | **+16.2** | **+16.2** |
| production | +0.2 | **−8.9** | 0.00 | 0.00 |

- **Saez now separates — on revenue**, not welfare. It extracts 1.6–2× the
  heuristic's revenue at every elasticity level (a real, non-noise separation
  that did not exist while it optimized the empty bracket).
- **But it never wins on welfare, and at `lc=0.3` it loses** (−0.64 welfare,
  −8.9 production): its higher rates suppress the now-elastic labor.
- At `lc ≥ 0.5` everything saturates at the CRRA labor floor (6.11 / 91), so no
  planner separates regardless.

## Why — objective mismatch, not a bug

Saez's `tau* = 1/(1 + a·e)` is the **revenue-maximizing** top rate by
construction. The benchmark's welfare objective is `eq_times_prod`
(mean_income × (1 − Gini)). These are different games:

- Revenue is a transfer — it does **not** enter `eq_times_prod`, so Saez's
  extra extraction yields zero welfare credit.
- On an inelastic workforce (`lc=0.15`) Saez's high rates are free revenue
  (labor doesn't respond) but still welfare-neutral.
- On an elastic workforce (`lc=0.3`) those same high rates **cut production**,
  so optimizing revenue actively *costs* welfare.

So the corrected Saez is welfare-dominated by the naive heuristic — not because
it is broken, but because it optimizes the right rate for the *wrong objective*.

## Net across the bd-cjx → bd-kwc arc

- The "planners don't separate on welfare" headline (bd-cjx, bd-anv) **stands**,
  and is now understood three layers deep:
  1. inelastic default workforce pins production → welfare can't move (bd-cjx);
  2. raising elasticity collapses output uniformly, not selectively (bd-anv);
  3. the one planner with a non-trivial rule (Saez) optimizes revenue, which is
     orthogonal-to-harmful for the welfare objective (bd-kwc).
- Saez is now correct and inspectable; its elasticity estimate is live and its
  rate targets a real bracket. It is the right tool for a *revenue* objective.

## Follow-ups

- **Welfare-objective Saez.** Derive the planner's top rate from the configured
  `objective` (eq_times_prod / utilitarian) instead of the fixed
  revenue-maximizing Saez formula, then re-benchmark. This is the change that
  could finally let a principled planner beat the heuristic.
- **Revenue-objective benchmark.** Under `objective=welfare` (which weights
  revenue-funded redistribution differently) Saez's extra extraction might pay
  off — worth a dedicated cell.
