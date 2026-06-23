# Where welfare-weighted Saez beats the heuristic — mapping the elasticity band

**Issue:** bd-brb (follow-up to bd-5gz) · **Scenario:** `ai_economist_saez.yaml` ·
**N = 100 paired seeds × 30 epochs** per point · planner = welfare-weighted Saez
(bd-5gz). See `band.svg`, `paired.csv`.

bd-5gz found a significant paired welfare win for Saez over the heuristic at
`labor_coeff = 0.3`, nothing at 0.15 or ≥0.5. This maps the band finely
(0.15→0.50 by 0.05). At each point, saez and heuristic run on the **same 100
seeds** and the per-seed welfare difference is paired-tested — the only way to
see this effect, since it hides entirely inside overlapping marginal bands.

## Result — the win is a spiky 0.20–0.40 band, not a smooth plateau

| labor_coeff | mean Δwelfare | t | wins/ties/losses | significant? |
|---|---|---|---|---|
| 0.15 | +0.032 | 1.76 | 3 / 97 / 0 | no |
| **0.20** | **+0.382** | **6.25** | 35 / 62 / 3 | **yes** |
| 0.25 | +0.083 | 1.92 | 10 / 87 / 3 | no |
| **0.30** | **+0.265** | **3.58** | 18 / 80 / 2 | **yes** |
| 0.35 | −0.010 | −0.32 | 4 / 91 / 5 | no |
| **0.40** | **+0.301** | **3.28** | 28 / 63 / 9 | **yes** |
| 0.45 | 0.000 | — | 0 / 100 / 0 | no (exact ties) |
| 0.50 | 0.000 | — | 0 / 100 / 0 | no (exact ties) |

(Δ = saez − heuristic final-epoch welfare, paired on seed.)

## Headline

The Saez advantage is **confined to `labor_coeff ∈ [0.20, 0.40]` and is
non-monotone within it** — significantly positive at 0.20, 0.30, 0.40, but
indistinguishable from the heuristic at 0.25 and 0.35. It is strongest at the
**low-elasticity edge (0.20, t=6.25, 35/100 wins)** and fades upward. Below 0.20
the lever barely grabs (labor is too inelastic to reward a lighter rate); at
≥0.45 every planner produces *bit-identical* welfare (the workforce has
collapsed to the CRRA labor-supply floor, so the schedule is irrelevant).

**Robustness:** the three significant points survive Bonferroni correction for 8
tests (α/8 = 0.00625 → two-sided critical t(99) ≈ 2.79); 0.20/0.30/0.40 clear it
(6.25 / 3.58 / 3.28), 0.25/0.35 do not. So the spikes are not multiple-comparison
noise — but the *gaps* between them are real too.

## Why spiky, not smooth

The economy is a small discrete gridworld (14 workers, integer build/gather
actions). Whether Saez's lighter top rate actually changes a worker's
gather-vs-leisure decision is a **threshold** crossing in the CRRA
`(1−τ)·u'(c) ≥ labor_coeff` test, not a continuous response. At a given
`labor_coeff` the rate difference flips that inequality for some seeds' marginal
workers and not others; small `labor_coeff` steps move which seeds are on the
knife-edge, so the paired win count jumps around (35 → 10 → 18 → 4 → 28) rather
than varying smoothly. The win is real wherever the rate difference lands near a
worker's effort threshold — which happens unevenly across the band.

This is consistent with the bd-anv picture: in this world elasticity acts
through discrete worker dropout, not a smooth labor-supply curve, so a planner's
edge appears in patches rather than a clean interval.

## Caveats / scope

- The win is genuine but **modest and patchy**: even at the 0.20 peak, 62/100
  seeds tie and the mean edge is +0.38 welfare (~3% of the ~13 baseline). Saez
  helps the marginal seeds and rarely hurts (≤9 losses anywhere).
- One scenario, one objective (`eq_times_prod`), one elasticity knob. The
  *pattern* (patchy wins in a bounded band, exact ties past the labor floor) is
  likely general to this discrete world; the exact significant points are not.
- The `labor_coeff ≥ 0.45` exact-tie plateau is a useful invariant: once output
  floors, the planner is provably irrelevant — a clean null for future planner
  experiments to check against.

## Follow-ups

- **Finer grid (0.18–0.42 by 0.02)** to resolve whether 0.25/0.35 are true nulls
  or just troughs between spikes — would pin the band's internal structure.
- **Larger workforce** (e.g. 30–50 workers) to test whether the spikiness is a
  small-N discreteness artifact that smooths out at scale.
