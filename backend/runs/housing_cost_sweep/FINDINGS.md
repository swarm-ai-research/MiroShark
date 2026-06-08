# Housing-cost sweep: can we break the rent attractor?

*Closes bd-yy1 (`Add gini / welfare-sensitivity sweep for housing cost parameters`).*

## Setup

- Scenario: `worlds/gather_trade_build/scenarios/ai_economist_full.yaml`
  default lineup (honest + gaming + evasive + collusive, 14 workers).
- Sweep: 3 × 3 × 3 = 27 cells over
  - `build.wood_cost` ∈ {3, 6, 12}
  - `build.stone_cost` ∈ {3, 6, 12}
  - `build.income_per_house_per_step` ∈ {0.25, 0.5, 1.0}
- Seeds: 20 per cell.
- Length: 30 epochs × 10 steps each.
- Harness: `python -m scripts.sweep_gtb`. 540 jobs in 6.8s.
- Estimator: late_rent_share = mean rent fraction over the last 5
  epochs (same upper-bound estimator as bd-vel).

## Results sorted by final welfare

| wood | stone | inc/s | rent% | welfare | Gini |
|---:|---:|---:|---:|---:|---:|
| **3** | **3** | **1.0** | **0.96** | **34.8** | 0.551 |
| 3 | 3 | 0.5 | 0.91 | 18.9 | 0.505 |
| 3 | 3 | 0.25 | 0.83 | 10.8 | 0.455 |
| 3 | 6 | 1.0 | 0.61 | 4.2 | 0.736 |
| 6 | 3 | 1.0 | 0.64 | 4.0 | 0.746 |
| 6 | 3 | 0.5 | 0.51 | 2.7 | 0.709 |
| 3 | 6 | 0.5 | 0.50 | 2.6 | 0.684 |
| (15 more cells, all with welfare < 2 and rent% < 0.40) | | | | | |
| 6 | 6 | * | 0.00 | 0.7 | 0.744 |
| 6 | 12 | * | 0.00 | 0.7 | 0.744 |
| 12 | 6 | * | 0.00 | 0.7 | 0.744 |
| 12 | 12 | * | 0.00 | 0.7 | 0.744 |

(* = collapsed identically across all three `income_per_step` values
because no houses get built; the income param is moot when the build
action never fires.)

## What the data says

1. **There is no parameter region where rent stays moderate AND
   welfare stays high.** The Pareto front in (rent_share, welfare)
   space is degenerate: the only cells with decent welfare are the
   ones where rent dominates ≥ 83%, and the only cells where rent
   stays below 60% have welfare collapse by 8×–10×.

2. **Raising either cost to 6 (with the other at 3) cuts welfare by
   ~90%.** From welfare=34.8 at the default to welfare~4 at any
   half-doubled cost. Workers can still afford the occasional house
   (rent_share ~60%) but the economy as a whole stops producing.

3. **Raising both costs to ≥ 6 collapses the economy entirely.**
   welfare = 0.7, Gini = 0.744 across nine cells — identical numbers
   because identical behaviour: no worker can afford to build, so the
   "build N houses then idle" attractor never starts, but workers also
   can't generate enough wood + stone to keep producing other things.
   The economy dies.

4. **The rent-attractor is structural, not a parameter quirk.** The
   build action creates an infinite per-step income stream. Any
   parameter region where building is cheap enough to fire generates
   the attractor; any region where it's expensive enough to suppress
   building also suppresses the rest of the economy. There is no
   parameter region where labor stays competitive AND welfare > 5.

5. **Lowering `income_per_house_per_step` softens but doesn't break
   the attractor.** At the default (wood=3, stone=3), going from
   inc/step=1.0 → 0.25 drops welfare from 34.8 → 10.8 but rent share
   only drops 96% → 83%. The proportional income reduction makes
   labor relatively MORE competitive but doesn't help in absolute
   terms — total output collapses faster than rent share retreats.

## Implications

- **bd-vel's "build dominates" finding is robust to housing-cost
  parameters.** You can't tune it away inside the build mechanism.
- **Breaking the rent attractor requires structural changes**, not
  parameter sweeps: per-step house decay, per-house income caps that
  fall with rank, or a max-houses-per-agent below 2. None of these
  are in the current env.
- **The audit-rate frontier (bd-an2) is now more interpretable.**
  Default scenario welfare of ~72 with no audits = ~95% rent share.
  Most of the audit-rate damage comes from confiscating reserves the
  workers would have invested in building MORE houses, slightly
  lowering future rent. The Laffer-curve effect at audit_prob=0.025
  is the tiny window before this confiscation feedback kicks in.

## Sibling questions worth filing

- **Houses-per-agent cap sweep.** Sweep `build.max_houses_per_agent` ∈
  {1, 2, 5, 10} and see if a low cap breaks the attractor without
  collapsing welfare. Hypothesis: cap=2 gives the cleanest break.
- **Per-step house decay arm.** Modify the env to subtract a small
  maintenance cost per step from house income. Should restore an
  upper bound on rent share.
- **Income-by-rank cap.** Houses owned by an agent with rank > N pay
  reduced income. Restores the "diminishing returns" property the
  current sim lacks.
- **Mobility / land-rights arm.** Currently houses persist on the
  grid forever. What if other workers could squat / take over abandoned
  houses? Should redistribute rent endogenously.

These don't require new harnessing — bd-cec's sweep_gtb can hit any
of them if the env supports the parameter. The first two are config-
only and could ship as a new scenario YAML.

## Reproduction

```bash
cd backend
uv run python -m scripts.housing_cost_sweep --n-seeds 20 --epochs 30 --steps 10
# or re-aggregate from existing cells/:
uv run python -m scripts.housing_cost_sweep --skip-sweep
```
