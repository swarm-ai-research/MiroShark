# Does the `/polymarket` headline `yes_probability` predict outcomes?

*Closes bd-mit (`Test whether the prediction-market headline actually predicts outcomes`).*

## Setup

For every market that resolves YES or NO, we want the `yes_probability`
the polymarket envelope reported at the epoch the market was *first
seen by a consumer* (the snapshot a downstream Polymarket bot would
have sized positions against). Brier score = (yes_prob − truth)^2,
grouped by `confidence_source` (the bd-xc2 field).

- Worker population: `BALANCED_LLM_LINEUP` (Ada / Bo / Cy / Dax — the
  3-LLM + 1-contrarian default that PR #1's round-2 smoke established
  produces YES/NO splits).
- LLM model: `gemini-2.5-flash-lite` (the bd-0t5 finding: 15× faster
  than `gemini-2.5-flash` at identical parse quality on this workload).
  Also re-ran with `gemini-2.5-flash` for comparison.
- Three configs tried:
  - **15 seeds × 8 epochs × 3 steps** (flash-lite, fast)
  - **3 seeds × 6 epochs × 3 steps** (flash, control)
  - **8 seeds × 6 epochs × 6 steps** (flash-lite, longer epochs)

## Results

| run | resolved pairs | two_sided | one_sided | no_stakes Brier | no_stakes yes_rate | mean_yes_prob |
|---|---:|---:|---:|---:|---:|---:|
| flash-lite, 15s×8e×3s | 276 | 0 | 0 | 0.2500 | 0.543 | 0.500 |
| flash, 3s×6e×3s | 42 | 0 | 0 | 0.2500 | 0.357 | 0.500 |
| flash-lite, 8s×6e×6s | 112 | 0 | 0 | 0.2500 | 0.384 | 0.500 |

**Every resolved market in every config landed in `no_stakes`.** Brier
is exactly 0.25 because `yes_probability` is the Laplace prior
(`0.5`) for every envelope — no agent ever placed a stake.

## What the data says

1. **The headline `yes_probability` is uninformative as a forecast at
   default LLM-agent configuration.** Brier = 0.25 is exactly chance.
   `mean_yes_prob = 0.500` for all 430 observations because no agent
   ever pushed the stake pool in either direction.

2. **LLM agents in `BALANCED_LLM_LINEUP` don't reliably emit the
   optional `market_stake` field at default temperature 0.4.** PR
   #1's round-2 smoke showed 14 stakes from 12 ticks at the same
   personas — apparently a fragile property of the specific scenario
   that doesn't reproduce when:
   - the seed changes (different RNG draws),
   - the epoch length changes (different observation frequency), or
   - the model changes (different sampling distribution).
   The `market_stake` field is optional, so the model is free to
   omit it; at default sampling it does.

3. **The bd-xc2 `confidence_source` caveat is validated at population
   scale.** Of 430 envelopes observed across three runs, **0** would
   pass the bd-xc2 `two_sided` filter. A consumer that drops
   `one_sided` and `no_stakes` envelopes (as bd-xc2 recommends) would
   correctly receive zero forecast signal.

4. **The base rate is 38–54% YES across configs.** The market
   generator produces questions that resolve YES somewhere between a
   coin flip and 1.5× chance, depending on how the metric stream
   behaves. So there IS predictive opportunity in this market — it's
   just that the swarm's `yes_probability` isn't capturing it.

5. **The yes_probability would only become predictive if the swarm
   actually stakes.** Until that happens, the `headline` field on
   `/polymarket` is effectively decorative.

## Implications

- **The bd-xc2 `confidence_source` field is essential**, not optional.
  Without it, a downstream consumer would treat every 0.5 yes_prob as
  "the swarm has no opinion" when in reality it's "no one has
  expressed an opinion." Different things.
- **Forcing the swarm to stake requires either explicit nudging in
  the persona prompts or a separate staker policy.** Currently
  `CONTRARIAN_BEAR` Dax says "stakes NO on welfare-upside markets
  others pile YES on" but at default temp the model doesn't follow
  through.
- **The PR #1 round-2 smoke result (14 stakes / 12 ticks) was
  not a stable equilibrium.** It was a path-dependent outcome that
  doesn't reproduce. Future demos that show LLM agents
  "spontaneously trading on prediction markets" should sanity-check
  with multiple seeds.

## Sibling questions worth filing

- **Stake-forcing persona prompt** — write an "ALWAYS_STAKE_BULLISH"
  persona whose prompt demands a `market_stake` every tick on the
  highest-confidence welfare-upside market. Test whether forcing
  one side causes the contrarian to actually counter-stake.
- **Higher-temperature persona arm** — try temp=0.8 to see if
  variance increases stake frequency.
- **External-staker harness** — drive stakes from a separate process
  via `POST /api/gtb/<sim>/stake`. Lets the LLM workforce focus on
  physical actions while an "external bot" provides liquidity. Tests
  whether the env handles inbound stakes from non-worker actors
  cleanly (which it should per bd-cec / phase 7).
- **Tax-aware policy with stake hook** — extend the rule-based
  honest policy with a deterministic stake heuristic (e.g., stake YES
  on the highest welfare-upside market if your coin > 5). Cheap
  fallback that produces two-sided pools without LLM cost.

## How to reproduce

```bash
cd backend
# Default: 15 seeds × 8 epochs × 3 steps with flash-lite
uv run python -m scripts.market_predictiveness_experiment

# Longer epochs:
uv run python -m scripts.market_predictiveness_experiment --n-seeds 8 --epochs 6 --steps 6

# Different model via env override:
LLM_MODEL_NAME=gemini-2.5-flash uv run python -m scripts.market_predictiveness_experiment --n-seeds 3
```

Raw per-pair envelopes land in `runs/market_predictiveness/pairs.json`;
summary in `summary.json`.
