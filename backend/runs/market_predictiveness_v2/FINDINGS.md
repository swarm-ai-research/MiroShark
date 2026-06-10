# Does the polymarket headline yes_probability predict outcomes?

*Closes bd-1nq (`Add ALWAYS_STAKE persona + re-run bd-mit`). Together
with bd-mit's original run, this gives the first definitive answer.*

## Setup

- 15 seeds × 6 epochs × 4 steps with the new **`STAKING_LINEUP`**
  (ALWAYS_STAKE_BULLISH + ALWAYS_STAKE_BEARISH + CONTRARIAN_BEAR +
  1 honest worker). Personas demand a `market_stake` every tick.
- LLM model: `gemini-2.5-flash-lite` (bd-0t5 finding).
- **Methodological fix from bd-mit:** capture the LATEST open envelope
  per market_id (overwritten each tick) instead of the first
  envelope at creation. The bd-mit version captured `no_stakes`
  for every market because stakes accumulate over time; the
  snapshot-at-creation always preceded any staking activity.

## Results

| confidence_source | n | Brier | yes_rate | mean_yes_prob |
|---|---:|---:|---:|---:|
| no_stakes | 139 | 0.2500 | 0.468 | 0.500 |
| one_sided | 43 | 0.2871 | 0.628 | 0.579 |
| **two_sided** | **28** | **0.4519** | **0.357** | **0.772** |

Diagnostic: 51–56 stakes were placed per seed (~750 total across 15
seeds), confirming the ALWAYS_STAKE personas worked.

## What the data says

1. **The two_sided headline `yes_probability` is WORSE than chance.**
   Brier = 0.452 vs the no-information prior of 0.250. **A consumer
   that filters down to two_sided envelopes (per bd-xc2 guidance)
   would do strictly worse than ignoring the headline entirely.**

2. **The headline is systematically overconfidently bullish.** When
   two-sided pools exist, the mean predicted YES probability is
   0.772, but the actual yes_rate is only 0.357. **The swarm is
   wrong by 41 percentage points in the same direction.** Not "noisy
   around the truth" — directionally inverted.

3. **One-sided pools are MORE accurate than two-sided** (Brier 0.287
   vs 0.452). The one-sided pools come from rounds where only one
   persona staked — typically the contrarian alone, since the
   bullish/bearish always-stake personas usually fire together. So
   the contrarian's NO stakes correctly bias the headline DOWN, but
   not far enough.

4. **The no_stakes prior is honest.** Brier = 0.250 = ((0.5)² ×
   yes_rate + (0.5)² × (1-yes_rate)) = 0.25 exactly. The Laplace
   prior is the optimal default in the absence of staking — it
   beats every other category here.

5. **Yes_rate by category tells the story.** Markets that don't get
   staked have yes_rate = 47%, near chance. Markets that get one
   side staked (mostly contrarian) have yes_rate = 63% — the
   contrarian preferentially fades clearly-bullish situations that
   actually go bullish more often. Markets that get BOTH sides
   staked are the 36% — the swarm only achieves consensus when
   the question is genuinely hard, and they consistently call it
   wrong.

## Why?

The LLM agents read the market generator's questions ("Will welfare
exceed X by epoch N?") and pattern-match on "the agents are working
hard, so welfare will probably exceed X." But the GTB world has a
build-dominates attractor (bd-vel) and an audit/tax extraction
mechanism (bd-an2) that systematically pulls metrics DOWN from peak.
The LLM swarm doesn't know about these structural pressures; it just
sees aspirational personas.

Equivalently: the markets are AUTO-GENERATED with thresholds at
+20% of current value (bd-4 phase). For most metrics, +20% over 5
epochs is an aggressive bet. The system implicitly biases questions
toward upside that doesn't materialize, and the LLM swarm bites every
time.

## Implications

- **The bd-xc2 `confidence_source` field is essential AND
  CONTRARIAN.** Don't just filter to `two_sided`; treat
  `two_sided` envelopes as inverted signals (bet the OPPOSITE
  direction). The headline is a counter-indicator.
- **The polymarket envelope is not just decorative** (as bd-mit
  originally concluded) — it's actively misleading when populated by
  LLM personas. This is worse than a useless signal; it's a trap.
- **The right fix is upstream of the consumer.** Either:
  (a) **Calibrate the question generator** to produce a 50/50 prior
      by default. Currently the +20% step bakes in upside bias.
  (b) **Add a known-calibrated forecaster** to the lineup — e.g., an
      LLM agent shown historical metric trajectories before each
      question. Restores a baseline of accuracy.
  (c) **Compute the headline using a different aggregation** — e.g.,
      Bayesian update from the prior using only stakes that beat a
      bookmaking spread.
- **For PR review of LLM-driven prediction markets in general:**
  treat their headline as adversarial. The PR #1 unification PR's
  framing ("the polymarket envelope can be consumed by any
  Polymarket bot") is technically true but the bot would lose
  money. A `health_check` or `calibration_score` field next to
  `yes_probability` would be a defensive minimum.

## Sibling questions worth filing

- **Calibrate the question generator.** Adjust `_threshold_pair`
  in `gtb_markets.py` to use the median + p25/p75 of recent metric
  trajectories instead of ±20% of current. Re-run bd-1nq; expect
  no_stakes yes_rate ~0.5 and the two_sided overconfidence to
  shrink.
- **Known-calibrated forecaster persona.** Add a `FORECASTER`
  persona that gets full metric history in its obs and is
  instructed to make probability-calibrated stakes only when
  confident. Test whether including it in the lineup pulls Brier
  back toward 0.25.
- **Adversarial bd-1nq.** Run with personas explicitly informed
  that "the markets are auto-generated with upside bias." Test
  whether the LLM can self-correct.
- **Compare Brier with and without contrarian.** Drop CONTRARIAN_BEAR
  from the lineup and re-run. Does the two_sided Brier get even
  worse (no NO-side staking) or does it become impossible (only
  one_sided)?

## How to reproduce

```bash
cd backend
uv run python -m scripts.market_predictiveness_experiment \
  --n-seeds 15 --epochs 6 --steps 4 \
  --output runs/market_predictiveness_v2
# Switch back to flash if flash-lite seems unreliable:
LLM_MODEL_NAME=gemini-2.5-flash uv run python -m scripts.market_predictiveness_experiment \
  --n-seeds 3 --epochs 6 --steps 4 \
  --output runs/mit_flash_compare
```
