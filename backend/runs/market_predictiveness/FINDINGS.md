# Does the `/polymarket` headline `yes_probability` predict outcomes?

*Closes bd-mit (`Test whether the prediction-market headline actually predicts outcomes`).*

## Setup

For every market that resolves YES or NO, we record the **latest open
envelope** the polymarket layer emitted before resolution — the
most-informed `yes_probability` a downstream consumer could have acted
on. Brier score = (yes_prob − truth)^2, grouped by `confidence_source`
(the bd-xc2 field).

> **Methodological note (PR #3 review catch).** The first version of
> this experiment captured the envelope at market *creation* and never
> updated it. A just-created market cannot have stakes yet, so every
> pair landed in `no_stakes` with `yes_probability = 0.5` *by
> construction*, and the run measured nothing about staking behavior.
> All numbers below are from re-runs with the fixed capture (latest
> open snapshot per market, plus a direct stake-event count from the
> stake history so "did anyone stake" is observed, not inferred).

- Worker population: `BALANCED_LLM_LINEUP` (Ada / Bo / Cy / Dax — the
  3-LLM + 1-contrarian default that PR #1's round-2 smoke established
  produces YES/NO splits).
- LLM model: `gemini-2.5-flash-lite` (the bd-0t5 finding: 15× faster
  than `gemini-2.5-flash` at identical parse quality on this workload).
  Also re-ran with `gemini-2.5-flash` for comparison.
- Three configs:
  - **15 seeds × 8 epochs × 3 steps** (flash-lite, fast)
  - **3 seeds × 6 epochs × 3 steps** (flash, control)
  - **8 seeds × 6 epochs × 6 steps** (flash-lite, longer epochs — the
    config whose `summary.json` / `pairs.json` are committed here)

## Results

| run | pairs | stakes placed | source | n | Brier | yes_rate | mean_yes_prob |
|---|---:|---:|---|---:|---:|---:|---:|
| flash-lite, 15s×8e×3s | 260 | 497 | two_sided | 18 | 0.4212 | 0.111 | 0.654 |
| | | | one_sided | 61 | 0.3052 | 0.541 | 0.760 |
| | | | no_stakes | 181 | 0.2500 | 0.541 | 0.500 |
| flash, 3s×6e×3s | 42 | 125 | two_sided | 3 | 0.4345 | 0.000 | 0.659 |
| | | | one_sided | 4 | 0.2887 | 0.250 | 0.634 |
| | | | no_stakes | 35 | 0.2500 | 0.486 | 0.500 |
| flash-lite, 8s×6e×6s | 112 | 243 | two_sided | 11 | 0.3752 | 0.364 | 0.668 |
| | | | one_sided | 33 | 0.3368 | 0.515 | 0.811 |
| | | | no_stakes | 68 | 0.2500 | 0.441 | 0.500 |

## What the data says

1. **The BALANCED_LLM_LINEUP stakes routinely.** 8–47 stakes per seed
   (243 total in the committed config). The earlier conclusion that
   "no agent ever placed a stake" was a measurement artifact of the
   capture-at-creation bug, not a property of the agents. The PR #1
   round-2 smoke (14 stakes / 12 ticks) was representative after all.

2. **Where the swarm stakes, the headline is WORSE than chance.**
   `two_sided` Brier is 0.375–0.435 across configs vs the
   no-information prior of 0.250; `one_sided` is 0.289–0.337. A
   consumer that filtered down to staked envelopes — the intuitively
   "high-signal" ones — would do strictly worse than ignoring the
   headline entirely.

3. **The failure mode is systematic bullish overconfidence.** Mean
   predicted YES probability on staked envelopes is 0.63–0.81, while
   actual yes_rates are 0.00–0.54. In the largest config the
   two_sided markets resolved YES only 11% of the time against a mean
   prediction of 65%. Not noise around the truth — directionally
   inverted.

4. **The `no_stakes` Laplace prior is the best performer.** Brier =
   0.250 exactly, by construction (`yes_probability = 0.5`). The
   "uninformative" default beats every staked category.

5. **The bd-xc2 `confidence_source` field is validated, but with a
   sign flip.** The field correctly separates sentiment from prior —
   but the right consumer policy is not "filter to two_sided", it is
   "treat staked envelopes as a counter-indicator". Betting against
   the two_sided headline (yes_rate 11% at mean prediction 65%)
   would have beaten chance in every config.

## Why bullish?

The LLM agents read the auto-generated questions ("Will welfare exceed
X by epoch N?") and pattern-match on "the agents are working hard, so
the metric will go up." But the market generator sets thresholds at
+20% of current value — an aggressive bet over a 5-epoch horizon in a
world whose build-dominates attractor and tax/audit extraction pull
metrics down from peak. The question stream is biased toward upside
that doesn't materialize, and the persona swarm bites every time.

## Implications

- **The headline is not decorative — it is actively misleading.** A
  Polymarket bot consuming these envelopes at face value would lose
  money on exactly the markets that look most informative.
- **`confidence_source` is essential**, both to flag the sentiment
  cases and to identify which envelopes to invert.
- **The fix is upstream of the consumer:** calibrate the question
  generator (thresholds from recent metric trajectories instead of a
  flat +20%), or add a calibrated forecaster persona, or aggregate
  stakes through something better than pool ratio.

## Sibling questions worth filing

- **Stake-forcing persona arm** (ALWAYS_STAKE personas) — does forcing
  both sides to stake every tick change the calibration, or just the
  volume? (Picked up as bd-1nq.)
- **Calibrated question generator** — set thresholds from the
  median/p25/p75 of recent metric trajectories; expect the bullish
  bias to shrink.
- **Higher-temperature persona arm** — try temp=0.8 to see how stake
  frequency and calibration move.
- **External-staker harness** — drive stakes from a separate process
  via `POST /api/gtb/<sim>/stake`; tests inbound stakes from
  non-worker actors.

## How to reproduce

```bash
cd backend
# Committed config: 8 seeds × 6 epochs × 6 steps with flash-lite
uv run python -m scripts.market_predictiveness_experiment --n-seeds 8 --epochs 6 --steps 6

# Fast default: 15 seeds × 8 epochs × 3 steps
uv run python -m scripts.market_predictiveness_experiment

# Different model via env override:
LLM_MODEL_NAME=gemini-2.5-flash uv run python -m scripts.market_predictiveness_experiment --n-seeds 3 --epochs 6 --steps 3
```

Requires `LLM_API_KEY` / `LLM_BASE_URL` pointed at an OpenAI-compatible
endpoint (the runs above used Gemini's). Raw per-pair envelopes land in
`runs/market_predictiveness/pairs.json`; summary in `summary.json`.
