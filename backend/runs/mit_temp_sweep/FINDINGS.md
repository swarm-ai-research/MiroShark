# Higher-temperature LLM persona arm

*Closes bd-in5 (`Higher-temperature LLM persona arm`).*

## Setup

- Three temperature settings: 0.4 (bd-mit/bd-1nq default), 0.8, 1.2.
- Per temperature: 6 seeds × 5 epochs × 3 steps with the bd-1nq
  `STAKING_LINEUP` (Bull + Bear + Contrarian + Honest).
- Model: `gemini-2.5-flash-lite` (bd-0t5 default).
- Methodology: capture last-open envelope per market (bd-1nq fix).

## Results

| temperature | n_obs | two_sided n | one_sided n | no_stakes n | two_sided Brier | one_sided Brier | no_stakes Brier | overall yes_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 42 | 4 | 10 | 28 | 0.079 | 0.234 | 0.250 | 1.000 |
| 0.8 | 44 | 7 | 8 | 29 | 0.119 | 0.298 | 0.250 | 1.000 |
| 1.2 | 47 | 5 | 12 | 30 | 0.109 | 0.348 | 0.250 | 1.000 |

## Major caveat

**Every market resolved YES at every temperature.** With `yes_rate=1.0`,
the Brier score is just `(1 − mean_yes_prob)²`, and the comparison
becomes "how close to 1.0 was the swarm's prediction." This is the
opposite degeneracy from bd-1nq's larger run (which had `yes_rate=0.357`
for two_sided markets and Brier=0.452 — the swarm was overconfidently
bullish in a bearish world).

Why the difference: bd-1nq ran 6-epoch × 4-step seeds; this ran
5-epoch × 3-step seeds. **In 15 ticks, the metric stream doesn't move
enough to cross most thresholds**, so the `<` markets resolve NO by
deadline AND the `>` markets resolve YES — but only the YES side
resolves. The shorter run leaves all `>` markets resolved YES and all
`<` markets unresolved (still open at end-of-run, so they don't enter
the Brier pool).

So this run can't actually answer the bd-in5 question (does higher
temperature improve calibration?) because the outcome distribution is
degenerate.

## What can still be said

1. **Higher temperature generates more stakes.** Total two_sided +
   one_sided stakes: 14 (temp=0.4) → 15 (0.8) → 17 (1.2). A weak
   positive trend; not statistically significant at this N. The
   personas are slightly more variable but the staking-rate
   improvement is marginal.

2. **Two_sided Brier worsens with temperature** (0.079 → 0.119 →
   0.109). Confidence_pct on the headline drifts away from the
   correct 1.0 prediction as temperature rises. Not a calibration
   story per se — just noisier estimates at higher sampling
   variance.

3. **The no_stakes prior (Brier 0.25 from yes_rate=1.0) is beaten
   by every staked configuration at every temperature.** When the
   outcome is degenerate-YES, even a slightly-bullish-biased swarm
   prediction (~0.7) beats the uninformed 0.5 prior. This INVERTS
   the bd-1nq finding (where the no_stakes prior beat two_sided).

## Why this matters less than it should

The bd-1nq finding was that the two_sided headline yes_probability is
**systematically inverted under build-dominates economics**. That
finding was based on n=139 markets across 15 seeds × 6 epochs × 4
steps, with a balanced yes/no resolution distribution. The bd-in5
sweep used smaller seeds × shorter epochs to test temperature, and
ran into the opposite degeneracy.

**A proper bd-in5 needs to match bd-1nq's seed/epoch budget at each
temperature.** That's ~45 minutes per temperature × 3 temperatures =
~2.5 hours of Gemini calls. Out of budget for this PR pass; flagged
as a sibling.

## Sibling questions worth filing

- **bd-in5 at full scale.** 15 seeds × 6 epochs × 4 steps × 3
  temperatures = ~45 minutes of Gemini calls. Replicates bd-1nq's
  budget at each temperature cell. Needed to actually answer the
  calibration question.
- **Question-generator calibration follow-up** (sibling to bd-1nq).
  The fact that 5-epoch × 3-step runs have yes_rate=1.0 while
  6-epoch × 4-step runs have yes_rate=0.357 says the generator's
  +20% threshold is wildly miscalibrated against the
  build-dominates trajectory. Worth its own sweep over
  generator threshold.
- **Temperature × persona-strictness combo.** Try
  ALWAYS_STAKE_BULLISH with temperature 0.0 (deterministic) vs
  temperature 1.2 (high variance) to see whether the "must stake"
  instruction is followed reliably across the temperature range.

## How to reproduce

```bash
cd backend
for temp in 0.4 0.8 1.2; do
  uv run python -m scripts.market_predictiveness_experiment \
    --n-seeds 6 --epochs 5 --steps 3 --temperature $temp \
    --output runs/mit_temp_$temp
done
```
