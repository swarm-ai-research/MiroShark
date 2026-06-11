# LLM × bot_makers arm

*Closes bd-hov (`LLM × bot_makers arm`). Was blocked on bd-3wh and
bd-2e2's env-extension fixes that also exposed `policy: "maker"` to
the in-process GTBWorldService factory.*

## Setup

- Two arms × 5 seeds × 6 epochs × 4 steps with `gemini-2.5-flash-lite`:
  1. **llm_only** — BALANCED_LLM_LINEUP (4 LLM personas).
  2. **llm_plus_makers** — BALANCED_LLM_LINEUP + 2 MakerWorkerPolicy.

Drives `GTBWorldService` directly, no Flask boot. Took ~3 minutes wall
clock total.

## Required env-factory addition

The in-process `GTBWorldService._make_policy` factory didn't previously
know about `policy: "maker"`, `policy: "market_aware"`, or
`policy: "tax_aware"` — only `runner._create_policy` did. Added those
three branches to keep the two factories in sync. No behaviour change
for existing scenarios.

## Results

| arm | total actions (mean) | orders placed (mean) | executed (mean) | executed_rate | welfare |
|---|---:|---:|---:|---:|---:|
| llm_only | 144 | 17.2 | 0 | 0.0000 | 0.06 |
| llm_plus_makers | 144 | 63.8 | 0 | 0.0000 | -0.11 |

## What the data says

1. **LLM agents DO place orders.** 17.2 per seed (~3-4 per worker
   across 24 ticks). This is a meaningful difference from the
   bd-4jr baseline where rule-based HonestWorkerPolicy never
   trades — the LLM agents engage with the market mechanism on
   their own.

2. **But still 0 executions in either arm.** Adding bot makers
   pushes orders from 17.2 → 63.8 (the 2 makers contribute ~46 of
   those) without producing a single crossing. The single-tick
   order book constraint (env clears every step, see bd-8dj
   discovery) is the binding constraint, not policy choice.

3. **LLM-with-makers has LOWER welfare** (-0.11 vs +0.06) because
   the makers occupy 2 worker slots without contributing to gather
   or build. The bd-4jr finding ("makers reduce welfare without
   activating trade") carries through to LLM populations.

4. **Both welfares are near zero** — far below bd-an2's baseline of
   ~70. The reason: small N (4-6 workers vs 14 default), short
   horizon (6 epochs × 4 steps = 24 ticks vs 30 × 15 = 450). LLM
   workers gather at a rule-based pace, so they need similar runway
   to accumulate. Don't compare to bd-an2 absolute numbers.

5. **The bd-8dj env bug is the binding constraint.** Until the order
   book persists across steps, no policy combination can produce
   executions. This is a vendored env change worth tackling as a
   sibling.

## Implications

- **bd-4jr's "trade is structurally dead" finding extends to LLM
  populations.** It's not a rule-based-policy limitation; it's an
  env design choice (single-tick order book).
- **LLM agents engage with the market mechanism by default.** Worth
  noting because the bd-mit market-stake engagement was much more
  reluctant (required ALWAYS_STAKE personas). Resource-market
  trading via TRADE_BUY/TRADE_SELL is apparently easier to elicit
  than prediction-market staking via the optional market_stake
  field.
- **The makers' contribution is pure overhead.** 46 extra orders
  per seed (the makers' contribution) produces 0 executions and
  reduces welfare. Suggests the MakerWorkerPolicy as written is
  net-negative for any population.

## Sibling questions worth filing

- **Persistent order book** (env change). Stop clearing buy/sell
  orders each step; let them age N ticks or until matched. This
  is the binding constraint behind bd-4jr, bd-8dj, and bd-hov —
  fixing it should reactivate trade across all three.
- **Maker policy that gathers first.** Modify MakerWorkerPolicy to
  spend M ticks gathering before posting asks. Pairs with the
  persistent-book fix above; together they'd give two-sided
  liquidity.
- **Larger-N LLM run.** Re-run bd-hov with 10+ LLM workers to see
  if order rate scales linearly and if execution emerges
  stochastically.

## How to reproduce

```bash
cd backend
uv run python -m scripts.llm_x_makers_experiment --n-seeds 5 --epochs 6 --steps 4
```

Per-seed counts land in `runs/llm_x_makers/results.json`. Total wall
clock ~3 minutes (2 arms × 5 seeds × 24 LLM-batch calls per seed,
all on flash-lite).
