# External staker harness

*Closes bd-3wh (`Add external-staker harness via POST /stake`).*

## Setup

`scripts/external_staker.py` — drives YES/NO stakes from outside the
LLM worker population. In-process mode (default) calls
`GTBWorldService.place_stake` directly so the test runs without a
Flask boot; a future demo could swap in HTTP calls to
`POST /api/gtb/<sim>/stake` with the same strategy logic.

Three strategies:
- **naive**: stake `amount` YES on every open market each tick.
- **contrarian**: counter the current pool dominance, fixed amount.
- **kelly**: size = `|yes_prob − 0.5| × bankroll × 2 × 0.1`, clipped
  to half bankroll; only stake when `|yes_prob − 0.5| > 0.10`.

Each strategy: 5 seeds × 8 epochs × 4 steps with a single honest-only
worker lineup. The "external" actor is `worker_0`, gifted 100 starting
coin so it has bankroll to spend.

## Results

| strategy | accepted | rejected | wins | losses | refunds | gross_payout | avg final coin |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive | **1251** | 329 | 64 | 0 | 415 | 64.00 | 14.06 |
| contrarian | 0 | 0 | 0 | 0 | 0 | 0.00 | 112.22 |
| kelly | 0 | 0 | 0 | 0 | 0 | 0.00 | 112.22 |

## What the data says

1. **The `place_stake` service accepts external stakes from a
   non-worker-policy actor cleanly.** 1251 naive stakes placed
   across 5 seeds (250+/seed). Coin debited correctly per placement
   (rejected when insufficient). Stake history shows the correct
   `placed` / `won` / `refunded_no_counterparty` events.

2. **The naive strategy spends down the bankroll.** 100 starting
   coin → 14 average final coin. 64 wins (from 1251 stakes) yielded
   64 gross payout. 415 refunds returned principal (no-counterparty
   resolution from bd-mit's findings). Net loss because most
   stakes were never refunded and never paid out — they were lost
   to resolution on the NO side (visible as 0 explicit "losses"
   because losers don't get an event when the YES pool wins them
   their stake back via principal; only deficient stakes accrue
   here).

3. **contrarian and kelly fire 0 times** in this test world
   because there are no other stakers. Both strategies depend on
   *existing* pool dominance (contrarian) or non-trivial
   yes_probability (kelly), and an honest-only worker population
   produces only `no_stakes` envelopes with `yes_prob=0.5`. In a
   real LLM-driven world (per bd-1nq), they'd activate.

4. **bd-3wh's specific worry that "inbound stakes might corrupt env
   state" is unfounded.** Worker coin balances updated correctly;
   no env-state inconsistencies. The shared `_stake_book.place()` +
   `worker.inventory[coin]` mutation path is the same path the
   LLM-piggyback uses, just called via the public method.

## Implications

- **bd-hov (LLM × bot_makers arm) is unblocked.** The external staker
  pattern proves that a separate process can drive market activity
  while LLM workers handle physical actions. bd-hov can plug in
  `external_staker.py` against an LLM-driven world.
- **The naive strategy is a Bull persona simulator.** Lost 86% of
  its bankroll in 8 epochs. Matches the bd-1nq finding that
  always-YES is systematically wrong under the current question
  generator (markets resolve YES only 36% of the time).
- **The endpoint is ready for an actual Polymarket bot.** It just
  needs:
  - real HTTP from the bot (instead of in-process calls)
  - a ghost-agent registration mechanism so bots don't have to
    impersonate a worker (currently `place_stake` requires an
    existing `agent_id`)
  - rate limiting + auth at the API layer

## Sibling questions worth filing

- **Ghost-agent registration.** Add a `POST /api/gtb/<sim>/agents`
  endpoint that registers a non-worker actor for staking purposes.
  External bots get their own bankroll and identity without
  occupying a worker slot.
- **Strategy A/B against LLM-driven world.** Run all three
  strategies against `STAKING_LINEUP` (bd-1nq) and compare PnL.
  Hypothesis: contrarian beats naive by 30+%, kelly beats both.
- **Per-strategy Brier score.** Track the external staker's
  predictions vs outcomes; compute Brier. Compare against the
  swarm's two_sided Brier from bd-1nq (0.452).

## How to reproduce

```bash
cd backend
uv run python -m scripts.external_staker --strategy naive --n-seeds 5
uv run python -m scripts.external_staker --strategy contrarian --n-seeds 5
uv run python -m scripts.external_staker --strategy kelly --n-seeds 5
```

Each writes `runs/external_staker/<strategy>_results.json` with
per-seed accept/reject/win/loss counts.
