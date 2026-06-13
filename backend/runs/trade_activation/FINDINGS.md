# Trade activation: can the wood/stone market actually clear?

*Closes bd-4jr (`Trade activation experiment`).*

## Setup

Three arms × 20 seeds × 25 epochs × 10 steps each:

1. **baseline** — default scenario, default agent lineup (14 workers,
   honest + gaming + evasive + collusive). The PR #1 smokes already
   established that rule-based agents never use the market; this arm
   re-establishes the floor.
2. **hetero_skill** — replace the lineup with 12 honest workers split
   into three skill cohorts (`skill_gather` ∈ {0.5, 1.0, 1.5}, 4 each).
   Should create gains-from-specialization that a functioning market
   would route as the bad gatherers buy from the good gatherers.
3. **bot_makers** — default lineup PLUS 2 new `MakerWorkerPolicy`
   agents (vendored into `agents.py` for this experiment) that post
   passive limit orders — buy at 1 − 0.2 mid when inventory is low,
   sell at 1 + 0.2 mid when inventory is above target. Provides
   passive liquidity that other agents can cross.

## Results

| arm | orders_placed / actions | trades_executed / actions | final welfare | final tax |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 31.01 | 151.07 |
| hetero_skill | 0.0000 | 0.0000 | 32.66 | 31.33 |
| **bot_makers** | **0.1338** | **0.0000** | 26.86 | 150.51 |

## What the data says

1. **Trade does not activate under any of the three arms.** Executed
   trade rate is 0.0000 in all 60 runs. The market is structurally
   dead, not just empirically underused.

2. **Heterogeneous skill alone is not enough.** Even when ~33% of
   workers are bad gatherers, none of them try to buy resources from
   the high-skill ones. The rule-based `HonestWorkerPolicy` has no
   trade branch at all (we already confirmed this with a `grep TRADE_`
   over `agents.py` — zero hits in the upstream policies).

3. **Bot makers post 13% of all actions as orders, but nothing
   crosses them.** The maker bots ARE trying to trade — they emit
   `order_placed` events at a healthy clip — but no other worker
   reads the order book to fill an order. So the order book grows
   unboundedly with one-sided liquidity that never executes.

4. **Adding bot makers REDUCES welfare** (31.01 → 26.86). The 2
   makers occupy worker slots that would otherwise gather + build, so
   total production drops by ~13%. The market layer is a net-cost
   feature under the current agent population.

5. **Tax revenue collapses in the hetero_skill arm** (151 → 31). With
   skill heterogeneity, total gross income drops by ~5× — most
   workers can't earn enough to be taxable. This is a striking side
   finding: the GTB world's tax system implicitly assumes a roughly
   homogeneous agent productivity distribution. Heterogeneity breaks
   it.

## Why doesn't trade activate?

There are three barriers, only one of which is a policy bug; the
other two are structural to the env design:

- **Rule-based policies don't read the order book.** The honest /
  gaming / evasive / collusive policies decide actions from their own
  inventory + energy only. None of them queries `obs["market_orders"]`
  or similar. **A `MarketAwareHonestPolicy` is the cheapest fix and
  is filed as a sibling issue below.**
- **Building dominates as a strategy** (bd-vel). If holding wood and
  stone is worth more as house construction material than as tradeable
  resources, no agent wants to sell. The price floor under wood-as-
  raw-material is effectively the build break-even, not zero.
- **One-sided liquidity is not informative.** Even with 13% order rate
  in the bot_makers arm, only buy orders fire (the makers only post
  sells when inventory > target=5, and the gather rate doesn't get
  them there). So the bot makers contribute one-sided pressure that
  no informed counterparty exists to balance.

## Implications

- **bd-vel's "build dominates" finding is upstream of this one.**
  Until building stops dominating (which bd-yy1 showed is hard within
  the current env), no rational trade activates.
- **The market layer is essentially decorative in the default
  scenario.** It exists, it accepts orders, it could clear — but no
  agent has both an inventory imbalance and a policy that responds to
  prices.
- **LLM-driven worker populations might activate it.** The phase-3
  `LLMWorkerPolicy` already sees the open-markets list in its obs;
  the obvious next experiment is whether an LLM population spontaneously
  starts crossing orders if seeded with maker bots. Cost: real, but
  bounded (we'd need ~30 LLM ticks × 4 seeds = ~50 minutes at Gemini-
  flash latency).

## Sibling questions worth filing

- **MarketAwareHonestPolicy.** A rule-based policy that buys
  resources from the cheapest sell order when its inventory is below
  build threshold and coin is plentiful. Cheapest fix. Would let bd-4jr
  re-run with executed_rate > 0.
- **LLM × bot_makers arm.** Same as the bot_makers arm above but with
  4 `BALANCED_LLM_LINEUP` agents replacing 4 of the rule-based
  workers. Hypothesis: LLM speculators cross the maker spread when
  they have spare coin from house rent.
- **Build cost above gather rate.** Make wood expensive enough
  (per bd-yy1) that gathering alone can't fund a build. Forces
  workers to consider trade. Risks tanking welfare (per bd-yy1
  results), so the win condition is narrow.

## How to reproduce

```bash
cd backend
uv run python -m scripts.trade_activation_experiment --n-seeds 20 --epochs 25 --steps 10
# Re-aggregate from existing runs without re-running the sweeps:
uv run python -m scripts.trade_activation_experiment --skip-sweep
```

Per-arm sweeps land under `runs/trade_activation/{baseline,
hetero_skill, bot_makers}/`. The aggregated comparison is in
`trade_rate_by_arm.csv`.
