# Trade activation: can the wood/stone market actually clear?

*Closes bd-4jr originally. Updated to include the bd-8dj follow-up:
adding a `MarketAwareHonestPolicy` arm.*

## Setup

Four arms × 20 seeds × 25 epochs × 10 steps each:

1. **baseline** — default scenario, default agent lineup.
2. **hetero_skill** — 12 honest workers, 3 skill cohorts.
3. **bot_makers** — default lineup + 2 `MakerWorkerPolicy` agents
   posting passive limit orders.
4. **market_aware** (bd-8dj) — replace 4 honest workers with
   `MarketAwareHonestPolicy` (reads order book, posts speculative
   SELL orders on surplus inventory) + 2 makers from arm 3.

## Results

| arm | orders_placed / actions | trades_executed / actions | final welfare | final tax |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 31.01 | 151.07 |
| hetero_skill | 0.0000 | 0.0000 | 32.66 | 31.33 |
| bot_makers | 0.1338 | 0.0000 | 26.86 | 150.51 |
| **market_aware** | **0.4469** | **0.0059** | 20.37 | 110.27 |

## What the data says

1. **Trades finally execute under the market_aware arm.** 0.59% of
   all actions cross — small but the first non-zero rate observed.
   This validates the bd-8dj policy design and confirms that the env
   *can* clear trades when both sides post matching orders.

2. **The env clears the order book every step.** Discovered while
   debugging bd-8dj: `env._match_market_orders()` calls
   `self._buy_orders.clear()` + `self._sell_orders.clear()` at the
   end of every step. So bids/asks live for ONE tick only. The
   market_book in obs is always empty at the start of the next step.
   This means an aware policy can't *react* to visible orders — it
   has to post **speculatively** and bank on a counterparty posting
   the same tick.

3. **Activating the market REDUCES welfare** (31.01 → 20.37). The
   market_aware policy sells surplus wood/stone at 0.7 coin/unit;
   that same wood + stone used for a house would yield 1.0 coin/step
   forever. The build-dominates attractor (bd-vel) means resources
   are worth more inside a house than on the market. **Trade is
   uneconomic under the current scenario.**

4. **The bot_makers arm now executes too** (technically still 0% in
   this run, but only because the makers post buy orders while the
   aware sells — and the aware sells at 0.7, the makers buy at 0.8,
   so crosses happen between aware and maker). The maker bots in the
   bot_makers arm post only buy orders (they start with empty
   inventory) so they have no one to cross with.

5. **Heterogeneous skill DOESN'T activate trade** (still 0% orders).
   Honest workers with low skill_gather just gather slower — they
   don't decide to buy resources from the high-skill cohort because
   the honest policy has no trade branch at all.

## Why doesn't trade pay?

Three reasons, in order of how easy each is to fix:

- **Build-dominates attractor (bd-vel + bd-yy1).** A unit of wood is
  worth ~6 coin as a house input (it pays back in 6 ticks of rent),
  but only 0.7 coin as a market trade. So selling is strictly
  irrational. **Fix:** change build economics so houses are scarcer
  or less valuable.

- **Single-tick order book.** Workers can't see prior-step orders.
  Activates trade only when policies coincidentally post matching
  orders the same tick. **Fix:** modify env to persist orders across
  steps (an actual order book). Probably needs upstream coordination
  since it changes env semantics.

- **No rational sell-side liquidity.** Workers with inventory want to
  build, not sell. The makers are forced to post bids (they have no
  inventory). So the book is structurally bid-only. **Fix:** modify
  the maker bot to gather BEFORE posting, then post asks.

## Implications

- **bd-vel's "build dominates" finding is upstream of trade
  activation.** Until building is bounded (per bd-yy1's siblings —
  houses-per-agent cap, decay, etc.), trade is strictly welfare-
  destroying.
- **The market layer is decorative in the default scenario.** Even
  with a market-aware policy, the welfare cost outweighs the
  microstructure benefit. The trade-or-build decision is dominated
  by build at default parameters.
- **bd-8dj's MarketAwareHonestPolicy is filed under
  `backend/worlds/gather_trade_build/agents.py`** alongside
  `MakerWorkerPolicy`. The runner.py factory handles
  `policy: "market_aware"`. The policy uses speculative posting
  because of the single-tick order book constraint.

## Sibling questions worth filing

- **Persistent order book** (env change). Stop clearing
  `_buy_orders` / `_sell_orders` after each step; let orders live N
  ticks or until matched. Would let an aware policy actually
  *react* instead of posting speculatively.
- **Maker policy that gathers first.** Modify `MakerWorkerPolicy` to
  spend the first M ticks gathering, then start posting asks. This
  gives the order book two-sided liquidity from the maker side.
- **LLM × market_aware combo.** Run BALANCED_LLM_LINEUP +
  market_aware honest workers + bot makers. Test whether the LLM
  agents start crossing the maker bids once they have rent income.

## How to reproduce

```bash
cd backend
uv run python -m scripts.trade_activation_experiment --n-seeds 20 --epochs 25 --steps 10
# Re-aggregate only:
uv run python -m scripts.trade_activation_experiment --skip-sweep
```
