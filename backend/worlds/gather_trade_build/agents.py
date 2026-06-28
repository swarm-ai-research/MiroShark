"""GTB worker agent types for the AI Economist scenario.

Provides baseline and adversarial worker policies that operate in the
GTB gridworld. Each agent type implements a decide() method that returns
a GTBAction given an observation dict.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from worlds.gather_trade_build.entities import (
    Direction,
    GTBActionType,
    ResourceType,
)
from worlds.gather_trade_build.env import GTBAction


class GTBWorkerPolicy(ABC):
    """Base class for GTB worker policies."""

    def __init__(self, agent_id: str, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self._rng = random.Random(seed)

    @abstractmethod
    def decide(self, obs: dict) -> GTBAction:
        """Choose an action given the current observation.

        Args:
            obs: Observation dict from GTBEnvironment.obs().

        Returns:
            GTBAction to execute.
        """

    def _random_direction(self) -> Direction:
        return self._rng.choice(list(Direction))

    def _find_resource_direction(
        self, obs: dict, resource: Optional[str] = None
    ) -> Optional[Direction]:
        """Find direction toward nearest visible resource.

        ``resource`` (a ResourceType value like "compute") restricts the
        search to that commodity; None matches any resource. On the base
        class so any policy (e.g. a COMPUTE futures hedger) can navigate to
        the tile it produces from.
        """
        pos = obs.get("position", (0, 0))
        r, c = pos[0], pos[1]
        best_dist = float("inf")
        best_dir = None

        for cell in obs.get("visible_cells", []):
            if "resource" not in cell or cell.get("amount", 0) <= 0:
                continue
            if resource is not None and cell.get("resource") != resource:
                continue
            cr, cc = cell["pos"]
            dist = abs(cr - r) + abs(cc - c)
            if 0 < dist < best_dist:
                best_dist = dist
                dr = cr - r
                dc = cc - c
                if abs(dr) >= abs(dc):
                    best_dir = Direction.DOWN if dr > 0 else Direction.UP
                else:
                    best_dir = Direction.RIGHT if dc > 0 else Direction.LEFT

        return best_dir


class HonestWorkerPolicy(GTBWorkerPolicy):
    """Honest worker: gathers resources, builds houses, trades.

    Never misreports or shifts income. Follows a simple gather-build cycle.
    """

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)
        houses = obs.get("houses_built", 0)

        # If we have enough resources, build
        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0 and houses < 10:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        # Try to gather from current cell
        visible = obs.get("visible_cells", [])
        pos = obs.get("position", (0, 0))

        # Check if current cell has resources
        for cell in visible:
            if cell.get("pos") == tuple(pos) and "resource" in cell:
                if cell.get("amount", 0) > 0 and energy >= 1.0:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )

        # Move towards nearest resource
        best_dir = self._find_resource_direction(obs)
        if best_dir and energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=best_dir,
            )

        # Fallback: random move
        if energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class RationalWorkerPolicy(HonestWorkerPolicy):
    """Labor-responsive worker: works only while one more unit of work is
    worth it under isoelastic preferences.

    Each step it compares the post-tax marginal utility of one unit of
    income against the labor disutility of earning it:

        (1 - marginal_rate) * unit_income * coin^(-eta)  >=  labor_coeff

    and takes leisure (NOOP) otherwise. This supplies the central AI
    Economist mechanism the other scripted baselines lack: higher
    marginal tax rates reduce labor supply (substitution effect) and
    higher accumulated wealth does too (income effect via CRRA).
    """

    def __init__(self, agent_id: str, eta: float = 0.35,
                 labor_coeff: float = 0.15,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self._eta = eta
        self._labor_coeff = labor_coeff

    def decide(self, obs: dict) -> GTBAction:
        if not self._work_is_worth_it(obs):
            return GTBAction(agent_id=self.agent_id,
                             action_type=GTBActionType.NOOP)
        return super().decide(obs)

    def _work_is_worth_it(self, obs: dict) -> bool:
        from worlds.gather_trade_build.reward import crra_marginal

        coin = obs.get("inventory", {}).get(ResourceType.COIN.value, 0.0)
        gross = obs.get("gross_income", 0.0)
        brackets = obs.get("tax_schedule", {}).get("brackets", [])
        marginal_rate = 0.0
        for b in brackets:
            if gross >= b.get("threshold", 0.0):
                marginal_rate = b.get("rate", 0.0)

        unit_income = 1.0  # one gather's worth of income
        net_marginal_utility = (
            (1.0 - marginal_rate) * unit_income * crra_marginal(coin, self._eta)
        )
        return net_marginal_utility >= self._labor_coeff


class GamingWorkerPolicy(GTBWorkerPolicy):
    """Strategic worker that uses income shifting to reduce tax burden.

    Bunches income near bracket boundaries by deferring excess income.
    """

    def __init__(self, agent_id: str, shift_fraction: float = 0.2,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self._shift_fraction = shift_fraction

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        gross = obs.get("gross_income", 0.0)
        schedule = obs.get("tax_schedule", {})
        brackets = schedule.get("brackets", [])

        # Check if income shifting would help (near a bracket boundary)
        if gross > 0 and brackets and energy >= 0.5:
            for b in brackets:
                thr = b.get("threshold", 0)
                if thr > 0 and gross > thr and gross < thr * 1.3:
                    # Near a bracket: shift income down
                    shift = min(gross - thr + 0.5, gross * self._shift_fraction)
                    if shift > 0.1:
                        return GTBAction(
                            agent_id=self.agent_id,
                            action_type=GTBActionType.SHIFT_INCOME,
                            shift_amount=shift,
                        )

        # Otherwise behave like honest worker
        return self._fallback_action(obs)

    def _fallback_action(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)

        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        if energy >= 1.0:
            # Check current cell for resources
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class EvasiveWorkerPolicy(GTBWorkerPolicy):
    """Worker that misreports income to evade taxes.

    Under-reports a fraction of income each epoch, risking audit penalties.
    """

    def __init__(self, agent_id: str, underreport_fraction: float = 0.3,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self._underreport_fraction = underreport_fraction
        self._reported_this_epoch = False

    def decide(self, obs: dict) -> GTBAction:
        gross = obs.get("gross_income", 0.0)
        step = obs.get("step", 0)

        # Misreport once per epoch (mid-epoch)
        if gross > 1.0 and not self._reported_this_epoch and step > 3:
            self._reported_this_epoch = True
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MISREPORT,
                underreport_fraction=self._underreport_fraction,
            )

        # Otherwise gather and build
        return self._gather_build(obs)

    def reset_epoch(self) -> None:
        self._reported_this_epoch = False

    def _gather_build(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)

        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )
        if energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )
        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class ZITraderPolicy(GTBWorkerPolicy):
    """Zero-intelligence-constrained trader (Gode & Sunder 1993).

    Posts random-priced limit orders constrained to be individually
    rational around a private resource valuation: bids are drawn at or
    below the private value, asks at or above it. Heterogeneous private
    values across a trader population make bids cross asks, providing
    organic two-sided market flow and price discovery toward the value
    band — Gode & Sunder showed ZI-C populations reach near-efficient
    prices without any strategic intelligence.

    When short of inventory the trader gathers like an honest worker so
    it always has something to sell.
    """

    def __init__(self, agent_id: str, value_estimate: float = 2.0,
                 value_jitter: float = 0.5, order_qty: float = 1.0,
                 min_coin_reserve: float = 2.0,
                 resources: Optional[tuple] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        # Private value: heterogeneous across traders so books cross
        self._value = value_estimate * (
            1.0 + self._rng.uniform(-value_jitter, value_jitter)
        )
        self._order_qty = order_qty
        self._min_coin_reserve = min_coin_reserve
        # Commodities this trader makes two-sided flow in (bd ja2). Defaults
        # to wood/stone; set e.g. (COMPUTE,) for an H100 spot desk.
        self._resources = resources or (ResourceType.WOOD, ResourceType.STONE)

    @property
    def private_value(self) -> float:
        return self._value

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        coin = inventory.get(ResourceType.COIN.value, 0.0)

        if energy >= 0.5:
            # Sell surplus above the private value
            for rtype in self._resources:
                held = inventory.get(rtype.value, 0.0)
                if held >= self._order_qty:
                    ask = self._value * (1.0 + self._rng.random())
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.TRADE_SELL,
                        resource_type=rtype,
                        quantity=min(self._order_qty, held),
                        price=ask,
                    )
            # Buy below the private value while keeping a coin reserve
            affordable = coin - self._min_coin_reserve
            if affordable >= self._value * self._order_qty:
                rtype = self._rng.choice(list(self._resources))
                bid = self._value * self._rng.uniform(0.5, 1.0)
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_BUY,
                    resource_type=rtype,
                    quantity=self._order_qty,
                    price=bid,
                )

        # Restock: gather like an honest worker
        if energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    if cell.get("amount", 0) > 0:
                        return GTBAction(
                            agent_id=self.agent_id,
                            action_type=GTBActionType.GATHER,
                        )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class CartelWorkerPolicy(GTBWorkerPolicy):
    """Price-fixing cartel member — collusion with a real economic channel.

    Every member of the coalition quotes the same elevated ask price
    (the cartel agreement) and never undercuts. When the cartel controls
    enough of the sell-side liquidity, buyers must pay the cartel price,
    raising members' sale income above the competitive level. Unlike
    CollusiveWorkerPolicy (which merely synchronizes action strings and
    gains nothing), this cartel profits, and is detectable from public
    market records via the price-fixing signature: near-identical asks
    across members. Requires a persistent order book
    (market.order_ttl_steps > 0) and buy-side demand to bite.
    """

    def __init__(self, agent_id: str, coalition_id: str,
                 cartel_price: float = 4.0, order_qty: float = 1.0,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self.coalition_id = coalition_id
        self._cartel_price = cartel_price
        self._order_qty = order_qty

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})

        # Quote the cartel price on any sellable surplus
        if energy >= 0.5:
            for rtype in (ResourceType.WOOD, ResourceType.STONE):
                held = inventory.get(rtype.value, 0.0)
                if held >= self._order_qty:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.TRADE_SELL,
                        resource_type=rtype,
                        quantity=min(self._order_qty, held),
                        price=self._cartel_price,
                    )

        # Restock like an honest gatherer
        if energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    if cell.get("amount", 0) > 0:
                        return GTBAction(
                            agent_id=self.agent_id,
                            action_type=GTBActionType.GATHER,
                        )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class CollusiveWorkerPolicy(GTBWorkerPolicy):
    """Worker that coordinates with coalition members.

    Coalition members synchronize gathering locations, share trade profits,
    and may collectively withhold effort.
    """

    def __init__(self, agent_id: str, coalition_id: str,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self.coalition_id = coalition_id

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        step = obs.get("step", 0)

        # Collusive pattern: all coalition members do the same action
        # at the same step (synchronized behavior)
        action_cycle = [
            GTBActionType.GATHER,
            GTBActionType.GATHER,
            GTBActionType.MOVE,
            GTBActionType.GATHER,
            GTBActionType.GATHER,
        ]
        idx = step % len(action_cycle)
        chosen = action_cycle[idx]

        if chosen == GTBActionType.GATHER and energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            # No resource here, move instead
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        if chosen == GTBActionType.MOVE and energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        # Build if possible
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)
        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class MakerWorkerPolicy(GTBWorkerPolicy):
    """Passive market-maker: posts limit sell orders on hoarded resources,
    crosses the spread to buy if cheap. Added for bd-4jr to test whether
    seeding the market with passive liquidity activates LLM/rule-based
    counterparty trading. Does not gather or build — pure microstructure
    agent.
    """

    def __init__(self, agent_id: str, seed=None, sell_markup: float = 0.2,
                 buy_discount: float = 0.2, target_inventory: float = 5.0) -> None:
        super().__init__(agent_id, seed)
        self._sell_markup = sell_markup
        self._buy_discount = buy_discount
        self._target_inventory = target_inventory

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        inv = obs.get("inventory", {})
        wood = inv.get(ResourceType.WOOD.value, 0.0)
        stone = inv.get(ResourceType.STONE.value, 0.0)
        coin = inv.get(ResourceType.COIN.value, 0.0)
        step = obs.get("step", 0)
        # Round-robin: sell wood on even steps, sell stone on odd steps,
        # buy if inventory is low and coin is plentiful.
        if step % 2 == 0:
            if wood > self._target_inventory:
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_SELL,
                    resource_type=ResourceType.WOOD,
                    quantity=1.0,
                    price=1.0 + self._sell_markup,
                )
            if coin >= 1.0:
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_BUY,
                    resource_type=ResourceType.WOOD,
                    quantity=1.0,
                    price=max(0.1, 1.0 - self._buy_discount),
                )
        else:
            if stone > self._target_inventory:
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_SELL,
                    resource_type=ResourceType.STONE,
                    quantity=1.0,
                    price=1.0 + self._sell_markup,
                )
            if coin >= 1.0:
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_BUY,
                    resource_type=ResourceType.STONE,
                    quantity=1.0,
                    price=max(0.1, 1.0 - self._buy_discount),
                )
        # Default to gathering so we have inventory to post.
        return GTBAction(
            agent_id=self.agent_id,
            action_type=GTBActionType.GATHER,
        )


class MarketAwareHonestPolicy(HonestWorkerPolicy):
    """Honest worker that ALSO reads the order book.

    Closes bd-8dj. Behaviour:
      - SELL surplus inventory (wood/stone above build_threshold + 2)
        whenever a bid exists at sell_min_price_ratio × fair_value.
        Triggers once house rent has accumulated coin and the worker
        has gathered more than it needs for its next build.
      - BUY missing resources to ANTICIPATE the next house build, even
        if the current inventory is non-zero — accept any ask up to
        buy_max_price_ratio × fair_value.
      - Otherwise fall through to honest gather/move/build.

    The buy condition triggers on "not currently ready to build AND
    has coin", not "below build threshold". The naive form never
    fires because honest workers gather their own resources first.
    """

    def __init__(self, agent_id: str, seed=None,
                 build_wood_threshold: float = 3.0,
                 build_stone_threshold: float = 3.0,
                 trade_energy_cost: float = 0.5,
                 fair_value: float = 1.0,
                 buy_max_price_ratio: float = 1.5,
                 sell_min_price_ratio: float = 0.7) -> None:
        super().__init__(agent_id, seed)
        self._build_wood = build_wood_threshold
        self._build_stone = build_stone_threshold
        self._trade_energy = trade_energy_cost
        self._fair_value = fair_value
        self._buy_max = buy_max_price_ratio
        self._sell_min = sell_min_price_ratio

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        inv = obs.get("inventory", {})
        wood = inv.get(ResourceType.WOOD.value, 0.0)
        stone = inv.get(ResourceType.STONE.value, 0.0)
        coin = inv.get(ResourceType.COIN.value, 0.0)
        energy = obs.get("energy", 0.0)
        book = obs.get("market_book", {})

        # IMPORTANT: the env clears its order books at the end of every
        # step (env.py:_match_market_orders), so the market_book in obs
        # is always empty at the start of the next step. We post
        # speculatively, banking on bot makers (or other aware workers)
        # to post matching orders the same tick.
        #
        # Priority: gather honestly first to build up surplus. Only
        # trade once we have real inventory to sell. Don't pre-emptively
        # buy with our starting coin or the policy never accumulates
        # anything.
        if energy >= self._trade_energy:
            for resource_name, holding, threshold in (
                (ResourceType.WOOD.value, wood, self._build_wood + 2.0),
                (ResourceType.STONE.value, stone, self._build_stone + 2.0),
            ):
                if holding < threshold:
                    continue
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.TRADE_SELL,
                    resource_type=ResourceType(resource_name),
                    quantity=1.0,
                    price=self._fair_value * self._sell_min,
                )
        return super().decide(obs)


class TaxAwareHonestPolicy(HonestWorkerPolicy):
    """Honest worker that adjusts effort by the marginal tax rate.

    Closes bd-2e2. Gives the planner a behavioural lever to react to.
    The default rule-based policies treat tax brackets as after-the-fact
    cash transfers; this one lets the planner shape effort decisions.

    Behaviour:
      - Compute the marginal rate at the worker's CURRENT epoch income
        (from obs["tax_schedule"] + obs["gross_income"]).
      - If marginal rate >= rate_threshold, suppress gather/build with
        probability `effort_suppression`: substitute with NOOP (or
        SHIFT_INCOME if near a bracket boundary).
      - Otherwise fall through to honest behaviour.
    """

    def __init__(self, agent_id: str, seed: Optional[int] = None,
                 rate_threshold: float = 0.30,
                 effort_suppression: float = 0.7,
                 shift_fraction: float = 0.2) -> None:
        super().__init__(agent_id, seed)
        self._rate_threshold = rate_threshold
        self._effort_suppression = effort_suppression
        self._shift_fraction = shift_fraction

    def _marginal_rate(self, gross: float, brackets: list) -> float:
        """Marginal rate at this income level."""
        if not brackets:
            return 0.0
        sorted_b = sorted(brackets, key=lambda b: b.get("threshold", 0))
        rate = float(sorted_b[0].get("rate", 0.0))
        for b in sorted_b:
            thr = float(b.get("threshold", 0))
            if gross >= thr:
                rate = float(b.get("rate", rate))
            else:
                break
        return rate

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        gross = float(obs.get("gross_income", 0.0))
        schedule = obs.get("tax_schedule", {})
        brackets = schedule.get("brackets", [])
        mr = self._marginal_rate(gross, brackets)

        if mr >= self._rate_threshold:
            # In a high-tax band — suppress effort with the given probability.
            if self._rng.random() < self._effort_suppression:
                # If near a bracket boundary, prefer SHIFT_INCOME to NOOP.
                for b in brackets:
                    thr = float(b.get("threshold", 0))
                    if thr > 0 and gross > thr and gross < thr * 1.25:
                        shift = min(gross - thr + 0.5, gross * self._shift_fraction)
                        if shift > 0.1:
                            return GTBAction(
                                agent_id=self.agent_id,
                                action_type=GTBActionType.SHIFT_INCOME,
                                shift_amount=shift,
                            )
                return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        return super().decide(obs)


class FuturesMakerPolicy(GTBWorkerPolicy):
    """Two-sided futures liquidity provider (bd-c7i).

    Each step posts one resting forward quote on a single delivery date
    (``epoch + horizon``), anchored to the current spot: a long (bid) at
    ``spot * (1 - spread)`` on even steps, a short (ask) at
    ``spot * (1 + spread)`` on odd steps. Over a few steps the book holds
    a two-sided forward quote for takers/hedgers to cross. Does not gather
    or build — a pure microstructure agent (parallel to MakerWorkerPolicy).
    """

    def __init__(self, agent_id: str, seed=None, horizon: int = 3,
                 spread: float = 0.1, qty: float = 1.0,
                 fair_value: float = 1.0,
                 resource: Optional[ResourceType] = None) -> None:
        super().__init__(agent_id, seed)
        self._horizon = horizon
        self._spread = spread
        self._qty = qty
        self._fair_value = fair_value
        # If set, quote only this commodity (e.g. COMPUTE for an H100 desk);
        # otherwise rotate the WOOD/STONE book as the bd-c7i validation does.
        self._resource = resource

    def _spot(self, obs: dict, resource: ResourceType) -> float:
        info = (obs.get("market_info") or {}).get(resource.value) or {}
        last = info.get("last_price")
        return last if last and last > 0 else self._fair_value

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        epoch = obs.get("epoch", 0)
        step = obs.get("step", 0)
        resource = self._resource or (
            ResourceType.WOOD if step % 4 < 2 else ResourceType.STONE)
        spot = self._spot(obs, resource)
        settle = epoch + self._horizon
        if step % 2 == 0:
            return GTBAction(
                agent_id=self.agent_id, action_type=GTBActionType.FUTURES_BUY,
                resource_type=resource, quantity=self._qty,
                price=spot * (1.0 - self._spread), settlement_epoch=settle,
            )
        return GTBAction(
            agent_id=self.agent_id, action_type=GTBActionType.FUTURES_SELL,
            resource_type=resource, quantity=self._qty,
            price=spot * (1.0 + self._spread), settlement_epoch=settle,
        )


class FuturesTakerPolicy(GTBWorkerPolicy):
    """Crosses resting forward quotes to generate matched contracts (bd-c7i).

    Reads the forward curve in obs; lifts the best ask (goes long) or hits
    the best bid (goes short) on whichever bucket has a quotable side,
    pricing to cross. Picks a side at random when both exist so contracts
    form on both directions. Pure microstructure (no gather/build)."""

    def __init__(self, agent_id: str, seed=None, qty: float = 1.0) -> None:
        super().__init__(agent_id, seed)
        self._qty = qty

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        curve = obs.get("futures_curve") or {}
        liftable = [c for c in curve.values()
                    if c.get("best_ask") is not None
                    or c.get("best_bid") is not None]
        if not liftable:
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        entry = self._rng.choice(liftable)
        resource = ResourceType(entry["resource"])
        settle = entry["settlement_epoch"]
        ask, bid = entry.get("best_ask"), entry.get("best_bid")
        # Choose a crossable side; prefer a random one when both exist.
        go_long = ask is not None and (bid is None or self._rng.random() < 0.5)
        if go_long:
            return GTBAction(
                agent_id=self.agent_id, action_type=GTBActionType.FUTURES_BUY,
                resource_type=resource, quantity=self._qty,
                price=ask, settlement_epoch=settle,
            )
        return GTBAction(
            agent_id=self.agent_id, action_type=GTBActionType.FUTURES_SELL,
            resource_type=resource, quantity=self._qty,
            price=bid, settlement_epoch=settle,
        )


class FuturesHedgerPolicy(GTBWorkerPolicy):
    """Spot-selling producer that optionally hedges with futures (bd-c7i).

    A producer with genuine spot-price exposure: it gathers ``WOOD`` and
    sells it on the spot market, so its revenue is ``spot * qty`` and
    varies with the (volatile) spot price. With ``hedge=True`` it also
    periodically shorts a forward at a price near spot — locking in a sale
    price, so realized revenue tends toward ``forward * qty`` regardless of
    where spot lands. With ``hedge=False`` it is the unhedged control:
    same gather/spot-sell behavior, no futures. The bd-c7i validation runs
    both arms side by side and compares revenue variance across seeds; the
    toggle keeps the comparison apples-to-apples (the futures hedge is the
    only difference).
    """

    def __init__(self, agent_id: str, seed=None, hedge: bool = True,
                 horizon: int = 3, hedge_qty: float = 1.0,
                 hedge_every: int = 4, fair_value: float = 1.0,
                 resource: ResourceType = ResourceType.WOOD) -> None:
        super().__init__(agent_id, seed)
        self._hedge = hedge
        self._horizon = horizon
        self._hedge_qty = hedge_qty
        self._hedge_every = hedge_every
        self._fair_value = fair_value
        # Commodity this producer gathers, spot-sells, and hedges (bd ja2):
        # COMPUTE makes it an H100 datacenter operator hedging GPU revenue.
        self._resource = resource

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        step = obs.get("step", 0)
        inv = obs.get("inventory", {})
        held = inv.get(self._resource.value, 0.0)
        info = (obs.get("market_info") or {}).get(self._resource.value) or {}
        spot = info.get("last_price") or self._fair_value

        # Hedge on a cadence (only when hedging is enabled).
        if self._hedge and step > 0 and step % self._hedge_every == 0:
            return GTBAction(
                agent_id=self.agent_id, action_type=GTBActionType.FUTURES_SELL,
                resource_type=self._resource, quantity=self._hedge_qty,
                price=spot, settlement_epoch=obs.get("epoch", 0) + self._horizon,
            )
        # Spot-sell accumulated inventory (the spot-priced revenue being hedged).
        if held >= 1.0:
            return GTBAction(
                agent_id=self.agent_id, action_type=GTBActionType.TRADE_SELL,
                resource_type=self._resource, quantity=1.0,
                price=max(0.1, spot * 0.95),  # cross down to ensure a sale
            )
        # Otherwise produce more: walk to the nearest tile of the commodity
        # (e.g. a datacenter for COMPUTE) if one is visible, else gather in
        # place (we may already be standing on one).
        direction = self._find_resource_direction(obs, self._resource.value)
        if direction is not None:
            return GTBAction(agent_id=self.agent_id,
                             action_type=GTBActionType.MOVE, direction=direction)
        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.GATHER)


class MatchedHedgerPolicy(GTBWorkerPolicy):
    """Producer that hedges its realized spot exposure by *crossing* the
    resting forward book (bd 5ij).

    Two changes from ``FuturesHedgerPolicy``, the two things bd k9w found
    missing between its check A (mechanism = 100% variance reduction) and
    check C (naive policy realizes none of it):

    1. **Matched sizing.** It keeps cumulative forwards short at
       ``hedge_ratio`` of its throughput (compute already sold + currently
       held), instead of a fixed cadence ``hedge_qty``. The hedge tracks the
       exposure it is actually building.
    2. **Through-the-book execution.** It hedges by hitting the best resting
       forward *bid* on the curve (a crossing ``FUTURES_SELL`` priced at that
       bid), so the order actually fills and pays the prevailing spread —
       rather than resting at its own mark and never matching. As it consumes
       the top of the book, deeper/worse bids remain, so large hedges realize
       genuine slippage.

    With ``hedge_ratio=0`` it is the unhedged control that still gathers and
    spot-sells (genuine, equal exposure) — fixing the near-inert control that
    made k9w check C degenerate.

    The policy can't see its own open contracts in ``obs``, so it tracks the
    hedged/sold quantities internally, assuming a crossing order fills (it is
    priced to cross the resting bid it just read). Slight drift is possible if
    the bid vanishes between read and match; acceptable for a scripted agent.
    """

    def __init__(self, agent_id: str, seed=None,
                 resource: ResourceType = ResourceType.COMPUTE,
                 hedge_ratio: float = 1.0, horizon: int = 3, lot: float = 1.0,
                 fair_value: float = 1.0) -> None:
        super().__init__(agent_id, seed)
        self._resource = resource
        self._hedge_ratio = max(0.0, hedge_ratio)
        self._horizon = horizon
        self._lot = lot
        self._fair_value = fair_value
        self._hedged = 0.0    # cumulative forward units shorted (assumed filled)
        self._sold = 0.0      # cumulative compute spot-sold

    def _best_forward_bid(self, obs: dict):
        """Best resting forward bid for our resource + its settlement epoch,
        or (None, None) if nobody is bidding the forward."""
        curve = obs.get("futures_curve") or {}
        best = None
        for entry in curve.values():
            if entry.get("resource") != self._resource.value:
                continue
            bid = entry.get("best_bid")
            if bid is None:
                continue
            if best is None or bid > best[0]:
                best = (bid, entry.get("settlement_epoch"))
        return best if best is not None else (None, None)

    def decide(self, obs: dict) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
        inv = obs.get("inventory", {})
        held = inv.get(self._resource.value, 0.0)
        info = (obs.get("market_info") or {}).get(self._resource.value) or {}
        spot = info.get("last_price") or self._fair_value

        # 1. Hedge to match exposure (throughput so far + inventory on hand),
        #    executed by crossing the best resting forward bid.
        if self._hedge_ratio > 0:
            target = self._hedge_ratio * (self._sold + held)
            gap = target - self._hedged
            bid, settle = self._best_forward_bid(obs)
            if gap >= self._lot and bid is not None:
                qty = min(gap, self._lot)
                self._hedged += qty
                return GTBAction(
                    agent_id=self.agent_id,
                    action_type=GTBActionType.FUTURES_SELL,
                    resource_type=self._resource, quantity=qty,
                    price=bid,  # cross the resting bid -> fills at the bid
                    settlement_epoch=settle,
                )

        # 2. Spot-sell accumulated inventory (the exposure being hedged).
        if held >= 1.0:
            self._sold += 1.0
            return GTBAction(
                agent_id=self.agent_id, action_type=GTBActionType.TRADE_SELL,
                resource_type=self._resource, quantity=1.0,
                price=max(0.1, spot * 0.95),  # cross down to ensure a sale
            )

        # 3. Produce more: walk to a datacenter tile, else gather in place.
        direction = self._find_resource_direction(obs, self._resource.value)
        if direction is not None:
            return GTBAction(agent_id=self.agent_id,
                             action_type=GTBActionType.MOVE, direction=direction)
        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.GATHER)
