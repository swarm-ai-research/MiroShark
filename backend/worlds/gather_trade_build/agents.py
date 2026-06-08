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

    def _find_resource_direction(self, obs: dict) -> Optional[Direction]:
        """Find direction toward nearest visible resource."""
        pos = obs.get("position", (0, 0))
        r, c = pos[0], pos[1]
        best_dist = float("inf")
        best_dir = None

        for cell in obs.get("visible_cells", []):
            if "resource" not in cell or cell.get("amount", 0) <= 0:
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
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        # Private value: heterogeneous across traders so books cross
        self._value = value_estimate * (
            1.0 + self._rng.uniform(-value_jitter, value_jitter)
        )
        self._order_qty = order_qty
        self._min_coin_reserve = min_coin_reserve

    @property
    def private_value(self) -> float:
        return self._value

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        coin = inventory.get(ResourceType.COIN.value, 0.0)

        if energy >= 0.5:
            # Sell surplus above the private value
            for rtype in (ResourceType.WOOD, ResourceType.STONE):
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
                rtype = self._rng.choice(
                    [ResourceType.WOOD, ResourceType.STONE]
                )
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
