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
