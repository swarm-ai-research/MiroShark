"""GTB gridworld environment: state, step semantics, and market."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from worlds.gather_trade_build.config import GTBConfig
from worlds.gather_trade_build.entities import (
    Direction,
    FuturesContract,
    FuturesOrder,
    GTBActionType,
    GTBEvent,
    GTBGridCell,
    House,
    MarketOrder,
    Resource,
    ResourceType,
    WorkerState,
)
from worlds.gather_trade_build.tax_schedule import TaxSchedule

logger = logging.getLogger(__name__)

# Direction deltas: (row_delta, col_delta)
_DIR_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


@dataclass
class EpochResult:
    """Result from end_epoch(), containing events and a pre-reset worker snapshot."""

    events: List[GTBEvent] = field(default_factory=list)
    snapshot: Dict[str, WorkerState] = field(default_factory=dict)


@dataclass
class GTBAction:
    """A worker's action for one step."""

    agent_id: str = ""
    action_type: GTBActionType = GTBActionType.NOOP
    direction: Direction = Direction.UP
    resource_type: ResourceType = ResourceType.WOOD
    quantity: float = 0.0
    price: float = 1.0
    shift_amount: float = 0.0
    underreport_fraction: float = 0.0
    # Commodity futures (bd-af2): for FUTURES_BUY/FUTURES_SELL the order
    # reuses resource_type, quantity, and price (as the forward price) and
    # targets this settlement epoch. 0 = unset / not a futures order. The
    # matching book (bd-oo7) reads these; the bare env ignores them.
    settlement_epoch: int = 0
    # Optional side-channel: place a YES/NO stake on an open prediction
    # market in the same tick. Env ignores these fields; the GTB world
    # service intercepts them after apply_actions and updates a separate
    # stake book. Keeps the env's invariants untouched.
    stake_market_id: str = ""
    stake_side: str = ""  # "" | "yes" | "no"
    stake_amount: float = 0.0


class GTBEnvironment:
    """Gather-Trade-Build gridworld environment.

    Provides:
      - Grid with resource tiles and houses
      - Worker state management (inventory, position, energy)
      - Centralized market for trading resources
      - Tax collection via TaxSchedule at epoch boundaries
      - Income shifting and misreporting mechanics
      - Audit pipeline
      - Collusion detection scaffolding
      - Event logging
    """

    def __init__(self, config: GTBConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._tax_schedule = TaxSchedule(config.taxation)

        # Grid
        self._height = config.map.height
        self._width = config.map.width
        self._grid: List[List[GTBGridCell]] = []
        self._houses: List[House] = []

        # Workers
        self._workers: Dict[str, WorkerState] = {}

        # Market
        self._buy_orders: List[MarketOrder] = []
        self._sell_orders: List[MarketOrder] = []
        self._last_trade_price: Dict[str, float] = {}
        self._volume_this_epoch: Dict[str, float] = {}

        # Futures market (bd-af2): resting long/short orders, matched
        # contracts, and last forward print per (resource, settlement_epoch).
        self._futures_buy_orders: List[FuturesOrder] = []
        self._futures_sell_orders: List[FuturesOrder] = []
        self._futures_contracts: List[FuturesContract] = []
        self._last_forward_price: Dict[str, float] = {}
        self._futures_volume_this_epoch: Dict[str, float] = {}
        self._futures_seq: int = 0

        # Events
        self._events: List[GTBEvent] = []

        # Step / epoch counters
        self._current_step = 0
        self._current_epoch = 0

        # Collusion tracking
        self._action_traces: Dict[str, List[str]] = {}  # agent_id -> recent actions
        self._ask_history: Dict[str, List[float]] = {}  # agent_id -> ask prices this epoch

        # Misreport fractions per agent for current epoch (applied to future income)
        self._misreport_fractions: Dict[str, float] = {}

        # Collusion response state
        self._collusion_audit_boost: Dict[str, float] = {}  # agent_id -> extra audit prob
        self._trade_restricted: Dict[str, int] = {}  # agent_id -> restriction_end_epoch

        # Frozen agents (from audit penalties)
        self._frozen_agents: Dict[str, int] = {}  # agent_id -> unfreeze_epoch

        # Treasury: taxes/fines/debt payments accumulate here and fund
        # redistribution (taxation.redistribution) and treasury-mode house
        # income (build.house_income_mode). With both off it just grows,
        # mirroring the legacy "burned" semantics.
        self._treasury = 0.0

        # Coin ledger: every mint/burn itemized by source so conservation
        # can be verified (sum of worker coin == minted - burned).
        self._coin_minted: Dict[str, float] = {}
        self._coin_burned: Dict[str, float] = {}
        # Gross coin actually received per worker this epoch (house income,
        # sale proceeds). Compared against gross_income_this_epoch at epoch
        # close to surface the income-vs-coin gap.
        self._coin_earned_this_epoch: Dict[str, float] = {}

        self._init_grid()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_grid(self) -> None:
        """Initialize the grid with resource tiles."""
        self._grid = []
        for r in range(self._height):
            row = []
            for c in range(self._width):
                cell = GTBGridCell(position=(r, c))
                # Randomly place resources
                roll = self._rng.random()
                if roll < self._config.map.wood_density:
                    cell.resource = Resource(
                        resource_type=ResourceType.WOOD,
                        amount=self._config.map.resource_max_amount,
                        position=(r, c),
                        regen_rate=self._config.map.resource_regen_rate,
                    )
                elif roll < self._config.map.wood_density + self._config.map.stone_density:
                    cell.resource = Resource(
                        resource_type=ResourceType.STONE,
                        amount=self._config.map.resource_max_amount,
                        position=(r, c),
                        regen_rate=self._config.map.resource_regen_rate,
                    )
                row.append(cell)
            self._grid.append(row)

    def add_worker(self, agent_id: str, skill_gather: float = 1.0,
                   skill_build: float = 1.0) -> WorkerState:
        """Register a worker in the environment."""
        row = self._rng.randint(0, self._height - 1)
        col = self._rng.randint(0, self._width - 1)
        worker = WorkerState(
            agent_id=agent_id,
            position=(row, col),
            energy=self._config.energy_per_step,
            max_energy=self._config.energy_per_step,
            skill_gather=skill_gather,
            skill_build=skill_build,
        )
        worker.add_resource(ResourceType.COIN, 10.0)  # starting endowment
        self._ledger_mint("endowment", 10.0)
        self._workers[agent_id] = worker
        self._grid[row][col].occupants.append(agent_id)
        self._action_traces[agent_id] = []
        return worker

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def obs(self, agent_id: str) -> Dict[str, Any]:
        """Build observation for an agent."""
        worker = self._workers[agent_id]
        r, c = worker.position
        # Visible neighborhood (5x5 centered on agent)
        visible = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._height and 0 <= nc < self._width:
                    cell = self._grid[nr][nc]
                    cell_info: Dict[str, Any] = {"pos": (nr, nc)}
                    if cell.resource and cell.resource.amount > 0:
                        cell_info["resource"] = cell.resource.resource_type.value
                        cell_info["amount"] = cell.resource.amount
                    if cell.house:
                        cell_info["house_owner"] = cell.house.owner_id
                    cell_info["occupants"] = list(cell.occupants)
                    visible.append(cell_info)

        return {
            "agent_id": agent_id,
            "position": worker.position,
            "inventory": dict(worker.inventory),
            "energy": worker.energy,
            "effort_this_epoch": worker.effort_this_epoch,
            "houses_built": worker.houses_built,
            "gross_income": worker.gross_income_this_epoch,
            "deferred_income": worker.deferred_income,
            "epoch": self._current_epoch,
            "step": self._current_step,
            "tax_schedule": self._tax_schedule.to_dict(),
            "visible_cells": visible,
            "frozen": agent_id in self._frozen_agents,
            "market_info": self._market_info(),
            # Top-of-book snapshot for market-aware policies (bd-8dj).
            # Per-resource: 3 lowest asks + 3 highest bids, oldest first
            # within the same price. Empty lists if no orders on that
            # side. Policies that don't read the order book ignore this.
            "market_book": self._market_book_snapshot(),
            # Forward curve: per (resource, settlement_epoch) best bid/ask
            # and last forward print (bd-oo7). Empty when futures disabled.
            "futures_curve": self._futures_curve(),
        }

    def _futures_curve(self) -> Dict[str, Dict[str, Any]]:
        """Forward-curve snapshot keyed by ``resource@settlement_epoch``.

        Per bucket: best bid (highest long), best ask (lowest short), last
        traded forward, and resting order counts. The basis vs spot
        (forward - spot) is left to consumers, who have ``market_info``.
        """
        curve: Dict[str, Dict[str, Any]] = {}
        keys = {
            self._futures_key(o.resource_type, o.settlement_epoch)
            for o in self._futures_buy_orders + self._futures_sell_orders
        }
        keys.update(self._last_forward_price.keys())
        for key in keys:
            res_val, _, settle_s = key.partition("@")
            settle = int(settle_s) if settle_s else 0
            bids = [o.forward_price for o in self._futures_buy_orders
                    if self._futures_key(o.resource_type, o.settlement_epoch) == key]
            asks = [o.forward_price for o in self._futures_sell_orders
                    if self._futures_key(o.resource_type, o.settlement_epoch) == key]
            curve[key] = {
                "resource": res_val,
                "settlement_epoch": settle,
                "best_bid": max(bids) if bids else None,
                "best_ask": min(asks) if asks else None,
                "last_forward": self._last_forward_price.get(key),
                "n_bids": len(bids),
                "n_asks": len(asks),
            }
        return curve

    def _market_info(self) -> Dict[str, Any]:
        """Per-resource price discovery info: last trade price, best
        bid/ask on the resting book, and volume this epoch."""
        info: Dict[str, Any] = {}
        for rtype in (ResourceType.WOOD, ResourceType.STONE):
            key = rtype.value
            bids = [o.price_per_unit for o in self._buy_orders
                    if o.resource_type == rtype and o.quantity > 1e-12]
            asks = [o.price_per_unit for o in self._sell_orders
                    if o.resource_type == rtype and o.quantity > 1e-12]
            info[key] = {
                "last_price": self._last_trade_price.get(key),
                "best_bid": max(bids) if bids else None,
                "best_ask": min(asks) if asks else None,
                "volume_this_epoch": self._volume_this_epoch.get(key, 0.0),
            }
        return info

    def _market_book_snapshot(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Compact top-of-book view by resource type, for policy obs."""
        snap: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for rt in (ResourceType.WOOD, ResourceType.STONE):
            sells = [o for o in self._sell_orders if o.resource_type == rt]
            buys = [o for o in self._buy_orders if o.resource_type == rt]
            sells = sorted(sells, key=lambda o: (o.price_per_unit, o.step))[:3]
            buys = sorted(buys, key=lambda o: (-o.price_per_unit, o.step))[:3]
            snap[rt.value] = {
                "asks": [
                    {"agent_id": o.agent_id, "quantity": o.quantity, "price": o.price_per_unit}
                    for o in sells
                ],
                "bids": [
                    {"agent_id": o.agent_id, "quantity": o.quantity, "price": o.price_per_unit}
                    for o in buys
                ],
            }
        return snap

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def apply_actions(self, actions: Dict[str, GTBAction]) -> List[GTBEvent]:
        """Apply a dict of agent actions for one step.

        Args:
            actions: Mapping from agent_id to their chosen action.

        Returns:
            List of events generated this step.
        """
        step_events: List[GTBEvent] = []

        # Regenerate resources
        self._regenerate_resources()

        # Process each agent's action
        for agent_id, action in actions.items():
            if agent_id in self._frozen_agents:
                step_events.append(GTBEvent(
                    event_type="frozen_skip",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                ))
                continue

            worker = self._workers.get(agent_id)
            if worker is None:
                continue

            # Track action for collusion detection
            self._action_traces.setdefault(agent_id, []).append(
                action.action_type.value
            )

            if action.action_type == GTBActionType.MOVE:
                evt = self._handle_move(worker, action.direction)
            elif action.action_type == GTBActionType.GATHER:
                evt = self._handle_gather(worker)
            elif action.action_type == GTBActionType.BUILD:
                evt = self._handle_build(worker)
            elif action.action_type == GTBActionType.TRADE_BUY:
                evt = self._handle_trade_buy(worker, action)
            elif action.action_type == GTBActionType.TRADE_SELL:
                evt = self._handle_trade_sell(worker, action)
            elif action.action_type in (GTBActionType.FUTURES_BUY,
                                        GTBActionType.FUTURES_SELL):
                evt = self._handle_futures_order(worker, action)
            elif action.action_type == GTBActionType.SHIFT_INCOME:
                evt = self._handle_shift_income(worker, action)
            elif action.action_type == GTBActionType.MISREPORT:
                evt = self._handle_misreport(worker, action)
            else:
                evt = GTBEvent(
                    event_type="noop",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                )

            if evt:
                step_events.append(evt)

        # Distribute house income
        house_events = self._distribute_house_income()
        step_events.extend(house_events)

        # Match market orders
        trade_events = self._match_market_orders()
        step_events.extend(trade_events)

        # Match futures orders (bd-oo7)
        if self._config.market.futures_enabled:
            step_events.extend(self._match_futures_orders())

        self._events.extend(step_events)
        self._current_step += 1
        return step_events

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _spend_energy(worker: WorkerState, cost: float) -> None:
        """Deduct energy and book it as labor effort (for labor disutility)."""
        worker.energy -= cost
        worker.effort_this_epoch += cost
        worker.cumulative_effort += cost

    def _handle_move(self, worker: WorkerState, direction: Direction) -> GTBEvent:
        delta = _DIR_DELTA.get(direction)
        if delta is None or worker.energy < self._config.energy_cost_move:
            return GTBEvent(
                event_type="move_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
            )
        old_r, old_c = worker.position
        new_r = max(0, min(self._height - 1, old_r + delta[0]))
        new_c = max(0, min(self._width - 1, old_c + delta[1]))

        # Update grid occupants
        self._grid[old_r][old_c].occupants = [
            a for a in self._grid[old_r][old_c].occupants if a != worker.agent_id
        ]
        self._grid[new_r][new_c].occupants.append(worker.agent_id)
        worker.position = (new_r, new_c)
        self._spend_energy(worker, self._config.energy_cost_move)

        return GTBEvent(
            event_type="move", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"from": (old_r, old_c), "to": (new_r, new_c)},
        )

    def _handle_gather(self, worker: WorkerState) -> GTBEvent:
        if worker.energy < self._config.energy_cost_gather:
            return GTBEvent(
                event_type="gather_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        r, c = worker.position
        cell = self._grid[r][c]
        if cell.resource is None or cell.resource.amount <= 0:
            return GTBEvent(
                event_type="gather_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_resource"},
            )

        gathered = min(cell.resource.amount, 1.0 * worker.skill_gather)
        cell.resource.amount -= gathered
        worker.add_resource(cell.resource.resource_type, gathered)
        self._spend_energy(worker, self._config.energy_cost_gather)

        if self._config.ledger_mode == "legacy":
            # Legacy: gathering books taxable income 1:1 with no coin
            # behind it (the income/coin gap in epoch_ledger events).
            income = gathered
            worker.gross_income_this_epoch += income
            worker.reported_income_this_epoch += income
            worker.cumulative_income += income
        else:
            # coin mode: gathering yields resources only; income arises
            # when they are sold or when houses pay out.
            income = 0.0

        return GTBEvent(
            event_type="gather", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={
                "resource": cell.resource.resource_type.value,
                "amount": gathered, "income": income,
            },
        )

    def _handle_build(self, worker: WorkerState) -> GTBEvent:
        cfg = self._config.build
        if worker.energy < self._config.energy_cost_build:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        if worker.houses_built >= cfg.max_houses_per_agent:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "max_houses"},
            )
        r0, c0 = worker.position
        if self._grid[r0][c0].house is not None:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "cell_occupied"},
            )
        wood_ok = worker.remove_resource(ResourceType.WOOD, cfg.wood_cost)
        if not wood_ok:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "insufficient_wood"},
            )
        stone_ok = worker.remove_resource(ResourceType.STONE, cfg.stone_cost)
        if not stone_ok:
            # Refund wood
            worker.add_resource(ResourceType.WOOD, cfg.wood_cost)
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "insufficient_stone"},
            )

        r, c = worker.position
        house = House(
            owner_id=worker.agent_id,
            position=(r, c),
            wood_cost=cfg.wood_cost,
            stone_cost=cfg.stone_cost,
            income_per_step=cfg.income_per_house_per_step,
            build_step=self._current_step,
        )
        self._houses.append(house)
        self._grid[r][c].house = house
        worker.houses_built += 1
        self._spend_energy(worker, self._config.energy_cost_build)

        return GTBEvent(
            event_type="build", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"position": (r, c), "houses_total": worker.houses_built},
        )

    def _handle_trade_buy(self, worker: WorkerState, action: GTBAction) -> GTBEvent:
        # Enforce collusion-triggered trade restrictions
        restrict_until = self._trade_restricted.get(worker.agent_id, 0)
        if self._current_epoch < restrict_until:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "trade_restricted_collusion"},
            )
        if worker.energy < self._config.energy_cost_trade:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        price = max(self._config.market.price_floor,
                     min(action.price, self._config.market.price_ceiling))
        qty = max(0.0, action.quantity)
        ttl = self._config.market.order_ttl_steps
        order = MarketOrder(
            agent_id=worker.agent_id,
            resource_type=action.resource_type,
            quantity=qty,
            price_per_unit=price,
            is_buy=True,
            step=self._current_step,
            expiry_step=self._current_step + ttl,
        )
        if ttl > 0:
            # Persistent book: escrow the full worst-case cost at post
            # time so a resting bid can always settle (no spoofing).
            escrow = qty * price * (1.0 + self._config.market.transaction_fee_rate)
            if qty <= 0 or worker.get_resource(ResourceType.COIN) < escrow:
                return GTBEvent(
                    event_type="trade_fail", step=self._current_step,
                    epoch=self._current_epoch, agent_id=worker.agent_id,
                    details={"reason": "insufficient_coin_escrow",
                              "required": escrow},
                )
            worker.remove_resource(ResourceType.COIN, escrow)
            order.escrowed_coin = escrow
        self._buy_orders.append(order)
        self._spend_energy(worker, self._config.energy_cost_trade)
        return GTBEvent(
            event_type="order_placed", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"side": "buy", "resource": action.resource_type.value,
                      "qty": action.quantity, "price": price},
        )

    def _handle_trade_sell(self, worker: WorkerState, action: GTBAction) -> GTBEvent:
        # Enforce collusion-triggered trade restrictions
        restrict_until = self._trade_restricted.get(worker.agent_id, 0)
        if self._current_epoch < restrict_until:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "trade_restricted_collusion"},
            )
        if worker.energy < self._config.energy_cost_trade:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        price = max(self._config.market.price_floor,
                     min(action.price, self._config.market.price_ceiling))
        qty = max(0.0, action.quantity)
        ttl = self._config.market.order_ttl_steps
        order = MarketOrder(
            agent_id=worker.agent_id,
            resource_type=action.resource_type,
            quantity=qty,
            price_per_unit=price,
            is_buy=False,
            step=self._current_step,
            expiry_step=self._current_step + ttl,
        )
        if ttl > 0:
            # Persistent book: escrow the resource at post time.
            if qty <= 0 or worker.get_resource(action.resource_type) < qty:
                return GTBEvent(
                    event_type="trade_fail", step=self._current_step,
                    epoch=self._current_epoch, agent_id=worker.agent_id,
                    details={"reason": "insufficient_resource_escrow",
                              "required": qty},
                )
            worker.remove_resource(action.resource_type, qty)
        self._sell_orders.append(order)
        self._ask_history.setdefault(worker.agent_id, []).append(price)
        self._spend_energy(worker, self._config.energy_cost_trade)
        return GTBEvent(
            event_type="order_placed", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"side": "sell", "resource": action.resource_type.value,
                      "qty": action.quantity, "price": price},
        )

    # ------------------------------------------------------------------
    # Futures market (bd-af2 first cut)
    # ------------------------------------------------------------------

    def _handle_futures_order(self, worker: WorkerState,
                              action: GTBAction) -> GTBEvent:
        """Post a resting futures limit order (FUTURES_BUY=long / SELL=short).

        Escrows ``futures_margin_rate * qty * forward_price`` in coin at
        post time (both sides escrow coin — the first cut is cash-settled,
        so the short need not hold the physical resource). Rejected if the
        settlement epoch is not in the future or the worker can't fund the
        margin.
        """
        is_buy = action.action_type == GTBActionType.FUTURES_BUY
        if not self._config.market.futures_enabled:
            return self._futures_fail(worker, "futures_disabled")
        if worker.energy < self._config.energy_cost_trade:
            return self._futures_fail(worker, "no_energy")
        qty = max(0.0, action.quantity)
        if qty <= 0:
            return self._futures_fail(worker, "non_positive_qty")
        if action.settlement_epoch <= self._current_epoch:
            return self._futures_fail(worker, "settlement_not_in_future")
        forward = max(self._config.market.price_floor,
                      min(action.price, self._config.market.price_ceiling))
        margin = self._config.market.futures_margin_rate * qty * forward
        if worker.get_resource(ResourceType.COIN) < margin:
            return self._futures_fail(worker, "insufficient_margin", margin)

        worker.remove_resource(ResourceType.COIN, margin)
        order = FuturesOrder(
            agent_id=worker.agent_id,
            resource_type=action.resource_type,
            qty=qty,
            forward_price=forward,
            settlement_epoch=action.settlement_epoch,
            is_buy=is_buy,
            step=self._current_step,
            margin=margin,
        )
        (self._futures_buy_orders if is_buy
         else self._futures_sell_orders).append(order)
        self._spend_energy(worker, self._config.energy_cost_trade)
        return GTBEvent(
            event_type="futures_order_placed", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"side": "long" if is_buy else "short",
                     "resource": action.resource_type.value, "qty": qty,
                     "forward_price": forward,
                     "settlement_epoch": action.settlement_epoch},
        )

    def _futures_fail(self, worker: WorkerState, reason: str,
                      required: float = 0.0) -> GTBEvent:
        details: Dict[str, Any] = {"reason": reason}
        if required:
            details["required"] = required
        return GTBEvent(
            event_type="futures_order_fail", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details=details,
        )

    @staticmethod
    def _futures_key(resource: ResourceType, settlement_epoch: int) -> str:
        return f"{resource.value}@{settlement_epoch}"

    def _match_futures_orders(self) -> List[GTBEvent]:
        """Match crossing futures orders into FuturesContracts.

        Grouped by (resource, settlement_epoch) — only orders for the same
        delivery date are fungible. Highest-bid / lowest-ask first; forward
        price is the midpoint of the crossing pair (mirrors the spot book).
        Each match mints one open contract carrying both sides' escrowed
        margin; settlement is bd-dog.
        """
        events: List[GTBEvent] = []
        # Distinct (resource, settlement_epoch) buckets present on the book.
        buckets = {
            (o.resource_type, o.settlement_epoch)
            for o in self._futures_buy_orders + self._futures_sell_orders
        }
        for resource, settle in buckets:
            buys = sorted(
                [o for o in self._futures_buy_orders
                 if o.resource_type == resource
                 and o.settlement_epoch == settle and o.qty > 1e-12],
                key=lambda o: -o.forward_price,
            )
            sells = sorted(
                [o for o in self._futures_sell_orders
                 if o.resource_type == resource
                 and o.settlement_epoch == settle and o.qty > 1e-12],
                key=lambda o: o.forward_price,
            )
            # For each buyer (best price first), match against the cheapest
            # crossable sellers, skipping self-orders without consuming them
            # for other buyers (bd-pfc — a simple two-pointer stranded a
            # seller when it self-matched the current buyer).
            for buy in buys:
                for sell in sells:
                    if buy.qty <= 1e-12:
                        break
                    if sell.qty <= 1e-12:
                        continue
                    if buy.agent_id == sell.agent_id:
                        continue
                    if buy.forward_price < sell.forward_price:
                        break  # sells sorted ascending: none cheaper ahead
                    qty = min(buy.qty, sell.qty)
                    forward = (buy.forward_price + sell.forward_price) / 2.0
                    # Move each side's pro-rata margin onto the contract.
                    m_long = buy.margin * (qty / buy.qty) if buy.qty > 0 else 0.0
                    m_short = sell.margin * (qty / sell.qty) if sell.qty > 0 else 0.0
                    buy.margin -= m_long
                    sell.margin -= m_short
                    self._futures_seq += 1
                    contract = FuturesContract(
                        contract_id=f"fc-{self._futures_seq}",
                        resource_type=resource,
                        qty=qty,
                        forward_price=forward,
                        settlement_epoch=settle,
                        long_agent_id=buy.agent_id,
                        short_agent_id=sell.agent_id,
                        margin_long=m_long,
                        margin_short=m_short,
                        created_epoch=self._current_epoch,
                    )
                    self._futures_contracts.append(contract)
                    buy.qty -= qty
                    sell.qty -= qty
                    key = self._futures_key(resource, settle)
                    self._last_forward_price[key] = forward
                    self._futures_volume_this_epoch[key] = (
                        self._futures_volume_this_epoch.get(key, 0.0) + qty
                    )
                    events.append(GTBEvent(
                        event_type="futures_matched", step=self._current_step,
                        epoch=self._current_epoch, agent_id=buy.agent_id,
                        details={"contract_id": contract.contract_id,
                                 "resource": resource.value, "qty": qty,
                                 "forward_price": forward,
                                 "settlement_epoch": settle,
                                 "long": buy.agent_id, "short": sell.agent_id},
                    ))
        # Drop fully-filled resting orders.
        self._futures_buy_orders = [o for o in self._futures_buy_orders
                                    if o.qty > 1e-12]
        self._futures_sell_orders = [o for o in self._futures_sell_orders
                                     if o.qty > 1e-12]
        return events

    def _refund_expired_futures_orders(self) -> List[GTBEvent]:
        """Refund margin on resting futures orders whose settlement epoch
        has arrived unmatched, so escrowed coin is not stranded. Matched
        contracts are settled separately (bd-dog)."""
        events: List[GTBEvent] = []
        kept_buys, kept_sells = [], []
        for order in self._futures_buy_orders:
            if order.settlement_epoch <= self._current_epoch:
                self._refund_futures_order(order, events)
            else:
                kept_buys.append(order)
        for order in self._futures_sell_orders:
            if order.settlement_epoch <= self._current_epoch:
                self._refund_futures_order(order, events)
            else:
                kept_sells.append(order)
        self._futures_buy_orders = kept_buys
        self._futures_sell_orders = kept_sells
        return events

    def _refund_futures_order(self, order: FuturesOrder,
                              events: List[GTBEvent]) -> None:
        worker = self._workers.get(order.agent_id)
        if worker is not None and order.margin > 0:
            worker.add_resource(ResourceType.COIN, order.margin)
        events.append(GTBEvent(
            event_type="futures_order_expired", step=self._current_step,
            epoch=self._current_epoch, agent_id=order.agent_id,
            details={"resource": order.resource_type.value,
                     "qty": order.qty, "refunded_margin": order.margin,
                     "settlement_epoch": order.settlement_epoch},
        ))
        order.margin = 0.0

    def open_futures_contracts(self) -> List[FuturesContract]:
        """Contracts not yet settled (mirrors GTBMarketBook.open_markets)."""
        return [c for c in self._futures_contracts if c.status == "open"]

    def futures_summary(self) -> Dict[str, float]:
        """Stock snapshot for metrics (bd-2qe): open interest, open notional,
        and mean basis (forward - spot) over resources that have both a last
        forward print and a last spot trade."""
        live = self.open_futures_contracts()
        open_notional = sum(c.qty * c.forward_price for c in live)
        bases = []
        for rt in (ResourceType.WOOD, ResourceType.STONE):
            spot = self._last_trade_price.get(rt.value)
            # Front-month basis: pick the nearest-dated forward print (the
            # smallest settlement epoch), not dict-insertion order. The key
            # encodes the date as ``<resource>@<epoch>``.
            dated = [(int(k.split("@", 1)[1]), self._last_forward_price[k])
                     for k in self._last_forward_price
                     if k.startswith(f"{rt.value}@")]
            if spot is not None and dated:
                _, nearest_forward = min(dated, key=lambda d: d[0])
                bases.append(nearest_forward - spot)
        basis = sum(bases) / len(bases) if bases else 0.0
        return {
            "open_interest": len(live),
            "open_notional": open_notional,
            "basis": basis,
        }

    def _settle_futures_contracts(self) -> List[GTBEvent]:
        """Cash-settle every open contract whose settlement epoch arrived.

        Deterministic: iterate in creation order. For each, the spot
        reference is ``_last_trade_price`` for the resource (falls back to
        the forward price — zero P&L — if the resource never traded spot,
        so a futures-only run still settles cleanly). Then:

        1. Return ``margin_long``/``margin_short`` to each side (restores
           the coin escrowed at match — coin-neutral).
        2. Transfer ``pnl = (spot - forward) * qty`` short->long (long
           profits when spot rose above the locked forward). Pure zero-sum
           transfer, clamped to the payer's available coin so no balance
           goes negative (first cut has no margin-call/liquidation — that
           is the deferred daily-MTM child). Records the realized P&L on
           each side's ``cumulative_income`` symmetrically.

        Coin is conserved: step 1 returns escrow, step 2 is a transfer.
        """
        events: List[GTBEvent] = []
        for c in self._futures_contracts:
            if c.status != "open" or c.settlement_epoch > self._current_epoch:
                continue
            long_w = self._workers.get(c.long_agent_id)
            short_w = self._workers.get(c.short_agent_id)

            # 1. Release margins (coin-neutral restore of escrow).
            if long_w is not None and c.margin_long > 0:
                long_w.add_resource(ResourceType.COIN, c.margin_long)
            if short_w is not None and c.margin_short > 0:
                short_w.add_resource(ResourceType.COIN, c.margin_short)

            # 2. P&L transfer, clamped so no balance goes negative.
            # ``realized`` is the long's signed P&L; the move is a pure
            # zero-sum coin transfer between the two workers.
            spot = self._last_trade_price.get(c.resource_type.value,
                                              c.forward_price)
            pnl = (spot - c.forward_price) * c.qty
            realized = 0.0
            if long_w is not None and short_w is not None:
                if pnl >= 0:  # long wins, short pays (clamped to short's coin)
                    transfer = min(pnl, short_w.get_resource(ResourceType.COIN))
                    short_w.remove_resource(ResourceType.COIN, transfer)
                    long_w.add_resource(ResourceType.COIN, transfer)
                    realized = transfer
                else:  # long loses, long pays short (clamped to long's coin)
                    transfer = min(-pnl, long_w.get_resource(ResourceType.COIN))
                    long_w.remove_resource(ResourceType.COIN, transfer)
                    short_w.add_resource(ResourceType.COIN, transfer)
                    realized = -transfer
                long_w.cumulative_income += realized
                short_w.cumulative_income -= realized

            c.status = "settled"
            c.settled_epoch = self._current_epoch
            c.settle_spot_price = spot
            c.margin_long = 0.0
            c.margin_short = 0.0
            events.append(GTBEvent(
                event_type="futures_settled", step=self._current_step,
                epoch=self._current_epoch, agent_id=c.long_agent_id,
                details={"contract_id": c.contract_id,
                         "resource": c.resource_type.value, "qty": c.qty,
                         "forward_price": c.forward_price, "spot": spot,
                         "pnl_long": realized, "long": c.long_agent_id,
                         "short": c.short_agent_id},
            ))
        return events

    def _handle_shift_income(self, worker: WorkerState,
                              action: GTBAction) -> GTBEvent:
        cfg = self._config.gaming
        if not cfg.income_shifting_enabled:
            return GTBEvent(
                event_type="shift_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "disabled"},
            )
        max_shift = worker.gross_income_this_epoch * cfg.max_shift_fraction_per_epoch
        shift = min(action.shift_amount, max_shift)
        if shift <= 0:
            return GTBEvent(
                event_type="shift_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "nothing_to_shift"},
            )

        friction = shift * cfg.gaming_cost_fraction
        net_shift = shift - friction
        worker.gross_income_this_epoch -= shift
        worker.reported_income_this_epoch -= shift
        worker.deferred_income += net_shift

        return GTBEvent(
            event_type="shift_income", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"gross_shifted": shift, "friction": friction,
                      "net_deferred": net_shift},
        )

    def _handle_misreport(self, worker: WorkerState,
                           action: GTBAction) -> GTBEvent:
        cfg = self._config.misreporting
        if not cfg.enabled:
            return GTBEvent(
                event_type="misreport_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "disabled"},
            )
        frac = min(action.underreport_fraction, cfg.max_underreport_fraction)
        frac = max(0.0, frac)
        if cfg.semantics == "stance":
            # The fraction is a per-epoch stance; reported income is
            # derived once at epoch close as gross * (1 - fraction),
            # so the hidden share cannot depend on action ordering.
            self._misreport_fractions[worker.agent_id] = frac
            return GTBEvent(
                event_type="misreport", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"underreport_fraction": frac, "semantics": "stance"},
            )

        # Legacy event semantics: rewrite reported income at this instant
        hidden = worker.gross_income_this_epoch * frac
        worker.reported_income_this_epoch = worker.gross_income_this_epoch - hidden
        # Track fraction so future income (house income) also applies it
        self._misreport_fractions[worker.agent_id] = frac

        return GTBEvent(
            event_type="misreport", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"underreport_fraction": frac, "hidden_income": hidden},
        )

    # ------------------------------------------------------------------
    # House income distribution
    # ------------------------------------------------------------------

    def _distribute_house_income(self) -> List[GTBEvent]:
        events = []
        if not self._houses:
            return events

        # treasury mode: payouts come out of collected taxes/fines instead
        # of being minted; pro-rate when the treasury can't cover demand.
        treasury_mode = self._config.build.house_income_mode == "treasury"
        scale = 1.0
        if treasury_mode:
            demand = sum(
                h.income_per_step for h in self._houses
                if h.owner_id in self._workers
            )
            scale = min(1.0, self._treasury / demand) if demand > 0 else 0.0
            if scale <= 0:
                return events

        for house in self._houses:
            worker = self._workers.get(house.owner_id)
            if worker is None:
                continue
            income = house.income_per_step * scale
            worker.add_resource(ResourceType.COIN, income)
            if treasury_mode:
                self._treasury -= income
                self._ledger_mint("house_income_treasury", income)
            else:
                self._ledger_mint("house_income", income)
            self._coin_earned_this_epoch[house.owner_id] = (
                self._coin_earned_this_epoch.get(house.owner_id, 0.0) + income
            )
            worker.gross_income_this_epoch += income
            # Legacy event semantics: the underreport fraction applies to
            # all income including house income (prevents partial undo).
            # In stance mode reported income is derived at epoch close,
            # so it accrues at full value here.
            if self._config.misreporting.semantics == "stance":
                misreport_frac = 0.0
            else:
                misreport_frac = self._misreport_fractions.get(worker.agent_id, 0.0)
            reported_portion = income * (1.0 - misreport_frac)
            worker.reported_income_this_epoch += reported_portion
            worker.cumulative_income += income
            events.append(GTBEvent(
                event_type="house_income", step=self._current_step,
                epoch=self._current_epoch, agent_id=house.owner_id,
                details={"income": income, "reported": reported_portion,
                          "house_pos": house.position},
            ))
        return events

    # ------------------------------------------------------------------
    # Market matching
    # ------------------------------------------------------------------

    def _match_market_orders(self) -> List[GTBEvent]:
        """Simple centralized market matching: match buy and sell orders."""
        events: List[GTBEvent] = []
        fee_rate = self._config.market.transaction_fee_rate

        # Group by resource type
        for rtype in ResourceType:
            if rtype == ResourceType.COIN:
                continue
            buys = sorted(
                [o for o in self._buy_orders if o.resource_type == rtype],
                key=lambda o: -o.price_per_unit,
            )
            sells = sorted(
                [o for o in self._sell_orders if o.resource_type == rtype],
                key=lambda o: o.price_per_unit,
            )

            # For each buyer (best price first), match against the cheapest
            # crossable sellers, skipping self-orders without consuming them
            # for other buyers (bd-pfc — the old two-pointer stranded a
            # seller when it self-matched the current buyer).
            for buy in buys:
                buyer = self._workers.get(buy.agent_id)
                if buyer is None:
                    continue
                for sell in sells:
                    if buy.quantity <= 0:
                        break
                    if sell.quantity <= 0:
                        continue
                    if buy.agent_id == sell.agent_id:
                        continue
                    if buy.price_per_unit < sell.price_per_unit:
                        break  # sells sorted ascending: none cheaper ahead
                    seller = self._workers.get(sell.agent_id)
                    if seller is None:
                        continue

                    qty = min(buy.quantity, sell.quantity)
                    price = (buy.price_per_unit + sell.price_per_unit) / 2.0
                    total = qty * price
                    fee = total * fee_rate

                    persistent = self._config.market.order_ttl_steps > 0
                    if persistent:
                        # Coin and resource were escrowed at post time; the
                        # midpoint price <= bid, so escrow always covers it.
                        buy.escrowed_coin -= total + fee
                        seller.add_resource(ResourceType.COIN, total - fee)
                    else:
                        # Legacy same-tick book: check balances at match time.
                        # The cheapest available seller is first; if the buyer
                        # can't afford it, it can't afford a pricier one.
                        if buyer.get_resource(ResourceType.COIN) < total + fee:
                            break
                        if seller.get_resource(rtype) < qty:
                            continue
                        buyer.remove_resource(ResourceType.COIN, total + fee)
                        seller.add_resource(ResourceType.COIN, total - fee)
                        seller.remove_resource(rtype, qty)
                    # Buyer pays total+fee, seller receives total-fee: both
                    # fee halves leave circulation.
                    self._ledger_burn("trade_fees", 2.0 * fee)
                    self._coin_earned_this_epoch[sell.agent_id] = (
                        self._coin_earned_this_epoch.get(sell.agent_id, 0.0)
                        + (total - fee)
                    )
                    buyer.add_resource(rtype, qty)

                    # Price discovery info exposed in observations
                    self._last_trade_price[rtype.value] = price
                    self._volume_this_epoch[rtype.value] = (
                        self._volume_this_epoch.get(rtype.value, 0.0) + qty
                    )

                    # Trade income for seller
                    seller.gross_income_this_epoch += total - fee
                    seller.reported_income_this_epoch += total - fee
                    seller.cumulative_income += total - fee

                    if self._config.ledger_mode == "coin":
                        # Purchases are an expense: income is net coin flow.
                        # (compute_tax floors at 0 if an epoch nets negative.)
                        buyer.gross_income_this_epoch -= total + fee
                        buyer.reported_income_this_epoch -= total + fee
                        buyer.cumulative_income -= total + fee
                        # Mirror in the gap tracker so income == net coin
                        # flow closes the epoch_ledger gap to zero.
                        self._coin_earned_this_epoch[buy.agent_id] = (
                            self._coin_earned_this_epoch.get(buy.agent_id, 0.0)
                            - (total + fee)
                        )

                    buy.quantity -= qty
                    sell.quantity -= qty

                    events.append(GTBEvent(
                        event_type="trade", step=self._current_step,
                        epoch=self._current_epoch,
                        details={
                            "buyer": buy.agent_id, "seller": sell.agent_id,
                            "resource": rtype.value, "quantity": qty,
                            "price": price, "fee": fee,
                        },
                    ))

        if self._config.market.order_ttl_steps > 0:
            # Persistent book: drop filled orders, expire stale ones
            # (refunding escrow), keep the rest resting.
            events.extend(self._expire_orders())
        else:
            # Legacy: order books are wiped every step
            self._buy_orders.clear()
            self._sell_orders.clear()
        return events

    def _expire_orders(self, force_all: bool = False) -> List[GTBEvent]:
        """Remove filled/expired orders, refunding escrow to their owners.

        With force_all=True every resting order is cancelled (epoch close).
        """
        events: List[GTBEvent] = []

        def _expired(o: MarketOrder) -> bool:
            return force_all or o.quantity <= 1e-12 or self._current_step >= o.expiry_step

        kept_buys: List[MarketOrder] = []
        for o in self._buy_orders:
            if not _expired(o):
                kept_buys.append(o)
                continue
            refund = max(0.0, o.escrowed_coin)
            if refund > 0:
                worker = self._workers.get(o.agent_id)
                if worker is not None:
                    worker.add_resource(ResourceType.COIN, refund)
            if o.quantity > 1e-12:
                events.append(GTBEvent(
                    event_type="order_expired", step=self._current_step,
                    epoch=self._current_epoch, agent_id=o.agent_id,
                    details={"side": "buy", "resource": o.resource_type.value,
                              "unfilled_qty": o.quantity,
                              "escrow_refund": refund},
                ))
        self._buy_orders = kept_buys

        kept_sells: List[MarketOrder] = []
        for o in self._sell_orders:
            if not _expired(o):
                kept_sells.append(o)
                continue
            if o.quantity > 1e-12:
                worker = self._workers.get(o.agent_id)
                if worker is not None:
                    worker.add_resource(o.resource_type, o.quantity)
                events.append(GTBEvent(
                    event_type="order_expired", step=self._current_step,
                    epoch=self._current_epoch, agent_id=o.agent_id,
                    details={"side": "sell", "resource": o.resource_type.value,
                              "unfilled_qty": o.quantity},
                ))
        self._sell_orders = kept_sells
        return events

    # ------------------------------------------------------------------
    # Resource regeneration
    # ------------------------------------------------------------------

    def _regenerate_resources(self) -> None:
        for row in self._grid:
            for cell in row:
                if cell.resource is not None:
                    cell.resource.amount = min(
                        cell.resource.amount + cell.resource.regen_rate,
                        self._config.map.resource_max_amount,
                    )

    # ------------------------------------------------------------------
    # Epoch boundary: taxes, audits, income shifting resolution
    # ------------------------------------------------------------------

    def end_epoch(self) -> "EpochResult":
        """Process epoch boundary: taxes, audits, deferred income.

        Returns an EpochResult containing the events and a pre-reset
        snapshot of worker states (for metrics computation).
        """
        events: List[GTBEvent] = []

        # 0. Persistent book: cancel all resting orders and refund escrow
        # BEFORE taxes, so locked coin is available for the tax bill.
        if self._config.market.order_ttl_steps > 0:
            events.extend(self._expire_orders(force_all=True))

        # 0a. Futures: refund margin on resting orders whose settlement
        # epoch has arrived unmatched (bd-oo7), before taxes.
        if self._config.market.futures_enabled:
            events.extend(self._refund_expired_futures_orders())
            # 0a2. Cash-settle contracts reaching their settlement epoch
            # (bd-dog): release margins + transfer (spot - forward)*qty
            # short->long. Before the snapshot so settled coin is captured.
            events.extend(self._settle_futures_contracts())

        # 0b. Stance-mode misreporting: derive reported income uniformly
        # from the declared fraction, before taxes are computed on it.
        if self._config.misreporting.semantics == "stance":
            for agent_id, frac in self._misreport_fractions.items():
                worker = self._workers.get(agent_id)
                if worker is not None and frac > 0:
                    worker.reported_income_this_epoch = (
                        worker.gross_income_this_epoch * (1.0 - frac)
                    )

        # 1. Collect taxes (only what can actually be paid)
        debt_enabled = self._config.taxation.debt_enabled
        for agent_id, worker in self._workers.items():
            coin_balance = worker.get_resource(ResourceType.COIN)

            # Outstanding debt from prior epochs is collected first.
            debt_collected = 0.0
            if debt_enabled and worker.tax_debt > 0:
                debt_collected = min(worker.tax_debt, coin_balance)
                worker.remove_resource(ResourceType.COIN, debt_collected)
                self._ledger_burn("debt_payments", debt_collected)
                self._treasury += debt_collected
                worker.tax_debt -= debt_collected
                coin_balance -= debt_collected

            tax = self._tax_schedule.compute_tax(worker.reported_income_this_epoch)
            actual_tax = min(tax, coin_balance)
            worker.remove_resource(ResourceType.COIN, actual_tax)
            self._ledger_burn("taxes", actual_tax)
            self._treasury += actual_tax
            worker.tax_paid_this_epoch = actual_tax
            shortfall = tax - actual_tax
            if debt_enabled and shortfall > 0:
                worker.tax_debt += shortfall
            events.append(GTBEvent(
                event_type="tax", epoch=self._current_epoch,
                agent_id=agent_id,
                details={
                    "gross_income": worker.gross_income_this_epoch,
                    "reported_income": worker.reported_income_this_epoch,
                    "tax_owed": tax,
                    "tax_paid": actual_tax,
                    "shortfall": shortfall,
                    "debt_collected": debt_collected,
                    "debt_outstanding": worker.tax_debt,
                    "effective_rate": actual_tax / max(worker.reported_income_this_epoch, 1e-9),
                },
            ))

        # 2. Audits
        audit_events = self._run_audits()
        events.extend(audit_events)

        # 2b. Redistribute the treasury lump-sum (AI Economist semantics:
        # collected revenue returns equally to all workers). Transfers are
        # not taxable income.
        if (self._config.taxation.redistribution == "lump_sum"
                and self._treasury > 0 and self._workers):
            share = self._treasury / len(self._workers)
            for agent_id, worker in self._workers.items():
                worker.add_resource(ResourceType.COIN, share)
                events.append(GTBEvent(
                    event_type="redistribution", epoch=self._current_epoch,
                    agent_id=agent_id, details={"amount": share},
                ))
            self._ledger_mint("redistribution", self._treasury)
            self._treasury = 0.0

        # 3. Unfreeze agents whose freeze expired
        to_unfreeze = [
            aid for aid, unfreeze_epoch in self._frozen_agents.items()
            if self._current_epoch >= unfreeze_epoch
        ]
        for aid in to_unfreeze:
            del self._frozen_agents[aid]
            events.append(GTBEvent(
                event_type="unfreeze", epoch=self._current_epoch,
                agent_id=aid,
            ))

        # Ledger coherence report. Two distinct checks:
        #   - coin conservation (hard invariant): worker coin == minted - burned
        #   - income/coin gap (known issue, tracked for the Phase 1 ledger
        #     rework): gathering and deferred-income carry-ins book gross
        #     income with no corresponding coin inflow, so this gap is
        #     expected to be positive until income == net coin flow.
        income_coin_gap = sum(
            w.gross_income_this_epoch
            - self._coin_earned_this_epoch.get(aid, 0.0)
            for aid, w in self._workers.items()
        )
        ledger = self.coin_ledger()
        events.append(GTBEvent(
            event_type="epoch_ledger", epoch=self._current_epoch,
            details={
                "income_coin_gap": income_coin_gap,
                "coin_conserved": abs(ledger["discrepancy"]) <= 1e-6,
                "discrepancy": ledger["discrepancy"],
                "minted": ledger["minted"],
                "burned": ledger["burned"],
                "treasury": self._treasury,
                "total_tax_debt": sum(
                    w.tax_debt for w in self._workers.values()
                ),
            },
        ))
        if abs(ledger["discrepancy"]) > 1e-6:
            logger.warning(
                "GTB coin conservation violated at epoch %d: %s",
                self._current_epoch, ledger,
            )

        # Snapshot worker state AFTER taxes/audits but BEFORE reset.
        # This gives metrics the accurate post-tax, pre-reset view.
        snapshot = self.snapshot_epoch_data()

        # 4. Reset epoch accumulators and materialize deferred income
        for worker in self._workers.values():
            deferred = worker.deferred_income
            worker.reset_epoch()
            # Carry deferred income as taxable income in the new epoch
            worker.gross_income_this_epoch += deferred
            worker.reported_income_this_epoch += deferred
            worker.deferred_income = 0.0
            worker.energy = worker.max_energy

        # Clear per-epoch misreport fractions
        self._misreport_fractions.clear()
        self._coin_earned_this_epoch.clear()
        self._volume_this_epoch.clear()
        self._ask_history.clear()

        self._current_epoch += 1
        self._current_step = 0

        # Trim action traces to detection window
        window = self._config.collusion.detection_window_steps
        for aid in self._action_traces:
            self._action_traces[aid] = self._action_traces[aid][-window:]

        return EpochResult(events=events, snapshot=snapshot)

    def _apply_audit_catch(self, agent_id: str, worker: WorkerState,
                            discrepancy: float) -> List[GTBEvent]:
        """Settle a successful audit: fine, debt, freeze. Shared by both
        selection modes."""
        cfg = self._config.misreporting
        events: List[GTBEvent] = []
        evaded_tax = self._tax_schedule.compute_tax(
            worker.gross_income_this_epoch
        ) - self._tax_schedule.compute_tax(worker.reported_income_this_epoch)
        fine_owed = evaded_tax * cfg.fine_multiplier
        coin_balance = worker.get_resource(ResourceType.COIN)
        fine_paid = min(fine_owed, coin_balance)
        worker.remove_resource(ResourceType.COIN, fine_paid)
        self._ledger_burn("fines", fine_paid)
        self._treasury += fine_paid
        if self._config.taxation.debt_enabled and fine_owed > fine_paid:
            worker.tax_debt += fine_owed - fine_paid
        worker.times_caught += 1
        worker.total_fines += fine_paid

        events.append(GTBEvent(
            event_type="audit_caught", epoch=self._current_epoch,
            agent_id=agent_id,
            details={
                "discrepancy": discrepancy,
                "evaded_tax": evaded_tax,
                "fine_owed": fine_owed,
                "fine": fine_paid,
                "shortfall": fine_owed - fine_paid,
                "times_caught": worker.times_caught,
            },
        ))

        if cfg.freeze_on_repeat and worker.times_caught >= cfg.freeze_after_n_catches:
            self._frozen_agents[agent_id] = (
                self._current_epoch + cfg.freeze_duration_epochs
            )
            events.append(GTBEvent(
                event_type="freeze", epoch=self._current_epoch,
                agent_id=agent_id,
                details={"until_epoch": self._current_epoch + cfg.freeze_duration_epochs},
            ))
        return events

    def _run_audits_observable(self) -> List[GTBEvent]:
        """Observable-information audit pipeline (selection_mode=observable).

        The tax authority risk-scores only what it can see: reported
        income vs coin verifiably received this epoch (house payouts and
        market sale proceeds — both on public records). Income from
        unobserved activity (gathering, in legacy ledger mode) generates
        no suspicion, so evasion on it is genuinely hard to catch.
        Anyone can be selected; audits of honest workers surface as
        audit_false_positive, and audits of misreporters succeed only
        with probability detection_power (audit_miss otherwise).
        """
        cfg = self._config.misreporting
        events: List[GTBEvent] = []
        for agent_id, worker in self._workers.items():
            observable = self._coin_earned_this_epoch.get(agent_id, 0.0)
            reported = worker.reported_income_this_epoch
            if observable > 1e-9:
                suspicion = max(0.0, (observable - reported) / observable)
            else:
                suspicion = 0.0
            audit_prob = cfg.audit_probability * (
                1.0 + cfg.risk_based_audit_multiplier * suspicion
            )
            collusion_boost = self._collusion_audit_boost.get(agent_id, 0.0)
            if collusion_boost > 0:
                audit_prob += collusion_boost * cfg.audit_probability
            audit_prob = min(1.0, audit_prob)

            if self._rng.random() >= audit_prob:
                continue
            worker.times_audited += 1

            discrepancy = (
                worker.gross_income_this_epoch - worker.reported_income_this_epoch
            )
            if discrepancy < 1e-9:
                events.append(GTBEvent(
                    event_type="audit_false_positive",
                    epoch=self._current_epoch, agent_id=agent_id,
                    details={"suspicion": suspicion,
                              "collusion_boost": collusion_boost},
                ))
                continue
            if self._rng.random() >= cfg.detection_power:
                events.append(GTBEvent(
                    event_type="audit_miss", epoch=self._current_epoch,
                    agent_id=agent_id,
                    details={"discrepancy": discrepancy,
                              "detection_power": cfg.detection_power},
                ))
                continue
            events.extend(self._apply_audit_catch(agent_id, worker, discrepancy))
        return events

    def _run_audits(self) -> List[GTBEvent]:
        """Run audit pipeline for the current epoch."""
        cfg = self._config.misreporting
        if not cfg.enabled:
            return []
        if cfg.selection_mode == "observable":
            return self._run_audits_observable()

        events: List[GTBEvent] = []
        for agent_id, worker in self._workers.items():
            discrepancy = worker.gross_income_this_epoch - worker.reported_income_this_epoch
            if discrepancy < 1e-9:
                continue  # Nothing to catch

            # Compute audit probability (base + risk-based + collusion boost)
            audit_prob = cfg.audit_probability
            if discrepancy > 0:
                ratio = discrepancy / max(worker.gross_income_this_epoch, 1e-9)
                audit_prob = min(1.0, audit_prob * (1.0 + ratio * cfg.risk_based_audit_multiplier))
            # Apply collusion-triggered audit boost
            collusion_boost = self._collusion_audit_boost.get(agent_id, 0.0)
            if collusion_boost > 0:
                audit_prob = min(1.0, audit_prob + collusion_boost * cfg.audit_probability)

            # Sample audit SELECTION first. If not selected, this worker is
            # never observed — no event, no times_audited increment, no
            # enforcement cost. Codex review noted the prior code
            # incremented times_audited for every discrepant worker and
            # only used audit_prob to decide caught-vs-miss, which made
            # total_audits identical to the misreporter count regardless
            # of audit_probability and broke experiments that tune the
            # audit rate.
            if self._rng.random() >= audit_prob:
                continue
            worker.times_audited += 1

            # Selected for audit: discrepancy is observed -> always caught.
            events.extend(self._apply_audit_catch(agent_id, worker, discrepancy))

        # False-positive pass: audit honest agents flagged by collusion boost.
        # These agents have no actual discrepancy, so they can never be "caught",
        # but selecting them for audit is itself a governance backfire.
        for agent_id, worker in self._workers.items():
            discrepancy = worker.gross_income_this_epoch - worker.reported_income_this_epoch
            if discrepancy >= 1e-9:
                continue  # Already handled in the misreporter pass above
            collusion_boost = self._collusion_audit_boost.get(agent_id, 0.0)
            if collusion_boost <= 0:
                continue  # Not flagged; skip
            fp_audit_prob = min(1.0, collusion_boost * cfg.audit_probability)
            if self._rng.random() < fp_audit_prob:
                worker.times_audited += 1
                events.append(GTBEvent(
                    event_type="audit_false_positive",
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                    details={"collusion_boost": collusion_boost},
                ))

        return events

    # ------------------------------------------------------------------
    # Collusion detection
    # ------------------------------------------------------------------

    def _apply_collusion_response(self, aid_a: str, aid_b: str) -> List[str]:
        """Apply configured responses (audit boost, trade restriction) to a
        flagged pair. Returns the list of responses applied."""
        cfg = self._config.collusion
        responses_applied: List[str] = []
        for aid in (aid_a, aid_b):
            current_extra = self._collusion_audit_boost.get(aid, 0.0)
            boost = cfg.response_audit_multiplier - 1.0
            self._collusion_audit_boost[aid] = min(current_extra + boost, 5.0)
            responses_applied.append("audit_boost")
        if cfg.response_trade_restriction_epochs > 0:
            restrict_until = (
                self._current_epoch + cfg.response_trade_restriction_epochs
            )
            for aid in (aid_a, aid_b):
                self._trade_restricted[aid] = max(
                    self._trade_restricted.get(aid, 0), restrict_until,
                )
            responses_applied.append("trade_restriction")
        return responses_applied

    def _detect_price_fixing(self) -> List[GTBEvent]:
        """Market-based collusion detector: flags pairs that each posted
        several sell orders this epoch at near-identical prices — the
        signature of a price-fixing cartel. Works from public market
        records only, so it can be wrong in both directions: independent
        traders can quote alike (false positive), cartels quoting with
        jitter slip through (false negative). Detector quality is
        measured downstream as precision/recall against true coalition
        labels.
        """
        cfg = self._config.collusion
        events: List[GTBEvent] = []
        candidates = {
            aid: prices for aid, prices in self._ask_history.items()
            if len(prices) >= cfg.price_fixing_min_asks
        }
        aids = sorted(candidates)
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                aid_a, aid_b = aids[i], aids[j]
                mean_a = sum(candidates[aid_a]) / len(candidates[aid_a])
                mean_b = sum(candidates[aid_b]) / len(candidates[aid_b])
                denom = max(mean_a, mean_b, 1e-9)
                rel_gap = abs(mean_a - mean_b) / denom
                if rel_gap > cfg.price_fixing_price_tolerance:
                    continue
                worker_a = self._workers.get(aid_a)
                worker_b = self._workers.get(aid_b)
                if worker_a is None or worker_b is None:
                    continue
                same_coalition = (
                    worker_a.coalition_id is not None
                    and worker_a.coalition_id == worker_b.coalition_id
                )
                responses = self._apply_collusion_response(aid_a, aid_b)
                events.append(GTBEvent(
                    event_type="collusion_detected",
                    epoch=self._current_epoch,
                    details={
                        "agents": [aid_a, aid_b],
                        "method": "price_fixing",
                        "mean_ask_a": mean_a,
                        "mean_ask_b": mean_b,
                        "relative_gap": rel_gap,
                        "suspicion_score": 1.0 - rel_gap,
                        "same_coalition": same_coalition,
                        "responses": responses,
                    },
                ))
        return events

    def detect_collusion(self) -> List[GTBEvent]:
        """Detect potential collusion among workers.

        Two detectors:
          - action-trace similarity over a rolling window (legacy)
          - price-fixing signature on posted asks (collusion.detect_price_fixing)

        Returns:
            List of collusion detection events.
        """
        cfg = self._config.collusion
        if not cfg.enabled:
            return []

        events: List[GTBEvent] = []
        if cfg.detect_price_fixing:
            events.extend(self._detect_price_fixing())
        agent_ids = list(self._action_traces.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                aid_a = agent_ids[i]
                aid_b = agent_ids[j]
                trace_a = self._action_traces[aid_a]
                trace_b = self._action_traces[aid_b]

                if not trace_a or not trace_b:
                    continue

                # Compute Jaccard similarity on action distribution
                min_len = min(len(trace_a), len(trace_b))
                if min_len == 0:
                    continue

                matches = sum(
                    1 for a, b in zip(trace_a[-min_len:], trace_b[-min_len:], strict=False)
                    if a == b
                )
                similarity = matches / min_len

                if similarity >= cfg.similarity_threshold:
                    # Compute suspicion score
                    worker_a = self._workers[aid_a]
                    worker_b = self._workers[aid_b]
                    same_coalition = (
                        worker_a.coalition_id is not None
                        and worker_a.coalition_id == worker_b.coalition_id
                    )
                    suspicion = similarity
                    if same_coalition:
                        suspicion = min(1.0, suspicion * 1.3)

                    if suspicion >= cfg.suspicion_score_threshold:
                        responses_applied = self._apply_collusion_response(
                            aid_a, aid_b,
                        )
                        events.append(GTBEvent(
                            event_type="collusion_detected",
                            epoch=self._current_epoch,
                            details={
                                "agents": [aid_a, aid_b],
                                "method": "action_trace",
                                "similarity": similarity,
                                "suspicion_score": suspicion,
                                "same_coalition": same_coalition,
                                "responses": responses_applied,
                            },
                        ))

        return events

    # ------------------------------------------------------------------
    # Coin ledger
    # ------------------------------------------------------------------

    def _ledger_mint(self, source: str, amount: float) -> None:
        self._coin_minted[source] = self._coin_minted.get(source, 0.0) + amount

    def _ledger_burn(self, sink: str, amount: float) -> None:
        self._coin_burned[sink] = self._coin_burned.get(sink, 0.0) + amount

    def register_external_coin(self, delta: float, source: str) -> None:
        """Record a coin change made outside the env (e.g. prediction-market
        stake escrow/payouts in the world service) so conservation checks
        stay balanced. Positive delta = coin added to a worker, negative =
        coin removed."""
        if delta >= 0:
            self._ledger_mint(source, delta)
        else:
            self._ledger_burn(source, -delta)

    def coin_ledger(self) -> Dict[str, Any]:
        """Itemized mint/burn totals and the conservation discrepancy.

        discrepancy == 0 (within float tolerance) means every coin held by
        workers is accounted for by an itemized mint minus an itemized burn.
        """
        total_coin = sum(
            w.get_resource(ResourceType.COIN) for w in self._workers.values()
        )
        # Coin locked in resting buy orders is still worker-owned
        escrowed = sum(max(0.0, o.escrowed_coin) for o in self._buy_orders)
        total_coin += escrowed
        minted = sum(self._coin_minted.values())
        burned = sum(self._coin_burned.values())
        return {
            "total_worker_coin": total_coin,
            "escrowed_coin": escrowed,
            "minted": dict(self._coin_minted),
            "burned": dict(self._coin_burned),
            "minted_total": minted,
            "burned_total": burned,
            "discrepancy": total_coin - (minted - burned),
        }

    def verify_coin_conservation(self, tolerance: float = 1e-6) -> bool:
        return abs(self.coin_ledger()["discrepancy"]) <= tolerance

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def snapshot_epoch_data(self) -> Dict[str, "WorkerState"]:
        """Create a snapshot of per-epoch worker metrics before reset.

        Returns a dict of lightweight WorkerState copies with the
        per-epoch fields preserved. Call this BEFORE end_epoch().
        """
        import copy
        snapshot = {}
        for aid, w in self._workers.items():
            ws = copy.copy(w)
            # Shallow copy is sufficient -- we only read scalar fields
            # and the inventory dict (which won't be mutated by reset_epoch)
            ws.inventory = dict(w.inventory)
            snapshot[aid] = ws
        return snapshot

    @property
    def treasury(self) -> float:
        return self._treasury

    @property
    def tax_schedule(self) -> TaxSchedule:
        return self._tax_schedule

    @property
    def workers(self) -> Dict[str, WorkerState]:
        return dict(self._workers)

    @property
    def houses(self) -> List[House]:
        return list(self._houses)

    @property
    def events(self) -> List[GTBEvent]:
        return list(self._events)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def config(self) -> GTBConfig:
        return self._config

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Compute aggregate stats for planner observation (live workers).

        NOTE: after end_epoch() the per-epoch counters are zeroed; use
        stats_from_snapshot() with the EpochResult snapshot to give the
        planner the epoch that actually just closed.
        """
        return self._aggregate_stats(self._workers)

    def stats_from_snapshot(
        self, snapshot: Dict[str, WorkerState],
    ) -> Dict[str, float]:
        """Aggregate stats over a pre-reset epoch snapshot (see end_epoch)."""
        return self._aggregate_stats(snapshot)

    def _aggregate_stats(
        self, workers: Dict[str, WorkerState],
    ) -> Dict[str, float]:
        from worlds.gather_trade_build.metrics import compute_gini
        from worlds.gather_trade_build.reward import compute_isoelastic_utility

        ws = list(workers.values())
        incomes = [w.gross_income_this_epoch for w in ws]
        coins = [w.get_resource(ResourceType.COIN) for w in ws]
        n = len(ws) or 1

        total_income = sum(incomes)
        mean_income = total_income / n
        total_tax = sum(w.tax_paid_this_epoch for w in ws)
        total_houses = sum(w.houses_built for w in ws)

        # Wealth = coin + house replacement value (build cost as proxy)
        house_value = self._config.build.wood_cost + self._config.build.stone_cost
        wealths = [
            w.get_resource(ResourceType.COIN) + house_value * w.houses_built
            for w in ws
        ]

        # Isoelastic utility (Phase 2 preferences)
        mean_utility = (
            sum(compute_isoelastic_utility(w, self._config.utility) for w in ws) / n
        )

        # Top-tail stats for the Saez planner. z* is the income threshold
        # above which the top marginal rate applies — normally the top
        # bracket edge. But when the scenario's top bracket sits above the
        # economy's realized income scale (gridworld per-epoch incomes are
        # single digits while a top-bracket edge may be 50), NO worker ever
        # lands in the top bracket, so top_mean_income is 0 every epoch.
        # That silently breaks the Saez planner: with zm==0 the online
        # elasticity estimate never updates (it is guarded on zm>0) and the
        # inverse-elasticity rule collapses to a constant tau*=0.5 target,
        # blind to the workforce (bd-kk5). Fall back to the observed top
        # quintile so the planner always sees a real tail of >=2 earners to
        # estimate the Pareto shape and elasticity from. Divergence from the
        # vendored kernel; documented per CLAUDE.md.
        thresholds = self._tax_schedule.bracket_thresholds
        top_threshold = thresholds[-1] if thresholds else 0.0
        top_incomes = [inc for inc in incomes if inc >= top_threshold]
        if len(top_incomes) < 2 and len(incomes) >= 2:
            ordered = sorted(incomes)
            cutoff = ordered[int(0.8 * len(ordered))]  # ~top quintile edge
            data_top = [inc for inc in incomes if inc >= cutoff]
            if len(data_top) >= 2:
                top_threshold = cutoff
                top_incomes = data_top
        top_mean_income = (
            sum(top_incomes) / len(top_incomes) if top_incomes else 0.0
        )

        # Marginal social welfare weight on the top tail, for a
        # welfare-weighted Saez rate tau* = (1-g)/(1-g + a*e) instead of
        # the revenue-maximizing g=0 limit 1/(1+a*e). Under the world's
        # CRRA utility the social value of a marginal coin to worker i is
        # u'(c_i) = c_i^(-eta); g normalizes the top tail's mean weight by
        # the population mean. g -> 0 means society places ~no value on top
        # earners' marginal consumption (revenue-max, taxes hard); g -> 1
        # means it values them like the average worker (no redistributive
        # case, taxes lightly). This lets Saez optimize toward the
        # configured welfare objective rather than pure revenue (bd-5gz).
        from worlds.gather_trade_build.reward import crra_marginal
        eta = self._config.utility.eta
        top_indices = [i for i, inc in enumerate(incomes) if inc >= top_threshold]
        if len(top_indices) >= 2:
            mu = [crra_marginal(c, eta) for c in coins]
            mean_mu = (sum(mu) / n) or 1.0
            top_welfare_weight = (
                sum(mu[i] for i in top_indices) / len(top_indices)
            ) / mean_mu
        else:
            top_welfare_weight = 0.0

        return {
            "total_income": total_income,
            "mean_income": mean_income,
            "gini": compute_gini(incomes),
            "gini_wealth": compute_gini(wealths),
            "mean_utility": mean_utility,
            "total_tax_revenue": total_tax,
            "total_houses": total_houses,
            "mean_coin": sum(coins) / n,
            "n_workers": n,
            "n_frozen": len(self._frozen_agents),
            "top_threshold": top_threshold,
            "top_mean_income": top_mean_income,
            "top_welfare_weight": top_welfare_weight,
            "n_top": float(len(top_incomes)),
        }

    def compute_incomes(self) -> Dict[str, float]:
        """Return current epoch gross income per worker."""
        return {aid: w.gross_income_this_epoch for aid, w in self._workers.items()}
