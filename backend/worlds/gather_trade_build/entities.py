"""Entity definitions for the Gather-Trade-Build domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ResourceType(Enum):
    """Types of resources in the GTB world."""

    WOOD = "wood"
    STONE = "stone"
    COIN = "coin"
    # H100 GPU compute, measured in compute-hours. A first-class tradeable
    # commodity (bd ja2): "datacenter" tiles (map.compute_density) regenerate
    # H100 capacity that workers gather, spot-trade, and write dated forward
    # contracts on through the existing resource-generic futures engine.
    COMPUTE = "compute"


# Resources that can be gathered, spot-traded, and written as dated forwards
# — everything except COIN, the numéraire. Canonical display/iteration order;
# add new commodities here so per-resource market views stay in sync.
TRADEABLE_RESOURCES = (
    ResourceType.WOOD,
    ResourceType.STONE,
    ResourceType.COMPUTE,
)


class GTBActionType(Enum):
    """Actions a worker can take in the GTB world."""

    MOVE = "move"
    GATHER = "gather"
    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    BUILD = "build"
    NOOP = "noop"
    # Strategic / adversarial actions
    SHIFT_INCOME = "shift_income"
    MISREPORT = "misreport"
    # Commodity futures: go long / short a dated forward contract on a
    # resource (bd-af2 first cut). FUTURES_BUY = long (agrees to buy at the
    # forward price at settlement), FUTURES_SELL = short. See FuturesContract.
    FUTURES_BUY = "futures_buy"
    FUTURES_SELL = "futures_sell"


class Direction(Enum):
    """Movement directions on the grid."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Resource:
    """A resource tile on the grid."""

    resource_type: ResourceType
    amount: float
    position: Tuple[int, int] = (0, 0)
    regen_rate: float = 0.0  # Per-step regeneration


@dataclass
class House:
    """A built house that generates income."""

    owner_id: str
    position: Tuple[int, int] = (0, 0)
    wood_cost: float = 0.0
    stone_cost: float = 0.0
    income_per_step: float = 1.0
    build_step: int = 0


@dataclass
class MarketOrder:
    """An order in the centralized market."""

    agent_id: str
    resource_type: ResourceType
    quantity: float
    price_per_unit: float  # In coins
    is_buy: bool = True  # True=buy, False=sell
    step: int = 0
    # Persistent-book fields (market.order_ttl_steps > 0)
    expiry_step: int = 0  # step after which the order is cancelled
    escrowed_coin: float = 0.0  # buyer coin locked at post time


@dataclass
class FuturesOrder:
    """A resting futures limit order (bd-oo7).

    A worker's standing offer to go long (``is_buy``) or short a dated
    forward on ``resource_type``: ``qty`` units at ``forward_price`` per
    unit, settling at ``settlement_epoch``. ``margin`` coin is escrowed at
    post time (parallel to ``MarketOrder.escrowed_coin``) so a resting
    order can always be honored when matched. Refunded if the order is
    never matched by its settlement epoch.
    """

    agent_id: str
    resource_type: ResourceType
    qty: float
    forward_price: float
    settlement_epoch: int
    is_buy: bool  # True = long, False = short
    step: int = 0
    margin: float = 0.0  # coin escrowed at post


@dataclass
class FuturesContract:
    """A matched dated forward contract on a resource (bd-af2 first cut).

    One contract pairs a long (``long_agent_id``, agreed to buy ``qty`` of
    ``resource_type`` at ``forward_price`` per unit at ``settlement_epoch``)
    with a short (``short_agent_id``, the counterparty). Both sides escrow
    margin at match time. The first cut is cash-settled at expiry: at
    ``settlement_epoch`` the env transfers ``(spot - forward_price) * qty``
    in coin from short to long (long profits if spot rose above the locked
    forward), then releases both margins. ``status`` mirrors
    ``gtb_markets.Market``: "open" until settled.

    This dataclass is types-only (bd-e0t). The matching book (bd-oo7) mints
    these; the settlement pass (bd-dog) resolves them.
    """

    contract_id: str
    resource_type: ResourceType
    qty: float
    forward_price: float  # coins per unit, locked at match
    settlement_epoch: int
    long_agent_id: str
    short_agent_id: str
    margin_long: float = 0.0
    margin_short: float = 0.0
    status: str = "open"  # "open" | "settled"
    created_epoch: int = 0
    settled_epoch: Optional[int] = None
    settle_spot_price: Optional[float] = None  # spot ref used at settlement

    def to_dict(self) -> Dict:
        return {
            "contract_id": self.contract_id,
            "resource_type": self.resource_type.value,
            "qty": self.qty,
            "forward_price": self.forward_price,
            "settlement_epoch": self.settlement_epoch,
            "long_agent_id": self.long_agent_id,
            "short_agent_id": self.short_agent_id,
            "margin_long": self.margin_long,
            "margin_short": self.margin_short,
            "status": self.status,
            "created_epoch": self.created_epoch,
            "settled_epoch": self.settled_epoch,
            "settle_spot_price": self.settle_spot_price,
        }


@dataclass
class TradeResult:
    """Result of a completed trade."""

    buyer_id: str
    seller_id: str
    resource_type: ResourceType
    quantity: float
    price_per_unit: float
    total_price: float
    step: int = 0


@dataclass
class WorkerState:
    """State of a single worker agent in the GTB world."""

    agent_id: str
    position: Tuple[int, int] = (0, 0)

    # Inventory
    inventory: Dict[str, float] = field(default_factory=lambda: {
        ResourceType.WOOD.value: 0.0,
        ResourceType.STONE.value: 0.0,
        ResourceType.COIN.value: 0.0,
    })

    # Income tracking
    gross_income_this_epoch: float = 0.0
    reported_income_this_epoch: float = 0.0
    cumulative_income: float = 0.0
    tax_paid_this_epoch: float = 0.0

    # Income shifting (deferred income account)
    deferred_income: float = 0.0

    # Unpaid taxes/fines carried across epochs (taxation.debt_enabled)
    tax_debt: float = 0.0

    # Houses built
    houses_built: int = 0

    # Energy budget
    energy: float = 100.0
    max_energy: float = 100.0

    # Labor effort (energy actually spent on actions), for labor
    # disutility in the isoelastic utility function
    effort_this_epoch: float = 0.0
    cumulative_effort: float = 0.0

    # Skill multiplier (heterogeneous agents)
    skill_gather: float = 1.0
    skill_build: float = 1.0

    # Audit history
    times_audited: int = 0
    times_caught: int = 0
    total_fines: float = 0.0

    # Collusion
    coalition_id: Optional[str] = None

    def get_resource(self, rtype: ResourceType) -> float:
        """Get amount of a resource."""
        return self.inventory.get(rtype.value, 0.0)

    def add_resource(self, rtype: ResourceType, amount: float) -> None:
        """Add resource to inventory."""
        key = rtype.value
        self.inventory[key] = self.inventory.get(key, 0.0) + amount

    def remove_resource(self, rtype: ResourceType, amount: float) -> bool:
        """Remove resource from inventory. Returns False if insufficient."""
        key = rtype.value
        current = self.inventory.get(key, 0.0)
        if current < amount - 1e-9:
            return False
        self.inventory[key] = max(0.0, current - amount)
        return True

    def reset_epoch(self) -> None:
        """Reset per-epoch accumulators."""
        self.gross_income_this_epoch = 0.0
        self.reported_income_this_epoch = 0.0
        self.tax_paid_this_epoch = 0.0
        self.effort_this_epoch = 0.0


@dataclass
class GTBGridCell:
    """A single cell in the GTB grid."""

    position: Tuple[int, int]
    resource: Optional[Resource] = None
    house: Optional[House] = None
    occupants: List[str] = field(default_factory=list)


@dataclass
class GTBEvent:
    """An event in the GTB simulation for logging."""

    event_type: str  # gather, trade, build, tax, audit, shift, misreport, collusion
    step: int = 0
    epoch: int = 0
    agent_id: str = ""
    details: Dict = field(default_factory=dict)
