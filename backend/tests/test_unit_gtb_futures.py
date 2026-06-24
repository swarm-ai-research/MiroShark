"""Unit tests for the commodity-futures types (bd-e0t, first cut of bd-af2).

bd-e0t is types-only: the FuturesContract dataclass, the FUTURES_BUY /
FUTURES_SELL action types, and the settlement_epoch field on GTBAction.
No matching or settlement behavior yet (those are bd-oo7 / bd-dog) — these
tests pin the shape the later children build on.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.entities import (  # noqa: E402
    FuturesContract,
    GTBActionType,
    ResourceType,
)
from worlds.gather_trade_build.env import GTBAction  # noqa: E402


def test_futures_action_types_exist():
    assert GTBActionType.FUTURES_BUY.value == "futures_buy"
    assert GTBActionType.FUTURES_SELL.value == "futures_sell"


def test_gtb_action_has_settlement_epoch_default_zero():
    a = GTBAction(action_type=GTBActionType.NOOP)
    assert a.settlement_epoch == 0  # 0 = not a futures order

    fut = GTBAction(
        agent_id="w0",
        action_type=GTBActionType.FUTURES_BUY,
        resource_type=ResourceType.WOOD,
        quantity=2.0,
        price=1.3,            # forward price per unit
        settlement_epoch=5,
    )
    assert fut.settlement_epoch == 5
    assert fut.price == 1.3
    assert fut.resource_type == ResourceType.WOOD


def test_futures_contract_construction_and_defaults():
    c = FuturesContract(
        contract_id="fc-0",
        resource_type=ResourceType.STONE,
        qty=3.0,
        forward_price=1.1,
        settlement_epoch=8,
        long_agent_id="w1",
        short_agent_id="w2",
    )
    assert c.status == "open"
    assert c.margin_long == 0.0 and c.margin_short == 0.0
    assert c.settled_epoch is None
    assert c.settle_spot_price is None


def test_futures_contract_to_dict_round_trips_fields():
    c = FuturesContract(
        contract_id="fc-1",
        resource_type=ResourceType.WOOD,
        qty=2.5,
        forward_price=1.25,
        settlement_epoch=6,
        long_agent_id="w3",
        short_agent_id="w4",
        margin_long=0.5,
        margin_short=0.5,
        created_epoch=2,
    )
    d = c.to_dict()
    assert d["contract_id"] == "fc-1"
    assert d["resource_type"] == "wood"  # serialized to the enum value
    assert d["qty"] == 2.5
    assert d["forward_price"] == 1.25
    assert d["settlement_epoch"] == 6
    assert d["long_agent_id"] == "w3" and d["short_agent_id"] == "w4"
    assert d["status"] == "open"
    assert d["settled_epoch"] is None
