"""Unit tests for the commodity-futures types (bd-e0t, first cut of bd-af2).

bd-e0t is types-only: the FuturesContract dataclass, the FUTURES_BUY /
FUTURES_SELL action types, and the settlement_epoch field on GTBAction.
No matching or settlement behavior yet (those are bd-oo7 / bd-dog) — these
tests pin the shape the later children build on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    FuturesContract,
    GTBActionType,
    ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402


def _env(**market):
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 1.0,
                "stone_density": 0.0},
        "market": market,
    })
    env = GTBEnvironment(cfg)
    for aid in ("w0", "w1", "w2"):
        env.add_worker(aid)  # 10 coin endowment each
    return env


def _fut(agent, side, qty, price, settle, resource=ResourceType.WOOD):
    at = GTBActionType.FUTURES_BUY if side == "long" else GTBActionType.FUTURES_SELL
    return GTBAction(agent_id=agent, action_type=at, resource_type=resource,
                     quantity=qty, price=price, settlement_epoch=settle)


def _coin(env, aid):
    return env._workers[aid].get_resource(ResourceType.COIN)


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


# ----------------------------------------------------------------------
# bd-oo7: futures order book + matching
# ----------------------------------------------------------------------

def test_crossing_pair_mints_one_contract_with_margin_escrowed():
    env = _env()
    c0, c1 = _coin(env, "w0"), _coin(env, "w1")
    env.apply_actions({
        "w0": _fut("w0", "long", 2.0, 1.2, 3),
        "w1": _fut("w1", "short", 2.0, 1.0, 3),
    })
    assert len(env._futures_contracts) == 1
    c = env._futures_contracts[0]
    assert c.long_agent_id == "w0" and c.short_agent_id == "w1"
    assert c.forward_price == pytest.approx(1.1)  # midpoint of 1.2/1.0
    assert c.qty == 2.0 and c.status == "open"
    # margin escrowed at each order's posted price (0.2 * qty * price)
    assert c.margin_long == pytest.approx(0.2 * 2.0 * 1.2)
    assert c.margin_short == pytest.approx(0.2 * 2.0 * 1.0)
    assert _coin(env, "w0") == pytest.approx(c0 - 0.2 * 2.0 * 1.2)
    assert _coin(env, "w1") == pytest.approx(c1 - 0.2 * 2.0 * 1.0)
    # books cleared
    assert env._futures_buy_orders == [] and env._futures_sell_orders == []


def test_non_crossing_orders_rest_without_matching():
    env = _env()
    env.apply_actions({
        "w0": _fut("w0", "long", 1.0, 0.9, 3),   # bid below
        "w1": _fut("w1", "short", 1.0, 1.5, 3),  # ask above
    })
    assert env._futures_contracts == []
    assert len(env._futures_buy_orders) == 1
    assert len(env._futures_sell_orders) == 1
    curve = env._futures_curve()["wood@3"]
    assert curve["best_bid"] == 0.9 and curve["best_ask"] == 1.5


def test_different_settlement_epochs_do_not_cross():
    env = _env()
    env.apply_actions({
        "w0": _fut("w0", "long", 1.0, 1.5, 3),
        "w1": _fut("w1", "short", 1.0, 1.0, 5),  # different delivery date
    })
    assert env._futures_contracts == []


def test_self_match_skipped():
    env = _env()
    env.apply_actions({"w0": _fut("w0", "long", 1.0, 1.5, 3)})
    env.apply_actions({"w0": _fut("w0", "short", 1.0, 1.0, 3)})
    assert env._futures_contracts == []  # same agent can't trade with itself


def test_settlement_must_be_in_future():
    env = _env()
    evs = env.apply_actions({"w0": _fut("w0", "long", 1.0, 1.0, 0)})
    assert env._futures_buy_orders == []
    assert any(e.event_type == "futures_order_fail"
               and e.details.get("reason") == "settlement_not_in_future"
               for e in evs)


def test_insufficient_margin_rejected():
    env = _env()
    # margin = 0.2 * 100 * 1.0 = 20 > 10 endowment
    evs = env.apply_actions({"w0": _fut("w0", "long", 100.0, 1.0, 3)})
    assert env._futures_buy_orders == []
    assert _coin(env, "w0") == pytest.approx(10.0)  # nothing escrowed
    assert any(e.details.get("reason") == "insufficient_margin" for e in evs)


def test_partial_fill_leaves_residual_resting():
    env = _env()
    env.apply_actions({
        "w0": _fut("w0", "long", 3.0, 1.2, 3),
        "w1": _fut("w1", "short", 1.0, 1.0, 3),
    })
    assert len(env._futures_contracts) == 1
    assert env._futures_contracts[0].qty == 1.0
    # 2 units of the long order still rest
    assert len(env._futures_buy_orders) == 1
    assert env._futures_buy_orders[0].qty == pytest.approx(2.0)


def test_unmatched_order_margin_refunded_at_settlement_epoch():
    env = _env()
    c0 = _coin(env, "w0")
    env.apply_actions({"w0": _fut("w0", "long", 1.0, 1.0, 1)})
    assert _coin(env, "w0") < c0  # margin escrowed
    # advance to the settlement epoch unmatched -> refund at epoch close
    env._current_epoch = 1
    env.end_epoch()
    assert _coin(env, "w0") == pytest.approx(c0)  # fully refunded
    assert env._futures_buy_orders == []


def test_futures_disabled_ignores_orders():
    env = _env(futures_enabled=False)
    evs = env.apply_actions({"w0": _fut("w0", "long", 1.0, 1.0, 3)})
    assert env._futures_buy_orders == []
    assert any(e.details.get("reason") == "futures_disabled" for e in evs)


# ----------------------------------------------------------------------
# bd-dog: cash settlement at expiry
# ----------------------------------------------------------------------

def _total_coin(env):
    return sum(w.get_resource(ResourceType.COIN) for w in env._workers.values())


def _match_and_settle(env, spot, settle_epoch=1, long_px=1.2, short_px=1.0, qty=2.0):
    """Post a crossing pair, set the spot reference, settle at expiry."""
    env.apply_actions({
        "w0": _fut("w0", "long", qty, long_px, settle_epoch),
        "w1": _fut("w1", "short", qty, short_px, settle_epoch),
    })
    assert len(env._futures_contracts) == 1
    if spot is not None:
        env._last_trade_price["wood"] = spot
    env._current_epoch = settle_epoch
    env.end_epoch()
    return env._futures_contracts[0]


def test_settlement_long_profits_when_spot_above_forward():
    env = _env()
    w0b, w1b = _coin(env, "w0"), _coin(env, "w1")
    c = _match_and_settle(env, spot=2.0)  # forward midpoint 1.1
    assert c.status == "settled" and c.settled_epoch == 1
    assert c.settle_spot_price == 2.0
    pnl = (2.0 - 1.1) * 2.0  # 1.8 to long
    # each side gets its margin back; long +pnl, short -pnl
    assert _coin(env, "w0") == pytest.approx(w0b + pnl)
    assert _coin(env, "w1") == pytest.approx(w1b - pnl)


def test_settlement_short_profits_when_spot_below_forward():
    env = _env()
    w0b, w1b = _coin(env, "w0"), _coin(env, "w1")
    c = _match_and_settle(env, spot=0.5)  # below forward 1.1
    pnl = (0.5 - 1.1) * 2.0  # -1.2 (short wins)
    assert _coin(env, "w0") == pytest.approx(w0b + pnl)   # long loses
    assert _coin(env, "w1") == pytest.approx(w1b - pnl)   # short gains


def test_settlement_no_spot_reference_is_zero_pnl():
    env = _env()
    w0b, w1b = _coin(env, "w0"), _coin(env, "w1")
    c = _match_and_settle(env, spot=None)  # wood never traded spot
    assert c.settle_spot_price == pytest.approx(c.forward_price)
    assert _coin(env, "w0") == pytest.approx(w0b)  # margins back, no P&L
    assert _coin(env, "w1") == pytest.approx(w1b)


def test_settlement_conserves_total_coin():
    for spot in (0.2, 1.1, 5.0):
        env = _env()
        before = _total_coin(env)
        _match_and_settle(env, spot=spot)
        assert _total_coin(env) == pytest.approx(before), f"spot={spot}"


def test_settlement_never_drives_balance_negative():
    env = _env()
    before = _total_coin(env)  # 3 workers x 10 endowment
    # Huge adverse move for the short: spot=50 on a forward ~1.1, qty 2 ->
    # nominal P&L ~98, far beyond the short's ~10 coin. Must clamp.
    _match_and_settle(env, spot=50.0)
    assert _coin(env, "w1") >= 0.0
    assert _total_coin(env) == pytest.approx(before)  # still conserved


def test_open_futures_contracts_excludes_settled():
    env = _env()
    _match_and_settle(env, spot=1.5)
    assert env.open_futures_contracts() == []
    assert len(env._futures_contracts) == 1  # kept, marked settled
