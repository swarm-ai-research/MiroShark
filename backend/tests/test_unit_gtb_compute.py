"""Unit tests for H100 compute as a first-class kernel resource (bd ja2).

ResourceType.COMPUTE is a tradeable commodity: "datacenter" tiles
(map.compute_density) regenerate compute-hours that workers gather and
spot-trade, and the resource-generic futures engine writes/settles dated
forwards on it. These tests pin:

  * the resource + its presence in the per-resource market views,
  * compute_density grid placement (and the 0.0 default → no perturbation
    of existing wood/stone scenarios),
  * a COMPUTE spot trade printing a last price (price discovery),
  * a COMPUTE futures contract cash-settling zero-sum with no negative
    balances — the same invariants the wood/stone futures tests assert.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
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
    TRADEABLE_RESOURCES,
    GTBActionType,
    ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402


def _env(compute_density=0.0, **market):
    cfg = GTBConfig.from_dict({
        "map": {"height": 4, "width": 4, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": compute_density},
        "market": market,
    })
    env = GTBEnvironment(cfg)
    for aid in ("w0", "w1", "w2"):
        env.add_worker(aid)  # 10 coin endowment each
    return env


def _coin(env, aid):
    return env._workers[aid].get_resource(ResourceType.COIN)


def _total_coin(env):
    return sum(w.get_resource(ResourceType.COIN) for w in env._workers.values())


def _fut(agent, side, qty, price, settle, resource=ResourceType.COMPUTE):
    at = GTBActionType.FUTURES_BUY if side == "long" else GTBActionType.FUTURES_SELL
    return GTBAction(agent_id=agent, action_type=at, resource_type=resource,
                     quantity=qty, price=price, settlement_epoch=settle)


# ----------------------------------------------------------------------
# Resource identity + market-view coverage
# ----------------------------------------------------------------------

def test_compute_resource_exists_and_is_tradeable():
    assert ResourceType.COMPUTE.value == "compute"
    assert ResourceType.COMPUTE in TRADEABLE_RESOURCES
    assert ResourceType.COIN not in TRADEABLE_RESOURCES


def test_market_views_include_compute():
    env = _env()
    assert "compute" in env._market_info()
    assert "compute" in env._market_book_snapshot()


# ----------------------------------------------------------------------
# Grid placement (compute_density) + the 0.0-default invariant
# ----------------------------------------------------------------------

def _compute_tiles(env):
    return [
        cell
        for row in env._grid
        for cell in row
        if cell.resource is not None
        and cell.resource.resource_type == ResourceType.COMPUTE
    ]


def test_compute_density_zero_places_no_datacenter_tiles():
    # The default — guarantees existing wood/stone scenarios are unperturbed.
    env = _env(compute_density=0.0)
    assert _compute_tiles(env) == []


def test_compute_density_seeds_datacenter_tiles():
    env = _env(compute_density=1.0)  # every tile is a datacenter
    tiles = _compute_tiles(env)
    assert len(tiles) == 4 * 4
    assert all(t.resource.amount > 0 for t in tiles)


# ----------------------------------------------------------------------
# Spot price discovery
# ----------------------------------------------------------------------

def test_compute_spot_trade_prints_last_price():
    env = _env()
    # Seed the seller with compute-hours to sell; buyer pays in coin.
    env._workers["w0"].add_resource(ResourceType.COMPUTE, 5.0)
    env.apply_actions({
        "w0": GTBAction(agent_id="w0", action_type=GTBActionType.TRADE_SELL,
                        resource_type=ResourceType.COMPUTE, quantity=2.0,
                        price=1.0),
        "w1": GTBAction(agent_id="w1", action_type=GTBActionType.TRADE_BUY,
                        resource_type=ResourceType.COMPUTE, quantity=2.0,
                        price=1.0),
    })
    assert env._last_trade_price.get("compute") is not None
    assert env._market_info()["compute"]["last_price"] is not None
    # Compute moved seller -> buyer.
    assert env._workers["w1"].get_resource(ResourceType.COMPUTE) > 0


# ----------------------------------------------------------------------
# Futures: match + zero-sum cash settlement on COMPUTE
# ----------------------------------------------------------------------

def test_compute_futures_match_and_settle_zero_sum():
    env = _env()
    coin_before = _total_coin(env)
    env.apply_actions({
        "w0": _fut("w0", "long", 2.0, 1.2, 1),
        "w1": _fut("w1", "short", 2.0, 1.0, 1),
    })
    assert len(env._futures_contracts) == 1
    c = env._futures_contracts[0]
    assert c.resource_type == ResourceType.COMPUTE

    # Spot rises above the locked forward (1.1 midpoint) -> long profits.
    env._last_trade_price["compute"] = 2.0
    env._current_epoch = 1
    env.end_epoch()

    c = env._futures_contracts[0]
    assert c.status == "settled"
    assert c.settle_spot_price == 2.0
    # Cash settlement is a pure transfer: total coin conserved, no negatives.
    assert _total_coin(env) == pytest.approx(coin_before)
    assert all(_coin(env, a) >= 0 for a in ("w0", "w1", "w2"))
    # Long's gain equals short's loss (zero-sum).
    long_pnl = env._workers["w0"].cumulative_income
    short_pnl = env._workers["w1"].cumulative_income
    assert long_pnl == pytest.approx(-short_pnl)
    assert long_pnl > 0  # spot rose, the long won


def test_coin_conservation_holds_with_inflight_futures_margin():
    # Regression for the bd ja2 ledger fix: the conservation diagnostic must
    # count in-flight futures margin (resting orders + open contracts) as
    # worker-owned escrow, else it false-positives a violation.
    env = _env()
    # A matched contract (both sides' margin on the contract)...
    env.apply_actions({
        "w0": _fut("w0", "long", 2.0, 1.2, 3),
        "w1": _fut("w1", "short", 2.0, 1.0, 3),
    })
    # ...plus a resting unmatched order (margin on the order).
    env.apply_actions({"w2": _fut("w2", "long", 1.0, 0.5, 3)})
    assert env._futures_contracts and env._futures_buy_orders
    led = env.coin_ledger()
    assert led["discrepancy"] == pytest.approx(0.0, abs=1e-9)
    assert env.verify_coin_conservation()


def test_compute_futures_curve_keyed_by_compute():
    env = _env()
    env.apply_actions({
        "w0": _fut("w0", "long", 1.0, 0.9, 3),    # bid
        "w1": _fut("w1", "short", 1.0, 1.5, 3),   # ask, no cross
    })
    curve = env._futures_curve()
    assert "compute@3" in curve
    assert curve["compute@3"]["best_bid"] == 0.9
    assert curve["compute@3"]["best_ask"] == 1.5


# ----------------------------------------------------------------------
# bd 5ij: MatchedHedgerPolicy — matched sizing + through-the-book execution
# ----------------------------------------------------------------------

from worlds.gather_trade_build.agents import MatchedHedgerPolicy  # noqa: E402


def _obs(held, *, bid=None, settle=3, spot=3.0, frozen=False):
    """Minimal obs for a compute producer; optional resting forward bid."""
    curve = {}
    if bid is not None:
        curve["compute@%d" % settle] = {
            "resource": "compute", "settlement_epoch": settle,
            "best_bid": bid, "best_ask": None, "last_forward": None,
            "n_bids": 1, "n_asks": 0,
        }
    return {
        "frozen": frozen,
        "inventory": {"compute": held, "coin": 1000.0},
        "market_info": {"compute": {"last_price": spot}},
        "futures_curve": curve,
    }


def test_matched_hedger_sizes_to_exposure_and_crosses_bid():
    pol = MatchedHedgerPolicy("p", resource=ResourceType.COMPUTE,
                              hedge_ratio=1.0, lot=2.0)
    # Holds 4 compute -> target hedge 4 units, lot 2: two FUTURES_SELLs that
    # CROSS the resting bid (priced at the bid, 3.0), then stops hedging.
    a1 = pol.decide(_obs(4.0, bid=3.0))
    assert a1.action_type == GTBActionType.FUTURES_SELL
    assert a1.quantity == pytest.approx(2.0)
    assert a1.price == pytest.approx(3.0)        # hits the bid (slippage-bearing)
    assert a1.settlement_epoch == 3
    a2 = pol.decide(_obs(4.0, bid=3.0))
    assert a2.action_type == GTBActionType.FUTURES_SELL
    assert a2.quantity == pytest.approx(2.0)
    # Exposure now fully hedged (4/4) -> switches to spot-selling.
    a3 = pol.decide(_obs(4.0, bid=3.0))
    assert a3.action_type == GTBActionType.TRADE_SELL
    assert a3.resource_type == ResourceType.COMPUTE


def test_matched_hedger_zero_ratio_is_active_unhedged_control():
    # hedge_ratio=0: never hedges, but still spot-sells (genuine exposure) —
    # the non-degenerate control bd k9w check C lacked.
    pol = MatchedHedgerPolicy("p", hedge_ratio=0.0)
    for _ in range(3):
        a = pol.decide(_obs(4.0, bid=3.0))
        assert a.action_type == GTBActionType.TRADE_SELL


def test_matched_hedger_without_book_cannot_hedge_but_still_sells():
    # No resting bid to cross -> the hedge can't execute; the producer doesn't
    # post an unmatched order, it just sells its exposure.
    pol = MatchedHedgerPolicy("p", hedge_ratio=1.0, lot=2.0)
    a = pol.decide(_obs(4.0, bid=None))
    assert a.action_type == GTBActionType.TRADE_SELL


def test_matched_hedger_fills_at_bid_through_real_engine():
    # End-to-end: a resting long bid at 2.0; the producer's matched hedge
    # crosses it and the contract prints AT the bid (midpoint of bid and a
    # sell priced at the bid) — i.e. the producer pays the spread, not its mark.
    env = _env()
    env._workers["p"] = env._workers.pop("w0")  # reuse an endowed worker as producer
    env._workers["p"].agent_id = "p"
    env._workers["p"].add_resource(ResourceType.COMPUTE, 10.0)
    # w1 rests a long bid for compute at 2.0, settle 3.
    env.apply_actions({"w1": _fut("w1", "long", 5.0, 2.0, 3)})
    pol = MatchedHedgerPolicy("p", hedge_ratio=1.0, lot=3.0)
    act = pol.decide(env.obs("p"))
    assert act.action_type == GTBActionType.FUTURES_SELL
    env.apply_actions({"p": act})
    assert len(env._futures_contracts) == 1
    c = env._futures_contracts[0]
    assert c.short_agent_id == "p" and c.long_agent_id == "w1"
    assert c.forward_price == pytest.approx(2.0)  # filled at the resting bid
