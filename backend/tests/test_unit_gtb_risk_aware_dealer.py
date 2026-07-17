"""Unit + integration tests for RiskAwareDealerPolicy and the §7.3 obs (bd rzt).

The dealer quotes a SKU forward priced off its inventory and hedges the basket,
running through the real order book. These pin:

  * SKU-aware futures curve + own-position obs (the §7.3 enablers),
  * the dealer quotes two-sided SKU forwards at ±(base spread) when flat,
  * its half-spread widens with warehoused net inventory,
  * it hedges net SKU inventory on the generic basket book (short basket when
    net-long SKU), toward −hedge_ratio·net,
  * end-to-end: a principal crossing the dealer's SKU quote leaves the dealer
    with a net SKU position that it then offsets on the basket book.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.agents import RiskAwareDealerPolicy  # noqa: E402
from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType, ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402


def _obs(step, own=None, curve=None, last=4.0, epoch=0):
    return {"step": step, "epoch": epoch, "frozen": False,
            "market_info": {"compute": {"last_price": last}},
            "own_futures_position": own or {},
            "futures_curve": curve or {}}


def _dealer(**kw):
    return RiskAwareDealerPolicy("d", seed=0, sku="a", horizon=3,
                                 base_spread=0.05, risk_factor=0.02,
                                 hedge_ratio=1.0, **{"qty": 2.0, **kw})


def test_quotes_two_sided_sku_when_flat():
    d = _dealer()
    bid = d.decide(_obs(0))                      # phase 0 -> bid
    ask = d.decide(_obs(1))                      # phase 1 -> ask
    assert bid.action_type == GTBActionType.FUTURES_BUY and bid.sku == "a"
    assert ask.action_type == GTBActionType.FUTURES_SELL and ask.sku == "a"
    assert abs(bid.price - 4.0 * 0.95) < 1e-9    # anchor·(1−base_spread)
    assert abs(ask.price - 4.0 * 1.05) < 1e-9
    assert bid.settlement_epoch == 3             # epoch + horizon


def test_spread_widens_with_inventory():
    d = _dealer()
    flat = d.decide(_obs(0))
    heavy = d.decide(_obs(0, own={"compute/a@3": 5.0}))   # net long 5
    # spread = base + risk·|net| = 0.05 + 0.02·5 = 0.15 -> lower bid
    assert heavy.price < flat.price
    assert abs(heavy.price - 4.0 * (1.0 - 0.15)) < 1e-9


def test_hedges_net_long_sku_by_shorting_basket():
    d = _dealer()
    act = d.decide(_obs(2, own={"compute/a@3": 4.0}))     # phase 2, net long SKU
    assert act.action_type == GTBActionType.FUTURES_SELL  # short the basket
    assert act.sku == ""                                  # generic basket book
    assert act.resource_type == ResourceType.COMPUTE
    assert act.quantity > 0 and act.settlement_epoch == 3


def test_no_hedge_when_flat_falls_through_to_quote():
    d = _dealer()
    act = d.decide(_obs(2, own={}))              # flat -> nothing to hedge
    assert act.sku == "a"                        # quotes the SKU instead


def test_obs_exposes_sku_curve_and_own_position():
    cfg = GTBConfig.from_dict({"map": {"height": 3, "width": 3,
                                       "compute_density": 1.0}})
    cfg.seed = 0
    env = GTBEnvironment(cfg)
    for n in ("long", "short"):
        env.add_worker(n)
        env._workers[n].add_resource(ResourceType.COIN, 100_000.0)
    for a, t in (("long", GTBActionType.FUTURES_BUY),
                 ("short", GTBActionType.FUTURES_SELL)):
        env.apply_actions({a: GTBAction(agent_id=a, action_type=t,
            resource_type=ResourceType.COMPUTE, quantity=3.0, price=4.0,
            settlement_epoch=2, sku="a")})
    curve = env.obs("long")["futures_curve"]
    assert "compute/a@2" in curve and curve["compute/a@2"]["sku"] == "a"
    assert env.obs("long")["own_futures_position"]["compute/a@2"] == 3.0    # long
    assert env.obs("short")["own_futures_position"]["compute/a@2"] == -3.0  # short


def test_dealer_ends_up_hedging_a_real_sku_fill():
    """End-to-end: a principal lifts the dealer's SKU ask (goes long SKU), so
    the dealer is left net-short the SKU and then buys the basket to hedge."""
    cfg = GTBConfig.from_dict({"map": {"height": 3, "width": 3,
                                       "compute_density": 1.0}})
    cfg.seed = 0
    env = GTBEnvironment(cfg)
    for n in ("dealer", "principal", "cp"):
        env.add_worker(n)
        env._workers[n].add_resource(ResourceType.COIN, 100_000.0)
    env._last_trade_price[ResourceType.COMPUTE.value] = 4.0
    dealer = _dealer(qty=3.0)

    got_basket_hedge = False
    for step in range(6):
        env._current_step = step
        obs = env.obs("dealer")
        act = dealer.decide(obs)
        env.apply_actions({"dealer": act})
        # Principal crosses the dealer's resting SKU ask -> goes long the SKU.
        curve = env.obs("principal")["futures_curve"].get("compute/a@3") or {}
        if curve.get("best_ask") is not None:
            env.apply_actions({"principal": GTBAction(agent_id="principal",
                action_type=GTBActionType.FUTURES_BUY,
                resource_type=ResourceType.COMPUTE, quantity=3.0,
                price=curve["best_ask"], settlement_epoch=3, sku="a")})
        # cp provides the other side of the dealer's basket hedge.
        bcurve = env.obs("cp")["futures_curve"].get("compute@3") or {}
        if bcurve.get("best_bid") is not None:
            env.apply_actions({"cp": GTBAction(agent_id="cp",
                action_type=GTBActionType.FUTURES_SELL,
                resource_type=ResourceType.COMPUTE, quantity=3.0,
                price=bcurve["best_bid"], settlement_epoch=3)})
        pos = env.obs("dealer")["own_futures_position"]
        if pos.get("compute@3", 0.0) > 0:        # dealer went long the basket
            got_basket_hedge = True

    pos = env.obs("dealer")["own_futures_position"]
    assert pos.get("compute/a@3", 0.0) < 0       # net short the SKU (sold to principal)
    assert got_basket_hedge                       # hedged by buying the basket
