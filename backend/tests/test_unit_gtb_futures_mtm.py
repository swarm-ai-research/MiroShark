"""Unit tests for daily mark-to-market + margin calls / liquidation (bd 1z3).

The legacy futures engine cash-settles once at expiry. With
``market.futures_daily_mtm=True`` each epoch remarks open contracts to the
current spot, transfers **variation margin** long<->short from free coin, and
**liquidates** a side that cannot fund a margin call — the Pirrong funding-
liquidity risk (a position can be hedged in value yet fail on cash flow). These
pin:

  * **path independence**: absent liquidation, walking the spot over several
    epochs transfers the *same* total P&L as a single-shot settlement (MTM only
    changes the timing),
  * variation margin is a zero-sum coin transfer each epoch (conserved),
  * a well-funded short meets margin calls as spot rises; an **underfunded**
    short is liquidated — its free coin drains to the long, the contract closes
    early, and the long bears the shortfall,
  * MTM off (default) is byte-identical single-shot settlement.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType, ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402


def _env(mtm: bool, coin_dealer: float = 1_000_000.0):
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
        "market": {"futures_daily_mtm": mtm},
    })
    cfg.seed = 0
    env = GTBEnvironment(cfg)
    env.add_worker("long"); env._workers["long"].add_resource(ResourceType.COIN, 1_000_000.0)
    env.add_worker("short"); env._workers["short"].add_resource(ResourceType.COIN, coin_dealer)
    return env


def _open(env, settle_epoch=3, qty=10.0, fwd=3.0):
    for agent, atype in (("long", GTBActionType.FUTURES_BUY),
                         ("short", GTBActionType.FUTURES_SELL)):
        env.apply_actions({agent: GTBAction(
            agent_id=agent, action_type=atype,
            resource_type=ResourceType.COMPUTE, quantity=qty, price=fwd,
            settlement_epoch=settle_epoch)})
    assert len(env._futures_contracts) == 1


def _advance(env, epoch, spot):
    """Move to `epoch` with the given compute spot, run the boundary, return the
    epoch's events."""
    env._last_trade_price[ResourceType.COMPUTE.value] = spot
    env._current_epoch = epoch
    return env.end_epoch().events


def _coin(env, name):
    return env._workers[name].get_resource(ResourceType.COIN)


def test_mtm_path_independent_absent_liquidation():
    """A spot path that ends at 5.0 transfers the same total P&L under daily MTM
    (marked each epoch) as a single-shot settlement — timing differs, total
    doesn't. Long P&L = (5−3)·10 = +20."""
    single = _env(mtm=False)
    l0 = _coin(single, "long")
    _open(single)
    _advance(single, 3, 5.0)                     # settle once at expiry
    single_pnl = _coin(single, "long") - l0

    daily = _env(mtm=True)
    l0 = _coin(daily, "long")
    _open(daily)
    for ep, spot in ((1, 4.0), (2, 4.5), (3, 5.0)):   # marked each epoch
        _advance(daily, ep, spot)
    daily_pnl = _coin(daily, "long") - l0

    assert abs(single_pnl - 20.0) < 1e-9
    assert abs(daily_pnl - single_pnl) < 1e-9    # path-independent total


def test_variation_margin_is_zero_sum():
    """Each MTM epoch, the long's gain equals the short's loss — total worker
    coin is unchanged (measured post-escrow, on non-settling epochs)."""
    env = _env(mtm=True)
    _open(env)
    total = _coin(env, "long") + _coin(env, "short")   # after escrow posted
    for ep, spot in ((1, 3.7), (2, 3.2)):              # not epoch 3 (settles)
        _advance(env, ep, spot)
        assert abs((_coin(env, "long") + _coin(env, "short")) - total) < 1e-9


def test_well_funded_short_meets_margin_calls():
    env = _env(mtm=True, coin_dealer=1_000_000.0)
    l0 = _coin(env, "long")
    _open(env)
    for ep, spot in ((1, 6.0), (2, 9.0), (3, 12.0)):   # spot ramps hard against short
        _advance(env, ep, spot)
    c = env._futures_contracts[0]
    assert c.status == "settled"                 # never liquidated
    assert abs((_coin(env, "long") - l0) - (12.0 - 3.0) * 10.0) < 1e-9  # +90 total


def test_underfunded_short_is_liquidated():
    """A short with a thin coin buffer cannot fund the variation-margin call as
    spot spikes and is liquidated — free coin drains to the long, the contract
    closes early, and the long bears the shortfall (Pirrong funding risk)."""
    # short holds only 30 free coin; a jump to spot 12 owes (12−3)·10 = 90.
    env = _env(mtm=True, coin_dealer=30.0)
    _open(env, qty=10.0, fwd=3.0)
    short_free_before = _coin(env, "short")
    events = _advance(env, 1, 12.0)              # margin call 90 > 30 free
    c = env._futures_contracts[0]
    assert c.status == "liquidated" and c.settled_epoch == 1
    assert _coin(env, "short") < short_free_before  # free coin drained by the call
    liq = [e for e in events if e.event_type == "futures_liquidated"]
    assert liq and liq[-1].details["shortfall"] > 0
    # contract drops out — no further settlement at expiry.
    _advance(env, 3, 12.0)
    assert c.status == "liquidated"


def _system_coin(env):
    """All worker coin + all escrow held on open contracts — invariant absent
    mint/burn (variation margin is a transfer, liquidation returns escrow)."""
    wc = sum(w.get_resource(ResourceType.COIN) for w in env._workers.values())
    esc = sum(c.margin_long + c.margin_short for c in env._futures_contracts
              if c.status == "open")
    return wc + esc


def test_coin_conserved_across_mtm_and_liquidation():
    """No coin is created or destroyed by MTM variation margin or by a
    liquidation — worker coin + open-contract escrow stays constant."""
    env = _env(mtm=True, coin_dealer=45.0)       # thin buffer → liquidation
    base = _system_coin(env)
    _open(env)
    assert abs(_system_coin(env) - base) < 1e-9  # escrow just moved off worker coin
    for ep, spot in ((1, 5.0), (2, 20.0), (3, 20.0)):   # epoch 2 spikes → liquidation
        _advance(env, ep, spot)
        assert abs(_system_coin(env) - base) < 1e-9
    assert env._futures_contracts[0].status == "liquidated"


def test_mtm_off_is_single_shot_default():
    """Default (MTM off): intermediate epochs transfer nothing; all P&L lands at
    settlement — byte-identical legacy behavior."""
    env = _env(mtm=False)
    l0 = _coin(env, "long")
    _open(env)
    post = _coin(env, "long")                     # after escrow
    _advance(env, 1, 100.0)                       # huge intermediate spot: ignored
    assert abs(_coin(env, "long") - post) < 1e-9  # nothing transferred pre-expiry
    _advance(env, 3, 5.0)
    assert abs((_coin(env, "long") - l0) - 20.0) < 1e-9   # escrow round-trips out
