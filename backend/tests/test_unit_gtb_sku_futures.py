"""Unit tests for the SKU-keyed futures book (bd umm §7 increment 1).

The vendored futures engine keys its order book on (resource, settlement_epoch).
This adds an optional ``sku`` label so only same-SKU orders are fungible — the
foundation for SKU-specific compute forwards hedged against a generic basket
(design §7). These pin:

  * a non-empty SKU partitions the book — differently-SKU'd orders at the same
    (resource, epoch) do NOT cross; same-SKU orders do,
  * sku="" (default) is byte-identical to the pre-SKU generic book,
  * a SKU contract cash-settles zero-sum, using a per-SKU spot ref when one is
    published (else the generic resource spot),
  * two SKUs on the same resource settle independently on their own spot.

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

_ENDOW = 100_000.0
_SETTLE = 2


def _env():
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
    })
    cfg.seed = 0
    env = GTBEnvironment(cfg)
    for name in ("lab", "cloud", "lab2", "cloud2"):
        env.add_worker(name)
        env._workers[name].add_resource(ResourceType.COIN, _ENDOW)
    return env


def _order(env, agent, action_type, qty, price, sku=""):
    env.apply_actions({agent: GTBAction(
        agent_id=agent, action_type=action_type,
        resource_type=ResourceType.COMPUTE, quantity=qty, price=price,
        settlement_epoch=_SETTLE, sku=sku)})


def test_sku_partitions_the_book():
    """A long on SKU 'a' and a crossing short on SKU 'b' do NOT match (different
    books); a short on SKU 'a' then matches — one contract, carrying sku='a'."""
    env = _env()
    _order(env, "lab", GTBActionType.FUTURES_BUY, 10, 3.5, sku="a")
    _order(env, "cloud", GTBActionType.FUTURES_SELL, 10, 3.5, sku="b")
    assert env._futures_contracts == []          # different SKUs never cross
    _order(env, "cloud2", GTBActionType.FUTURES_SELL, 10, 3.5, sku="a")
    assert len(env._futures_contracts) == 1
    c = env._futures_contracts[0]
    assert c.sku == "a" and c.long_agent_id == "lab" and c.short_agent_id == "cloud2"


def test_default_sku_matches_as_before():
    """sku='' is the generic book: a long and crossing short cross into one
    contract with an empty SKU — unchanged pre-SKU behavior."""
    env = _env()
    _order(env, "lab", GTBActionType.FUTURES_BUY, 5, 4.0, sku="")
    _order(env, "cloud", GTBActionType.FUTURES_SELL, 5, 4.0, sku="")
    assert len(env._futures_contracts) == 1
    assert env._futures_contracts[0].sku == ""


def test_futures_key_backward_compatible():
    """The book key is byte-identical to the pre-SKU form when sku='' and only
    inserts a /sku segment otherwise."""
    k = GTBEnvironment._futures_key
    assert k(ResourceType.COMPUTE, 3) == "compute@3"
    assert k(ResourceType.COMPUTE, 3, "") == "compute@3"
    assert k(ResourceType.COMPUTE, 3, "h200-use1") == "compute/h200-use1@3"


def _settle_at(env, resource_spot, sku_spots=None):
    env._last_trade_price[ResourceType.COMPUTE.value] = resource_spot
    for sku, px in (sku_spots or {}).items():
        env._last_trade_price[f"{ResourceType.COMPUTE.value}/{sku}"] = px
    env._current_epoch = _SETTLE
    env.end_epoch()


def _coin(env, name):
    return env._workers[name].get_resource(ResourceType.COIN)


def test_sku_contract_settles_zero_sum_on_sku_spot():
    """A SKU contract cash-settles zero-sum and uses the published per-SKU spot
    (not the generic resource spot). P&L measured as each side's coin delta
    (margin round-trips out, so the delta is pure realized P&L)."""
    env = _env()
    c0 = {n: _coin(env, n) for n in ("lab", "cloud")}
    _order(env, "lab", GTBActionType.FUTURES_BUY, 10, 3.0, sku="a")
    _order(env, "cloud", GTBActionType.FUTURES_SELL, 10, 3.0, sku="a")
    _settle_at(env, resource_spot=99.0, sku_spots={"a": 5.0})   # sku spot ≠ resource
    c = env._futures_contracts[0]
    assert c.status == "settled" and c.settle_spot_price == 5.0
    long_pnl = _coin(env, "lab") - c0["lab"]
    short_pnl = _coin(env, "cloud") - c0["cloud"]
    assert abs(long_pnl + short_pnl) < 1e-9          # zero-sum
    assert abs(long_pnl - 20.0) < 1e-9               # (5−3)×10, on the SKU spot


def test_two_skus_settle_independently():
    """Two SKUs on the same resource settle on their own spot — different P&L
    from the same forward."""
    env = _env()
    c0 = {n: _coin(env, n) for n in ("lab", "lab2")}
    _order(env, "lab", GTBActionType.FUTURES_BUY, 10, 3.0, sku="a")
    _order(env, "cloud", GTBActionType.FUTURES_SELL, 10, 3.0, sku="a")
    _order(env, "lab2", GTBActionType.FUTURES_BUY, 10, 3.0, sku="b")
    _order(env, "cloud2", GTBActionType.FUTURES_SELL, 10, 3.0, sku="b")
    _settle_at(env, resource_spot=3.0, sku_spots={"a": 4.0, "b": 2.0})
    assert abs((_coin(env, "lab") - c0["lab"]) - 10.0) < 1e-9   # sku a: (4−3)×10
    assert abs((_coin(env, "lab2") - c0["lab2"]) + 10.0) < 1e-9  # sku b: (2−3)×10
