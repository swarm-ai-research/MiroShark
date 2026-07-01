"""Unit tests for the bilateral compute marketplace (arkhai simple-compute-market port).

Pins the three ported microstructure pieces from
``worlds.gather_trade_build.compute_market``:

  * the listings **registry** discovery filters (GPU model / rate / size, and
    the paused/depleted exclusions),
  * the buyer-driven **bisection negotiation** — converges inside the bargaining
    range when one exists, exits when it doesn't, and is a deterministic
    reduction over its inputs,
  * **escrow settlement** — conserves coin + compute on the happy path, and
    refunds/defaults (moving nothing) when a leg is unfundable,
  * the ``ComputeMarketplace`` end-to-end discover→negotiate→settle flow.

Bypass-Flask by design (CLAUDE.md): imports only the world package, runs on
pyyaml + pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.compute_market import (  # noqa: E402
    ComputeDeal,
    ComputeListing,
    ComputeMarketplace,
    ComputeRegistry,
    GpuModel,
    negotiate,
    settle_escrow,
)
from worlds.gather_trade_build.entities import ResourceType, WorkerState  # noqa: E402


def _worker(aid, coin=0.0, compute=0.0):
    w = WorkerState(agent_id=aid)
    w.add_resource(ResourceType.COIN, coin)
    w.add_resource(ResourceType.COMPUTE, compute)
    return w


def _listing(lid="cl-0", seller="s", model=GpuModel.H100, qty=10.0,
             reserve=1.0, advertised=2.0):
    return ComputeListing(listing_id=lid, seller_id=seller, gpu_model=model,
                          quantity=qty, reserve_rate=reserve,
                          advertised_rate=advertised)


# ----------------------------------------------------------------------
# Registry discovery
# ----------------------------------------------------------------------


def test_registry_query_filters_by_gpu_model_rate_and_size():
    reg = ComputeRegistry()
    reg.publish(_listing("a", model=GpuModel.H100, advertised=2.0, qty=10))
    reg.publish(_listing("b", model=GpuModel.V100, advertised=1.0, qty=10))
    reg.publish(_listing("c", model=GpuModel.H100, advertised=9.0, qty=10))
    reg.publish(_listing("d", model=GpuModel.H100, advertised=2.5, qty=1))

    # H100 only, ask <= 3.0, at least 5 units: only "a" survives all three.
    got = reg.query(gpu_model=GpuModel.H100, max_rate=3.0, min_quantity=5.0)
    assert [l.listing_id for l in got] == ["a"]


def test_registry_query_sorts_cheapest_ask_first():
    reg = ComputeRegistry()
    reg.publish(_listing("hi", advertised=5.0))
    reg.publish(_listing("lo", advertised=2.0))
    reg.publish(_listing("mid", advertised=3.0))
    got = reg.query(gpu_model=GpuModel.H100)
    assert [l.listing_id for l in got] == ["lo", "mid", "hi"]


def test_registry_excludes_paused_and_depleted():
    reg = ComputeRegistry()
    reg.publish(_listing("paused", qty=10))
    reg.pause("paused")
    reg.publish(_listing("empty", qty=0.0))
    reg.publish(_listing("live", qty=5.0))
    assert [l.listing_id for l in reg.query()] == ["live"]


def test_hidden_reserve_listing_opens_at_floor():
    lst = _listing(advertised=None, reserve=1.5)
    assert lst.opening_ask == 1.5
    # max_rate filters on the floor when the reserve is hidden.
    reg = ComputeRegistry()
    reg.publish(lst)
    assert reg.query(max_rate=1.0) == []
    assert reg.query(max_rate=2.0) == [lst]


# ----------------------------------------------------------------------
# Negotiation: convergence, exit, determinism
# ----------------------------------------------------------------------


def test_negotiation_converges_inside_bargaining_range():
    # floor 1.0, reservation 2.0 -> a deal must land in [1.0, 2.0].
    res = negotiate(seller_floor=1.0, buyer_reservation=2.0,
                    opening_ask=2.0, opening_bid=0.7)
    assert res.is_deal
    assert 1.0 - 1e-9 <= res.agreed_rate <= 2.0 + 1e-9


def test_negotiation_seller_accepts_a_bid_above_floor_immediately():
    # Buyer opens above the (hidden) floor -> seller accepts round 1 at the bid.
    res = negotiate(seller_floor=1.0, buyer_reservation=3.0,
                    opening_ask=1.0, opening_bid=1.2)
    assert res.is_deal
    assert res.agreed_rate == pytest.approx(1.2)
    assert res.rounds == 1
    assert res.thread[-1].outcome == "accept"


def test_negotiation_no_overlap_exits_without_deal():
    # Seller needs >= 2.0, buyer will pay <= 1.0: empty bargaining range.
    res = negotiate(seller_floor=2.0, buyer_reservation=1.0,
                    opening_ask=2.5, opening_bid=0.7)
    assert not res.is_deal
    assert res.agreed_rate is None
    assert res.thread[-1].outcome in ("exit",)


def test_negotiation_respects_max_rounds_when_marginally_infeasible():
    # Tiny gap, no overlap, gap ratio never exceeds 1.5x -> runs out of rounds.
    res = negotiate(seller_floor=1.1, buyer_reservation=1.0,
                    opening_ask=1.2, opening_bid=1.0, max_rounds=4)
    assert not res.is_deal
    assert res.rounds == 4


def test_negotiation_is_deterministic_reduction():
    kw = dict(seller_floor=1.0, buyer_reservation=2.0,
              opening_ask=2.0, opening_bid=0.7)
    a = negotiate(**kw)
    b = negotiate(**kw)
    assert a.agreed_rate == b.agreed_rate
    assert [m.to_dict() for m in a.thread] == [m.to_dict() for m in b.thread]


# ----------------------------------------------------------------------
# Escrow settlement
# ----------------------------------------------------------------------


def _deal(qty=3.0, rate=2.0):
    return ComputeDeal(listing_id="cl-0", buyer_id="b", seller_id="s",
                       gpu_model=GpuModel.H100, quantity=qty, rate=rate, rounds=1)


def test_settlement_conserves_coin_and_compute():
    buyer = _worker("b", coin=100.0)
    seller = _worker("s", compute=10.0)
    total_coin = buyer.get_resource(ResourceType.COIN) + seller.get_resource(ResourceType.COIN)
    total_compute = buyer.get_resource(ResourceType.COMPUTE) + seller.get_resource(ResourceType.COMPUTE)

    deal = _deal(qty=3.0, rate=2.0)   # amount = 6.0 coin
    assert settle_escrow(deal, buyer, seller) is True
    assert deal.settled and not deal.defaulted

    # Buyer paid 6 coin, got 3 compute; seller mirror. Totals conserved.
    assert buyer.get_resource(ResourceType.COIN) == pytest.approx(94.0)
    assert buyer.get_resource(ResourceType.COMPUTE) == pytest.approx(3.0)
    assert seller.get_resource(ResourceType.COIN) == pytest.approx(6.0)
    assert seller.get_resource(ResourceType.COMPUTE) == pytest.approx(7.0)
    assert (buyer.get_resource(ResourceType.COIN)
            + seller.get_resource(ResourceType.COIN)) == pytest.approx(total_coin)
    assert (buyer.get_resource(ResourceType.COMPUTE)
            + seller.get_resource(ResourceType.COMPUTE)) == pytest.approx(total_compute)


def test_settlement_defaults_when_buyer_underfunded():
    buyer = _worker("b", coin=1.0)          # needs 6.0
    seller = _worker("s", compute=10.0)
    deal = _deal(qty=3.0, rate=2.0)
    assert settle_escrow(deal, buyer, seller) is False
    assert deal.defaulted and not deal.settled
    # Nothing moved.
    assert buyer.get_resource(ResourceType.COIN) == pytest.approx(1.0)
    assert buyer.get_resource(ResourceType.COMPUTE) == pytest.approx(0.0)
    assert seller.get_resource(ResourceType.COMPUTE) == pytest.approx(10.0)


def test_settlement_refunds_when_seller_cannot_deliver():
    buyer = _worker("b", coin=100.0)
    seller = _worker("s", compute=1.0)      # can't deliver 3
    deal = _deal(qty=3.0, rate=2.0)
    assert settle_escrow(deal, buyer, seller) is False
    assert deal.defaulted
    # Buyer made whole (escrow refunded), no compute delivered.
    assert buyer.get_resource(ResourceType.COIN) == pytest.approx(100.0)
    assert buyer.get_resource(ResourceType.COMPUTE) == pytest.approx(0.0)
    assert seller.get_resource(ResourceType.COIN) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Marketplace end-to-end
# ----------------------------------------------------------------------


def test_marketplace_discover_negotiate_settle_end_to_end():
    workers = {"seller": _worker("seller", compute=10.0),
               "buyer": _worker("buyer", coin=100.0)}
    mkt = ComputeMarketplace(workers)
    lst = mkt.list_capacity("seller", GpuModel.H100, quantity=10.0,
                            reserve_rate=1.0, advertised_rate=2.0)
    assert lst is not None

    deal = mkt.buy("buyer", GpuModel.H100, quantity=4.0, max_rate=2.0)
    assert deal is not None and deal.settled
    assert 1.0 - 1e-9 <= deal.rate <= 2.0 + 1e-9
    # Buyer received compute, seller received coin, listing decremented.
    assert workers["buyer"].get_resource(ResourceType.COMPUTE) == pytest.approx(4.0)
    assert mkt.registry.get(lst.listing_id).quantity == pytest.approx(6.0)
    snap = mkt.snapshot()
    assert snap["n_deals"] == 1 and snap["volume"] == pytest.approx(4.0)
    assert snap["vwap_rate"] == pytest.approx(deal.rate)


def test_marketplace_no_matching_listing_returns_none():
    workers = {"seller": _worker("seller", compute=10.0),
               "buyer": _worker("buyer", coin=100.0)}
    mkt = ComputeMarketplace(workers)
    mkt.list_capacity("seller", GpuModel.V100, quantity=10.0, reserve_rate=1.0)
    # Buyer wants H100 — none listed.
    assert mkt.buy("buyer", GpuModel.H100, quantity=1.0, max_rate=5.0) is None


def test_marketplace_depleted_listing_is_unpublished():
    workers = {"seller": _worker("seller", compute=5.0),
               "buyer": _worker("buyer", coin=100.0)}
    mkt = ComputeMarketplace(workers)
    lst = mkt.list_capacity("seller", GpuModel.H100, quantity=5.0,
                            reserve_rate=1.0, advertised_rate=2.0)
    deal = mkt.buy("buyer", GpuModel.H100, quantity=5.0, max_rate=2.0)
    assert deal is not None and deal.settled
    assert mkt.registry.get(lst.listing_id) is None
    assert mkt.registry.query(gpu_model=GpuModel.H100) == []


def test_marketplace_does_not_self_trade():
    # The only listing is the buyer's own -> no counterparty, no deal.
    workers = {"a": _worker("a", coin=100.0, compute=10.0)}
    mkt = ComputeMarketplace(workers)
    mkt.list_capacity("a", GpuModel.H100, quantity=10.0, reserve_rate=1.0,
                      advertised_rate=2.0)
    assert mkt.buy("a", GpuModel.H100, quantity=1.0, max_rate=2.0) is None
