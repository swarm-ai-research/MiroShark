"""Bilateral compute marketplace ported from arkhai-io/simple-compute-market.

The GTB world already trades ``ResourceType.COMPUTE`` two ways: a continuous
double auction (the spot book) and a dated-forward futures book. The
``simple-compute-market`` reference implementation models a *third*, structurally
different microstructure for the same commodity, and this module ports it so the
three can be compared head-to-head under the seed-sweep harness (CLAUDE.md
methodology):

  1. **Listings registry (discovery).** Sellers publish GPU-capacity *listings*
     to an index; buyers *query* it (filter by GPU model / rate / size) instead
     of resting orders on a shared book. Mirrors upstream's ``core/registry``
     FastAPI index and its ``filter-spec.yaml`` (``equals`` / ``upper_bound`` /
     ``lower_bound`` operators).
  2. **Buyer-driven bilateral negotiation.** For a chosen listing, buyer and
     seller run a bounded-round, buyer-driven message thread. Each side concedes
     by **bisecting** between its own limit (seller floor / buyer reservation)
     and the peer's standing quote, accepts once the peer's quote clears its
     limit, and exits when the two are more than ``gap_ratio`` apart with no
     overlap. Mirrors upstream's ``bisection`` + ``listed_price`` +
     ``max_rounds_guard`` negotiation middlewares.
  3. **Escrow-backed settlement.** The agreed deal is settled through an escrow
     hold: the buyer's coin is locked, released to the seller on delivery of the
     compute-hours, or refunded if the seller cannot deliver. Mirrors upstream's
     Alkahest ``EscrowTerms`` obligation lifecycle, with
     ``amount = rate * duration``.

Divergences from upstream (documented per CLAUDE.md — this is a trusted,
single-process simulation, not a P2P protocol):

  * No signed HTTP and no on-chain Alkahest. The registry is an in-process index;
    "escrow" is a coin hold on the buyer's ``WorkerState`` inventory. Upstream's
    byte-compare settlement collapses to a direct inventory transfer.
  * The negotiation chain keeps only the *price-bearing* middlewares that move a
    deal (inventory guard, GPU/escrow-shape guard, bisection, listed-price,
    max-rounds). Signature/codec middlewares have no analog in-sim.
  * ``duration_hours`` maps onto COMPUTE-hours 1:1 (rate is coin per
    compute-hour), so ``amount = rate * qty`` mirrors upstream's
    ``rate * duration_seconds / 3600``.

Pure Python, no Flask, no RNG at module scope — safe to import from the
bypass-Flask unit tests and to drive from the seed-sweep harness.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from worlds.gather_trade_build.entities import ResourceType, WorkerState

logger = logging.getLogger(__name__)

_EPS = 1e-9


class GpuModel(Enum):
    """GPU SKUs a listing can advertise.

    A small, ordered subset of upstream's ``filter-spec.yaml`` model enum
    (Blackwell → Volta). The label is opaque to the marketplace mechanics —
    it only gates discovery (a buyer shopping for ``H100`` won't match a
    ``V100`` listing) — but keeps the compute commodity's provenance explicit,
    matching the GTB ``COMPUTE`` = H100-hours framing (bd ja2).
    """

    B200 = "B200"      # Blackwell
    H200 = "H200"      # Hopper (refresh)
    H100 = "H100"      # Hopper — the GTB kernel's canonical compute unit
    A100 = "A100"      # Ampere
    V100 = "V100"      # Volta


# ----------------------------------------------------------------------
# Listings registry (discovery)
# ----------------------------------------------------------------------


@dataclass
class ComputeListing:
    """A seller's advertisement of GPU capacity — one row in the registry.

    Ports upstream's ``Listing`` (``offer_resource`` + ``accepted_escrows``)
    collapsed to the fields that drive price in-sim:

      * ``reserve_rate`` is the seller's floor (coin per compute-hour) — the
        ``primary_rate`` extracted from ``accepted_escrows[0].rates`` upstream,
        or ``[seller.pricing].default_min_price`` for a hidden-reserve listing.
        Never disclosed to the buyer.
      * ``advertised_rate`` is the public ask the buyer sees before negotiating
        (the listed rate). ``None`` for a hidden-reserve listing, in which case
        the opening ask collapses to the floor.
    """

    listing_id: str
    seller_id: str
    gpu_model: GpuModel
    quantity: float                      # compute-hours available on this slice
    reserve_rate: float                  # seller floor, coin/compute-hour (private)
    advertised_rate: Optional[float] = None  # public ask, coin/compute-hour
    paused: bool = False

    @property
    def opening_ask(self) -> float:
        """Public ask used to open a negotiation: the advertised rate, or the
        floor when the reserve is hidden (never below the floor)."""
        if self.advertised_rate is None:
            return self.reserve_rate
        return max(self.reserve_rate, self.advertised_rate)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpu_model"] = self.gpu_model.value
        return d


class ComputeRegistry:
    """In-process listings index — the discovery surface.

    Publish/update/pause/unpublish plus a spec-driven ``query`` mirroring
    upstream's ``GET /listings`` filters: ``equals`` on ``gpu_model``,
    ``upper_bound`` on the advertised rate, ``lower_bound`` on quantity. Paused
    (or depleted) listings are never returned.
    """

    def __init__(self) -> None:
        self._listings: Dict[str, ComputeListing] = {}
        self._next_idx: int = 0

    def publish(self, listing: ComputeListing) -> str:
        """Insert or replace a listing (upsert by ``listing_id``)."""
        self._listings[listing.listing_id] = listing
        return listing.listing_id

    def new_listing_id(self) -> str:
        lid = f"cl-{self._next_idx:04d}"
        self._next_idx += 1
        return lid

    def get(self, listing_id: str) -> Optional[ComputeListing]:
        return self._listings.get(listing_id)

    def pause(self, listing_id: str, paused: bool = True) -> bool:
        lst = self._listings.get(listing_id)
        if lst is None:
            return False
        lst.paused = paused
        return True

    def unpublish(self, listing_id: str) -> bool:
        return self._listings.pop(listing_id, None) is not None

    def query(
        self,
        gpu_model: Optional[GpuModel] = None,
        max_rate: Optional[float] = None,
        min_quantity: float = 0.0,
    ) -> List[ComputeListing]:
        """Discovery query. Returns live, matching listings sorted cheapest-ask
        first (a buyer shops the best price). ``max_rate`` bounds the *public*
        ask, so a hidden-reserve listing (ask == floor) is filtered on its
        floor — the only rate a buyer can see pre-negotiation."""
        out: List[ComputeListing] = []
        for lst in self._listings.values():
            if lst.paused or lst.quantity <= _EPS:
                continue
            if gpu_model is not None and lst.gpu_model is not gpu_model:
                continue
            if lst.quantity + _EPS < min_quantity:
                continue
            if max_rate is not None and lst.opening_ask > max_rate + _EPS:
                continue
            out.append(lst)
        out.sort(key=lambda l: (l.opening_ask, l.listing_id))
        return out

    def snapshot(self) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in self._listings.values()]


# ----------------------------------------------------------------------
# Negotiation (buyer-driven, bounded-round bisection)
# ----------------------------------------------------------------------


@dataclass
class NegotiationMessage:
    """One turn in the negotiation transcript (the immutable message thread)."""

    round: int
    sender: str          # "buyer" | "seller"
    rate: float          # the quote on the wire, coin/compute-hour
    outcome: str         # "open" | "counter" | "accept" | "exit"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NegotiationResult:
    """Reduction of a negotiation thread to an agreed rate (or ``None``)."""

    agreed_rate: Optional[float]
    rounds: int
    thread: List[NegotiationMessage]

    @property
    def is_deal(self) -> bool:
        return self.agreed_rate is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreed_rate": self.agreed_rate,
            "rounds": self.rounds,
            "is_deal": self.is_deal,
            "thread": [m.to_dict() for m in self.thread],
        }


def negotiate(
    *,
    seller_floor: float,
    buyer_reservation: float,
    opening_ask: float,
    opening_bid: float,
    max_rounds: int = 5,
    gap_ratio: float = 1.5,
) -> NegotiationResult:
    """Buyer-driven, bounded-round bisection negotiation.

    A pure, deterministic reduction over a message thread (upstream: "negotiate
    is a pure reduction over a shared message history"). Both sides concede by
    bisecting between their own limit and the peer's standing quote:

      * The **seller** accepts once the buyer's bid clears its ``seller_floor``;
        otherwise it counters at ``(ask + bid) / 2`` (never below the floor).
      * The **buyer** accepts once the seller's ask clears its
        ``buyer_reservation``; otherwise it counters at ``(bid + ask) / 2``
        (never above the reservation).
      * If the standing quotes are more than ``gap_ratio`` apart *and* the
        bargaining range is empty (``seller_floor > buyer_reservation``), the
        party on turn exits — the ``bisection`` middleware's "gap exceeds 1.5×"
        bail, restricted to genuinely infeasible pairs so a wide-but-feasible
        opening still converges.

    When a positive bargaining range exists (``seller_floor <=
    buyer_reservation``) the deal price lands inside it; when it does not, the
    negotiation exits with no deal. Deterministic in its inputs, so the same
    ``(floor, reservation, ask, bid)`` always reduces to the same rate.
    """
    bid = float(opening_bid)
    ask = max(float(seller_floor), float(opening_ask))
    thread: List[NegotiationMessage] = [
        NegotiationMessage(0, "buyer", bid, "open")
    ]
    no_overlap = seller_floor > buyer_reservation + _EPS

    for rnd in range(1, max_rounds + 1):
        # --- seller responds to the standing buyer bid ---
        if bid + _EPS >= seller_floor:
            thread.append(NegotiationMessage(rnd, "seller", bid, "accept"))
            return NegotiationResult(bid, rnd, thread)
        if no_overlap and ask > bid * gap_ratio + _EPS:
            thread.append(NegotiationMessage(rnd, "seller", ask, "exit"))
            return NegotiationResult(None, rnd, thread)
        ask = max(seller_floor, (ask + bid) / 2.0)
        thread.append(NegotiationMessage(rnd, "seller", ask, "counter"))

        # --- buyer responds to the standing seller ask ---
        if ask <= buyer_reservation + _EPS:
            thread.append(NegotiationMessage(rnd, "buyer", ask, "accept"))
            return NegotiationResult(ask, rnd, thread)
        if no_overlap and ask > buyer_reservation * gap_ratio + _EPS:
            thread.append(NegotiationMessage(rnd, "buyer", ask, "exit"))
            return NegotiationResult(None, rnd, thread)
        bid = min(buyer_reservation, (bid + ask) / 2.0)
        thread.append(NegotiationMessage(rnd, "buyer", bid, "counter"))

    return NegotiationResult(None, max_rounds, thread)


# ----------------------------------------------------------------------
# Escrow-backed settlement
# ----------------------------------------------------------------------


@dataclass
class ComputeDeal:
    """A negotiated, settled (or defaulted) compute purchase."""

    listing_id: str
    buyer_id: str
    seller_id: str
    gpu_model: GpuModel
    quantity: float                 # compute-hours transacted
    rate: float                     # coin per compute-hour
    rounds: int                     # negotiation rounds to agreement
    settled: bool = False
    defaulted: bool = False

    @property
    def amount(self) -> float:
        """Total coin: ``rate * duration`` (duration == compute-hours here)."""
        return self.rate * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpu_model"] = self.gpu_model.value
        d["amount"] = self.amount
        return d


def settle_escrow(
    deal: ComputeDeal, buyer: WorkerState, seller: WorkerState
) -> bool:
    """Settle a deal through an escrow hold.

    Lifecycle mirroring the Alkahest obligation: lock the buyer's coin
    (``amount = rate * qty``), then release it to the seller against delivery of
    the compute-hours. If either leg is unfundable (buyer short on coin, or
    seller can no longer deliver the capacity), nothing moves, the buyer is made
    whole, and the deal is marked defaulted — the sim analog of a failed
    provision / expired escrow. Conserves both total coin and total compute.

    Returns ``True`` on settlement, ``False`` on default.
    """
    amount = deal.amount
    # Fund the escrow: buyer must cover the full amount.
    if buyer.get_resource(ResourceType.COIN) + _EPS < amount:
        deal.defaulted = True
        return False
    # Seller must still be able to deliver the leased capacity.
    if seller.get_resource(ResourceType.COMPUTE) + _EPS < deal.quantity:
        deal.defaulted = True
        return False

    # Lock buyer coin into escrow.
    buyer.remove_resource(ResourceType.COIN, amount)
    # Deliver compute, release coin to seller.
    delivered = seller.remove_resource(ResourceType.COMPUTE, deal.quantity)
    if not delivered:
        # Race lost between the check and the debit — refund and default.
        buyer.add_resource(ResourceType.COIN, amount)
        deal.defaulted = True
        return False
    buyer.add_resource(ResourceType.COMPUTE, deal.quantity)
    seller.add_resource(ResourceType.COIN, amount)
    deal.settled = True
    return True


# ----------------------------------------------------------------------
# Marketplace orchestrator
# ----------------------------------------------------------------------


class ComputeMarketplace:
    """Registry + negotiation + escrow, over a set of ``WorkerState`` agents.

    Ties the three ports together into the buyer's end-to-end flow
    (``market listing list`` → ``market buy`` upstream): a seller lists capacity
    it holds; a buyer discovers the cheapest matching listing, negotiates a rate,
    and settles through escrow. Keeps a deal log for snapshotting.
    """

    def __init__(
        self,
        workers: Dict[str, WorkerState],
        *,
        max_rounds: int = 5,
        gap_ratio: float = 1.5,
    ) -> None:
        self._workers = workers
        self.registry = ComputeRegistry()
        self._max_rounds = max_rounds
        self._gap_ratio = gap_ratio
        self._deals: List[ComputeDeal] = []

    @property
    def deals(self) -> List[ComputeDeal]:
        return list(self._deals)

    def list_capacity(
        self,
        seller_id: str,
        gpu_model: GpuModel,
        quantity: float,
        reserve_rate: float,
        advertised_rate: Optional[float] = None,
    ) -> Optional[ComputeListing]:
        """Publish a listing for capacity the seller actually holds.

        Clamps the advertised quantity to the seller's on-hand compute (a slice
        is only advertised when a member can satisfy it, per upstream's
        derived-listing reconciliation). Returns ``None`` if the seller holds no
        compute to sell."""
        seller = self._workers.get(seller_id)
        if seller is None:
            return None
        held = seller.get_resource(ResourceType.COMPUTE)
        qty = min(quantity, held)
        if qty <= _EPS:
            return None
        listing = ComputeListing(
            listing_id=self.registry.new_listing_id(),
            seller_id=seller_id,
            gpu_model=gpu_model,
            quantity=qty,
            reserve_rate=reserve_rate,
            advertised_rate=advertised_rate,
        )
        self.registry.publish(listing)
        return listing

    def buy(
        self,
        buyer_id: str,
        gpu_model: GpuModel,
        quantity: float,
        max_rate: float,
        opening_bid: Optional[float] = None,
    ) -> Optional[ComputeDeal]:
        """Discover → negotiate → settle a compute purchase.

        Picks the cheapest live listing of ``gpu_model`` whose public ask is
        within ``max_rate`` and that can cover ``quantity``, negotiates a rate
        (buyer reservation ``max_rate``, opening bid defaulting to 70% of it),
        and settles through escrow. On settlement the listing's remaining
        quantity is decremented (and unpublished when depleted). Returns the
        ``ComputeDeal`` (settled or defaulted), or ``None`` if discovery/
        negotiation produced no agreement.
        """
        buyer = self._workers.get(buyer_id)
        if buyer is None:
            return None
        candidates = self.registry.query(
            gpu_model=gpu_model, max_rate=max_rate, min_quantity=quantity
        )
        for listing in candidates:
            if listing.seller_id == buyer_id:
                continue  # don't self-trade
            seller = self._workers.get(listing.seller_id)
            if seller is None:
                continue
            bid0 = opening_bid if opening_bid is not None else 0.7 * max_rate
            result = negotiate(
                seller_floor=listing.reserve_rate,
                buyer_reservation=max_rate,
                opening_ask=listing.opening_ask,
                opening_bid=bid0,
                max_rounds=self._max_rounds,
                gap_ratio=self._gap_ratio,
            )
            if not result.is_deal:
                continue
            deal = ComputeDeal(
                listing_id=listing.listing_id,
                buyer_id=buyer_id,
                seller_id=listing.seller_id,
                gpu_model=gpu_model,
                quantity=quantity,
                rate=result.agreed_rate,
                rounds=result.rounds,
            )
            ok = settle_escrow(deal, buyer, seller)
            self._deals.append(deal)
            if ok:
                listing.quantity -= quantity
                if listing.quantity <= _EPS:
                    self.registry.unpublish(listing.listing_id)
                return deal
            # Defaulted on this listing — try the next candidate.
        return None

    def snapshot(self) -> Dict[str, Any]:
        settled = [d for d in self._deals if d.settled]
        volume = sum(d.quantity for d in settled)
        notional = sum(d.amount for d in settled)
        vwap = notional / volume if volume > _EPS else None
        return {
            "listings": self.registry.snapshot(),
            "deals": [d.to_dict() for d in self._deals],
            "n_deals": len(settled),
            "n_defaults": sum(1 for d in self._deals if d.defaulted),
            "volume": volume,
            "vwap_rate": vwap,
        }
