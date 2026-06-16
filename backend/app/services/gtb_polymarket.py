"""Adapter: re-shape GTB markets + stake book into Polymarket's envelope.

The existing ``polymarket_service.compute_polymarket`` post-processes a
*finished* narrative simulation into ONE binary YES/NO payload. GTB
generates MANY forward-looking markets MID-RUN, with live stake books.
This module bridges those — same ``schema_version: "1"`` shape, same
``_confidence_tier`` bucket scale — so a downstream Polymarket bot
consumes both surfaces through one parser.

Per-market YES probability is derived from the live stake book using
Laplace smoothing so a brand-new market with zero stakes reads as a
coin flip (0.5) instead of a divide-by-zero. As stakes accumulate, the
estimate converges to ``yes_pool / (yes_pool + no_pool)``.

Resolved markets emit ``yes_probability`` of 1.0 / 0.0 (matching the
resolution) so a consumer doing post-hoc PnL has the same field
populated as for open markets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .polymarket_service import _confidence_tier

_SCHEMA_PATH = Path(__file__).parent / "schema" / "gtb_polymarket.schema.json"


_SCHEMA_VERSION = "1.1"
# Laplace smoothing: add this much to each side's pool so the prior is
# always 50/50 and a market with two opposing 1.0-coin stakes is still
# read as a near-coin-flip rather than overconfidently 50%.
_LAPLACE = 1.0

# Discriminates how yes_probability was derived. Lets a downstream consumer
# trust resolved/two_sided readings and discount the others rather than
# treating every envelope as a real forecast.
_CONFIDENCE_SOURCES = ("resolved", "no_stakes", "one_sided", "two_sided")


def _confidence_source(
    is_resolved: bool, yes_pool: float, no_pool: float
) -> str:
    if is_resolved:
        return "resolved"
    if yes_pool <= 0 and no_pool <= 0:
        return "no_stakes"
    if yes_pool <= 0 or no_pool <= 0:
        return "one_sided"
    return "two_sided"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _yes_prob_from_stakes(yes_pool: float, no_pool: float) -> float:
    total = yes_pool + no_pool + 2 * _LAPLACE
    if total <= 0:
        return 0.5
    return round((yes_pool + _LAPLACE) / total, 4)


def _direction(yes_prob: float) -> str:
    if yes_prob >= 0.6:
        return "Bullish"
    if yes_prob <= 0.4:
        return "Bearish"
    return "Neutral"


def _risk_tier(confidence_pct: float) -> str:
    if confidence_pct >= 50.0:
        return "low-risk"
    if confidence_pct >= 25.0:
        return "medium-risk"
    return "high-risk"


def _market_to_envelope(
    market: Dict[str, Any],
    sim_id: str,
    yes_pool: float,
    no_pool: float,
    is_resolved: bool,
    generated_at: str,
) -> Dict[str, Any]:
    if is_resolved:
        status = market.get("status")
        if status == "yes":
            yes_prob = 1.0
        elif status == "no":
            yes_prob = 0.0
        else:
            # expired / refunded — no resolution, treat as coin flip.
            yes_prob = 0.5
    else:
        yes_prob = _yes_prob_from_stakes(yes_pool, no_pool)

    no_prob = round(1.0 - yes_prob, 4)
    confidence_pct = round(abs(yes_prob - 0.5) * 200.0, 1)
    direction = _direction(yes_prob)

    return {
        "schema_version": _SCHEMA_VERSION,
        "simulation_id": sim_id,
        "source_sim_id": sim_id,
        "market_id": market.get("market_id"),
        "suggested_market_title": market.get("question"),
        "metric": market.get("metric"),
        "op": market.get("op"),
        "threshold": market.get("threshold"),
        "deadline_epoch": market.get("deadline_epoch"),
        "status": market.get("status", "open"),
        "resolved_epoch": market.get("resolved_epoch"),
        "resolved_value": market.get("resolved_value"),
        "yes_probability": yes_prob,
        "no_probability": no_prob,
        "yes_pool": round(yes_pool, 4),
        "no_pool": round(no_pool, 4),
        "direction": direction,
        "confidence_pct": confidence_pct,
        "confidence_tier": _confidence_tier(confidence_pct),
        "confidence_source": _confidence_source(is_resolved, yes_pool, no_pool),
        "risk_tier": _risk_tier(confidence_pct),
        "polymarket_generated_at": generated_at,
    }


def _stake_totals_by_market(stakes: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    open_stakes = (stakes or {}).get("open_stakes", {})
    for mid, lst in open_stakes.items():
        agg = {"yes": 0.0, "no": 0.0}
        for s in lst:
            side = s.get("side")
            amt = float(s.get("amount") or 0.0)
            if side in agg:
                agg[side] += amt
        out[mid] = agg
    return out


def compute_gtb_polymarket(
    state: Dict[str, Any], sim_id: str
) -> Optional[Dict[str, Any]]:
    """Re-shape a GTB world snapshot as a multi-market Polymarket payload.

    Returns ``None`` only if ``state`` is unusable (missing the markets
    block). Open and resolved markets both produce envelopes so a
    consumer can score positions across the whole history of the run.
    """
    if not isinstance(state, dict):
        return None
    markets_block = state.get("markets")
    if not isinstance(markets_block, dict):
        return None

    stakes_block = state.get("stakes") or {}
    totals = _stake_totals_by_market(stakes_block)
    now = _iso_now()
    envelopes: List[Dict[str, Any]] = []

    for m in markets_block.get("open", []):
        mid = m.get("market_id", "")
        agg = totals.get(mid, {"yes": 0.0, "no": 0.0})
        envelopes.append(_market_to_envelope(
            m, sim_id, agg["yes"], agg["no"], is_resolved=False, generated_at=now,
        ))
    for m in markets_block.get("resolved", []):
        # Resolved markets have no live stake pool (stakes were
        # distributed); pools are zero by definition here.
        envelopes.append(_market_to_envelope(
            m, sim_id, 0.0, 0.0, is_resolved=True, generated_at=now,
        ))

    headline = _pick_headline(envelopes)
    return {
        "schema_version": _SCHEMA_VERSION,
        "simulation_id": sim_id,
        "source_sim_id": sim_id,
        "polymarket_generated_at": now,
        "current_epoch": state.get("epoch"),
        "headline": headline,
        "markets": envelopes,
    }


@lru_cache(maxsize=1)
def _load_schema() -> Dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text())


def validate_payload(payload: Dict[str, Any]) -> None:
    """Validate a payload against gtb_polymarket.schema.json.

    Raises ``jsonschema.ValidationError`` on mismatch. Intended for tests
    and assertions on the producer side; we don't call it from hot paths
    so a schema bump doesn't risk blocking a live run on a producer bug.
    """
    # Import lazily so the module loads in environments that don't install
    # jsonschema (e.g. minimal containers that never emit envelopes).
    from jsonschema import validate

    validate(instance=payload, schema=_load_schema())


def _pick_headline(envelopes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the most-actionable open envelope for single-signal consumers.

    Ranking: highest |confidence_pct| (clearest belief) among open
    markets with at least some stake; falls back to the highest-
    confidence open market overall; returns ``None`` if every market
    is resolved.
    """
    open_envs = [e for e in envelopes if e["status"] == "open"]
    if not open_envs:
        return None
    staked = [e for e in open_envs if e["yes_pool"] + e["no_pool"] > 0]
    pool = staked or open_envs
    pool.sort(key=lambda e: e["confidence_pct"], reverse=True)
    return pool[0]
