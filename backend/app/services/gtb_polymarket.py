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

CAVEAT — one-sided liquidity is the default. The first live LLM smoke
on PR #1 surfaced that populations of persona-aligned LLM agents pile
stakes on one side (8/8 YES). When only one side has any stakes, the
yes_probability is NOT a forecast — it is a sentiment poll of the
swarm. A 95%-bullish swarm can be 95% bullish on a market that resolves
NO. Every envelope therefore exposes a ``confidence_source`` field:

    confidence_source = "two_sided"   # both yes_pool and no_pool > 0
                      | "one_sided"   # only one side has stakes
                      | "no_stakes"   # neither side has stakes
                      | "resolved"    # market already settled

External consumers (Polymarket bots, MCP tools, downstream
calibration code) SHOULD treat ``one_sided`` and ``no_stakes``
envelopes as un-priced sentiment, not as forecasts. Do not size
positions against them. The ``headline`` picker already prefers
``two_sided`` markets when available; consumers should filter
similarly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .polymarket_service import _confidence_tier


_SCHEMA_VERSION = "1"
# Laplace smoothing: add this much to each side's pool so the prior is
# always 50/50 and a market with two opposing 1.0-coin stakes is still
# read as a near-coin-flip rather than overconfidently 50%.
_LAPLACE = 1.0


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


def _confidence_source(yes_pool: float, no_pool: float, is_resolved: bool) -> str:
    """Classify how trustworthy the yes_probability is as a forecast.

    - ``resolved``: market already settled; yes_prob is 1.0 or 0.0.
    - ``two_sided``: both sides have stakes — actual price discovery.
    - ``one_sided``: only one side staked — sentiment, not a forecast.
    - ``no_stakes``: neither side; yes_prob is the Laplace prior (0.5).
    """
    if is_resolved:
        return "resolved"
    if yes_pool > 0 and no_pool > 0:
        return "two_sided"
    if yes_pool > 0 or no_pool > 0:
        return "one_sided"
    return "no_stakes"


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
        "risk_tier": _risk_tier(confidence_pct),
        # See module docstring: only `two_sided` envelopes are forecasts.
        # `one_sided` and `no_stakes` are population sentiment polls
        # that downstream consumers should NOT size positions on.
        "confidence_source": _confidence_source(yes_pool, no_pool, is_resolved),
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


def _pick_headline(envelopes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the most-actionable open envelope for single-signal consumers.

    Ranking (each tier strictly preferred over the next):
      1. ``two_sided`` markets — real price discovery with counterparty
      2. ``one_sided`` markets — sentiment poll, flagged as such
      3. ``no_stakes`` markets — Laplace prior, last resort

    Within each tier, sort by ``confidence_pct`` descending. Returns
    ``None`` only when every market is resolved.
    """
    open_envs = [e for e in envelopes if e["status"] == "open"]
    if not open_envs:
        return None
    two_sided = [e for e in open_envs if e["confidence_source"] == "two_sided"]
    one_sided = [e for e in open_envs if e["confidence_source"] == "one_sided"]
    no_stakes = [e for e in open_envs if e["confidence_source"] == "no_stakes"]
    for pool in (two_sided, one_sided, no_stakes):
        if pool:
            pool.sort(key=lambda e: e["confidence_pct"], reverse=True)
            return pool[0]
    return None
