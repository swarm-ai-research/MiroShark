"""URL-level Polymarket dispatcher.

Phase 9 unified the *schema* (GTB envelopes use the same
``schema_version: "1"`` shape as the narrative \`polymarket_service\`).
This blueprint unifies the *URL*: \`/api/sim/<sim_id>/polymarket\` serves
the right surface for the sim type — a live GTB world or a finished
narrative sim — so an external Polymarket bot has one place to point.

Implementation: a 307 redirect to the existing surface. Keeps both
upstream handlers as the single source of truth (publish gate, cache
TTL, content-disposition headers) and avoids re-implementing either
inline.
"""

from __future__ import annotations

from flask import Blueprint, redirect

from ..services.gtb_service import get_registry


sim_dispatcher_bp = Blueprint("sim_dispatcher", __name__)


@sim_dispatcher_bp.route("/<sim_id>/polymarket", methods=["GET"])
def polymarket(sim_id: str):
    """Redirect to the right Polymarket surface for this sim.

    GTB world registered → /api/gtb/<sim_id>/polymarket
    Otherwise → /api/simulation/<sim_id>/polymarket.json
    (the narrative surface handles its own 404 / 403 if the sim
    doesn't exist or isn't published).
    """
    if get_registry().get(sim_id) is not None:
        return redirect(f"/api/gtb/{sim_id}/polymarket", code=307)
    return redirect(f"/api/simulation/{sim_id}/polymarket.json", code=307)
