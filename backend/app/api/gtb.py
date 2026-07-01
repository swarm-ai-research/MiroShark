"""GTB world API — start/step/inspect a gather-trade-build economy.

Phase 2 surface: lifecycle (start/stop) + step + state snapshot.
Phase 3 will add per-agent action overrides driven by LLM tool calls.
"""

from __future__ import annotations

from flask import jsonify, request

from . import gtb_bp
from ..services.gtb_service import get_registry
from ..utils.logger import get_logger

logger = get_logger("miroshark.api.gtb")

# Upper bound on steps advanced in a single /step request, so one call can't
# pin the worker thread on an unbounded loop.
_MAX_STEP_BATCH = 1000


def _coerce_step_count(raw, maximum: int = _MAX_STEP_BATCH) -> int:
    """Coerce a requested step count to an int in ``[1, maximum]``.

    Raises ``ValueError``/``TypeError`` on non-numeric input (e.g. a string
    or list) so the route can return 400 instead of an unhandled 500.
    """
    n = int(raw)
    return max(1, min(n, maximum))


@gtb_bp.route("/<sim_id>/start", methods=["POST"])
def start_world(sim_id: str):
    payload = request.get_json(silent=True) or {}
    scenario = payload.get("scenario")
    overrides = payload.get("overrides")
    try:
        service = get_registry().start(sim_id, scenario, overrides)
    except FileNotFoundError as e:
        return jsonify({"error": f"scenario not found: {e}"}), 404
    except Exception as e:
        logger.exception("GTB start failed for %s", sim_id)
        return jsonify({"error": str(e)}), 400
    return jsonify({"sim_id": sim_id, "state": service.state()})


@gtb_bp.route("/<sim_id>/step", methods=["POST"])
def step_world(sim_id: str):
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    payload = request.get_json(silent=True) or {}
    try:
        n = _coerce_step_count(payload.get("n", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "'n' must be an integer"}), 400
    results = [service.step() for _ in range(n)]
    return jsonify({"sim_id": sim_id, "ticks": results})


@gtb_bp.route("/<sim_id>/state", methods=["GET"])
def get_state(sim_id: str):
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    return jsonify({"sim_id": sim_id, "state": service.state()})


@gtb_bp.route("/<sim_id>/stop", methods=["POST"])
def stop_world(sim_id: str):
    stopped = get_registry().stop(sim_id)
    return jsonify({"sim_id": sim_id, "stopped": stopped})


@gtb_bp.route("", methods=["GET"])
@gtb_bp.route("/", methods=["GET"])
def list_worlds():
    return jsonify({"sim_ids": get_registry().list_ids()})


@gtb_bp.route("/<sim_id>/markets", methods=["GET"])
def get_markets(sim_id: str):
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    return jsonify({"sim_id": sim_id, "markets": service.markets()})


@gtb_bp.route("/<sim_id>/markets/generate", methods=["POST"])
def generate_markets(sim_id: str):
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    new = service.generate_markets()
    return jsonify({"sim_id": sim_id, "new_markets": new, "all": service.markets()})


@gtb_bp.route("/<sim_id>/stake", methods=["POST"])
def place_stake(sim_id: str):
    """Inbound stake from an external client (UI button, Polymarket bot).

    Body: {agent_id, market_id, side: "yes"|"no", amount: float}.
    Coin is debited immediately from the worker's inventory; payouts
    arrive when the market resolves, same as the LLM-action piggyback
    path.
    """
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    payload = request.get_json(silent=True) or {}
    for field in ("agent_id", "market_id", "side", "amount"):
        if field not in payload:
            return jsonify({"error": f"missing field: {field}"}), 400
    result = service.place_stake(
        agent_id=str(payload["agent_id"]),
        market_id=str(payload["market_id"]),
        side=str(payload["side"]).lower(),
        amount=float(payload["amount"]),
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@gtb_bp.route("/<sim_id>/polymarket", methods=["GET"])
def polymarket_envelope(sim_id: str):
    """Polymarket-shaped view of the live GTB market book.

    Same schema_version + confidence_tier vocabulary as the existing
    polymarket_service surface, so downstream bots can parse both
    streams through one adapter. Open markets get a stake-derived YES
    probability; resolved markets get 0.0 / 1.0 reflecting the outcome.
    """
    service = get_registry().get(sim_id)
    if service is None:
        return jsonify({"error": "no world for sim_id"}), 404
    from ..services.gtb_polymarket import compute_gtb_polymarket
    payload = compute_gtb_polymarket(service.state(), sim_id)
    if payload is None:
        return jsonify({"error": "world has no market state yet"}), 404
    return jsonify(payload)
