"""LLM-driven worker policy for GTB.

Implements the same `.decide(obs)` interface as the rule-based policies in
`worlds.gather_trade_build.agents` so a GTBWorldService can mix LLM agents
and policy agents transparently.

Each tick, the policy renders the observation as a structured prompt, calls
chat_json on the configured LLM, parses the returned JSON into a GTBAction.
On any failure (missing key, parse error, invalid action) the policy falls
back to the HonestWorkerPolicy so a misconfigured deploy never bricks a tick.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from worlds.gather_trade_build.agents import HonestWorkerPolicy
from worlds.gather_trade_build.entities import Direction, GTBActionType, ResourceType
from worlds.gather_trade_build.env import GTBAction

logger = logging.getLogger(__name__)


_ACTION_SCHEMA_HINT = {
    "action_type": "one of: move, gather, build, trade_buy, trade_sell, shift_income, misreport, noop",
    "direction": "one of: up, down, left, right (only for action_type=move)",
    "resource_type": "one of: wood, stone (only for trade_buy/trade_sell)",
    "quantity": "number > 0 (for trade)",
    "price": "coins per unit (for trade)",
    "shift_amount": "coins to defer (for shift_income)",
    "underreport_fraction": "0..1 (for misreport)",
}


_SYSTEM_PROMPT = """You are a worker in a gather-trade-build economy.
Each tick you choose ONE action that maximizes your long-run utility:
post-tax income, energy budget, and house ownership.

Mechanics in one paragraph:
- Move on a grid to find wood / stone tiles, GATHER to collect them.
- BUILD a house on an empty cell adjacent to you (costs wood+stone, pays
  income each step thereafter).
- Trade in the central market: TRADE_BUY or TRADE_SELL wood/stone at a
  price you set.
- Income is taxed at epoch end via piecewise brackets. You may SHIFT_INCOME
  to defer earnings into a later epoch, or MISREPORT a fraction (risks
  audit, fine, and reputation loss).
- Energy depletes per action; conserve it.

Return JSON ONLY. Schema:
{action_schema}

Examples:
{{"action_type": "move", "direction": "up"}}
{{"action_type": "gather"}}
{{"action_type": "build"}}
{{"action_type": "trade_sell", "resource_type": "wood", "quantity": 2, "price": 1.5}}
{{"action_type": "noop"}}
"""


def _render_persona(persona: Dict[str, Any]) -> str:
    name = persona.get("name", "Worker")
    bio = persona.get("personality") or persona.get("bio") or ""
    objective = persona.get("objective", "maximize long-run post-tax income")
    return f"You are {name}. {bio}\nYour objective: {objective}"


def _render_obs(obs: Dict[str, Any]) -> str:
    visible = obs.get("visible_cells", [])
    res_cells = [c for c in visible if "resource" in c]
    return json.dumps({
        "position": obs.get("position"),
        "inventory": obs.get("inventory"),
        "energy": obs.get("energy"),
        "gross_income_this_epoch": obs.get("gross_income"),
        "deferred_income": obs.get("deferred_income"),
        "houses_built": obs.get("houses_built"),
        "epoch": obs.get("epoch"),
        "step": obs.get("step"),
        "frozen": obs.get("frozen"),
        "tax_brackets": obs.get("tax_schedule", {}).get("brackets", []),
        "visible_resources": res_cells[:10],
        "visible_cells_count": len(visible),
    }, indent=2)


def _parse_action(agent_id: str, raw: Dict[str, Any]) -> GTBAction:
    at = (raw.get("action_type") or "noop").lower()
    try:
        action_type = GTBActionType(at)
    except ValueError:
        action_type = GTBActionType.NOOP

    direction = Direction.UP
    if action_type == GTBActionType.MOVE:
        d = (raw.get("direction") or "up").lower()
        try:
            direction = Direction(d)
        except ValueError:
            direction = Direction.UP

    resource_type = ResourceType.WOOD
    if action_type in (GTBActionType.TRADE_BUY, GTBActionType.TRADE_SELL):
        rt = (raw.get("resource_type") or "wood").lower()
        try:
            resource_type = ResourceType(rt)
        except ValueError:
            resource_type = ResourceType.WOOD

    return GTBAction(
        agent_id=agent_id,
        action_type=action_type,
        direction=direction,
        resource_type=resource_type,
        quantity=float(raw.get("quantity") or 0.0),
        price=float(raw.get("price") or 1.0),
        shift_amount=float(raw.get("shift_amount") or 0.0),
        underreport_fraction=float(raw.get("underreport_fraction") or 0.0),
    )


class LLMWorkerPolicy:
    """Worker policy that asks an LLM for the next action each tick."""

    def __init__(
        self,
        agent_id: str,
        persona: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        llm_client: Any = None,
        temperature: float = 0.4,
    ) -> None:
        self._agent_id = agent_id
        self._persona = persona or {"name": agent_id}
        self._llm = llm_client
        self._temperature = temperature
        self._fallback = HonestWorkerPolicy(agent_id=agent_id, seed=seed)

    def _ensure_llm(self):
        if self._llm is None:
            from ..utils.llm_client import create_llm_client
            self._llm = create_llm_client()
        return self._llm

    def decide(self, obs: Dict[str, Any]) -> GTBAction:
        if obs.get("frozen"):
            return GTBAction(agent_id=self._agent_id, action_type=GTBActionType.NOOP)
        try:
            llm = self._ensure_llm()
            system = _SYSTEM_PROMPT.format(
                action_schema=json.dumps(_ACTION_SCHEMA_HINT, indent=2)
            )
            user = (
                _render_persona(self._persona)
                + "\n\nCurrent observation:\n"
                + _render_obs(obs)
                + "\n\nChoose your next action. JSON only."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            raw = llm.chat_json(messages=messages, temperature=self._temperature, max_tokens=300)
            return _parse_action(self._agent_id, raw)
        except Exception as e:
            logger.warning(
                "LLM action failed for %s (%s); falling back to honest",
                self._agent_id, e,
            )
            return self._fallback.decide(obs)


_BATCH_SYSTEM_PROMPT = """You are the decision layer for {n} workers in a
gather-trade-build economy. For EACH worker, choose ONE action this tick.

Mechanics: move on a grid to find wood / stone, GATHER to collect, BUILD a
house (costs wood+stone, pays income/step), TRADE_BUY / TRADE_SELL wood or
stone in the market at a price you set. Income is taxed at epoch end via
piecewise brackets; SHIFT_INCOME defers earnings, MISREPORT under-reports
(risks audit + fine). Energy depletes per action.

Return JSON ONLY, mapping agent_id -> action object. Action schema:
{action_schema}

Example:
{{
  "worker_0": {{"action_type": "gather"}},
  "worker_1": {{"action_type": "trade_sell", "resource_type": "wood", "quantity": 2, "price": 1.5}}
}}
"""


class BatchLLMDriver:
    """Single LLM call decides actions for a set of agents per tick.

    Avoids the N-agents × N-ticks call count of per-agent LLMWorkerPolicy.
    Tracks its own agent_id set; on parse failure falls back to honest per
    missing/invalid agent.
    """

    def __init__(
        self,
        agent_ids: Optional[list] = None,
        personas: Optional[Dict[str, Dict[str, Any]]] = None,
        llm_client: Any = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        self._agent_ids = list(agent_ids or [])
        self._personas = personas or {}
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._fallbacks: Dict[str, HonestWorkerPolicy] = {
            aid: HonestWorkerPolicy(agent_id=aid, seed=hash(aid) & 0xFFFF)
            for aid in self._agent_ids
        }

    @property
    def agent_ids(self) -> list:
        return list(self._agent_ids)

    def add_agent(self, agent_id: str, persona: Optional[Dict[str, Any]] = None) -> None:
        if agent_id not in self._agent_ids:
            self._agent_ids.append(agent_id)
        if persona is not None:
            self._personas[agent_id] = persona
        self._fallbacks.setdefault(
            agent_id, HonestWorkerPolicy(agent_id=agent_id, seed=hash(agent_id) & 0xFFFF)
        )

    def _ensure_llm(self):
        if self._llm is None:
            from ..utils.llm_client import create_llm_client
            self._llm = create_llm_client()
        return self._llm

    def decide_all(self, obs_by_agent: Dict[str, Dict[str, Any]]) -> Dict[str, GTBAction]:
        if not obs_by_agent:
            return {}
        out: Dict[str, GTBAction] = {}
        try:
            llm = self._ensure_llm()
            payload = {
                aid: {
                    "persona": self._personas.get(aid, {"name": aid}),
                    "obs": {
                        "position": obs.get("position"),
                        "inventory": obs.get("inventory"),
                        "energy": obs.get("energy"),
                        "gross_income": obs.get("gross_income"),
                        "houses_built": obs.get("houses_built"),
                        "epoch": obs.get("epoch"),
                        "step": obs.get("step"),
                        "frozen": obs.get("frozen"),
                        "tax_brackets": obs.get("tax_schedule", {}).get("brackets", []),
                        "visible_resources": [
                            c for c in obs.get("visible_cells", []) if "resource" in c
                        ][:8],
                    },
                }
                for aid, obs in obs_by_agent.items()
            }
            system = _BATCH_SYSTEM_PROMPT.format(
                n=len(obs_by_agent),
                action_schema=json.dumps(_ACTION_SCHEMA_HINT, indent=2),
            )
            user = (
                "Decide actions for these workers. Return one entry per worker.\n\n"
                + json.dumps(payload, indent=2)
            )
            raw = llm.chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            if not isinstance(raw, dict):
                raise ValueError(f"expected object, got {type(raw).__name__}")
            for aid in obs_by_agent:
                entry = raw.get(aid)
                if isinstance(entry, dict):
                    out[aid] = _parse_action(aid, entry)
                else:
                    out[aid] = self._fallbacks[aid].decide(obs_by_agent[aid])
        except Exception as e:
            logger.warning("Batch LLM failed (%s); falling back to honest for all", e)
            for aid, obs in obs_by_agent.items():
                fb = self._fallbacks.get(aid) or HonestWorkerPolicy(agent_id=aid)
                out[aid] = fb.decide(obs)
        return out
