"""GTB world service — owns a GTBEnvironment per simulation.

Wraps the headless `worlds.gather_trade_build` kernel so the Flask app can
start a world, advance ticks, close epochs, and snapshot state via HTTP.
Phase 2: policies drive workers (default: honest). Phase 3 will replace
policy-driven actions with LLM-emitted actions per agent.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from worlds.gather_trade_build.agents import (
    CollusiveWorkerPolicy,
    EvasiveWorkerPolicy,
    GamingWorkerPolicy,
    HonestWorkerPolicy,
)
from worlds.gather_trade_build.config import GTBConfig
from worlds.gather_trade_build.entities import ResourceType
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment
from worlds.gather_trade_build.metrics import GTBMetrics, compute_gtb_metrics
from worlds.gather_trade_build.planner import PlannerAgent

logger = logging.getLogger(__name__)


_DEFAULT_SCENARIO = (
    Path(__file__).resolve().parents[2]
    / "worlds"
    / "gather_trade_build"
    / "scenarios"
    / "ai_economist_full.yaml"
)


def _make_policy(spec: Dict[str, Any], agent_id: str, seed: int):
    kind = spec.get("policy", "honest")
    if kind == "honest":
        return HonestWorkerPolicy(agent_id=agent_id, seed=seed)
    if kind == "gaming":
        return GamingWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            shift_fraction=spec.get("shift_fraction", 0.2),
        )
    if kind == "evasive":
        return EvasiveWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            underreport_fraction=spec.get("underreport_fraction", 0.3),
        )
    if kind == "collusive":
        return CollusiveWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            coalition_id=spec.get("coalition_id", "default"),
        )
    if kind == "llm":
        from .gtb_llm_agent import LLMWorkerPolicy
        return LLMWorkerPolicy(
            agent_id=agent_id,
            persona=spec.get("persona"),
            seed=seed,
            temperature=spec.get("temperature", 0.4),
        )
    if kind == "llm_batched":
        # The world drives these via a single batched LLM call per tick.
        # The per-agent policy is honest-fallback used if the batch fails.
        return HonestWorkerPolicy(agent_id=agent_id, seed=seed)
    raise ValueError(f"Unknown policy: {kind}")


class GTBWorldService:
    """One GTB world instance, ticked by external callers."""

    def __init__(
        self,
        config: GTBConfig,
        agent_specs: List[Dict[str, Any]],
        steps_per_epoch: int = 15,
        seed: int = 42,
    ) -> None:
        self._config = config
        self._steps_per_epoch = steps_per_epoch
        self._seed = seed
        self._lock = threading.Lock()

        self._env = GTBEnvironment(config)
        self._policies = {}
        self._batch_personas: Dict[str, Dict[str, Any]] = {}
        next_id = 0
        for spec in agent_specs:
            for _ in range(spec.get("count", 1)):
                agent_id = f"worker_{next_id}"
                next_id += 1
                self._env.add_worker(
                    agent_id=agent_id,
                    skill_gather=spec.get("skill_gather", 1.0),
                    skill_build=spec.get("skill_build", 1.0),
                )
                self._policies[agent_id] = _make_policy(
                    spec, agent_id, seed + next_id
                )
                if spec.get("policy") == "collusive":
                    self._env.workers[agent_id].coalition_id = spec.get(
                        "coalition_id", "default"
                    )
                if spec.get("policy") == "llm_batched":
                    self._batch_personas[agent_id] = spec.get("persona") or {"name": agent_id}

        self._planner = PlannerAgent(
            config.planner, self._env.tax_schedule, seed=seed
        )
        self._epoch_metrics: List[GTBMetrics] = []
        self._action_overrides: Dict[str, GTBAction] = {}
        self._step_in_epoch = 0
        from .gtb_markets import GTBMarketBook
        self._market_book = GTBMarketBook()
        self._batch_driver = None
        if self._batch_personas:
            from .gtb_llm_agent import BatchLLMDriver
            self._batch_driver = BatchLLMDriver(
                agent_ids=list(self._batch_personas.keys()),
                personas=dict(self._batch_personas),
            )

    def attach_batch_driver(self, driver) -> None:
        """Replace the auto-created BatchLLMDriver (e.g. with a fake in tests)."""
        with self._lock:
            self._batch_driver = driver

    @property
    def epoch(self) -> int:
        return self._env.current_epoch

    @property
    def step_in_epoch(self) -> int:
        return self._step_in_epoch

    def set_action(self, agent_id: str, action: GTBAction) -> None:
        """Override the next-step action for one agent. Cleared after step()."""
        with self._lock:
            self._action_overrides[agent_id] = action

    def _obs_with_markets(self, agent_id: str) -> Dict[str, Any]:
        """Env obs + open markets, so LLM agents can reason about how
        their actions push the metric stream toward open YES/NO questions."""
        obs = self._env.obs(agent_id)
        obs["open_markets"] = [
            {
                "market_id": m.market_id,
                "question": m.question,
                "metric": m.metric,
                "op": m.op,
                "threshold": m.threshold,
                "deadline_epoch": m.deadline_epoch,
            }
            for m in self._market_book.open_markets
        ]
        return obs

    def step(self) -> Dict[str, Any]:
        """Advance one step. Returns a list of event dicts."""
        with self._lock:
            batch_actions: Dict[str, GTBAction] = {}
            if self._batch_driver and self._batch_driver.agent_ids:
                batch_obs = {
                    aid: self._obs_with_markets(aid)
                    for aid in self._batch_driver.agent_ids
                    if aid in self._policies
                }
                batch_actions = self._batch_driver.decide_all(batch_obs)

            actions: Dict[str, GTBAction] = {}
            for agent_id, policy in self._policies.items():
                if agent_id in self._action_overrides:
                    actions[agent_id] = self._action_overrides[agent_id]
                elif agent_id in batch_actions:
                    actions[agent_id] = batch_actions[agent_id]
                else:
                    obs = self._obs_with_markets(agent_id)
                    actions[agent_id] = policy.decide(obs)
            events = self._env.apply_actions(actions)
            self._action_overrides.clear()
            self._step_in_epoch += 1
            should_close = self._step_in_epoch >= self._steps_per_epoch
            event_dicts = [self._event_to_dict(e) for e in events]
            metrics_dict: Optional[Dict[str, Any]] = None
            if should_close:
                metrics_dict = self._close_epoch_locked()
            return {
                "epoch": self._env.current_epoch,
                "step_in_epoch": self._step_in_epoch,
                "events": event_dicts,
                "epoch_closed": should_close,
                "metrics": metrics_dict,
            }

    def _close_epoch_locked(self) -> Dict[str, Any]:
        epoch_events = list(self._env.detect_collusion())
        result = self._env.end_epoch()
        epoch_events.extend(result.events)
        for policy in self._policies.values():
            if isinstance(policy, EvasiveWorkerPolicy):
                policy.reset_epoch()
        metrics = compute_gtb_metrics(
            workers=result.snapshot,
            events=epoch_events,
            epoch=self._env.current_epoch - 1,
            bracket_thresholds=self._env.tax_schedule.bracket_thresholds,
            prod_weight=self._config.planner.prod_weight,
            ineq_weight=self._config.planner.ineq_weight,
        )
        self._epoch_metrics.append(metrics)
        metrics_dict = self._metrics_to_dict(metrics)
        # Resolve open markets and lazily seed new ones so there's
        # always something to bet on.
        self._market_book.on_epoch_close(metrics.epoch, metrics_dict)
        if not self._market_book.open_markets:
            self._market_book.generate(metrics.epoch, metrics_dict)
        if self._planner.should_update(self._env.current_epoch):
            stats = self._env.get_aggregate_stats()
            self._planner.update(stats)
        self._step_in_epoch = 0
        return metrics_dict

    def state(self) -> Dict[str, Any]:
        """JSON-serializable snapshot of the world."""
        with self._lock:
            workers = []
            for w in self._env.workers.values():
                workers.append({
                    "agent_id": w.agent_id,
                    "position": list(w.position),
                    "inventory": dict(w.inventory),
                    "gross_income_this_epoch": w.gross_income_this_epoch,
                    "tax_paid_this_epoch": w.tax_paid_this_epoch,
                    "cumulative_income": w.cumulative_income,
                    "houses_built": w.houses_built,
                    "energy": w.energy,
                    "skill_gather": w.skill_gather,
                    "skill_build": w.skill_build,
                    "times_audited": w.times_audited,
                    "times_caught": w.times_caught,
                    "coalition_id": w.coalition_id,
                    "policy": type(self._policies[w.agent_id]).__name__,
                })
            houses = [
                {
                    "owner_id": h.owner_id,
                    "position": list(h.position),
                    "build_step": h.build_step,
                }
                for h in self._env.houses
            ]
            brackets = [
                {"threshold": b.threshold, "rate": b.rate}
                for b in self._env.tax_schedule.brackets
            ]
            last_metrics = (
                self._metrics_to_dict(self._epoch_metrics[-1])
                if self._epoch_metrics
                else None
            )
            return {
                "config": {
                    "map": {
                        "height": self._config.map.height,
                        "width": self._config.map.width,
                    },
                    "steps_per_epoch": self._steps_per_epoch,
                    "seed": self._seed,
                },
                "epoch": self._env.current_epoch,
                "step_in_epoch": self._step_in_epoch,
                "workers": workers,
                "houses": houses,
                "tax_brackets": brackets,
                "resources": self._resource_grid(),
                "last_metrics": last_metrics,
                "metrics_history": [
                    self._metrics_to_dict(m) for m in self._epoch_metrics
                ],
                "markets": self._market_book.to_dict(),
            }

    def generate_markets(self) -> List[Dict[str, Any]]:
        """Force-seed markets from the latest metrics. Returns new market dicts."""
        with self._lock:
            if not self._epoch_metrics:
                return []
            last = self._epoch_metrics[-1]
            latest = self._metrics_to_dict(last)
            # Anchor on the metrics' own epoch so close-time seeding and
            # manual seeding produce identical question strings → dedup
            # is stable within an epoch.
            new = self._market_book.generate(last.epoch, latest)
            return [m.to_dict() for m in new]

    def markets(self) -> Dict[str, Any]:
        with self._lock:
            return self._market_book.to_dict()

    def _resource_grid(self) -> List[Dict[str, Any]]:
        out = []
        for row in self._env._grid:
            for cell in row:
                if cell.resource and cell.resource.amount > 0:
                    out.append({
                        "position": list(cell.position),
                        "type": cell.resource.resource_type.value,
                        "amount": cell.resource.amount,
                    })
        return out

    @staticmethod
    def _event_to_dict(evt) -> Dict[str, Any]:
        return {
            "event_type": evt.event_type,
            "epoch": evt.epoch,
            "step": evt.step,
            "agent_id": evt.agent_id,
            "details": evt.details,
        }

    @staticmethod
    def _metrics_to_dict(m: GTBMetrics) -> Dict[str, Any]:
        return {
            "epoch": m.epoch,
            "total_production": m.total_production,
            "gini_coefficient": m.gini_coefficient,
            "welfare": m.welfare,
            "total_tax_revenue": m.total_tax_revenue,
            "total_audits": m.total_audits,
            "total_catches": m.total_catches,
            "bunching_intensity": m.bunching_intensity,
        }


class GTBWorldRegistry:
    """Process-wide registry of running GTB worlds, keyed by sim_id."""

    def __init__(self) -> None:
        self._worlds: Dict[str, GTBWorldService] = {}
        self._lock = threading.Lock()

    def start(
        self,
        sim_id: str,
        scenario_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> GTBWorldService:
        path = Path(scenario_path) if scenario_path else _DEFAULT_SCENARIO
        with open(path) as f:
            data = yaml.safe_load(f)
        if overrides:
            data = {**data, **overrides}
        config = GTBConfig.from_dict(data.get("domain", {}))
        sim = data.get("simulation", {})
        seed = (overrides or {}).get("seed") or sim.get("seed", 42)
        steps_per_epoch = sim.get("steps_per_epoch", 15)
        config.seed = seed
        agent_specs = data.get("agents", [{"policy": "honest", "count": 5}])

        service = GTBWorldService(
            config=config,
            agent_specs=agent_specs,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
        )
        with self._lock:
            self._worlds[sim_id] = service
        logger.info("GTB world started for sim_id=%s (%d workers)",
                    sim_id, len(service._policies))
        return service

    def get(self, sim_id: str) -> Optional[GTBWorldService]:
        with self._lock:
            return self._worlds.get(sim_id)

    def stop(self, sim_id: str) -> bool:
        with self._lock:
            return self._worlds.pop(sim_id, None) is not None

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._worlds.keys())


_registry: Optional[GTBWorldRegistry] = None


def get_registry() -> GTBWorldRegistry:
    global _registry
    if _registry is None:
        _registry = GTBWorldRegistry()
    return _registry
