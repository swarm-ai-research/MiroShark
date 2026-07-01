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
    CartelWorkerPolicy,
    CollusiveWorkerPolicy,
    EvasiveWorkerPolicy,
    GamingWorkerPolicy,
    HonestWorkerPolicy,
    RationalWorkerPolicy,
    ZITraderPolicy,
)
from worlds.gather_trade_build.config import GTBConfig
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


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict-into-dict. Non-dict values (scalars, lists)
    are replaced, not merged — list-replace is the right default for
    explicit configs like 'agents:'. Only nested dicts compose.

    Used so a caller can do overrides={"simulation": {"seed": 7}}
    without wiping the rest of the simulation block from the scenario
    YAML.
    """
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _stats_from_snapshot(snapshot, env) -> Dict[str, float]:
    """Aggregate planner stats from a pre-reset worker snapshot.

    end_epoch() returns a snapshot taken BEFORE per-epoch counters are
    cleared. Reading from there gives the planner the closed-epoch
    income / tax / Gini it actually wants to react to. Delegates to the
    kernel so the service and the headless runner stay in lockstep.
    """
    return env.stats_from_snapshot(snapshot)


def _make_policy(spec: Dict[str, Any], agent_id: str, seed: int):
    kind = spec.get("policy", "honest")
    if kind == "honest":
        return HonestWorkerPolicy(agent_id=agent_id, seed=seed)
    if kind == "rational":
        return RationalWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            eta=spec.get("eta", 0.35),
            labor_coeff=spec.get("labor_coeff", 0.15),
        )
    if kind == "trader":
        return ZITraderPolicy(
            agent_id=agent_id,
            seed=seed,
            value_estimate=spec.get("value_estimate", 2.0),
            value_jitter=spec.get("value_jitter", 0.5),
            order_qty=spec.get("order_qty", 1.0),
        )
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
    if kind == "cartel":
        return CartelWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            coalition_id=spec.get("coalition_id", "default"),
            cartel_price=spec.get("cartel_price", 4.0),
            order_qty=spec.get("order_qty", 1.0),
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
    if kind == "maker":
        from worlds.gather_trade_build.agents import MakerWorkerPolicy
        return MakerWorkerPolicy(
            agent_id=agent_id,
            seed=seed,
            sell_markup=spec.get("sell_markup", 0.2),
            buy_discount=spec.get("buy_discount", 0.2),
            target_inventory=spec.get("target_inventory", 5.0),
        )
    if kind == "market_aware":
        from worlds.gather_trade_build.agents import MarketAwareHonestPolicy
        return MarketAwareHonestPolicy(
            agent_id=agent_id,
            seed=seed,
            build_wood_threshold=spec.get("build_wood_threshold", 3.0),
            build_stone_threshold=spec.get("build_stone_threshold", 3.0),
        )
    if kind == "tax_aware":
        from worlds.gather_trade_build.agents import TaxAwareHonestPolicy
        return TaxAwareHonestPolicy(
            agent_id=agent_id,
            seed=seed,
            rate_threshold=spec.get("rate_threshold", 0.30),
            effort_suppression=spec.get("effort_suppression", 0.7),
        )
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
        self._batch_temperatures: Dict[str, float] = {}
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
                if spec.get("policy") in ("collusive", "cartel"):
                    self._env.workers[agent_id].coalition_id = spec.get(
                        "coalition_id", "default"
                    )
                if spec.get("policy") == "llm_batched":
                    self._batch_personas[agent_id] = spec.get("persona") or {"name": agent_id}
                    self._batch_temperatures[agent_id] = float(spec.get("temperature", 0.4))

        self._planner = PlannerAgent(
            config.planner, self._env.tax_schedule, seed=seed
        )
        self._epoch_metrics: List[GTBMetrics] = []
        # Step events buffered for epoch metrics (trades, misreports,
        # shifts happen mid-epoch; metrics must see them, as the
        # headless runner does)
        self._step_event_buffer: List[Any] = []
        # Per-worker summary of the previous epoch, injected into
        # observations so agents (LLM especially) can reason about how
        # their tax strategy actually played out
        self._last_epoch_worker_summary: Dict[str, Dict[str, Any]] = {}
        self._action_overrides: Dict[str, GTBAction] = {}
        self._step_in_epoch = 0
        from .gtb_markets import GTBMarketBook, GTBStakeBook
        self._market_book = GTBMarketBook()
        self._stake_book = GTBStakeBook()
        self._batch_driver = None
        if self._batch_personas:
            from .gtb_llm_agent import BatchLLMDriver
            temps = set(self._batch_temperatures.values())
            if len(temps) > 1:
                raise ValueError(
                    f"BatchLLMDriver requires a uniform temperature across llm_batched "
                    f"personas; got {sorted(temps)}"
                )
            self._batch_driver = BatchLLMDriver(
                agent_ids=list(self._batch_personas.keys()),
                personas=dict(self._batch_personas),
                temperature=next(iter(temps)),
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

    def _place_stake_locked(
        self, agent_id: str, market_id: str, side: str, amount: float,
    ):
        """Validate + escrow one stake. CALLER MUST HOLD ``self._lock``.

        The single source of truth for both stake pathways — the HTTP
        ``place_stake()`` and the action-piggybacked ``_process_stakes_locked()``
        — so escrow/validation never diverge between them. Returns a tuple
        ``(ok, reason, stake, coin)``: on success ``(True, None, stake,
        coin_before)``; on failure ``(False, reason_str, None, coin_seen)``.
        """
        market = self._market_book.find_open(market_id)
        if market is None:
            return (False, "no_such_open_market", None, 0.0)
        worker = self._env.workers.get(agent_id)
        if worker is None:
            return (False, "no_such_agent", None, 0.0)
        coin = worker.inventory.get("coin", 0.0)
        amount = float(amount)
        stake = self._stake_book.place(
            agent_id=agent_id, market=market, side=side,
            amount=amount, worker_coin=coin, epoch=self._env.current_epoch,
        )
        if stake is None:
            return (False, "invalid_or_insufficient_coin", None, coin)
        # Escrow: debit the worker's coin now; payouts on resolve.
        worker.inventory["coin"] = max(0.0, coin - amount)
        self._env.register_external_coin(-amount, "stake_escrow")
        return (True, None, stake, coin)

    def place_stake(
        self, agent_id: str, market_id: str, side: str, amount: float,
    ) -> Dict[str, Any]:
        """Public stake-place entry point used by the inbound HTTP route.

        External Polymarket bots / UI clients call this to take a YES or
        NO position on an open GTB market without having to be one of
        the LLM-driven workers. Shares validation + escrow with the
        action-piggybacked path via ``_place_stake_locked``.
        """
        with self._lock:
            ok, reason, stake, coin = self._place_stake_locked(
                agent_id, market_id, side, amount,
            )
            if not ok:
                if reason == "no_such_open_market":
                    return {"ok": False, "reason": reason,
                            "market_id": market_id}
                if reason == "no_such_agent":
                    return {"ok": False, "reason": reason, "agent_id": agent_id}
                return {"ok": False, "reason": reason, "market_id": market_id,
                        "side": side, "amount": amount, "coin": coin}
            worker = self._env.workers[agent_id]
            return {"ok": True, "stake": stake.to_dict(),
                    "remaining_coin": worker.inventory["coin"]}

    def _process_stakes_locked(self, actions: Dict[str, GTBAction]) -> List[Any]:
        """Apply any stake intents piggy-backed on this tick's actions.

        Returns a list of GTBEvent-shaped dicts (we synthesize them locally
        rather than touching env's GTBEvent class). Shares validation +
        escrow with ``place_stake`` via ``_place_stake_locked``."""
        from worlds.gather_trade_build.entities import GTBEvent
        out: List[Any] = []
        for agent_id, action in actions.items():
            mid = (action.stake_market_id or "").strip()
            side = (action.stake_side or "").strip().lower()
            amount = float(action.stake_amount or 0.0)
            if not mid or not side or amount <= 0:
                continue
            ok, reason, _stake, coin = self._place_stake_locked(
                agent_id, mid, side, amount,
            )
            if not ok:
                out.append(GTBEvent(
                    event_type="stake_rejected",
                    epoch=self._env.current_epoch,
                    step=self._step_in_epoch,
                    agent_id=agent_id,
                    details={"reason": reason, "market_id": mid, "side": side,
                             "amount": amount, "coin": coin},
                ))
                continue
            out.append(GTBEvent(
                event_type="stake_placed",
                epoch=self._env.current_epoch,
                step=self._step_in_epoch,
                agent_id=agent_id,
                details={"market_id": mid, "side": side, "amount": amount},
            ))
        return out

    def _obs_with_markets(self, agent_id: str) -> Dict[str, Any]:
        """Env obs + open markets, so LLM agents can reason about how
        their actions push the metric stream toward open YES/NO questions."""
        obs = self._env.obs(agent_id)
        obs["last_epoch"] = self._last_epoch_worker_summary.get(agent_id)
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
            stake_events = self._process_stakes_locked(actions)
            events.extend(stake_events)
            self._step_event_buffer.extend(events)
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
        # Include this epoch's step events (trades, misreports, shifts)
        # so metrics see the whole epoch, matching the headless runner.
        epoch_events = list(self._step_event_buffer)
        self._step_event_buffer = []
        epoch_events.extend(self._env.detect_collusion())
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
            utility_config=self._config.utility,
            house_value=(
                self._config.build.wood_cost + self._config.build.stone_cost
            ),
        )
        self._epoch_metrics.append(metrics)
        self._last_epoch_worker_summary = {
            aid: {
                "gross_income": w.gross_income_this_epoch,
                "reported_income": w.reported_income_this_epoch,
                "tax_paid": w.tax_paid_this_epoch,
                "effective_tax_rate": (
                    w.tax_paid_this_epoch
                    / max(w.reported_income_this_epoch, 1e-9)
                ),
                "effort": w.effort_this_epoch,
                "times_audited_total": w.times_audited,
                "times_caught_total": w.times_caught,
            }
            for aid, w in result.snapshot.items()
        }
        metrics_dict = self._metrics_to_dict(metrics)
        # Resolve open markets and lazily seed new ones so there's
        # always something to bet on.
        resolved = self._market_book.on_epoch_close(metrics.epoch, metrics_dict)
        # Pay out stakes on every newly resolved market: credit the
        # winning side's coin directly into worker inventories.
        for m in resolved:
            payouts = self._stake_book.distribute(m)
            for agent_id, gross in payouts.items():
                w = self._env.workers.get(agent_id)
                if w is not None:
                    w.inventory["coin"] = w.inventory.get("coin", 0.0) + gross
                    self._env.register_external_coin(gross, "stake_payouts")
        if not self._market_book.open_markets:
            self._market_book.generate(metrics.epoch, metrics_dict)
        # IMPORTANT: end_epoch() above has already reset each worker's
        # per-epoch income/tax counters and advanced current_epoch, so
        # calling env.get_aggregate_stats() here would feed the planner
        # the *next* (fresh, empty) epoch's zeros. Compute stats from
        # result.snapshot — the pre-reset snapshot — instead, so the
        # heuristic / bandit planner updates on the epoch that actually
        # just closed.
        if self._planner.should_update(self._env.current_epoch):
            stats = _stats_from_snapshot(result.snapshot, self._env)
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
                "stakes": self._stake_book.to_dict(),
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
            "gini_wealth": m.gini_wealth,
            "welfare": m.welfare,
            "welfare_eq_prod": m.welfare_eq_prod,
            "welfare_utilitarian": m.welfare_utilitarian,
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
            data = _deep_merge(data, overrides)
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
