"""Unit tests for the GTB world service.

Pure offline — no Flask, no LLM API. Loads the service module directly
(bypassing app/__init__.py's Flask import) so the tests run anywhere the
worlds/ package + pyyaml are available.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent


def _load_module(name: str, relpath: str):
    """Load a module from its file, bypassing parent package __init__."""
    spec = importlib.util.spec_from_file_location(name, _BACKEND / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gtb_modules():
    # Stub the `app`, `app.services`, `app.utils` packages so the lazy
    # `from ..utils.llm_client import create_llm_client` in gtb_llm_agent
    # has a resolvable parent without dragging in Flask.
    for pkg, path in [
        ("app", "app"),
        ("app.services", "app/services"),
        ("app.utils", "app/utils"),
    ]:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [str(_BACKEND / path)]
            sys.modules[pkg] = stub
    sys.path.insert(0, str(_BACKEND))
    llm_mod = _load_module("app.services.gtb_llm_agent", "app/services/gtb_llm_agent.py")
    svc_mod = _load_module("app.services.gtb_service", "app/services/gtb_service.py")
    return svc_mod, llm_mod


@pytest.fixture
def small_world(gtb_modules):
    svc_mod, _ = gtb_modules
    from worlds.gather_trade_build.config import GTBConfig

    cfg = GTBConfig.from_dict({})
    cfg.seed = 13
    return svc_mod.GTBWorldService(
        config=cfg,
        agent_specs=[{"policy": "honest", "count": 4}],
        steps_per_epoch=3,
        seed=13,
    )


class TestServiceBasics:
    def test_start_initializes_workers(self, small_world):
        assert small_world.epoch == 0
        assert small_world.step_in_epoch == 0
        state = small_world.state()
        assert len(state["workers"]) == 4
        assert all(w["policy"] == "HonestWorkerPolicy" for w in state["workers"])
        assert state["tax_brackets"], "tax brackets should be exposed"

    def test_step_advances_and_emits_events(self, small_world):
        tick = small_world.step()
        assert tick["step_in_epoch"] == 1
        assert tick["epoch_closed"] is False
        # Each honest worker emits at least a noop-or-action event per step
        assert len(tick["events"]) >= 1

    def test_epoch_closes_at_steps_per_epoch(self, small_world):
        # Fresh function-scope fixture: steps_per_epoch=3, so the 3rd step closes.
        small_world.step()
        small_world.step()
        closing = small_world.step()
        assert closing["epoch_closed"] is True
        m = closing["metrics"]
        assert m is not None
        for k in ("total_production", "gini_coefficient", "welfare",
                  "total_tax_revenue", "total_audits", "bunching_intensity"):
            assert k in m
        # Service should have rolled over to next epoch.
        assert small_world.epoch == 1
        assert small_world.step_in_epoch == 0


class TestActionOverride:
    def test_set_action_overrides_policy_for_one_tick(self, gtb_modules):
        svc_mod, _ = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig
        from worlds.gather_trade_build.env import GTBAction
        from worlds.gather_trade_build.entities import GTBActionType, Direction

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg, agent_specs=[{"policy": "honest", "count": 2}],
            steps_per_epoch=10, seed=1,
        )
        svc.set_action(
            "worker_0",
            GTBAction(agent_id="worker_0", action_type=GTBActionType.MOVE,
                      direction=Direction.RIGHT),
        )
        tick = svc.step()
        evs = [e for e in tick["events"] if e["agent_id"] == "worker_0"]
        assert evs and evs[0]["event_type"] in ("move", "noop")  # noop if blocked


class TestLLMWorkerPolicy:
    def test_decide_uses_fake_llm(self, gtb_modules):
        _, llm_mod = gtb_modules

        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def chat_json(self, messages, temperature=0.3, max_tokens=300):
                self.calls += 1
                return {"action_type": "gather"}

        fake = FakeLLM()
        p = llm_mod.LLMWorkerPolicy(
            agent_id="w0", persona={"name": "Alice"}, llm_client=fake
        )
        obs = _stub_obs("w0")
        act = p.decide(obs)
        assert fake.calls == 1
        assert act.action_type.value == "gather"

    def test_decide_falls_back_on_llm_error(self, gtb_modules):
        _, llm_mod = gtb_modules

        class BrokenLLM:
            def chat_json(self, **kw):
                raise RuntimeError("API down")

        p = llm_mod.LLMWorkerPolicy(
            agent_id="w0", persona={"name": "X"}, llm_client=BrokenLLM()
        )
        act = p.decide(_stub_obs("w0"))
        # Honest fallback never errors and always returns a valid action.
        assert act.action_type.value in {
            "move", "gather", "build", "trade_buy", "trade_sell", "noop"
        }

    def test_frozen_agent_returns_noop_without_calling_llm(self, gtb_modules):
        _, llm_mod = gtb_modules

        class TrackingLLM:
            def __init__(self):
                self.calls = 0

            def chat_json(self, **kw):
                self.calls += 1
                return {"action_type": "gather"}

        llm = TrackingLLM()
        p = llm_mod.LLMWorkerPolicy(agent_id="w0", llm_client=llm)
        obs = _stub_obs("w0")
        obs["frozen"] = True
        act = p.decide(obs)
        assert act.action_type.value == "noop"
        assert llm.calls == 0


class TestBatchLLMDriver:
    def test_single_call_decides_for_all(self, gtb_modules):
        svc_mod, llm_mod = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig

        class FakeBatchLLM:
            def __init__(self):
                self.calls = 0

            def chat_json(self, messages, temperature=0.4, max_tokens=2048):
                self.calls += 1
                return {
                    "worker_0": {"action_type": "gather"},
                    "worker_1": {"action_type": "move", "direction": "left"},
                }

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg,
            agent_specs=[
                {"policy": "llm_batched", "persona": {"name": "Alice"}, "count": 1},
                {"policy": "llm_batched", "persona": {"name": "Bob"}, "count": 1},
                {"policy": "honest", "count": 1},
            ],
            steps_per_epoch=10,
            seed=2,
        )
        fake = FakeBatchLLM()
        svc.attach_batch_driver(
            llm_mod.BatchLLMDriver(
                agent_ids=["worker_0", "worker_1"],
                personas={"worker_0": {"name": "Alice"}, "worker_1": {"name": "Bob"}},
                llm_client=fake,
            )
        )

        # Three ticks: only three LLM calls (one per tick), not three × two.
        for _ in range(3):
            svc.step()
        assert fake.calls == 3, "batch driver should call LLM once per tick"

    def test_batch_falls_back_on_missing_entry(self, gtb_modules):
        svc_mod, llm_mod = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig

        class PartialLLM:
            def chat_json(self, messages, temperature=0.4, max_tokens=2048):
                # Returns an entry only for worker_0; worker_1 should
                # fall through to honest fallback.
                return {"worker_0": {"action_type": "gather"}}

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg,
            agent_specs=[
                {"policy": "llm_batched", "count": 2},
            ],
            steps_per_epoch=10,
            seed=3,
        )
        svc.attach_batch_driver(
            llm_mod.BatchLLMDriver(
                agent_ids=["worker_0", "worker_1"], llm_client=PartialLLM()
            )
        )
        tick = svc.step()
        events_by_agent = {e["agent_id"]: e for e in tick["events"]}
        assert "worker_0" in events_by_agent
        assert "worker_1" in events_by_agent  # honest fallback fired


class TestMarketsInObservation:
    def test_llm_policy_sees_open_markets(self, gtb_modules):
        svc_mod, llm_mod = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig

        captured = {}

        class CapturingLLM:
            def chat_json(self, messages, temperature=0.4, max_tokens=300):
                # Stash the user prompt so the test can read what the
                # service handed to the LLM.
                captured["user"] = messages[-1]["content"]
                return {"action_type": "gather"}

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "honest", "count": 1}],
            steps_per_epoch=1,  # close fast so markets seed quickly
            seed=99,
        )
        # Drive one epoch close so the market book seeds.
        svc.step()
        assert svc.markets()["open"], "precondition: markets should be open"

        # Install an LLM policy now and run a second step.
        svc._policies["worker_0"] = llm_mod.LLMWorkerPolicy(
            agent_id="worker_0", llm_client=CapturingLLM()
        )
        svc.step()
        assert "open_markets" in captured["user"], (
            "the LLM prompt should surface open_markets so agents can act on them"
        )

    def test_batch_driver_sees_open_markets(self, gtb_modules):
        svc_mod, llm_mod = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig

        captured = {}

        class CapturingBatchLLM:
            def chat_json(self, messages, temperature=0.4, max_tokens=2048):
                captured["user"] = messages[-1]["content"]
                return {"worker_0": {"action_type": "gather"}}

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "llm_batched", "count": 1}],
            steps_per_epoch=1,
            seed=101,
        )
        svc.step()  # seeds markets at epoch 0 close
        svc.attach_batch_driver(
            llm_mod.BatchLLMDriver(
                agent_ids=["worker_0"], llm_client=CapturingBatchLLM()
            )
        )
        svc.step()
        assert "open_markets" in captured["user"]


class TestPlannerStatsFromSnapshot:
    def test_planner_sees_closed_epoch_not_fresh_zeros(self, gtb_modules):
        svc_mod, _ = gtb_modules
        from worlds.gather_trade_build.config import GTBConfig
        from worlds.gather_trade_build.env import GTBAction
        from worlds.gather_trade_build.entities import GTBActionType

        cfg = GTBConfig.from_dict({})
        svc = svc_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "honest", "count": 3}],
            steps_per_epoch=1,
            seed=44,
        )
        # Capture every stats dict the planner sees.
        seen = []
        orig_update = svc._planner.update
        svc._planner.update = lambda stats: (seen.append(dict(stats)), orig_update(stats))[1]

        # Stage some non-zero gross income on each worker so end_epoch
        # produces real numbers, and verify the planner gets THOSE — not
        # the post-reset zeros that env.get_aggregate_stats() would yield.
        for w in svc._env.workers.values():
            w.gross_income_this_epoch = 5.0
            w.cumulative_income = 5.0
        svc.set_action("worker_0", GTBAction(
            agent_id="worker_0", action_type=GTBActionType.NOOP,
        ))
        svc.step()  # closes epoch 0 → calls planner.update

        assert seen, "planner.update must have been called on epoch close"
        last = seen[-1]
        # total_income reflects the staged 15.0 (3 workers × 5.0), NOT 0.
        assert last["total_income"] > 0, (
            f"planner saw post-reset zeros: {last}"
        )


class TestRegistry:
    def test_registry_round_trip(self, gtb_modules):
        svc_mod, _ = gtb_modules
        reg = svc_mod.GTBWorldRegistry()
        svc = reg.start(
            "sim-x", overrides={"simulation": {"steps_per_epoch": 4, "seed": 9}}
        )
        assert reg.get("sim-x") is svc
        assert "sim-x" in reg.list_ids()
        assert reg.stop("sim-x") is True
        assert reg.get("sim-x") is None


def _stub_obs(agent_id: str):
    return {
        "agent_id": agent_id,
        "position": (0, 0),
        "inventory": {"wood": 0.0, "stone": 0.0, "coin": 0.0},
        "energy": 100.0,
        "houses_built": 0,
        "gross_income": 0.0,
        "deferred_income": 0.0,
        "epoch": 0,
        "step": 0,
        "tax_schedule": {"brackets": [{"threshold": 0.0, "rate": 0.1}]},
        "visible_cells": [],
        "frozen": False,
    }
