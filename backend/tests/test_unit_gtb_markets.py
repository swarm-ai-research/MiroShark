"""Unit tests for GTB market generation + resolution."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gtb_mods():
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
    markets = _load("app.services.gtb_markets", "app/services/gtb_markets.py")
    _ = _load("app.services.gtb_llm_agent", "app/services/gtb_llm_agent.py")
    service = _load("app.services.gtb_service", "app/services/gtb_service.py")
    return service, markets


_METRICS_EPOCH_0 = {
    "epoch": 0,
    "total_production": 50.0,
    "gini_coefficient": 0.30,
    "welfare": 3.0,
    "total_tax_revenue": 5.0,
    "total_audits": 10,
    "total_catches": 1,
    "bunching_intensity": 0.10,
}


class TestGeneration:
    def test_seeds_two_markets_per_known_metric(self, gtb_mods):
        _, markets = gtb_mods
        book = markets.GTBMarketBook()
        out = book.generate(current_epoch=0, latest_metrics=_METRICS_EPOCH_0)
        # 7 known metrics × 2 sides (>, <) = 14 questions.
        assert len(out) == 14
        ops = sorted({m.op for m in out})
        assert ops == ["<", ">"]
        assert all(m.status == "open" for m in out)
        assert all(m.deadline_epoch == 5 for m in out), "default horizon=5"

    def test_no_duplicate_questions(self, gtb_mods):
        _, markets = gtb_mods
        book = markets.GTBMarketBook()
        book.generate(0, _METRICS_EPOCH_0)
        again = book.generate(0, _METRICS_EPOCH_0)
        assert again == [], "second call with same state should add nothing"

    def test_skips_missing_metrics(self, gtb_mods):
        _, markets = gtb_mods
        book = markets.GTBMarketBook()
        partial = {"welfare": 2.0, "gini_coefficient": 0.4, "epoch": 0}
        out = book.generate(0, partial)
        kinds = {m.metric for m in out}
        assert kinds == {"welfare", "gini_coefficient"}
        assert len(out) == 4


class TestResolution:
    def test_threshold_crossing_resolves_yes(self, gtb_mods):
        _, markets = gtb_mods
        book = markets.GTBMarketBook()
        book.generate(0, _METRICS_EPOCH_0)
        # Welfare jumps from 3.0 → 4.0 (> 3.6 upside threshold).
        m2 = dict(_METRICS_EPOCH_0, epoch=1, welfare=4.0)
        resolved = book.on_epoch_close(1, m2)
        welfare_yes = [
            m for m in resolved
            if m.metric == "welfare" and m.op == ">" and m.status == "yes"
        ]
        assert welfare_yes, "welfare > 3.6 should resolve YES on jump to 4.0"
        # The matching < market is still open (welfare didn't fall to 2.4).
        welfare_open = [
            m for m in book.open_markets if m.metric == "welfare" and m.op == "<"
        ]
        assert welfare_open

    def test_deadline_expires_open_markets(self, gtb_mods):
        _, markets = gtb_mods
        book = markets.GTBMarketBook(generator=markets.GTBMarketGenerator(horizon_epochs=2))
        book.generate(0, _METRICS_EPOCH_0)
        # Tick forward without moving metrics: same numbers, epoch 2.
        m2 = dict(_METRICS_EPOCH_0, epoch=2)
        resolved = book.on_epoch_close(2, m2)
        statuses = {m.status for m in resolved}
        # Whatever crossed crosses; the rest must be NO at deadline.
        assert "no" in statuses or "yes" in statuses
        assert all(m.status != "open" for m in resolved)
        assert not book.open_markets


class TestServiceIntegration:
    def test_step_seeds_and_resolves_markets(self, gtb_mods):
        service_mod, _ = gtb_mods
        from worlds.gather_trade_build.config import GTBConfig

        cfg = GTBConfig.from_dict({})
        svc = service_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "honest", "count": 4}],
            steps_per_epoch=2,
            seed=42,
        )
        # Step the world past the first epoch close.
        svc.step()
        tick = svc.step()
        assert tick["epoch_closed"] is True
        # After close, the service should have seeded markets.
        state = svc.state()
        assert state["markets"]["open"], "expected open markets after first close"
        questions = [m["question"] for m in state["markets"]["open"]]
        assert any("welfare" in q or "Gini" in q for q in questions)

    def test_stake_placed_debits_coin_and_payout_on_yes(self, gtb_mods):
        service_mod, _ = gtb_mods
        from worlds.gather_trade_build.config import GTBConfig
        from worlds.gather_trade_build.env import GTBAction
        from worlds.gather_trade_build.entities import GTBActionType

        cfg = GTBConfig.from_dict({})
        svc = service_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "honest", "count": 2}],
            steps_per_epoch=1,
            seed=21,
        )
        svc.step()  # epoch 0 closes, seeds markets
        open_markets = svc.markets()["open"]
        # Pick a "welfare >" market and force-resolve YES by jamming a
        # welfare value above threshold via a custom metrics_dict on the
        # next epoch close. We do this through the StakeBook + MarketBook
        # APIs directly to keep the test focused on payout math.
        target = next(m for m in open_markets if m["metric"] == "welfare" and m["op"] == ">")
        market_id = target["market_id"]

        # Worker 0 stakes YES, worker 1 stakes NO via action piggyback.
        for aid, side in (("worker_0", "yes"), ("worker_1", "no")):
            w = svc._env.workers[aid]
            w.inventory["coin"] = 10.0  # ensure stake-able
        svc.set_action("worker_0", GTBAction(
            agent_id="worker_0", action_type=GTBActionType.NOOP,
            stake_market_id=market_id, stake_side="yes", stake_amount=4.0,
        ))
        svc.set_action("worker_1", GTBAction(
            agent_id="worker_1", action_type=GTBActionType.NOOP,
            stake_market_id=market_id, stake_side="no", stake_amount=2.0,
        ))
        svc.step()
        # Coin should be debited by stake amount post-tick.
        assert abs(svc._env.workers["worker_0"].inventory["coin"] - 6.0) < 1e-6
        assert abs(svc._env.workers["worker_1"].inventory["coin"] - 8.0) < 1e-6
        # Force the market to resolve YES by injecting a sky-high welfare
        # tick into the market book directly (bypasses simulating physics).
        m_obj = svc._market_book.find_open(market_id)
        assert m_obj is not None
        synthetic_metrics = {**target, "welfare": 9999.0, m_obj.metric: 9999.0,
                             "total_production": 0, "gini_coefficient": 0,
                             "total_tax_revenue": 0, "total_audits": 0,
                             "total_catches": 0, "bunching_intensity": 0}
        resolved = svc._market_book.on_epoch_close(svc.epoch, synthetic_metrics)
        for rm in resolved:
            payouts = svc._stake_book.distribute(rm)
            for aid, gross in payouts.items():
                w = svc._env.workers[aid]
                w.inventory["coin"] = w.inventory.get("coin", 0.0) + gross
        # Worker 0 staked 4 YES, worker 1 staked 2 NO; YES wins → worker 0
        # gets back 4 (principal) + 2 (loser pool) = 6. Original 10 - 4 + 6 = 12.
        assert abs(svc._env.workers["worker_0"].inventory["coin"] - 12.0) < 1e-6
        # Worker 1 loses the 2 coin staked; ends at 8.
        assert abs(svc._env.workers["worker_1"].inventory["coin"] - 8.0) < 1e-6

    def test_stake_rejected_on_insufficient_coin(self, gtb_mods):
        service_mod, _ = gtb_mods
        from worlds.gather_trade_build.config import GTBConfig
        from worlds.gather_trade_build.env import GTBAction
        from worlds.gather_trade_build.entities import GTBActionType

        cfg = GTBConfig.from_dict({})
        svc = service_mod.GTBWorldService(
            config=cfg, agent_specs=[{"policy": "honest", "count": 1}],
            steps_per_epoch=1, seed=23,
        )
        svc.step()
        market_id = svc.markets()["open"][0]["market_id"]
        svc._env.workers["worker_0"].inventory["coin"] = 0.5
        svc.set_action("worker_0", GTBAction(
            agent_id="worker_0", action_type=GTBActionType.NOOP,
            stake_market_id=market_id, stake_side="yes", stake_amount=5.0,
        ))
        tick = svc.step()
        rejected = [e for e in tick["events"] if e["event_type"] == "stake_rejected"]
        assert rejected, "stake larger than coin must be rejected"
        # Coin unchanged.
        assert abs(svc._env.workers["worker_0"].inventory["coin"] - 0.5) < 1e-6

    def test_generate_is_idempotent_within_epoch(self, gtb_mods):
        service_mod, _ = gtb_mods
        from worlds.gather_trade_build.config import GTBConfig

        cfg = GTBConfig.from_dict({})
        svc = service_mod.GTBWorldService(
            config=cfg,
            agent_specs=[{"policy": "honest", "count": 2}],
            steps_per_epoch=2,
            seed=7,
        )
        # No metrics yet → generate is a no-op (no closed epoch to read).
        assert svc.generate_markets() == []
        # Drive one epoch close. lazy-seed fires in on_epoch_close.
        svc.step()
        svc.step()
        before = len(svc.markets()["open"])
        assert before > 0
        # Same epoch + same metrics → question strings dedup to no-op.
        again = svc.generate_markets()
        assert again == []
        assert len(svc.markets()["open"]) == before
