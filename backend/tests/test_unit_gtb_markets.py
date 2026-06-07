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
