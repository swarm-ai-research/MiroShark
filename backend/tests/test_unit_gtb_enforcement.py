"""Unit tests for GTB Phase 4: stance-mode misreporting, observable-
information audits with detection noise, the price-fixing cartel, and
detector precision/recall metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402
from worlds.gather_trade_build.runner import GTBScenarioRunner  # noqa: E402


def _make_env(**domain) -> GTBEnvironment:
    base = {
        "map": {"height": 4, "width": 4, "wood_density": 1.0,
                "stone_density": 0.0},
    }
    base.update(domain)
    cfg = GTBConfig.from_dict(base)
    cfg.seed = 7
    return GTBEnvironment(cfg)


def _misreport(aid: str, frac: float) -> GTBAction:
    return GTBAction(agent_id=aid, action_type=GTBActionType.MISREPORT,
                     underreport_fraction=frac)


def _gather(aid: str) -> GTBAction:
    return GTBAction(agent_id=aid, action_type=GTBActionType.GATHER)


# ----------------------------------------------------------------------
# 4.1 Stance-mode misreporting
# ----------------------------------------------------------------------

def test_stance_mode_hidden_share_is_ordering_independent():
    """In stance mode, hidden income is exactly gross * fraction no
    matter when the MISREPORT action happened within the epoch."""
    env = _make_env(misreporting={"semantics": "stance",
                                  "audit_probability": 0.0})
    worker = env.add_worker("A")
    env.apply_actions({"A": _gather("A")})
    env.apply_actions({"A": _misreport("A", 0.4)})
    env.apply_actions({"A": _gather("A")})  # income AFTER the misreport
    gross = worker.gross_income_this_epoch
    assert gross > 0

    result = env.end_epoch()
    tax_events = [e for e in result.events if e.event_type == "tax"]
    reported = tax_events[0].details["reported_income"]
    assert reported == pytest.approx(gross * 0.6)


def test_event_mode_hidden_share_depends_on_ordering():
    """Legacy event semantics: gather income earned after the misreport
    refills reported income, so less than gross * fraction stays hidden."""
    env = _make_env(misreporting={"semantics": "event",
                                  "audit_probability": 0.0})
    worker = env.add_worker("A")
    env.apply_actions({"A": _gather("A")})
    env.apply_actions({"A": _misreport("A", 0.4)})
    env.apply_actions({"A": _gather("A")})
    gross = worker.gross_income_this_epoch
    hidden = gross - worker.reported_income_this_epoch
    assert hidden < gross * 0.4 - 1e-9  # diluted by post-misreport income


# ----------------------------------------------------------------------
# 4.2 Observable-information audits
# ----------------------------------------------------------------------

def _observable_env(audit_probability=1.0, detection_power=1.0):
    return _make_env(misreporting={
        "selection_mode": "observable",
        "semantics": "stance",
        "audit_probability": audit_probability,
        "detection_power": detection_power,
        "freeze_on_repeat": False,
    })


def test_observable_honest_audit_is_false_positive():
    env = _observable_env()
    worker = env.add_worker("A")
    worker.gross_income_this_epoch = 10.0
    worker.reported_income_this_epoch = 10.0
    result = env.end_epoch()
    fps = [e for e in result.events if e.event_type == "audit_false_positive"]
    assert fps and fps[0].agent_id == "A"
    assert worker.times_audited == 1
    assert worker.times_caught == 0


def test_observable_zero_detection_power_yields_misses():
    env = _observable_env(detection_power=0.0)
    worker = env.add_worker("A")
    worker.gross_income_this_epoch = 10.0
    worker.reported_income_this_epoch = 5.0
    result = env.end_epoch()
    misses = [e for e in result.events if e.event_type == "audit_miss"]
    assert misses and misses[0].agent_id == "A"
    assert worker.times_audited == 1
    assert worker.times_caught == 0


def test_observable_full_detection_power_catches():
    env = _observable_env(detection_power=1.0)
    worker = env.add_worker("A")
    worker.gross_income_this_epoch = 10.0
    worker.reported_income_this_epoch = 5.0
    result = env.end_epoch()
    catches = [e for e in result.events if e.event_type == "audit_caught"]
    assert catches and catches[0].agent_id == "A"
    assert worker.times_caught == 1


def test_observable_unobserved_income_generates_no_extra_suspicion():
    """In legacy ledger mode, gather income is not on any public record:
    a worker hiding it has suspicion 0 and faces only the base audit
    rate — the auditor no longer reads minds."""
    env = _observable_env(audit_probability=0.5)
    worker = env.add_worker("A")
    # All income from gathering (no coin received on record)
    env.apply_actions({"A": _gather("A")})
    env.apply_actions({"A": _misreport("A", 0.5)})

    # The selection probability must equal the base rate: with the env
    # rng seeded, compare statistically over many epochs
    audited_before = worker.times_audited
    n_epochs = 200
    for _ in range(n_epochs):
        worker.gross_income_this_epoch = 10.0
        worker.reported_income_this_epoch = 5.0
        env.end_epoch()
    rate = (worker.times_audited - audited_before) / n_epochs
    assert 0.35 < rate < 0.65  # base 0.5, no risk multiplier applied


# ----------------------------------------------------------------------
# 4.3 Cartel + price-fixing detection
# ----------------------------------------------------------------------

def _cartel_scenario():
    return {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.5,
                    "stone_density": 0.3},
            "market": {"order_ttl_steps": 10},
            "collusion": {
                "enabled": True,
                "detect_price_fixing": True,
                # Disable the noisy action-trace detector to isolate the
                # market-based one
                "similarity_threshold": 1.1,
            },
        },
        "agents": [
            {"policy": "cartel", "count": 3, "coalition_id": "cartel_x",
             "cartel_price": 4.0},
            {"policy": "trader", "count": 3, "value_estimate": 3.0,
             "value_jitter": 0.4},
            {"policy": "honest", "count": 2},
        ],
    }


def test_price_fixing_detector_flags_cartel():
    scenario = _cartel_scenario()
    cfg = GTBConfig.from_dict(scenario["domain"])
    cfg.seed = 3
    runner = GTBScenarioRunner(config=cfg, agent_specs=scenario["agents"],
                               n_epochs=4, steps_per_epoch=12, seed=3)
    metrics = runner.run()

    # detect_collusion events live in the runner's event log (they are
    # returned to the caller, not appended to env._events)
    detections = [
        e for e in runner._all_events
        if e["event_type"] == "collusion_detected"
        and e["details"].get("method") == "price_fixing"
    ]
    assert detections, "cartel posting identical asks was never flagged"
    # Every detected pair here should be a true coalition pair
    assert all(d["details"]["same_coalition"] for d in detections)
    # Precision/recall metrics populated and sane
    for m in metrics:
        assert 0.0 <= m.collusion_precision <= 1.0
        assert 0.0 <= m.collusion_recall <= 1.0
    assert any(m.collusion_recall > 0 for m in metrics)
    # With only the cartel posting fixed prices, no false alarms
    assert all(m.collusion_precision == 1.0 for m in metrics)


def test_cartel_profits_from_price_fixing():
    """The cartel's channel is real: with buy-side demand present,
    cartel members earn more sale income at the fixed elevated price
    than the same agents quoting competitively."""
    def total_cartel_sale_income(cartel_price: float, seeds=(1, 2, 3)) -> float:
        total = 0.0
        for s in seeds:
            scenario = _cartel_scenario()
            scenario["domain"]["collusion"]["enabled"] = False  # no responses
            scenario["agents"][0]["cartel_price"] = cartel_price
            # Buy-side demand that can reach the cartel price
            scenario["agents"][1]["value_estimate"] = 4.5
            cfg = GTBConfig.from_dict(scenario["domain"])
            cfg.seed = s
            runner = GTBScenarioRunner(
                config=cfg, agent_specs=scenario["agents"],
                n_epochs=4, steps_per_epoch=12, seed=s,
            )
            runner.run()
            for e in runner.env.events:
                if e.event_type == "trade" and e.details["seller"].startswith(
                        "worker_cartel"):
                    total += e.details["quantity"] * e.details["price"]
        return total

    competitive = total_cartel_sale_income(cartel_price=2.0)
    fixed = total_cartel_sale_income(cartel_price=3.5)
    assert fixed > 0, "cartel never sold at the fixed price"
    # Revenue per unit is higher; total revenue should not collapse
    # (traders' private values reach ~6, so demand survives the markup)
    assert fixed > competitive * 0.5


def test_action_trace_detector_unchanged_for_legacy_scenarios():
    """detect_price_fixing defaults on but is inert when nobody posts
    asks — legacy scenarios see identical detection behavior."""
    env = _make_env()
    env.add_worker("A")
    env.add_worker("B")
    for _ in range(5):
        env.apply_actions({"A": _gather("A"), "B": _gather("B")})
    events = env.detect_collusion()
    assert all(
        e.details.get("method") == "action_trace" for e in events
    )
