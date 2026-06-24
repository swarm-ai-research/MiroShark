"""Unit tests for GTB Phase 2: isoelastic preferences, labor-responsive
workers, welfare objectives, wealth Gini, and the Saez planner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.agents import RationalWorkerPolicy  # noqa: E402
from worlds.gather_trade_build.config import (  # noqa: E402
    PlannerConfig,
    TaxBracket,
    TaxScheduleConfig,
    UtilityConfig,
)
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType,
    WorkerState,
)
from worlds.gather_trade_build.planner import PlannerAgent  # noqa: E402
from worlds.gather_trade_build.reward import (  # noqa: E402
    compute_isoelastic_utility,
    crra,
    crra_marginal,
)
from worlds.gather_trade_build.sweep import run_one_seed  # noqa: E402
from worlds.gather_trade_build.tax_schedule import TaxSchedule  # noqa: E402


# ----------------------------------------------------------------------
# CRRA utility
# ----------------------------------------------------------------------

def test_crra_is_increasing_and_concave():
    eta = 0.35
    xs = [1.0, 5.0, 20.0, 100.0]
    utils = [crra(x, eta) for x in xs]
    assert utils == sorted(utils)  # increasing
    marginals = [crra_marginal(x, eta) for x in xs]
    assert marginals == sorted(marginals, reverse=True)  # diminishing


def test_crra_eta_zero_is_linear():
    assert crra(10.0, 0.0) == pytest.approx(9.0)  # (x - 1) / 1
    assert crra_marginal(10.0, 0.0) == pytest.approx(1.0)


def test_crra_eta_one_is_log():
    import math
    assert crra(10.0, 1.0) == pytest.approx(math.log(10.0))


def test_isoelastic_utility_penalizes_effort():
    cfg = UtilityConfig(eta=0.35, labor_coeff=0.15, house_weight=5.0)
    rested = WorkerState(agent_id="a")
    rested.inventory["coin"] = 20.0
    tired = WorkerState(agent_id="b")
    tired.inventory["coin"] = 20.0
    tired.cumulative_effort = 50.0
    assert compute_isoelastic_utility(rested, cfg) > compute_isoelastic_utility(
        tired, cfg
    )


# ----------------------------------------------------------------------
# Labor-responsive worker
# ----------------------------------------------------------------------

def _obs(coin: float, gross: float, top_rate: float) -> dict:
    return {
        "position": (0, 0),
        "inventory": {"coin": coin, "wood": 0.0, "stone": 0.0},
        "energy": 10.0,
        "gross_income": gross,
        "houses_built": 0,
        "tax_schedule": {"brackets": [
            {"threshold": 0.0, "rate": 0.1},
            {"threshold": 5.0, "rate": top_rate},
        ]},
        "visible_cells": [
            {"pos": (0, 0), "resource": "wood", "amount": 3.0},
        ],
    }


def test_rational_worker_works_at_low_tax():
    policy = RationalWorkerPolicy("w", eta=0.35, labor_coeff=0.15, seed=1)
    action = policy.decide(_obs(coin=10.0, gross=10.0, top_rate=0.2))
    assert action.action_type != GTBActionType.NOOP


def test_rational_worker_rests_at_confiscatory_tax():
    """Substitution effect: a ~confiscatory marginal rate makes the next
    unit of work not worth its labor disutility."""
    policy = RationalWorkerPolicy("w", eta=0.35, labor_coeff=0.15, seed=1)
    action = policy.decide(_obs(coin=10.0, gross=10.0, top_rate=0.95))
    assert action.action_type == GTBActionType.NOOP


def test_rational_worker_rests_when_rich():
    """Income effect: diminishing marginal utility of coin makes wealthy
    workers choose leisure even at moderate rates."""
    policy = RationalWorkerPolicy("w", eta=0.35, labor_coeff=0.15, seed=1)
    action = policy.decide(_obs(coin=10_000.0, gross=10.0, top_rate=0.2))
    assert action.action_type == GTBActionType.NOOP


# ----------------------------------------------------------------------
# Laffer curve (Phase 2 acceptance)
# ----------------------------------------------------------------------

def _revenue_at_rate(rate: float, seeds=(1, 2, 3)) -> float:
    scenario = {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.4,
                    "stone_density": 0.2},
            "taxation": {"brackets": [{"threshold": 0.0, "rate": rate}]},
            # Static schedule: isolate the worker response. `rl` is now a
            # live REINFORCE planner (P1 trio), so hold it inert by pushing
            # the update interval past the 5-epoch horizon rather than
            # relying on the old no-op stub behaviour.
            "planner": {"planner_type": "rl", "update_interval_epochs": 999},
        },
        "agents": [{"policy": "rational", "count": 6}],
        "simulation": {"n_epochs": 5, "steps_per_epoch": 10},
    }
    total = 0.0
    for s in seeds:
        run = run_one_seed(scenario, seed=s)
        total += sum(m["total_tax_revenue"] for m in run["metrics"])
    return total / len(seeds)


def test_laffer_curve_has_interior_peak():
    """With labor-responsive workers, revenue cannot be monotone in the
    rate: zero at 0%, and collapsing near-confiscatory rates below the
    interior. (Impossible pre-Phase-2: effort never responded to taxes.)"""
    r0 = _revenue_at_rate(0.0)
    r_mid = _revenue_at_rate(0.5)
    r_high = _revenue_at_rate(0.95)
    assert r0 == pytest.approx(0.0)
    assert r_mid > r_high
    assert r_mid > 0


# ----------------------------------------------------------------------
# Welfare objectives + wealth Gini
# ----------------------------------------------------------------------

def test_metrics_include_wealth_gini_and_new_welfare():
    scenario = {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.3,
                    "stone_density": 0.2},
        },
        "agents": [
            {"policy": "honest", "count": 2, "skill_gather": 1.5},
            {"policy": "honest", "count": 2, "skill_gather": 0.7},
        ],
        "simulation": {"n_epochs": 3, "steps_per_epoch": 8},
    }
    run = run_one_seed(scenario, seed=3)
    last = run["metrics"][-1]
    assert 0.0 <= last["gini_wealth"] <= 1.0
    assert last["welfare_eq_prod"] == pytest.approx(
        (last["total_production"] / 4) * (1.0 - last["gini_wealth"])
    )
    assert "welfare_utilitarian" in last


def test_planner_welfare_objectives():
    schedule = TaxSchedule(TaxScheduleConfig())
    stats = {"mean_income": 10.0, "gini": 0.4, "gini_wealth": 0.6,
             "mean_utility": 2.5}

    eq = PlannerAgent(
        PlannerConfig(objective="eq_times_prod", inequality_measure="wealth"),
        schedule,
    )
    assert eq._compute_welfare(stats) == pytest.approx(10.0 * 0.4)

    util = PlannerAgent(PlannerConfig(objective="utilitarian"), schedule)
    assert util._compute_welfare(stats) == pytest.approx(2.5)


# ----------------------------------------------------------------------
# Saez planner
# ----------------------------------------------------------------------

def _saez_planner(top_rate: float = 0.45) -> PlannerAgent:
    schedule = TaxSchedule(TaxScheduleConfig(brackets=[
        TaxBracket(threshold=0.0, rate=0.1),
        TaxBracket(threshold=10.0, rate=0.2),
        TaxBracket(threshold=50.0, rate=top_rate),
    ]))
    return PlannerAgent(
        PlannerConfig(planner_type="saez", saez_rate_change_cap=0.05),
        schedule, seed=1,
    )


def test_saez_moves_top_rate_toward_inverse_elasticity_target():
    planner = _saez_planner(top_rate=0.45)
    # Thin top tail (zm close to z*): Pareto a is large -> tau* is low
    # -> top rate should fall, capped at 0.05 per update, and the lower
    # brackets must be untouched.
    stats = {"top_threshold": 50.0, "top_mean_income": 55.0}
    brackets = planner.update(stats)
    assert brackets[0].rate == pytest.approx(0.1)
    assert brackets[1].rate == pytest.approx(0.2)
    assert brackets[2].rate == pytest.approx(0.40)  # 0.45 - cap

    # Fat top tail (zm >> z*): a -> ~1, tau* = 1/(1+e) is high -> rises
    planner2 = _saez_planner(top_rate=0.2)
    stats2 = {"top_threshold": 50.0, "top_mean_income": 500.0}
    brackets2 = planner2.update(stats2)
    assert brackets2[2].rate > 0.2


def test_saez_top_rate_never_goes_non_monotone():
    planner = _saez_planner(top_rate=0.21)
    stats = {"top_threshold": 50.0, "top_mean_income": 51.0}  # tau* tiny
    for _ in range(10):
        brackets = planner.update(stats)
    # Monotone schedules enforced: top rate floored at the bracket below
    assert brackets[2].rate >= brackets[1].rate - 1e-9


def test_saez_elasticity_estimate_updates_and_stays_bounded():
    planner = _saez_planner()
    for zm in (60.0, 80.0, 70.0, 90.0):
        planner.update({"top_threshold": 50.0, "top_mean_income": zm})
    assert 0.05 <= planner.elasticity_estimate <= 2.0


def test_heuristic_and_bandit_never_emit_non_monotone_schedule():
    """bd-njq: heuristic pushes top brackets *down* when Gini<0.3 and
    production is healthy, which could drive a top rate below the bracket
    beneath it -> TaxSchedule._validate ValueError mid-run. bandit perturbs
    randomly with the same exposure. Both must floor to monotone first."""
    # Low Gini + healthy production + a large learning rate drives the
    # heuristic into the non-monotone regime in a single step.
    low_gini_high_prod = {"gini": 0.0, "mean_income": 20.0}

    for ptype in ("heuristic", "bandit"):
        schedule = TaxSchedule(TaxScheduleConfig(brackets=[
            TaxBracket(threshold=0.0, rate=0.1),
            TaxBracket(threshold=10.0, rate=0.5),
            TaxBracket(threshold=25.0, rate=0.5),
        ]))
        planner = PlannerAgent(
            PlannerConfig(planner_type=ptype, learning_rate=0.2,
                          exploration_rate=1.0),
            schedule, seed=1,
        )
        for _ in range(20):
            brackets = planner.update(low_gini_high_prod)  # must not raise
            rates = [b.rate for b in brackets]
            assert rates == sorted(rates), f"{ptype} emitted non-monotone {rates}"


def test_saez_welfare_weight_lowers_optimal_rate():
    """bd-5gz: a positive social welfare weight g on the top tail lowers the
    optimal Saez rate vs the g=0 (revenue-maximizing) limit. With a fat tail
    the revenue-max rule raises the top rate; a high g flips it to a cut."""
    rev_max = _saez_planner(top_rate=0.5)
    b0 = rev_max.update({"top_threshold": 50.0, "top_mean_income": 200.0})

    welfare = _saez_planner(top_rate=0.5)
    bg = welfare.update({
        "top_threshold": 50.0, "top_mean_income": 200.0,
        "top_welfare_weight": 0.7,
    })
    assert bg[-1].rate < b0[-1].rate


def test_saez_targets_highest_populated_bracket():
    """bd-anv/bd-kk5: when z* (the realized top-tail cutoff) sits below the
    configured top bracket, Saez must optimize the highest *populated*
    bracket (the one containing z*), not the empty top bracket."""
    planner = _saez_planner(top_rate=0.45)  # schedule [0.0, 10.0, 50.0]
    # z*=12 lands in the [10, 50) bracket (index 1); fat tail -> rate rises.
    for _ in range(3):
        brackets = planner.update({"top_threshold": 12.0, "top_mean_income": 200.0})
    assert brackets[0].rate == pytest.approx(0.1)   # bottom bracket untouched
    assert brackets[1].rate > 0.2                    # populated bracket optimized
    # empty top bracket carried up so the schedule stays monotone
    assert brackets[2].rate >= brackets[1].rate - 1e-9


def test_saez_top_tail_falls_back_when_top_bracket_unpopulated():
    """bd-kk5: when the configured top bracket sits above the economy's
    realized income scale, no worker reaches it and top_mean_income would
    be 0 every epoch — silently freezing the Saez elasticity estimate and
    collapsing it to a constant tau* controller. The env must fall back to
    the observed top quintile so the planner always sees a real tail."""
    from worlds.gather_trade_build.config import GTBConfig
    from worlds.gather_trade_build.env import GTBEnvironment

    cfg = GTBConfig.from_dict({
        "taxation": {"brackets": [
            {"threshold": 0.0, "rate": 0.1},
            {"threshold": 50.0, "rate": 0.45},  # unreachable in this economy
        ]},
    })
    env = GTBEnvironment(cfg)
    # Incomes 0..9 — every worker is far below the 50.0 top-bracket edge.
    workers = {}
    for i in range(10):
        w = WorkerState(agent_id=f"w{i}")
        w.gross_income_this_epoch = float(i)
        workers[f"w{i}"] = w

    stats = env.stats_from_snapshot(workers)

    # Pre-fix this was 0.0 (empty top bracket); post-fix it reflects the
    # observed top quintile, with a cutoff below the unreachable bracket.
    assert stats["top_mean_income"] > 0.0
    assert stats["top_threshold"] < 50.0
    assert stats["n_top"] >= 2
