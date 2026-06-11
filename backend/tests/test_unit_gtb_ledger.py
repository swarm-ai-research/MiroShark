"""Unit tests for GTB Phase 1 ledger coherence.

Covers ledger_mode=coin income accounting, lump-sum redistribution,
tax-debt carryforward, and treasury-funded house income.
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
    House,
    ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402
from worlds.gather_trade_build.sweep import run_one_seed  # noqa: E402


def _make_env(**domain_overrides) -> GTBEnvironment:
    """4x4 all-wood map so GATHER always succeeds wherever a worker is."""
    domain = {
        "map": {"height": 4, "width": 4, "wood_density": 1.0,
                "stone_density": 0.0},
        **domain_overrides,
    }
    cfg = GTBConfig.from_dict(domain)
    cfg.seed = 7
    return GTBEnvironment(cfg)


def test_coin_mode_gather_books_no_income():
    env = _make_env(ledger_mode="coin")
    worker = env.add_worker("A")
    env.apply_actions({
        "A": GTBAction(agent_id="A", action_type=GTBActionType.GATHER),
    })
    assert worker.get_resource(ResourceType.WOOD) > 0
    assert worker.gross_income_this_epoch == 0.0
    assert worker.reported_income_this_epoch == 0.0


def test_legacy_mode_gather_books_income():
    env = _make_env(ledger_mode="legacy")
    worker = env.add_worker("A")
    env.apply_actions({
        "A": GTBAction(agent_id="A", action_type=GTBActionType.GATHER),
    })
    assert worker.gross_income_this_epoch > 0.0


def test_coin_mode_trade_is_symmetric_and_gap_free():
    """Seller books net proceeds; buyer books the outlay as an expense.
    Income == net coin flow, so the epoch ledger gap is ~0."""
    env = _make_env(ledger_mode="coin")
    seller = env.add_worker("S")
    buyer = env.add_worker("B")
    env.apply_actions({
        "S": GTBAction(agent_id="S", action_type=GTBActionType.GATHER),
    })
    wood = seller.get_resource(ResourceType.WOOD)
    assert wood > 0
    env.apply_actions({
        "S": GTBAction(agent_id="S", action_type=GTBActionType.TRADE_SELL,
                       resource_type=ResourceType.WOOD, quantity=wood,
                       price=2.0),
        "B": GTBAction(agent_id="B", action_type=GTBActionType.TRADE_BUY,
                       resource_type=ResourceType.WOOD, quantity=wood,
                       price=2.0),
    })
    total = wood * 2.0
    fee = total * env.config.market.transaction_fee_rate
    assert seller.gross_income_this_epoch == pytest.approx(total - fee)
    assert buyer.gross_income_this_epoch == pytest.approx(-(total + fee))
    assert buyer.get_resource(ResourceType.WOOD) == pytest.approx(wood)

    result = env.end_epoch()
    ledger_evt = [e for e in result.events if e.event_type == "epoch_ledger"][0]
    assert ledger_evt.details["income_coin_gap"] == pytest.approx(0.0, abs=1e-9)
    assert env.verify_coin_conservation()


def test_coin_mode_honest_scenario_has_zero_income_coin_gap():
    """Whole-scenario acceptance: with honest agents in coin mode, every
    epoch's booked income equals coin actually received."""
    scenario = {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.3,
                    "stone_density": 0.2},
            "ledger_mode": "coin",
        },
        "agents": [{"policy": "honest", "count": 4}],
        "simulation": {"n_epochs": 4, "steps_per_epoch": 8},
    }
    run = run_one_seed(scenario, seed=11)
    assert run["coin_conserved"]
    for m in run["metrics"]:
        assert m["income_coin_gap"] == pytest.approx(0.0, abs=1e-9)


def test_lump_sum_redistribution_returns_all_revenue():
    scenario = {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.3,
                    "stone_density": 0.2},
            "taxation": {"redistribution": "lump_sum"},
        },
        "agents": [{"policy": "honest", "count": 4}],
        "simulation": {"n_epochs": 4, "steps_per_epoch": 8},
    }
    run = run_one_seed(scenario, seed=11)
    assert run["coin_conserved"]
    total_redistributed = sum(m["total_redistributed"] for m in run["metrics"])
    total_revenue = sum(
        m["total_tax_revenue"] + m["total_fines"] for m in run["metrics"]
    )
    assert total_redistributed == pytest.approx(total_revenue)
    assert total_redistributed > 0
    # Conservation itemization: everything burned as taxes/fines came back.
    ledger = run["ledger"]
    assert ledger["minted"].get("redistribution", 0.0) == pytest.approx(
        ledger["burned"].get("taxes", 0.0) + ledger["burned"].get("fines", 0.0)
    )


def test_redistribution_lowers_gini_vs_burn():
    base = {
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.3,
                    "stone_density": 0.2},
        },
        # Heterogeneous skills so there is inequality to redistribute away
        "agents": [
            {"policy": "honest", "count": 2, "skill_gather": 1.5},
            {"policy": "honest", "count": 2, "skill_gather": 0.7},
        ],
        "simulation": {"n_epochs": 6, "steps_per_epoch": 10},
    }
    import copy
    redis = copy.deepcopy(base)
    redis["domain"]["taxation"] = {"redistribution": "lump_sum"}

    seeds = [1, 2, 3, 4, 5]
    gini_burn = [run_one_seed(base, s)["metrics"][-1]["gini_coefficient"]
                 for s in seeds]
    gini_redis = [run_one_seed(redis, s)["metrics"][-1]["gini_coefficient"]
                  for s in seeds]
    # Wealth feeds back into production (houses), so income Gini should
    # fall on average when revenue is recycled instead of burned.
    assert sum(gini_redis) <= sum(gini_burn)


def test_tax_debt_accrues_and_is_collected():
    env = _make_env(taxation={"debt_enabled": True})
    worker = env.add_worker("A")
    # Owes more than the 10-coin endowment can cover
    worker.gross_income_this_epoch = 100.0
    worker.reported_income_this_epoch = 100.0
    owed = env.tax_schedule.compute_tax(100.0)
    assert owed > 10.0

    env.end_epoch()
    assert worker.get_resource(ResourceType.COIN) == pytest.approx(0.0)
    assert worker.tax_debt == pytest.approx(owed - 10.0)

    # Next epoch: flush with coin, debt is collected first
    worker.add_resource(ResourceType.COIN, 50.0)
    env.register_external_coin(50.0, "test_grant")
    env.end_epoch()
    assert worker.tax_debt == pytest.approx(0.0)
    assert worker.get_resource(ResourceType.COIN) == pytest.approx(
        50.0 - (owed - 10.0)
    )
    assert env.verify_coin_conservation()


def test_debt_disabled_shortfall_evaporates_legacy():
    env = _make_env()  # debt_enabled defaults False
    worker = env.add_worker("A")
    worker.gross_income_this_epoch = 100.0
    worker.reported_income_this_epoch = 100.0
    env.end_epoch()
    assert worker.tax_debt == 0.0


def test_treasury_house_income_capped_by_treasury():
    env = _make_env(build={"house_income_mode": "treasury",
                           "income_per_house_per_step": 1.0})
    worker = env.add_worker("A")
    env._houses.append(House(owner_id="A", position=(0, 0),
                             income_per_step=1.0))
    coin_before = worker.get_resource(ResourceType.COIN)

    # Empty treasury: no house income is paid
    env.apply_actions({"A": GTBAction(agent_id="A",
                                      action_type=GTBActionType.NOOP)})
    assert worker.get_resource(ResourceType.COIN) == pytest.approx(coin_before)

    # Fund the treasury: house income flows, capped at demand
    env._treasury = 5.0
    env.apply_actions({"A": GTBAction(agent_id="A",
                                      action_type=GTBActionType.NOOP)})
    assert worker.get_resource(ResourceType.COIN) == pytest.approx(
        coin_before + 1.0
    )
    assert env.treasury == pytest.approx(4.0)


def test_coherent_scenario_runs_and_conserves():
    import yaml
    path = (_BACKEND / "worlds" / "gather_trade_build" / "scenarios"
            / "ai_economist_coherent.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    run = run_one_seed(data, seed=42, n_epochs=5)
    assert run["coin_conserved"]
    assert sum(m["total_redistributed"] for m in run["metrics"]) > 0
