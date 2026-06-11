"""Unit tests for the GTB seed-sweep harness and coin-ledger checks.

Pure offline — exercises the worlds/ kernel directly, no Flask, no LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.sweep import (  # noqa: E402
    export_sweep,
    run_one_seed,
    run_sweep,
)


@pytest.fixture
def tiny_scenario(tmp_path) -> Path:
    """A small but mechanically complete scenario: honest + evasive +
    gaming agents on a small map, enough epochs for taxes and audits."""
    data = {
        "scenario_id": "tiny_test",
        "domain": {
            "map": {"height": 8, "width": 8, "wood_density": 0.3,
                    "stone_density": 0.2},
            "taxation": {
                "brackets": [
                    {"threshold": 0.0, "rate": 0.1},
                    {"threshold": 5.0, "rate": 0.3},
                ],
            },
            "misreporting": {"enabled": True, "audit_probability": 0.5},
        },
        "agents": [
            {"policy": "honest", "count": 2},
            {"policy": "evasive", "count": 2, "underreport_fraction": 0.4},
            {"policy": "gaming", "count": 1},
        ],
        "simulation": {"n_epochs": 4, "steps_per_epoch": 6, "seed": 42},
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_single_seed_is_deterministic(tiny_scenario):
    with open(tiny_scenario) as f:
        data = yaml.safe_load(f)
    a = run_one_seed(data, seed=7)
    b = run_one_seed(data, seed=7)
    assert a["metrics"] == b["metrics"]
    assert a["ledger"] == b["ledger"]


def test_different_seeds_differ(tiny_scenario):
    with open(tiny_scenario) as f:
        data = yaml.safe_load(f)
    a = run_one_seed(data, seed=1)
    b = run_one_seed(data, seed=2)
    assert a["metrics"] != b["metrics"]


def test_coin_conservation_holds(tiny_scenario):
    """Hard invariant: every coin held by workers is an itemized
    mint minus an itemized burn."""
    result = run_sweep(tiny_scenario, seeds=[1, 2, 3])
    assert result.all_conserved
    for run in result.runs:
        assert abs(run["ledger"]["discrepancy"]) <= 1e-6


def test_income_coin_gap_documents_known_incoherence(tiny_scenario):
    """Characterization test for the Phase 1 ledger issue: gathering books
    gross income with no coin inflow, so the gap must be positive. When the
    Phase 1 rework lands (income == net coin flow) this test should be
    inverted to assert the gap is ~0."""
    result = run_sweep(tiny_scenario, seeds=[1, 2, 3])
    gaps = result.metric_series("income_coin_gap")
    assert all(g >= 0 for g in gaps)
    assert sum(gaps) > 0, "expected a positive income/coin gap pre-Phase-1"


def test_sweep_aggregation_shape(tiny_scenario):
    result = run_sweep(tiny_scenario, seeds=[1, 2, 3])
    assert len(result.runs) == 3
    assert sorted(result.summary.keys()) == [0, 1, 2, 3]
    welfare = result.summary[3]["welfare"]
    for stat in ("mean", "std", "ci95", "min", "max"):
        assert stat in welfare
    assert welfare["min"] <= welfare["mean"] <= welfare["max"]
    assert welfare["ci95"] >= 0


def test_undetected_evasion_rate_in_unit_range(tiny_scenario):
    """With evasive agents and a 0.5 audit rate, the hidden-income-based
    rate must be a real fraction (the old audit_miss-based metric was
    structurally 0 because the env never emits audit_miss)."""
    result = run_sweep(tiny_scenario, seeds=list(range(1, 6)))
    rates = [
        m["undetected_evasion_rate"]
        for run in result.runs for m in run["metrics"]
    ]
    assert all(0.0 <= r <= 1.0 for r in rates)
    hidden_epochs = [r for r in rates if r > 0]
    caught_epochs = [
        m["total_catches"] for run in result.runs for m in run["metrics"]
        if m["total_catches"] > 0
    ]
    # Across 5 seeds with 2 evasive agents, both outcomes should occur:
    # some hidden income escapes, some audits catch.
    assert hidden_epochs, "no epoch with undetected evasion across 5 seeds"
    assert caught_epochs, "no audit catches across 5 seeds"


def test_overrides_change_behavior(tiny_scenario):
    base = run_sweep(tiny_scenario, seeds=[1, 2, 3])
    no_audit = run_sweep(
        tiny_scenario, seeds=[1, 2, 3],
        overrides={"domain.misreporting.audit_probability": 0.0},
    )
    assert sum(no_audit.metric_series("total_catches")) == 0
    # Identical seeds, so any difference is the override.
    assert sum(base.metric_series("total_audits")) >= sum(
        no_audit.metric_series("total_audits")
    )


def test_export_writes_manifest_and_csvs(tiny_scenario, tmp_path):
    result = run_sweep(tiny_scenario, seeds=[1, 2])
    out = export_sweep(result, tmp_path / "sweep_out")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["scenario_id"] == "tiny_test"
    assert manifest["seeds"] == [1, 2]
    assert manifest["config_hash"]
    assert manifest["coin_conserved_all_runs"] is True

    per_seed = (out / "per_seed_metrics.csv").read_text().splitlines()
    # header + 2 seeds * 4 epochs
    assert len(per_seed) == 1 + 2 * 4
    assert per_seed[0].startswith("seed,epoch,")

    summary = (out / "summary.csv").read_text().splitlines()
    assert len(summary) == 1 + 4  # header + 4 epochs
    assert "welfare_mean" in summary[0]
    assert "income_coin_gap_mean" in summary[0]
