"""Unit tests for the GTB → Polymarket envelope adapter."""

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
def gtb_poly():
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
    # Load the chain polymarket_service depends on.
    _load("app.services.signal_service", "app/services/signal_service.py")
    _load("app.services.polymarket_service", "app/services/polymarket_service.py")
    return _load("app.services.gtb_polymarket", "app/services/gtb_polymarket.py")


def _state(open_markets, resolved_markets=(), stakes_by_market=None, epoch=2):
    return {
        "epoch": epoch,
        "markets": {
            "open": list(open_markets),
            "resolved": list(resolved_markets),
        },
        "stakes": {
            "open_stakes": stakes_by_market or {},
            "history": [],
        },
    }


_M = lambda i, q, op=">", th=0.5, dl=5, status="open", rv=None, re=None: {
    "market_id": i,
    "question": q,
    "metric": "welfare",
    "op": op,
    "threshold": th,
    "deadline_epoch": dl,
    "created_epoch": 0,
    "status": status,
    "resolved_epoch": re,
    "resolved_value": rv,
}


class TestEnvelopeShape:
    def test_open_market_with_no_stakes_is_coin_flip(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Will welfare > 3?")]), sim_id="demo",
        )
        assert payload["schema_version"] == "1.1"
        assert payload["simulation_id"] == "demo"
        assert len(payload["markets"]) == 1
        env = payload["markets"][0]
        assert env["yes_probability"] == 0.5
        assert env["no_probability"] == 0.5
        assert env["direction"] == "Neutral"
        assert env["confidence_tier"] == "speculative"
        assert env["suggested_market_title"] == "Will welfare > 3?"

    def test_yes_pool_dominance_swings_envelope_bullish(self, gtb_poly):
        stakes = {"gtb-0000": [
            {"agent_id": "w0", "side": "yes", "amount": 8.0, "epoch": 1, "market_id": "gtb-0000"},
            {"agent_id": "w1", "side": "no",  "amount": 1.0, "epoch": 1, "market_id": "gtb-0000"},
        ]}
        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Will welfare > 3?")], stakes_by_market=stakes),
            sim_id="demo",
        )
        env = payload["markets"][0]
        assert env["direction"] == "Bullish"
        assert env["yes_probability"] > 0.6
        assert env["yes_pool"] == 8.0
        assert env["no_pool"] == 1.0
        # Confidence_pct = |yes - 0.5| * 200; positive band.
        assert env["confidence_pct"] > 20

    def test_resolved_yes_market_is_one(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(
            _state(
                [],
                resolved_markets=[_M("gtb-0001", "Q", status="yes", rv=4.2, re=3)],
            ),
            sim_id="demo",
        )
        env = payload["markets"][0]
        assert env["status"] == "yes"
        assert env["yes_probability"] == 1.0
        assert env["no_probability"] == 0.0
        assert env["direction"] == "Bullish"
        assert env["confidence_tier"] == "high-conviction"

    def test_resolved_no_market_is_zero(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(
            _state(
                [],
                resolved_markets=[_M("gtb-0002", "Q", status="no", rv=0.1, re=3)],
            ),
            sim_id="demo",
        )
        env = payload["markets"][0]
        assert env["yes_probability"] == 0.0
        assert env["direction"] == "Bearish"


class TestHeadline:
    def test_headline_is_highest_confidence_open_market(self, gtb_poly):
        # Two open markets: one with stakes far apart, one with no stakes.
        stakes = {"gtb-0000": [
            {"agent_id": "w0", "side": "yes", "amount": 9.0, "epoch": 1, "market_id": "gtb-0000"},
            {"agent_id": "w1", "side": "no",  "amount": 1.0, "epoch": 1, "market_id": "gtb-0000"},
        ]}
        payload = gtb_poly.compute_gtb_polymarket(
            _state(
                [_M("gtb-0000", "Sharp belief"), _M("gtb-0001", "Coin flip")],
                stakes_by_market=stakes,
            ),
            sim_id="demo",
        )
        assert payload["headline"]["market_id"] == "gtb-0000"

    def test_no_open_markets_means_no_headline(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(_state([]), sim_id="demo")
        assert payload["headline"] is None
        assert payload["markets"] == []


class TestGuards:
    def test_non_dict_state_returns_none(self, gtb_poly):
        assert gtb_poly.compute_gtb_polymarket(None, sim_id="demo") is None
        assert gtb_poly.compute_gtb_polymarket("oops", sim_id="demo") is None

    def test_missing_markets_block_returns_none(self, gtb_poly):
        assert gtb_poly.compute_gtb_polymarket({"epoch": 0}, sim_id="demo") is None


class TestConfidenceSource:
    def test_no_stakes_is_tagged(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Q")]), sim_id="demo",
        )
        assert payload["markets"][0]["confidence_source"] == "no_stakes"

    def test_one_sided_pile_on_is_tagged(self, gtb_poly):
        stakes = {"gtb-0000": [
            {"agent_id": "w0", "side": "yes", "amount": 5.0, "epoch": 1, "market_id": "gtb-0000"},
            {"agent_id": "w1", "side": "yes", "amount": 3.0, "epoch": 1, "market_id": "gtb-0000"},
        ]}
        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Q")], stakes_by_market=stakes), sim_id="demo",
        )
        assert payload["markets"][0]["confidence_source"] == "one_sided"

    def test_two_sided_real_forecast_is_tagged(self, gtb_poly):
        stakes = {"gtb-0000": [
            {"agent_id": "w0", "side": "yes", "amount": 4.0, "epoch": 1, "market_id": "gtb-0000"},
            {"agent_id": "w1", "side": "no",  "amount": 2.0, "epoch": 1, "market_id": "gtb-0000"},
        ]}
        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Q")], stakes_by_market=stakes), sim_id="demo",
        )
        assert payload["markets"][0]["confidence_source"] == "two_sided"

    def test_resolved_market_is_tagged(self, gtb_poly):
        payload = gtb_poly.compute_gtb_polymarket(
            _state([], resolved_markets=[_M("gtb-0001", "Q", status="yes", rv=4.2, re=3)]),
            sim_id="demo",
        )
        assert payload["markets"][0]["confidence_source"] == "resolved"


class TestSchemaValidation:
    def test_compute_output_validates_against_schema(self, gtb_poly):
        stakes = {"gtb-0000": [
            {"agent_id": "w0", "side": "yes", "amount": 4.0, "epoch": 1, "market_id": "gtb-0000"},
            {"agent_id": "w1", "side": "no",  "amount": 2.0, "epoch": 1, "market_id": "gtb-0000"},
        ]}
        payload = gtb_poly.compute_gtb_polymarket(
            _state(
                [_M("gtb-0000", "Open"), _M("gtb-0001", "No stakes")],
                resolved_markets=[_M("gtb-0002", "Done", status="yes", rv=4.2, re=3)],
                stakes_by_market=stakes,
            ),
            sim_id="demo",
        )
        gtb_poly.validate_payload(payload)  # raises on mismatch

    def test_validate_rejects_unknown_confidence_source(self, gtb_poly):
        from jsonschema import ValidationError

        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Q")]), sim_id="demo",
        )
        payload["markets"][0]["confidence_source"] = "bogus"
        with pytest.raises(ValidationError):
            gtb_poly.validate_payload(payload)

    def test_validate_rejects_old_schema_version(self, gtb_poly):
        from jsonschema import ValidationError

        payload = gtb_poly.compute_gtb_polymarket(
            _state([_M("gtb-0000", "Q")]), sim_id="demo",
        )
        payload["schema_version"] = "1"
        with pytest.raises(ValidationError):
            gtb_poly.validate_payload(payload)
