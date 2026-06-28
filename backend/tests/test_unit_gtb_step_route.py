"""Unit tests for the GTB ``/step`` route's step-count parsing (bd p4w).

Pre-fix the route did ``int(payload.get("n", 1))`` directly, so a non-numeric
``n`` (``{"n": "abc"}`` or ``{"n": [1,2,3]}``) raised ``ValueError`` and
propagated as a Flask 500 with a traceback. Now it coerces via
``_coerce_step_count`` and returns a clean 400, and clamps large values.

Mounts ``gtb_bp`` on a stub Flask app with a fake registry — no real world,
no Neo4j, no LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---- pure helper -----------------------------------------------------------

def test_coerce_step_count_pure():
    from app.api.gtb import _coerce_step_count, _MAX_STEP_BATCH

    assert _coerce_step_count(1) == 1
    assert _coerce_step_count(3) == 3
    assert _coerce_step_count("5") == 5        # numeric string is fine
    assert _coerce_step_count(2.9) == 2         # truncates toward zero
    assert _coerce_step_count(0) == 1           # floor at 1
    assert _coerce_step_count(-4) == 1          # floor at 1
    assert _coerce_step_count(10**9) == _MAX_STEP_BATCH  # clamp at ceiling
    for bad in ("abc", [1, 2, 3], {"x": 1}, None):
        with pytest.raises((TypeError, ValueError)):
            _coerce_step_count(bad)


# ---- route ----------------------------------------------------------------

class _FakeService:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1
        return {"tick": self.calls}


class _FakeRegistry:
    def __init__(self, service):
        self._service = service

    def get(self, sim_id):
        return self._service


@pytest.fixture
def client(monkeypatch):
    from flask import Flask
    import app.api.gtb as gtb_mod
    from app.api import gtb_bp

    service = _FakeService()
    monkeypatch.setattr(gtb_mod, "get_registry", lambda: _FakeRegistry(service))

    flask_app = Flask(__name__)
    flask_app.register_blueprint(gtb_bp, url_prefix="/api/gtb")
    c = flask_app.test_client()
    c._service = service  # expose for assertions
    return c


def test_step_non_numeric_n_returns_400_not_500(client):
    resp = client.post("/api/gtb/sim1/step", json={"n": "abc"})
    assert resp.status_code == 400
    assert "n" in resp.get_json()["error"]
    assert client._service.calls == 0  # never stepped


def test_step_list_n_returns_400(client):
    resp = client.post("/api/gtb/sim1/step", json={"n": [1, 2, 3]})
    assert resp.status_code == 400


def test_step_valid_n_runs_that_many(client):
    resp = client.post("/api/gtb/sim1/step", json={"n": 3})
    assert resp.status_code == 200
    assert len(resp.get_json()["ticks"]) == 3


def test_step_default_n_is_one(client):
    resp = client.post("/api/gtb/sim1/step", json={})
    assert resp.status_code == 200
    assert len(resp.get_json()["ticks"]) == 1


def test_step_large_n_is_clamped(client):
    from app.api.gtb import _MAX_STEP_BATCH

    resp = client.post("/api/gtb/sim1/step", json={"n": 10**7})
    assert resp.status_code == 200
    assert len(resp.get_json()["ticks"]) == _MAX_STEP_BATCH
