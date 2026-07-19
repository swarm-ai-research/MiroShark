"""Unit tests for checkpoint export (branch-per-sim audit snapshots).

Needs the git CLI (present in dev + CI); no Flask app boot for the service
tests, stub-mounted blueprint for the route tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services import checkpoint_export as cx  # noqa: E402

SIM = "sim_00ccbbaa1122"


def _make_sim_dir(tmp_path: Path, sim_id: str = SIM) -> Path:
    sim_dir = tmp_path / "sims" / sim_id
    sim_dir.mkdir(parents=True)
    (sim_dir / "state.json").write_text(json.dumps({"simulation_id": sim_id, "status": "running"}))
    (sim_dir / "simulation_config.json").write_text(json.dumps({"total_rounds": 5}))
    return sim_dir


def test_export_creates_branch_with_full_state(tmp_path):
    sim_dir = _make_sim_dir(tmp_path)
    repo = tmp_path / "checkpoints.git"

    result = cx.export_checkpoint(sim_dir, SIM, repo, label="baseline")

    assert result["branch"] == f"refs/heads/checkpoint/{SIM}"
    assert set(result["files"]) == {"state.json", "simulation_config.json"}
    assert result["checkpoint_index"] == 1
    # The branch exists in the bare repo and holds exactly one commit.
    out = subprocess.run(
        ["git", "log", "--oneline", f"checkpoint/{SIM}"],
        cwd=repo, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout
    assert len(out.splitlines()) == 1
    assert "baseline" in out


def test_repeated_exports_chain_commits(tmp_path):
    sim_dir = _make_sim_dir(tmp_path)
    repo = tmp_path / "checkpoints.git"

    first = cx.export_checkpoint(sim_dir, SIM, repo)
    (sim_dir / "state.json").write_text(json.dumps({"simulation_id": SIM, "status": "completed"}))
    second = cx.export_checkpoint(sim_dir, SIM, repo)

    history = cx.list_checkpoints(SIM, repo)
    assert [h["commit"] for h in history] == [first["commit"], second["commit"]]
    assert second["checkpoint_index"] == 2


def test_import_round_trips_state(tmp_path):
    sim_dir = _make_sim_dir(tmp_path)
    repo = tmp_path / "checkpoints.git"
    cx.export_checkpoint(sim_dir, SIM, repo)

    restored_dir = tmp_path / "restored"
    restored = cx.import_checkpoint(SIM, restored_dir, repo)

    assert "state.json" in restored
    assert "checkpoint_meta.json" in restored  # snapshot self-describes
    assert json.loads((restored_dir / "state.json").read_text()) == {
        "simulation_id": SIM, "status": "running",
    }
    meta = json.loads((restored_dir / "checkpoint_meta.json").read_text())
    assert meta["schema"] == "miroshark.checkpoint.v1"
    assert meta["note"].startswith("audit/recovery")


def test_import_specific_earlier_commit(tmp_path):
    sim_dir = _make_sim_dir(tmp_path)
    repo = tmp_path / "checkpoints.git"
    first = cx.export_checkpoint(sim_dir, SIM, repo)
    (sim_dir / "state.json").write_text(json.dumps({"status": "completed"}))
    cx.export_checkpoint(sim_dir, SIM, repo)

    restored_dir = tmp_path / "restored"
    cx.import_checkpoint(SIM, restored_dir, repo, commit=first["commit"])
    assert json.loads((restored_dir / "state.json").read_text())["status"] == "running"


def test_meta_folds_in_run_event_projection(tmp_path):
    sim_dir = _make_sim_dir(tmp_path)
    from app.services.run_events import EVENT_RUN_COMPLETED, EVENT_RUN_CREATED, emit

    emit(sim_dir, SIM, EVENT_RUN_CREATED)
    emit(sim_dir, SIM, EVENT_RUN_COMPLETED)
    repo = tmp_path / "checkpoints.git"
    cx.export_checkpoint(sim_dir, SIM, repo)

    restored_dir = tmp_path / "restored"
    restored = cx.import_checkpoint(SIM, restored_dir, repo)
    assert "run_events.jsonl" in restored
    meta = json.loads((restored_dir / "checkpoint_meta.json").read_text())
    assert meta["run_status"] == "completed"
    assert meta["run_events_last_seq"] == 1


def test_export_empty_sim_dir_raises(tmp_path):
    empty = tmp_path / "sims" / SIM
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="no snapshot-able"):
        cx.export_checkpoint(empty, SIM, tmp_path / "checkpoints.git")


def test_branches_are_per_sim(tmp_path):
    repo = tmp_path / "checkpoints.git"
    a_dir = _make_sim_dir(tmp_path, "sim_aaaaaaaaaaaa")
    b_dir = _make_sim_dir(tmp_path, "sim_bbbbbbbbbbbb")
    cx.export_checkpoint(a_dir, "sim_aaaaaaaaaaaa", repo)
    cx.export_checkpoint(b_dir, "sim_bbbbbbbbbbbb", repo)

    assert len(cx.list_checkpoints("sim_aaaaaaaaaaaa", repo)) == 1
    assert len(cx.list_checkpoints("sim_bbbbbbbbbbbb", repo)) == 1
    assert cx.list_checkpoints("sim_cccccccccccc", repo) == []


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    from flask import Flask

    from app.config import Config

    sims_root = tmp_path / "sims"
    sims_root.mkdir()
    monkeypatch.setattr(Config, "WONDERWALL_SIMULATION_DATA_DIR", str(sims_root))
    monkeypatch.setenv("MIROSHARK_CHECKPOINT_REPO", str(tmp_path / "checkpoints.git"))
    monkeypatch.setenv("MIROSHARK_ADMIN_TOKEN", "admin-sekrit")

    from app.api import simulation_bp

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    return app.test_client()


def test_checkpoint_route_requires_admin_token(client, tmp_path):
    _make_sim_dir(tmp_path)
    assert client.post(f"/api/simulation/{SIM}/checkpoint").status_code == 401


def test_checkpoint_route_roundtrip(client, tmp_path):
    _make_sim_dir(tmp_path)
    headers = {"Authorization": "Bearer admin-sekrit"}

    resp = client.post(
        f"/api/simulation/{SIM}/checkpoint", headers=headers,
        json={"label": "pre-branch"},
    )
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["branch"] == f"refs/heads/checkpoint/{SIM}"
    assert data["checkpoint_index"] == 1

    listing = client.get(f"/api/simulation/{SIM}/checkpoints")
    assert listing.status_code == 200
    entries = listing.get_json()["data"]["checkpoints"]
    assert len(entries) == 1
    assert "pre-branch" in entries[0]["subject"]


def test_checkpoint_route_404_and_409(client, tmp_path):
    headers = {"Authorization": "Bearer admin-sekrit"}
    assert client.post(
        "/api/simulation/sim_ffffffffffff/checkpoint", headers=headers
    ).status_code == 404

    empty = tmp_path / "sims" / "sim_dddddddddddd"
    empty.mkdir(parents=True)
    resp = client.post(
        "/api/simulation/sim_dddddddddddd/checkpoint", headers=headers
    )
    assert resp.status_code == 409  # nothing snapshot-able yet


def test_read_paths_tolerate_missing_repo(tmp_path):
    """Listing/reading before any checkpoint exists must not require the
    bare repo — only export creates it. (Found live: GET /checkpoints 500d
    on a fresh deployment.)"""
    missing = tmp_path / "never-created.git"
    assert cx.branch_head(SIM, missing) is None
    assert cx.list_checkpoints(SIM, missing) == []
    with pytest.raises(ValueError, match="no checkpoints"):
        cx.import_checkpoint(SIM, tmp_path / "dest", missing)
