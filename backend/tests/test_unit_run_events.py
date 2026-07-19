"""Unit tests for the unified RunEvent stream (fabro-store pattern).

Covers the three acceptance criteria of the stream:

1. **Replayable** — a full lifecycle folds back into the run's state from
   nothing but ``run_events.jsonl``.
2. **Resumable mid-round** — a checkpoint taken partway through the log
   resumes folding without re-applying earlier envelopes (idempotent).
3. **Lineage queryable from events** — ``run.created`` /
   ``run.parent_linked`` carry parent + kind, and the lineage/repro
   surfaces consume them without state.json heuristics.

Bypass-Flask by design: stdlib + pytest only.
"""

import json

from app.services import lineage_service, repro_export, run_events
from app.services.run_events import (
    EVENT_PLATFORM_COMPLETED,
    EVENT_PREPARE_COMPLETED,
    EVENT_PREPARE_STARTED,
    EVENT_ROUND_COMPLETED,
    EVENT_ROUND_STARTED,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_CREATED,
    EVENT_RUN_FAILED,
    EVENT_RUN_PARENT_LINKED,
    EVENT_RUN_RUNNING,
    EVENT_RUN_STARTED,
    Checkpoint,
    RunEvent,
    RunEventLog,
    emit,
)

SIM = "sim_0123456789ab"


def _lifecycle(sim_dir, sim_id=SIM, rounds=2):
    """Emit a full happy-path lifecycle and return the log."""
    emit(sim_dir, sim_id, EVENT_RUN_CREATED, lineage_kind="original",
         total_rounds=rounds)
    emit(sim_dir, sim_id, EVENT_PREPARE_STARTED)
    emit(sim_dir, sim_id, EVENT_PREPARE_COMPLETED, entities_count=5,
         profiles_count=5)
    emit(sim_dir, sim_id, EVENT_RUN_STARTED, total_rounds=rounds)
    emit(sim_dir, sim_id, EVENT_RUN_RUNNING, pid=4242)
    for r in range(1, rounds + 1):
        emit(sim_dir, sim_id, EVENT_ROUND_STARTED, platform="twitter", round=r)
        emit(sim_dir, sim_id, EVENT_ROUND_COMPLETED, platform="twitter",
             round=r, simulated_hours=r)
    emit(sim_dir, sim_id, EVENT_PLATFORM_COMPLETED, platform="twitter",
         total_rounds=rounds, total_actions=17)
    emit(sim_dir, sim_id, EVENT_RUN_COMPLETED, source="action_log")
    return RunEventLog(sim_dir)


# ---------------------------------------------------------------------------
# log basics
# ---------------------------------------------------------------------------
def test_append_is_byte_stable_jsonl_without_seq(tmp_path):
    log = RunEventLog(tmp_path)
    log.append(RunEvent.new(SIM, EVENT_RUN_CREATED))
    log.append(RunEvent.new(SIM, EVENT_RUN_STARTED))

    lines = log.path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "seq" not in record  # seq derived at replay, never persisted
        assert record["schema"] == run_events.EVENT_SCHEMA_V1

    envelopes = list(log.iter_envelopes())
    assert [e.seq for e in envelopes] == [0, 1]


def test_replay_skips_torn_writes(tmp_path):
    log = RunEventLog(tmp_path)
    log.append(RunEvent.new(SIM, EVENT_RUN_CREATED))
    with log.path.open("a") as f:
        f.write('{"schema": "miroshark.run_event.v1", "id": "evt_trunc')  # crash mid-line
    log.append(RunEvent.new(SIM, EVENT_RUN_STARTED))

    events = log.replay()
    assert [e.event for e in events] == [EVENT_RUN_CREATED, EVENT_RUN_STARTED]


def test_emit_never_raises(tmp_path):
    target = tmp_path / "not-a-dir"
    target.write_text("a file where a directory must be")
    assert emit(target / "sub", SIM, EVENT_RUN_CREATED) is None  # swallowed


# ---------------------------------------------------------------------------
# acceptance 1: replayable from the event log alone
# ---------------------------------------------------------------------------
def test_projection_reconstructs_full_lifecycle(tmp_path):
    log = _lifecycle(tmp_path, rounds=3)
    projection, checkpoint = log.project()

    assert projection.simulation_id == SIM
    assert projection.status == "completed"
    assert projection.current_round == {"twitter": 3}
    assert projection.completed_platforms == ["twitter"]
    assert projection.total_rounds == 3
    assert projection.created_at is not None
    assert projection.finished_at is not None
    assert checkpoint.last_seq == 12  # 13 events, 0-indexed


def test_projection_records_failure(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED)
    emit(tmp_path, SIM, EVENT_RUN_STARTED)
    emit(tmp_path, SIM, EVENT_RUN_FAILED, error="boom", phase="run")

    projection, _ = RunEventLog(tmp_path).project()
    assert projection.status == "failed"
    assert projection.error == "boom"


# ---------------------------------------------------------------------------
# acceptance 2: resumable mid-round
# ---------------------------------------------------------------------------
def test_checkpoint_resume_mid_round_is_idempotent(tmp_path):
    log = _lifecycle(tmp_path, rounds=2)

    # Snapshot partway through (mid-round: after round 1 started, seq 5).
    mid_state, mid_checkpoint = log.project(until_seq=5)
    assert mid_state.status == "running"
    assert mid_state.current_round == {"twitter": 1}
    assert mid_checkpoint.last_seq == 5

    # Persist + reload the checkpoint (monitor restart), then resume.
    ckpt_path = tmp_path / "projection.ckpt.json"
    mid_checkpoint.save(ckpt_path)
    resumed_from = Checkpoint.load(ckpt_path)
    final_state, final_checkpoint = log.project(resume=resumed_from)

    # Resumed fold must agree exactly with a from-scratch fold.
    scratch_state, scratch_checkpoint = log.project()
    assert final_state.to_dict() == scratch_state.to_dict()
    assert final_checkpoint.last_seq == scratch_checkpoint.last_seq


def test_resume_never_double_applies(tmp_path):
    log = _lifecycle(tmp_path, rounds=1)
    _, checkpoint = log.project()

    # Resuming at the tip applies nothing further and state is unchanged.
    state_again, checkpoint_again = log.project(resume=checkpoint)
    assert checkpoint_again.last_seq == checkpoint.last_seq
    assert state_again.to_dict() == checkpoint.state


# ---------------------------------------------------------------------------
# acceptance 3: lineage queryable from events
# ---------------------------------------------------------------------------
def test_projection_carries_fork_and_counterfactual_lineage(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED, lineage_kind="fork",
         parent_simulation_id="sim_parent00000")
    projection, _ = RunEventLog(tmp_path).project()
    assert projection.parent_simulation_id == "sim_parent00000"
    assert projection.lineage_kind == "fork"

    emit(tmp_path, SIM, EVENT_RUN_PARENT_LINKED,
         parent_simulation_id="sim_parent00000",
         lineage_kind="counterfactual", trigger_round=12, label="ceo_resigns")
    projection, _ = RunEventLog(tmp_path).project()
    assert projection.lineage_kind == "counterfactual"


def test_lineage_service_prefers_event_stream(tmp_path):
    data_dir = tmp_path
    child_dir = data_dir / "sim_child0000001"
    child_dir.mkdir()
    # state.json says fork-of-A; the event stream says counterfactual-of-B.
    (child_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_child0000001",
        "parent_simulation_id": "sim_aaaaaaaaaaaa",
        "is_public": True,
    }))
    emit(child_dir, "sim_child0000001", EVENT_RUN_CREATED,
         lineage_kind="counterfactual",
         parent_simulation_id="sim_bbbbbbbbbbbb")

    payload = lineage_service.build_lineage_payload(
        "sim_child0000001", str(data_dir)
    )
    assert payload["lineage_kind"] == "counterfactual"
    assert payload["parent"]["simulation_id"] == "sim_bbbbbbbbbbbb"

    # find_children follows the event-stream parent too.
    children = lineage_service.find_children("sim_bbbbbbbbbbbb", str(data_dir))
    assert [c["simulation_id"] for c in children] == ["sim_child0000001"]
    assert lineage_service.find_children("sim_aaaaaaaaaaaa", str(data_dir)) == []


def test_lineage_service_falls_back_without_stream(tmp_path):
    child_dir = tmp_path / "sim_child0000002"
    child_dir.mkdir()
    (child_dir / "state.json").write_text(json.dumps({
        "simulation_id": "sim_child0000002",
        "parent_simulation_id": "sim_parent00000",
        "is_public": True,
    }))
    payload = lineage_service.build_lineage_payload(
        "sim_child0000002", str(tmp_path)
    )
    assert payload["lineage_kind"] == "fork"
    assert payload["parent"]["simulation_id"] == "sim_parent00000"


# ---------------------------------------------------------------------------
# repro export consumes the stream
# ---------------------------------------------------------------------------
def test_repro_export_summarizes_run_events(tmp_path):
    _lifecycle(tmp_path, rounds=2)
    blob = repro_export.build_repro_config(
        {"simulation_id": SIM}, None, str(tmp_path)
    )
    section = blob["run_events"]
    assert section["replayable"] is True
    assert section["final_status"] == "completed"
    assert section["event_count"] == 11
    assert "_projection" not in section  # private key stripped
    assert repro_export.validate_blob(blob) == []


def test_repro_export_null_without_stream(tmp_path):
    blob = repro_export.build_repro_config(
        {"simulation_id": SIM}, None, str(tmp_path)
    )
    assert blob["run_events"] is None
    assert repro_export.validate_blob(blob) == []


def test_repro_export_lineage_prefers_event_stream(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED, lineage_kind="fork",
         parent_simulation_id="sim_evtparent000")
    blob = repro_export.build_repro_config(
        {"simulation_id": SIM, "parent_simulation_id": "sim_staleparent0"},
        None,
        str(tmp_path),
    )
    assert blob["lineage"]["parent_simulation_id"] == "sim_evtparent000"


# ---------------------------------------------------------------------------
# live tailing + SSE stream (u5fv.4)
# ---------------------------------------------------------------------------
def _drain(gen, n):
    """Pull up to n non-None items from a tail/stream generator."""
    out = []
    for item in gen:
        if item is not None:
            out.append(item)
        if len(out) >= n:
            break
    return out


def test_tail_replays_then_picks_up_live_appends(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED)
    log = RunEventLog(tmp_path)

    appended = {"done": False}

    def fake_sleep(_):
        # First idle cycle: append a new event mid-tail.
        if not appended["done"]:
            emit(tmp_path, SIM, EVENT_RUN_STARTED)
            appended["done"] = True

    gen = log.tail(sleep=fake_sleep)
    got = _drain(gen, 2)
    assert [(e.seq, e.event.event) for e in got] == [
        (0, EVENT_RUN_CREATED),
        (1, EVENT_RUN_STARTED),
    ]


def test_tail_respects_start_seq_and_skips_torn_tail(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED)
    emit(tmp_path, SIM, EVENT_RUN_STARTED)
    log = RunEventLog(tmp_path)
    with log.path.open("a") as f:
        f.write('{"torn')  # incomplete line, no newline

    stops = {"n": 0}

    def stop():
        stops["n"] += 1
        return stops["n"] > 2

    got = [e for e in log.tail(start_seq=1, stop=stop) if e is not None]
    # Only the complete event at seq 1; the torn tail is never consumed.
    assert [(e.seq, e.event.event) for e in got] == [(1, EVENT_RUN_STARTED)]


def test_sse_stream_formats_chunks_and_closes_on_terminal(tmp_path):
    _lifecycle(tmp_path, rounds=1)
    chunks = list(run_events.sse_stream(tmp_path))  # terminal event ends it

    assert chunks[0].startswith("id: 0\nevent: run.created\n")
    payload = json.loads(chunks[0].split("data: ", 1)[1].strip())
    assert payload["seq"] == 0
    assert payload["simulation_id"] == SIM
    # Last chunk is the terminal event; generator returned by itself.
    assert "event: run.completed\n" in chunks[-1]


def test_sse_stream_resumes_from_start_seq(tmp_path):
    _lifecycle(tmp_path, rounds=1)
    chunks = list(run_events.sse_stream(tmp_path, start_seq=7))
    assert chunks[0].startswith("id: 7\n")


def test_sse_stream_reconnect_after_terminal_closes_immediately(tmp_path):
    _lifecycle(tmp_path, rounds=1)
    _, checkpoint = RunEventLog(tmp_path).project()

    # Simulate EventSource auto-reconnect with Last-Event-ID at the tip.
    chunks = list(run_events.sse_stream(tmp_path, start_seq=checkpoint.last_seq + 1))
    assert len(chunks) == 1
    assert chunks[0].startswith("event: stream_end\n")


def test_sse_stream_emits_heartbeats_while_idle(tmp_path):
    emit(tmp_path, SIM, EVENT_RUN_CREATED)  # non-terminal: stream stays open

    clock = {"t": 0.0}
    stops = {"n": 0}

    def fake_clock():
        return clock["t"]

    def fake_sleep(_):
        clock["t"] += 20.0  # each idle cycle jumps past heartbeat_interval

    def stop():
        stops["n"] += 1
        return stops["n"] > 3

    chunks = list(run_events.sse_stream(
        tmp_path, clock=fake_clock, sleep=fake_sleep, stop=stop,
    ))
    assert any(c.startswith("event: heartbeat\n") for c in chunks)


def test_watch_page_wires_eventsource_with_polling_fallback():
    from app.services.watch_renderer import _broadcast_js

    js = _broadcast_js()
    assert "/events/stream" in js
    assert "EventSource" in js
    # Polling stays functional: demoted while SSE is live, restored on error.
    assert "SSE_FALLBACK_POLL_MS" in js
    assert "POLL_MS = 15000" in js
    # Terminal events close the stream deliberately (no reconnect loop).
    assert "run.completed" in js and "es.close()" in js


def test_sse_route_streams_finished_sim_over_http(tmp_path, monkeypatch):
    """Full HTTP path: mount simulation_bp, stream a finished sim's log."""
    from flask import Flask

    from app.config import Config

    sims_root = tmp_path / "sims"
    sim_dir = sims_root / SIM
    sim_dir.mkdir(parents=True)
    _lifecycle(sim_dir, rounds=1)
    monkeypatch.setattr(Config, "WONDERWALL_SIMULATION_DATA_DIR", str(sims_root))
    # Pin the keyless local-dev posture: with no internal key and DEBUG on,
    # the anonymous-spectator gate stays open (historic behavior). The gated
    # posture is covered by test_sse_route_gates_private_sim_for_anonymous.
    monkeypatch.delenv("MIROSHARK_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(Config, "DEBUG", True)

    from app.api import simulation_bp

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    client = app.test_client()

    resp = client.get(f"/api/simulation/{SIM}/events/stream")
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)  # terminal event closes the stream
    assert "event: run.created" in body
    assert "event: run.completed" in body
    assert "id: 0" in body

    # Reconnect with Last-Event-ID at the tip → immediate clean close.
    resp2 = client.get(
        f"/api/simulation/{SIM}/events/stream",
        headers={"Last-Event-ID": "10"},
    )
    assert "event: stream_end" in resp2.get_data(as_text=True)

    # Unknown sim → 404; pre-stream sim dir → single no_stream event.
    assert client.get("/api/simulation/sim_ffffffffffff/events/stream").status_code == 404
    bare = sims_root / "sim_barebones000"
    bare.mkdir()
    resp3 = client.get("/api/simulation/sim_barebones000/events/stream")
    assert "event: no_stream" in resp3.get_data(as_text=True)


def test_sse_route_gates_private_sim_for_anonymous(tmp_path, monkeypatch):
    """Keyed deployment: anonymous stream of a private sim gets no_stream.

    The spectator exemption lets this route past the blanket internal-key
    guard (EventSource cannot send headers), so the route itself must gate
    private sims for callers without the key — and serve keyed callers.
    """
    from flask import Flask

    from app.config import Config

    sims_root = tmp_path / "sims"
    sim_dir = sims_root / SIM
    sim_dir.mkdir(parents=True)
    _lifecycle(sim_dir, rounds=1)
    monkeypatch.setattr(Config, "WONDERWALL_SIMULATION_DATA_DIR", str(sims_root))
    monkeypatch.setenv("MIROSHARK_INTERNAL_KEY", "test-key-123")

    from app.api import simulation_bp

    app = Flask(__name__)
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    client = app.test_client()

    # Anonymous: gated with a clean no_stream event, not the run events.
    resp = client.get(f"/api/simulation/{SIM}/events/stream")
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)
    assert "no_stream" in body
    assert "not public" in body

    # Keyed operator: full stream.
    resp = client.get(
        f"/api/simulation/{SIM}/events/stream",
        headers={"x-miroshark-internal-key": "test-key-123"},
    )
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "no_stream" not in body
    assert "run.started" in body or "id:" in body
