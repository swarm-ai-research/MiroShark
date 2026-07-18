"""Unified, durable RunEvent stream for simulations (fabro-store pattern).

Simulation state today is scattered: inline logging in ``simulation_runner``,
``state.json`` / ``run_state.json`` snapshots, per-platform ``actions.jsonl``
files, and fire-and-forget webhooks. None of it is a single replayable
record of *what happened, in order*.

This module adds one: every lifecycle transition is appended to
``run_events.jsonl`` in the simulation directory as a :class:`RunEvent`. The
stream is:

- **Append-only** — events are never rewritten; ``seq`` is derived at replay
  time from file position, so the file stays byte-stable and diff-friendly.
- **Replayable** — :class:`RunProjection` folds the stream back into the
  run's current state (status, per-platform rounds, lineage) from nothing
  but the log.
- **Resumable** — :class:`Checkpoint` snapshots a projection at a ``seq``
  watermark; folding resumes from ``last_seq + 1`` and re-applied envelopes
  are skipped (idempotent), so a monitor restart mid-round rebuilds state
  without re-reading the whole log.
- **Lineage-carrying** — ``run.created`` records ``parent_simulation_id``
  and ``lineage_kind`` (original / fork / counterfactual), so lineage is
  queryable straight from the log without scanning ``state.json`` files.

The design mirrors fabro-store's ``EventEnvelope`` / ``RunProjectionReducer``
(and the SWARM port in ``swarm/logging/projection.py``): envelope = ``seq`` +
event, reducer = pure fold, checkpoint = ``{last_seq, state}``.

Emission is **best-effort by contract**: :func:`emit` swallows I/O errors so
the event stream can never break a live simulation. Stdlib-only on purpose —
unit tests run without Flask.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

RUN_EVENTS_FILENAME = "run_events.jsonl"
EVENT_SCHEMA_V1 = "miroshark.run_event.v1"

# Lifecycle event names. Dotted ``noun.verb`` like fabro's EventBody variants.
EVENT_RUN_CREATED = "run.created"
EVENT_RUN_PARENT_LINKED = "run.parent_linked"
EVENT_PREPARE_STARTED = "run.prepare_started"
EVENT_PREPARE_COMPLETED = "run.prepare_completed"
EVENT_RUN_STARTED = "run.started"
EVENT_RUN_RUNNING = "run.running"
EVENT_ROUND_STARTED = "round.started"
EVENT_ROUND_COMPLETED = "round.completed"
EVENT_PLATFORM_COMPLETED = "platform.completed"
EVENT_RUN_COMPLETED = "run.completed"
EVENT_RUN_STOPPED = "run.stopped"
EVENT_RUN_FAILED = "run.failed"


@dataclass(frozen=True)
class RunEvent:
    """One immutable lifecycle event for one simulation."""

    id: str
    ts: str
    simulation_id: str
    event: str
    platform: Optional[str] = None
    round: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {"schema": EVENT_SCHEMA_V1, **asdict(self)}
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunEvent":
        return cls(
            id=data.get("id", ""),
            ts=data.get("ts", ""),
            simulation_id=data.get("simulation_id", ""),
            event=data.get("event", ""),
            platform=data.get("platform"),
            round=data.get("round"),
            properties=dict(data.get("properties", {})),
        )

    @classmethod
    def new(
        cls,
        simulation_id: str,
        event: str,
        *,
        platform: Optional[str] = None,
        round: Optional[int] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        return cls(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            ts=datetime.now(timezone.utc).isoformat(),
            simulation_id=simulation_id,
            event=event,
            platform=platform,
            round=round,
            properties=dict(properties or {}),
        )


@dataclass(frozen=True)
class EventEnvelope:
    """A replayed event tagged with its 0-indexed position in the log.

    ``seq`` is derived at replay time and never written to disk, so existing
    logs remain readable byte-for-byte (same convention as fabro-store keys /
    the SWARM ``EventEnvelope`` port).
    """

    seq: int
    event: RunEvent


class RunEventLog:
    """Append-only JSONL event log for one simulation directory."""

    def __init__(self, sim_dir: Path | str) -> None:
        self.path = Path(sim_dir) / RUN_EVENTS_FILENAME

    def append(self, event: RunEvent) -> RunEvent:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Heal a torn tail (crash mid-line): if the file doesn't end in a
        # newline, terminate the partial line first so the corrupt record
        # stays isolated instead of swallowing this append too.
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event

    def replay(self) -> List[RunEvent]:
        if not self.path.exists():
            return []
        events: List[RunEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(RunEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                # A torn write (crash mid-line) must not make the whole run
                # unreplayable; skip the corrupt line.
                continue
        return events

    def iter_envelopes(
        self, *, start_seq: int = 0, until_seq: Optional[int] = None
    ) -> Iterator[EventEnvelope]:
        for seq, event in enumerate(self.replay()):
            if seq < start_seq:
                continue
            if until_seq is not None and seq > until_seq:
                break
            yield EventEnvelope(seq=seq, event=event)

    def project(
        self,
        *,
        until_seq: Optional[int] = None,
        resume: Optional["Checkpoint"] = None,
    ) -> Tuple["RunProjection", "Checkpoint"]:
        """Fold the log into a :class:`RunProjection` (checkpoint-resumable)."""

        reducer = RunProjectionReducer()
        start_seq = 0 if resume is None else resume.last_seq + 1
        return reducer.replay(
            self.iter_envelopes(start_seq=start_seq, until_seq=until_seq),
            resume=resume,
        )

    def tail(
        self,
        *,
        start_seq: int = 0,
        poll_interval: float = 0.5,
        stop: Optional[Callable[[], bool]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[Optional[EventEnvelope]]:
        """Follow the log live: replay from ``start_seq``, then poll for more.

        Yields :class:`EventEnvelope` for each new event and ``None`` once per
        idle poll cycle (so callers can interleave heartbeats without a second
        thread). ``seq`` numbering matches :meth:`iter_envelopes` exactly —
        corrupt lines are skipped without consuming a seq. Only complete
        (newline-terminated) lines are consumed; a torn tail stays buffered
        until the writer heals it. Runs until ``stop()`` returns true.
        """

        next_seq = 0
        pos = 0
        while True:
            emitted = False
            if self.path.exists():
                with self.path.open("rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                # Consume only complete lines; leave a torn tail in place.
                end = chunk.rfind(b"\n")
                if end >= 0:
                    for raw in chunk[:end].split(b"\n"):
                        pos += len(raw) + 1
                        if not raw.strip():
                            continue
                        try:
                            event = RunEvent.from_dict(json.loads(raw))
                        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                            continue
                        seq = next_seq
                        next_seq += 1
                        if seq >= start_seq:
                            emitted = True
                            yield EventEnvelope(seq=seq, event=event)
            if stop is not None and stop():
                return
            if not emitted:
                yield None
                sleep(poll_interval)


# ---------------------------------------------------------------------------
# projection (fold events -> current run state)
# ---------------------------------------------------------------------------
@dataclass
class RunProjection:
    """Current state of a run, reconstructed purely from the event stream."""

    simulation_id: str = ""
    status: str = "unknown"
    parent_simulation_id: Optional[str] = None
    lineage_kind: str = "original"
    current_round: Dict[str, int] = field(default_factory=dict)
    completed_platforms: List[str] = field(default_factory=list)
    total_rounds: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunProjection":
        return cls(**data)


@dataclass
class Checkpoint:
    """A projection snapshot at a ``seq`` watermark (``-1`` = empty state)."""

    last_seq: int
    state: Dict[str, Any]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_seq": self.last_seq, "state": self.state}, sort_keys=True)
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Checkpoint":
        data = json.loads(Path(path).read_text())
        return cls(last_seq=int(data["last_seq"]), state=dict(data["state"]))


class RunProjectionReducer:
    """Fold envelopes into a :class:`RunProjection` (idempotent on resume)."""

    _STATUS_BY_EVENT = {
        EVENT_RUN_CREATED: "created",
        EVENT_PREPARE_STARTED: "preparing",
        EVENT_PREPARE_COMPLETED: "ready",
        EVENT_RUN_STARTED: "starting",
        EVENT_RUN_RUNNING: "running",
        EVENT_RUN_COMPLETED: "completed",
        EVENT_RUN_STOPPED: "stopped",
        EVENT_RUN_FAILED: "failed",
    }

    def initial(self) -> RunProjection:
        return RunProjection()

    def apply(self, state: RunProjection, envelope: EventEnvelope) -> RunProjection:
        event = envelope.event
        if event.simulation_id and not state.simulation_id:
            state.simulation_id = event.simulation_id

        status = self._STATUS_BY_EVENT.get(event.event)
        if status is not None:
            state.status = status

        if event.event == EVENT_RUN_CREATED:
            state.created_at = event.ts
            props = event.properties
            state.parent_simulation_id = props.get("parent_simulation_id")
            state.lineage_kind = props.get("lineage_kind", "original")
            if props.get("total_rounds") is not None:
                state.total_rounds = props["total_rounds"]
        elif event.event == EVENT_RUN_PARENT_LINKED:
            state.parent_simulation_id = event.properties.get("parent_simulation_id")
            state.lineage_kind = event.properties.get("lineage_kind", state.lineage_kind)
        elif event.event in (EVENT_ROUND_STARTED, EVENT_ROUND_COMPLETED):
            if event.platform is not None and event.round is not None:
                state.current_round[event.platform] = event.round
            if event.properties.get("total_rounds") is not None:
                state.total_rounds = event.properties["total_rounds"]
        elif event.event == EVENT_PLATFORM_COMPLETED:
            if event.platform and event.platform not in state.completed_platforms:
                state.completed_platforms.append(event.platform)
        elif event.event == EVENT_RUN_FAILED:
            state.error = event.properties.get("error")

        if event.event in (EVENT_RUN_COMPLETED, EVENT_RUN_STOPPED, EVENT_RUN_FAILED):
            state.finished_at = event.ts
        return state

    def replay(
        self,
        envelopes: Iterable[EventEnvelope],
        *,
        resume: Optional[Checkpoint] = None,
    ) -> Tuple[RunProjection, Checkpoint]:
        if resume is None:
            state = self.initial()
            last_seq = -1
        else:
            state = RunProjection.from_dict(resume.state)
            last_seq = resume.last_seq
        for envelope in envelopes:
            if envelope.seq <= last_seq:
                continue  # idempotent resume: never double-apply
            state = self.apply(state, envelope)
            last_seq = envelope.seq
        return state, Checkpoint(last_seq=last_seq, state=state.to_dict())


# ---------------------------------------------------------------------------
# SSE streaming (consumed by the /events/stream route + watch page)
# ---------------------------------------------------------------------------
TERMINAL_EVENTS = frozenset({EVENT_RUN_COMPLETED, EVENT_RUN_STOPPED, EVENT_RUN_FAILED})


def sse_stream(
    sim_dir: Path | str,
    *,
    start_seq: int = 0,
    poll_interval: float = 0.5,
    heartbeat_interval: float = 15.0,
    stop: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Yield the run's event stream as Server-Sent-Events wire chunks.

    Replays from ``start_seq`` (pass ``Last-Event-ID + 1`` on reconnect so
    EventSource resumes exactly where it left off), then follows the log
    live. Each event chunk carries ``id: <seq>`` — the browser stores it and
    sends it back as ``Last-Event-ID`` on auto-reconnect, so a dropped
    connection (or a gunicorn worker timeout) is seamless. Heartbeat chunks
    keep intermediaries from closing the idle connection. The stream ends
    after a terminal event (completed / stopped / failed) — for already-
    finished sims the client gets one full replay and a clean close.
    """

    log = RunEventLog(sim_dir)
    # Reconnect-after-terminal guard: EventSource auto-reconnects with
    # Last-Event-ID after we close a finished stream. If the run is already
    # terminal and the client has every event, close again immediately —
    # otherwise the reconnect would tail a dead log forever.
    projection, checkpoint = log.project()
    if projection.status in ("completed", "stopped", "failed") and start_seq > checkpoint.last_seq:
        yield 'event: stream_end\ndata: {"reason": "run already finished"}\n\n'
        return

    last_beat = clock()
    for item in log.tail(
        start_seq=start_seq, poll_interval=poll_interval, stop=stop, sleep=sleep
    ):
        if item is not None:
            payload = {"seq": item.seq, **item.event.to_dict()}
            yield (
                f"id: {item.seq}\n"
                f"event: {item.event.event}\n"
                f"data: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n\n"
            )
            if item.event.event in TERMINAL_EVENTS:
                return
        now = clock()
        if now - last_beat >= heartbeat_interval:
            yield 'event: heartbeat\ndata: {}\n\n'
            last_beat = now


# ---------------------------------------------------------------------------
# best-effort emission (the hook the runner/manager call sites use)
# ---------------------------------------------------------------------------
def emit(
    sim_dir: Path | str,
    simulation_id: str,
    event: str,
    *,
    platform: Optional[str] = None,
    round: Optional[int] = None,
    **properties: Any,
) -> Optional[RunEvent]:
    """Append a lifecycle event; never raises (a failed emit must not break
    a live simulation). Returns the event, or ``None`` if the write failed."""

    try:
        return RunEventLog(sim_dir).append(
            RunEvent.new(
                simulation_id,
                event,
                platform=platform,
                round=round,
                properties=properties or None,
            )
        )
    except Exception:  # noqa: BLE001 — best-effort by contract
        return None
