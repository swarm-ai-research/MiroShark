"""Checkpoint export — sim state snapshots on a dedicated Git branch.

Adapts fabro-checkpoint's branch-per-run pattern for **audit and disaster
recovery, not version control**: each checkpoint commits the simulation's
JSON state artifacts to ``refs/heads/checkpoint/<simulation_id>`` in a
dedicated bare repository (default ``uploads/checkpoints.git``). Belief
state is snapshotted as opaque JSON blobs — deliberately NOT diffed as
code; the branch history answers "what did the run's state look like at
each checkpoint" and lets an operator restore a wrecked sim directory.

Implementation uses git plumbing only (``hash-object`` / ``mktree`` /
``commit-tree`` / ``update-ref``): no working tree, no checkouts, so
concurrent checkpoints of different sims can't race each other and the
live simulation directory is never touched by the exporter.

Every checkpoint also embeds a ``checkpoint_meta.json`` blob (when the sim
has a RunEvent stream, its projected status + last seq travel along), so a
snapshot is self-describing when inspected years later with plain git.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger("miroshark.checkpoint")

DEFAULT_REPO_DIRNAME = "checkpoints.git"

# The JSON state artifacts that make up a snapshot. Missing files are
# skipped (a sim that never ran has no run_state.json — that's fine).
SNAPSHOT_FILES = (
    "state.json",
    "run_state.json",
    "simulation_config.json",
    "run_events.jsonl",
    "counterfactual_injection.json",
    "quality.json",
    "trajectory.json",
    "resolution.json",
    "twitter_profiles.json",
    "reddit_profiles.json",
    "polymarket_profiles.json",
)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "miroshark-checkpoint",
    "GIT_AUTHOR_EMAIL": "checkpoint@miroshark.local",
    "GIT_COMMITTER_NAME": "miroshark-checkpoint",
    "GIT_COMMITTER_EMAIL": "checkpoint@miroshark.local",
}


def _git(repo: Path, args: List[str], *, input_bytes: Optional[bytes] = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **_GIT_ENV},
    )
    return result.stdout.decode("utf-8").strip()


def _branch_ref(simulation_id: str) -> str:
    return f"refs/heads/checkpoint/{simulation_id}"


def ensure_repo(repo_dir: Path) -> Path:
    """Initialise the dedicated bare checkpoint repository if missing."""

    repo_dir = Path(repo_dir)
    if not (repo_dir / "HEAD").exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        _git(repo_dir, ["init", "--bare", "-q", "."])
    return repo_dir


def export_checkpoint(
    sim_dir: Path | str,
    simulation_id: str,
    repo_dir: Path | str,
    *,
    label: str = "",
) -> Dict[str, Any]:
    """Snapshot the sim's JSON state onto its checkpoint branch.

    Returns ``{commit, branch, files, checkpoint_index}``. Raises
    ``ValueError`` when the sim directory has no snapshot-able artifacts.
    """

    sim_dir = Path(sim_dir)
    repo = ensure_repo(Path(repo_dir))
    ref = _branch_ref(simulation_id)

    # Blob every present artifact. Contents are read once and hashed via
    # stdin so a file being rewritten mid-export can't half-appear.
    tree_lines: List[str] = []
    files: List[str] = []
    for name in SNAPSHOT_FILES:
        path = sim_dir / name
        if not path.is_file():
            continue
        blob = _git(repo, ["hash-object", "-w", "--stdin"], input_bytes=path.read_bytes())
        tree_lines.append(f"100644 blob {blob}\t{name}")
        files.append(name)
    if not files:
        raise ValueError(f"no snapshot-able state files in {sim_dir}")

    parent = branch_head(simulation_id, repo)
    meta: Dict[str, Any] = {
        "schema": "miroshark.checkpoint.v1",
        "simulation_id": simulation_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "files": files,
        "parent_commit": parent,
        "note": "audit/recovery snapshot — not version control",
    }
    # Self-describing snapshots: fold the RunEvent projection in when the
    # sim has a stream, so the commit records how far the run had gotten.
    try:
        from .run_events import RunEventLog

        log = RunEventLog(sim_dir)
        if log.path.exists():
            projection, checkpoint = log.project()
            meta["run_status"] = projection.status
            meta["run_events_last_seq"] = checkpoint.last_seq
    except Exception:  # noqa: BLE001 — meta enrichment is best-effort
        pass
    meta_blob = _git(
        repo,
        ["hash-object", "-w", "--stdin"],
        input_bytes=(json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    tree_lines.append(f"100644 blob {meta_blob}\tcheckpoint_meta.json")

    tree = _git(repo, ["mktree"], input_bytes=("\n".join(tree_lines) + "\n").encode("utf-8"))
    message = f"checkpoint {simulation_id}" + (f": {label}" if label else "")
    commit_args = ["commit-tree", tree, "-m", message]
    if parent:
        commit_args += ["-p", parent]
    commit = _git(repo, commit_args)
    # Compare-and-swap on the old tip: a concurrent export of the same sim
    # fails loudly instead of silently dropping the other's commit.
    _git(repo, ["update-ref", ref, commit] + ([parent] if parent else []))

    index = len(list_checkpoints(simulation_id, repo))
    logger.info(f"checkpoint: {simulation_id} -> {commit[:12]} ({len(files)} files)")
    return {
        "commit": commit,
        "branch": ref,
        "files": files,
        "checkpoint_index": index,
    }


def branch_head(simulation_id: str, repo_dir: Path | str) -> Optional[str]:
    """Current tip of the sim's checkpoint branch, or ``None``.

    A repo that has never been initialised (no checkpoint taken yet) is the
    same as "no checkpoints": read paths must not require the bare repo to
    exist — only :func:`export_checkpoint` creates it.
    """

    repo = Path(repo_dir)
    if not (repo / "HEAD").exists():
        return None
    try:
        return _git(repo, ["rev-parse", "--verify", "-q", _branch_ref(simulation_id)])
    except subprocess.CalledProcessError:
        return None


def list_checkpoints(simulation_id: str, repo_dir: Path | str) -> List[Dict[str, Any]]:
    """All checkpoints for a sim, oldest first (empty when none)."""

    repo = Path(repo_dir)
    if branch_head(simulation_id, repo) is None:
        return []
    raw = _git(
        repo,
        ["log", "--reverse", "--format=%H%x00%cI%x00%s", _branch_ref(simulation_id)],
    )
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        commit, ts, subject = line.split("\x00", 2)
        out.append({"commit": commit, "committed_at": ts, "subject": subject})
    return out


def import_checkpoint(
    simulation_id: str,
    dest_dir: Path | str,
    repo_dir: Path | str,
    *,
    commit: Optional[str] = None,
) -> List[str]:
    """Restore a checkpoint's files into ``dest_dir`` (disaster recovery).

    Defaults to the branch tip; pass ``commit`` for an earlier snapshot.
    Returns the restored filenames (``checkpoint_meta.json`` included, so
    the restored directory records where it came from).
    """

    repo = Path(repo_dir)
    target = commit or branch_head(simulation_id, repo)
    if not target:
        raise ValueError(f"no checkpoints for {simulation_id}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    restored: List[str] = []
    for line in _git(repo, ["ls-tree", target]).splitlines():
        mode_type_sha, name = line.split("\t", 1)
        sha = mode_type_sha.split()[2]
        content = subprocess.run(
            ["git", "cat-file", "blob", sha],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        (dest / name).write_bytes(content)
        restored.append(name)
    return restored
