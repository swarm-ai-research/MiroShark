"""FeedOracle MCP connector — grounded data for simulation seeds.

The sibling `mirofish-oracle-seeds` repo exposes 484 MCP tools across 44
oracle servers (MiCA, DORA, macro, DEX liquidity, sanctions, carbon, etc.).
MiroShark templates can opt into a subset by declaring::

    "oracle_tools": [
        {"server": "feedoracle_core", "tool": "mica_status",     "args": {"token_symbol": "USDT"}},
        {"server": "feedoracle_core", "tool": "peg_deviation",   "args": {"token_symbol": "USDT"}},
        {"server": "feedoracle_core", "tool": "macro_risk",      "args": {}}
    ]

At template-use time, ``resolve_oracle_tools`` dispatches each call through
the shared MCP client layer (``agent_mcp_tools.MCPHttpSession``) and returns
a markdown-formatted evidence block to be appended to the template's
``seed_document`` before graph build.

Server resolution is config-first: when the entry's ``server`` name matches
an ``http`` server in the shared ``config/mcp_servers.yaml`` manifest, that
spec (url + auth headers) is used — adding a new oracle is config-only.
Unmatched names fall back to the legacy FeedOracle endpoint
(``FEEDORACLE_MCP_URL``, default ``https://mcp.feedoracle.io/mcp``) with
``FEEDORACLE_API_KEY`` injected as a tool argument, so existing templates
keep working with zero config changes.

Opt-in. Disabled by default (``ORACLE_SEED_ENABLED=false``). Silently
returns an empty block on any failure — the template still works without
oracle data.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:  # httpx is already pulled in via OpenAI SDK — but we import defensively.
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

from ..utils.logger import get_logger

logger = get_logger("miroshark.oracle_seed")


_DEFAULT_ENDPOINT = "https://mcp.feedoracle.io/mcp"
_DEFAULT_TIMEOUT_SEC = 15.0


def _enabled() -> bool:
    return (os.environ.get("ORACLE_SEED_ENABLED", "false").lower() == "true")


def _endpoint() -> str:
    return (os.environ.get("FEEDORACLE_MCP_URL") or _DEFAULT_ENDPOINT).rstrip("/")


def _api_key() -> Optional[str]:
    return os.environ.get("FEEDORACLE_API_KEY") or None


def _session_for(server: str, registry: Dict[str, Any]):
    """Resolve a server name to an (MCPHttpSession, inject_api_key) pair.

    Config-first: an ``http`` entry in the shared manifest wins (auth via its
    headers). Anything else — including the empty server name — falls back to
    a synthetic spec for the legacy FeedOracle endpoint, where the API key is
    injected as a tool argument (the FeedOracle convention).
    """
    from .agent_mcp_tools import MCPHttpSession, MCPServerSpec

    spec = registry.get(server)
    if spec is not None and getattr(spec, "transport", "stdio") == "http":
        return MCPHttpSession(spec, timeout_sec=_DEFAULT_TIMEOUT_SEC), False
    fallback = MCPServerSpec(
        name=server or "feedoracle",
        transport="http",
        url=_endpoint(),
    )
    return MCPHttpSession(fallback, timeout_sec=_DEFAULT_TIMEOUT_SEC), True


def resolve_oracle_tools(template: Dict[str, Any]) -> str:
    """Dispatch ``template['oracle_tools']`` and return a markdown evidence block.

    Returns an empty string if oracle seeds are disabled, the list is empty,
    or any dispatch fails — the caller can safely concatenate the result onto
    the seed document without worrying about None.
    """
    tools = template.get("oracle_tools") or []
    if not tools or not _enabled() or httpx is None:
        return ""

    from .agent_mcp_tools import load_manifest_registry

    api_key = _api_key()
    registry = load_manifest_registry()
    sessions: Dict[str, Any] = {}  # server name -> (session, inject_api_key)
    results: List[Dict[str, Any]] = []

    def _get_session(server: str):
        if server in sessions:
            return sessions[server]
        try:
            session, inject_key = _session_for(server, registry)
            session.initialize()
        except Exception as exc:
            logger.warning(f"oracle_seed: init for {server or 'feedoracle'} failed ({exc}) — skipping")
            sessions[server] = None
            return None
        sessions[server] = (session, inject_key)
        return sessions[server]

    for entry in tools:
        if not isinstance(entry, dict):
            continue
        server = (entry.get("server") or "").strip()
        name = (entry.get("tool") or "").strip()
        if not name:
            continue
        resolved = _get_session(server)
        if resolved is None:
            continue
        session, inject_key = resolved
        args = dict(entry.get("args") or {})
        if inject_key and api_key:
            args.setdefault("api_key", api_key)
        # Config-declared servers own their namespace: call the bare tool
        # name. The legacy shared endpoint multiplexes servers, so try the
        # namespaced form first, then bare (both forms exist in the wild).
        candidates = [name] if not inject_key else (
            [f"{server}__{name}", name] if server else [name]
        )
        data = None
        for i, candidate in enumerate(candidates):
            try:
                data = session.call_tool(candidate, args)
                break
            except Exception as exc:
                if i < len(candidates) - 1:
                    logger.info(f"oracle_seed: {candidate} failed ({exc}) — trying bare name")
                else:
                    logger.warning(f"oracle_seed: {name} failed ({exc}) — skipping this tool")
        else:
            continue
        results.append({"server": server, "tool": name, "args": args, "data": data})

    for resolved in sessions.values():
        if resolved is not None:
            resolved[0].close()

    if not results:
        return ""

    lines = ["", "## Oracle Evidence (live at ingest time)", ""]
    for r in results:
        label = f"{r['server']}/{r['tool']}" if r["server"] else r["tool"]
        args_str = ", ".join(f"{k}={v}" for k, v in (r["args"] or {}).items()) or "(no args)"
        lines.append(f"### {label}  —  {args_str}")
        data = r["data"]
        try:
            pretty = json.dumps(data, ensure_ascii=False, indent=2, default=str)[:1500]
        except Exception:
            pretty = str(data)[:1500]
        lines.append("```json")
        lines.append(pretty)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
