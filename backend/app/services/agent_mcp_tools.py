"""Per-agent MCP tools — OpenMiro-style pluggable toolset for personas.

Grants select personas (journalists, analysts, traders) access to real MCP
tools during simulation. Disabled by default: gated by
``MCP_AGENT_TOOLS_ENABLED`` at runtime, and further per-persona by the
``tools_enabled`` flag on :class:`WonderwallAgentProfile`.

## Wiring

1. Set ``MCP_AGENT_TOOLS_ENABLED=true`` in ``.env``.
2. Drop a YAML manifest at ``MCP_SERVERS_CONFIG`` (default ``config/mcp_servers.yaml``).
   Schema (OpenMiro-compatible; ``transport`` follows fabro-mcp's
   ``McpTransport`` — ``stdio`` subprocesses and remote ``http`` servers are
   declared side by side and dispatched through one client layer)::

       mcp_servers:
         - name: web_search              # transport defaults to stdio
           command: python
           args: ["-m", "mcp_web_search"]
           env:
             BRAVE_API_KEY: "${BRAVE_KEY}"
         - name: price_feed
           command: npx
           args: ["-y", "@feedoracle/mcp-remote"]
         - name: feedoracle_core         # remote MCP server, no subprocess
           transport: http
           url: https://mcp.feedoracle.io/mcp
           headers:
             Authorization: "Bearer ${FEEDORACLE_API_KEY}"

3. In preset templates or at simulation-config time, mark personas with
   ``tools_enabled: true`` and optionally ``allowed_tools: [name,...]``.

The simulation loop should call :func:`build_agent_toolset` per round and
pass the resulting dispatcher to the agent's prompt builder. Full Wonderwall
integration requires changes to the wonderwall runner; this module provides
the primitive so that work can proceed without blocking the feature flag.

Keeps dependencies soft: if ``pyyaml`` is missing, returns an empty registry
with a warning (rather than crashing the Flask app on import).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger("miroshark.agent_mcp")


@dataclass
class MCPServerSpec:
    """Parsed MCP server entry from the YAML manifest.

    ``transport`` mirrors fabro-mcp's ``McpTransport``: ``stdio`` entries
    carry ``command``/``args``/``env`` (spawned as a subprocess), ``http``
    entries carry ``url``/``headers`` (JSON-RPC over POST, session id via
    ``Mcp-Session-Id``). One registry, one dispatch layer, both transports.
    """
    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # stdio | http
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


def _enabled() -> bool:
    return os.environ.get("MCP_AGENT_TOOLS_ENABLED", "false").lower() == "true"


def _manifest_path() -> str:
    return os.environ.get("MCP_SERVERS_CONFIG") or os.path.join(
        os.getcwd(), "config", "mcp_servers.yaml"
    )


def _resolve_env_map(raw: Dict[str, str]) -> Dict[str, str]:
    """Resolve ``${ENVVAR}`` placeholders in a str→str mapping.

    Placeholders may be embedded ("Bearer ${KEY}") or the whole value
    ("${KEY}"); unset variables resolve to the empty string.
    """
    import re

    resolved = {}
    for k, v in (raw or {}).items():
        if isinstance(v, str):
            resolved[k] = re.sub(
                r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v
            )
        else:
            resolved[k] = str(v)
    return resolved


def load_registry() -> Dict[str, MCPServerSpec]:
    """Parse the manifest into a name → MCPServerSpec map.

    Returns an empty dict when the feature is off, the file is missing,
    ``pyyaml`` isn't installed, or the file is malformed. Never raises.
    """
    if not _enabled():
        return {}
    return load_manifest_registry()


def load_manifest_registry() -> Dict[str, MCPServerSpec]:
    """Parse the manifest without the ``MCP_AGENT_TOOLS_ENABLED`` gate.

    For consumers with their own opt-in flag (oracle seeding uses
    ``ORACLE_SEED_ENABLED``) that still want the shared server config.
    Same never-raises contract as :func:`load_registry`.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("agent_mcp: pyyaml not installed — per-agent MCP tools disabled")
        return {}

    path = _manifest_path()
    if not os.path.exists(path):
        logger.info(f"agent_mcp: manifest not found at {path} — no tools registered")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning(f"agent_mcp: failed to parse {path} ({exc})")
        return {}

    servers: Dict[str, MCPServerSpec] = {}
    entries = raw.get("mcp_servers") or raw.get("servers") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        transport = (entry.get("transport") or "stdio").strip().lower()
        command = (entry.get("command") or "").strip()
        url = (entry.get("url") or "").strip()
        if not name:
            continue
        if transport == "stdio" and not command:
            continue
        if transport == "http" and not url:
            logger.warning(f"agent_mcp: http server {name!r} missing url — skipped")
            continue
        if transport not in ("stdio", "http"):
            logger.warning(f"agent_mcp: server {name!r} has unknown transport {transport!r} — skipped")
            continue
        servers[name] = MCPServerSpec(
            name=name,
            command=command,
            args=list(entry.get("args") or []),
            env=_resolve_env_map(entry.get("env") or {}),
            transport=transport,
            url=url,
            headers=_resolve_env_map(entry.get("headers") or {}),
        )

    logger.info(f"agent_mcp: loaded {len(servers)} MCP server(s) from {path}")
    return servers


class MCPHttpSession:
    """MCP client for ``transport: http`` servers (JSON-RPC over POST).

    Interface-compatible with the bridge's stdio ``_MCPProcess`` —
    ``initialize`` / ``list_tools`` / ``call_tool`` / ``shutdown`` — so the
    dispatch layer treats both transports identically (fabro-mcp's
    ``McpTransport::Http`` counterpart). Tracks the ``Mcp-Session-Id``
    header across calls, per the streamable-HTTP MCP convention.
    """

    def __init__(self, spec: MCPServerSpec, timeout_sec: float = 15.0):
        import httpx  # deferred: soft dependency, matches oracle_seed

        self.name = spec.name
        self.url = spec.url.rstrip("/")
        self.headers = dict(spec.headers or {})
        self.session_id: Optional[str] = None
        self._client = httpx.Client(timeout=timeout_sec)
        self._initialized = False
        self._cached_tools: Optional[List[Dict[str, object]]] = None

    def _rpc(self, method: str, params: Dict[str, object], timeout: Optional[float] = None) -> Dict[str, object]:
        import json
        import uuid

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": method,
            "params": params,
        }
        resp = self._client.post(self.url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        data: Dict[str, object] = resp.json()
        return data

    def initialize(self, timeout: float = 10.0) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "miroshark", "version": "1.0"},
            },
            timeout=timeout,
        )
        self._initialized = True

    def list_tools(self, timeout: float = 10.0) -> List[Dict[str, object]]:
        if self._cached_tools is not None:
            return self._cached_tools
        resp = self._rpc("tools/list", {}, timeout=timeout)
        tools = (resp.get("result") or {}).get("tools") or []  # type: ignore[union-attr]
        self._cached_tools = tools
        return tools

    def call_tool(self, name: str, args: Dict[str, object], timeout: Optional[float] = None) -> object:
        import json

        resp = self._rpc("tools/call", {"name": name, "arguments": dict(args or {})}, timeout=timeout)
        result = resp.get("result") or {}
        # Standard MCP content shape: [{"type": "text", "text": "..."}]
        content = result.get("content") if isinstance(result, dict) else None
        if content and isinstance(content, list):
            first = content[0]
            if isinstance(first, dict) and first.get("text"):
                try:
                    return json.loads(first["text"])
                except json.JSONDecodeError:
                    return first["text"]
        return result

    def shutdown(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # oracle_seed's legacy client called this ``close``; keep the alias.
    close = shutdown


def build_agent_toolset(
    profile,  # WonderwallAgentProfile-like duck type
    registry: Optional[Dict[str, MCPServerSpec]] = None,
) -> Dict[str, MCPServerSpec]:
    """Return the subset of the registry this persona may call.

    An empty return means the agent has no tools (either globally off,
    persona opted out, or allowlist narrows to nothing). Callers don't need
    to check ``_enabled()`` themselves.
    """
    if not _enabled() or not getattr(profile, "tools_enabled", False):
        return {}
    reg = registry if registry is not None else load_registry()
    if not reg:
        return {}
    allowed = list(getattr(profile, "allowed_tools", None) or [])
    if not allowed:
        return dict(reg)  # persona with no allowlist → all tools
    return {name: spec for name, spec in reg.items() if name in allowed}


def summarize_toolset(tools: Dict[str, MCPServerSpec]) -> str:
    """Short human-readable listing suitable for an agent system prompt."""
    if not tools:
        return "(no MCP tools available this round)"
    lines = ["Available MCP tools:"]
    for name, spec in tools.items():
        if getattr(spec, "transport", "stdio") == "http":
            lines.append(f"  - {name}: {spec.url}")
        else:
            args_str = " ".join(spec.args)
            lines.append(f"  - {name}: {spec.command} {args_str}".rstrip())
    return "\n".join(lines)
