"""Unit tests for the per-agent MCP registry loader. No network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.services import agent_mcp_tools as amt


@dataclass
class _FakeProfile:
    tools_enabled: bool = False
    allowed_tools: list = None  # type: ignore


def test_load_registry_off_by_default(monkeypatch):
    monkeypatch.delenv("MCP_AGENT_TOOLS_ENABLED", raising=False)
    assert amt.load_registry() == {}


def test_load_registry_missing_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(tmp_path / "does_not_exist.yaml"))
    assert amt.load_registry() == {}


def test_load_registry_parses_manifest(monkeypatch, tmp_path: Path):
    pytest.importorskip("yaml")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
mcp_servers:
  - name: web_search
    command: python
    args: ["-m", "mcp_web"]
  - name: no_command
    args: []
"""
    )
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(manifest))
    reg = amt.load_registry()
    # Entry without command is dropped; valid entry survives.
    assert list(reg.keys()) == ["web_search"]
    assert reg["web_search"].command == "python"
    assert reg["web_search"].args == ["-m", "mcp_web"]


def test_build_toolset_respects_tools_enabled(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    reg = {
        "web_search": amt.MCPServerSpec(name="web_search", command="python"),
        "price_feed": amt.MCPServerSpec(name="price_feed", command="node"),
    }
    # Disabled persona → empty toolset even when registry is full.
    profile = _FakeProfile(tools_enabled=False)
    assert amt.build_agent_toolset(profile, registry=reg) == {}


def test_build_toolset_allowlist_filters(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    reg = {
        "web_search": amt.MCPServerSpec(name="web_search", command="python"),
        "price_feed": amt.MCPServerSpec(name="price_feed", command="node"),
    }
    profile = _FakeProfile(tools_enabled=True, allowed_tools=["price_feed"])
    got = amt.build_agent_toolset(profile, registry=reg)
    assert list(got.keys()) == ["price_feed"]


def test_build_toolset_no_allowlist_means_all(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    reg = {
        "web_search": amt.MCPServerSpec(name="web_search", command="python"),
    }
    profile = _FakeProfile(tools_enabled=True, allowed_tools=[])
    assert amt.build_agent_toolset(profile, registry=reg) == reg


def test_summarize_empty_tools():
    assert "no MCP tools" in amt.summarize_toolset({})


# ---------------------------------------------------------------------------
# http transport (fabro McpTransport pattern, u5fv.3)
# ---------------------------------------------------------------------------
def _write_manifest(tmp_path: Path, body: str) -> None:
    (tmp_path / "mcp_servers.yaml").write_text(body)


def test_load_registry_parses_http_transport(monkeypatch, tmp_path: Path):
    pytest.importorskip("yaml")
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(tmp_path / "mcp_servers.yaml"))
    monkeypatch.setenv("ORACLE_KEY", "sekrit")
    _write_manifest(tmp_path, """
mcp_servers:
  - name: local_search
    command: python
    args: ["-m", "mcp_search"]
  - name: feedoracle_core
    transport: http
    url: https://mcp.feedoracle.io/mcp
    headers:
      Authorization: "Bearer ${ORACLE_KEY}"
""")
    reg = amt.load_registry()
    assert set(reg) == {"local_search", "feedoracle_core"}
    assert reg["local_search"].transport == "stdio"
    fo = reg["feedoracle_core"]
    assert fo.transport == "http"
    assert fo.url == "https://mcp.feedoracle.io/mcp"
    assert fo.headers == {"Authorization": "Bearer sekrit"}


def test_load_registry_skips_invalid_transport_entries(monkeypatch, tmp_path: Path):
    pytest.importorskip("yaml")
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "true")
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(tmp_path / "mcp_servers.yaml"))
    _write_manifest(tmp_path, """
mcp_servers:
  - name: http_without_url
    transport: http
  - name: unknown_transport
    transport: carrier_pigeon
    command: python
  - name: stdio_without_command
  - name: ok
    command: python
""")
    assert set(amt.load_registry()) == {"ok"}


def test_load_manifest_registry_bypasses_enabled_gate(monkeypatch, tmp_path: Path):
    """oracle_seed has its own opt-in flag; it must see the config even when
    per-agent tools are globally off."""
    pytest.importorskip("yaml")
    monkeypatch.setenv("MCP_AGENT_TOOLS_ENABLED", "false")
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(tmp_path / "mcp_servers.yaml"))
    _write_manifest(tmp_path, """
mcp_servers:
  - name: weather
    transport: http
    url: https://w.example/mcp
""")
    assert amt.load_registry() == {}
    assert set(amt.load_manifest_registry()) == {"weather"}


def test_summarize_toolset_shows_url_for_http():
    tools = {
        "weather": amt.MCPServerSpec(name="weather", transport="http", url="https://w.example/mcp"),
        "search": amt.MCPServerSpec(name="search", command="python", args=["-m", "s"]),
    }
    out = amt.summarize_toolset(tools)
    assert "weather: https://w.example/mcp" in out
    assert "search: python -m s" in out


def test_bridge_dispatches_http_specs_to_http_session(monkeypatch):
    """MCPAgentBridge._ensure picks the transport from the spec; everything
    downstream (dispatch_calls) is transport-agnostic."""
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = str(_Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    import mcp_agent_bridge as bridge_mod

    created = []

    class _FakeHttpSession:
        def __init__(self, spec, timeout_sec=None):
            created.append(spec.name)

        def initialize(self, timeout=10.0):
            pass

        def call_tool(self, name, args, timeout=None):
            return {"echo": name}

        def shutdown(self):
            pass

    monkeypatch.setattr(
        "app.services.agent_mcp_tools.MCPHttpSession", _FakeHttpSession
    )
    registry = {
        "weather": amt.MCPServerSpec(name="weather", transport="http", url="https://w.example/mcp"),
    }
    bridge = bridge_mod.MCPAgentBridge(registry)
    results = bridge.dispatch_calls(
        [bridge_mod.MCPCallRequest(server="weather", tool="forecast", args={"city": "SG"})]
    )
    assert created == ["weather"]
    assert results[0].ok and results[0].data == {"echo": "forecast"}
    bridge.shutdown()
