"""Unit tests for the FeedOracle-seed connector. No network."""

from __future__ import annotations


from app.services import oracle_seed


def test_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "false")
    out = oracle_seed.resolve_oracle_tools(
        {"oracle_tools": [{"server": "x", "tool": "y", "args": {}}]}
    )
    assert out == ""


def test_returns_empty_when_no_tools(monkeypatch):
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")
    assert oracle_seed.resolve_oracle_tools({"oracle_tools": []}) == ""
    assert oracle_seed.resolve_oracle_tools({}) == ""


def test_returns_empty_when_httpx_missing(monkeypatch):
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")
    monkeypatch.setattr(oracle_seed, "httpx", None)
    out = oracle_seed.resolve_oracle_tools(
        {"oracle_tools": [{"server": "x", "tool": "y", "args": {}}]}
    )
    assert out == ""


def test_returns_empty_on_init_failure(monkeypatch):
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")

    def _boom(server, registry):
        raise RuntimeError("boom")

    monkeypatch.setattr(oracle_seed, "_session_for", _boom)
    out = oracle_seed.resolve_oracle_tools(
        {"oracle_tools": [{"server": "x", "tool": "y", "args": {}}]}
    )
    assert out == ""


class _FakeSession:
    def __init__(self, data=None, record=None):
        self._data = data if data is not None else {"price": 1.002, "deviation_bps": 20}
        self.record = record if record is not None else []

    def initialize(self):
        pass

    def call_tool(self, name, args, timeout=None):
        self.record.append((name, dict(args)))
        return self._data

    def close(self):
        pass


def test_formats_markdown_block(monkeypatch):
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")
    monkeypatch.setattr(
        oracle_seed, "_session_for", lambda server, registry: (_FakeSession(), True)
    )
    out = oracle_seed.resolve_oracle_tools({
        "oracle_tools": [
            {"server": "feedoracle_core", "tool": "peg_deviation", "args": {"token_symbol": "USDT"}},
        ],
    })
    assert "## Oracle Evidence" in out
    assert "feedoracle_core/peg_deviation" in out
    assert "token_symbol=USDT" in out
    assert "price" in out and "1.002" in out


def test_legacy_fallback_injects_api_key_and_namespaced_name(monkeypatch):
    """The un-configured path keeps the historical FeedOracle conventions."""
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")
    monkeypatch.setenv("FEEDORACLE_API_KEY", "sekrit")
    record = []
    monkeypatch.setattr(
        oracle_seed, "_session_for",
        lambda server, registry: (_FakeSession(record=record), True),
    )
    oracle_seed.resolve_oracle_tools({
        "oracle_tools": [{"server": "feedoracle_core", "tool": "macro_risk", "args": {}}],
    })
    assert record[0][0] == "feedoracle_core__macro_risk"  # namespaced first
    assert record[0][1].get("api_key") == "sekrit"


def test_config_declared_server_uses_bare_name_and_no_key_injection(monkeypatch):
    """A manifest-declared http server owns its namespace + auth headers."""
    monkeypatch.setenv("ORACLE_SEED_ENABLED", "true")
    monkeypatch.setenv("FEEDORACLE_API_KEY", "sekrit")
    record = []
    monkeypatch.setattr(
        oracle_seed, "_session_for",
        lambda server, registry: (_FakeSession(record=record), False),
    )
    oracle_seed.resolve_oracle_tools({
        "oracle_tools": [{"server": "weather", "tool": "forecast", "args": {"city": "SG"}}],
    })
    assert record == [("forecast", {"city": "SG"})]  # bare name, no api_key


def test_session_for_prefers_manifest_http_spec(monkeypatch):
    from app.services.agent_mcp_tools import MCPServerSpec

    class _FakeHttpSession:
        def __init__(self, spec, timeout_sec=None):
            self.spec = spec

    monkeypatch.setattr(
        "app.services.agent_mcp_tools.MCPHttpSession", _FakeHttpSession
    )
    registry = {
        "weather": MCPServerSpec(name="weather", transport="http", url="https://w.example/mcp"),
        "local_tool": MCPServerSpec(name="local_tool", command="python"),
    }
    session, inject = oracle_seed._session_for("weather", registry)
    assert session.spec.url == "https://w.example/mcp"
    assert inject is False

    # stdio manifest entries and unknown names both fall back to FeedOracle.
    monkeypatch.setenv("FEEDORACLE_MCP_URL", "https://legacy.example/mcp")
    for server in ("local_tool", "unknown", ""):
        session, inject = oracle_seed._session_for(server, registry)
        assert session.spec.url == "https://legacy.example/mcp"
        assert inject is True
