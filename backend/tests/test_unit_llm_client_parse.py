"""Unit tests for the chat_json greedy-extractor fallback.

Loads the helper directly (no Flask) so the tests run with only stdlib.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def extract():
    # Stub `app` + `app.utils` so the module's relative imports resolve
    # without dragging Flask + openai into the test path.
    for pkg, path in [("app", "app"), ("app.utils", "app/utils")]:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg); stub.__path__ = [str(_BACKEND / path)]
            sys.modules[pkg] = stub
    # The helper is module-level and pure — pull JUST the function out
    # by reading the source and exec'ing it in an isolated namespace.
    src = (_BACKEND / "app/utils/llm_client.py").read_text()
    # Locate the def and grab through its return at function end.
    start = src.index("def _extract_outermost_json")
    end_marker = "\n\n\n"
    end = src.find(end_marker, start)
    body = src[start:end if end > 0 else len(src)]
    ns: dict = {"Optional": __import__("typing").Optional}
    exec(body, ns)
    return ns["_extract_outermost_json"]


class TestExtractor:
    def test_clean_object_passes_through(self, extract):
        assert extract('{"a": 1}') == '{"a": 1}'

    def test_object_after_prose(self, extract):
        text = 'Sure! Here is your action:\n{"action_type": "gather"}'
        assert extract(text) == '{"action_type": "gather"}'

    def test_object_with_trailing_commentary(self, extract):
        text = '{"action_type": "build"}\nLet me know if you want me to elaborate.'
        assert extract(text) == '{"action_type": "build"}'

    def test_object_in_markdown_fence_residue(self, extract):
        # Fence regex partially stripped; closing fence text remains.
        text = '{"k": "v"}\n``` (and some extra text)'
        assert extract(text) == '{"k": "v"}'

    def test_nested_braces_balance_correctly(self, extract):
        text = 'noise {"outer": {"inner": [1, 2, 3]}} trailing'
        assert extract(text) == '{"outer": {"inner": [1, 2, 3]}}'

    def test_braces_inside_string_dont_throw_off_depth(self, extract):
        text = '{"q": "Will gini > 0.4 by epoch 5?", "n": 1}'
        assert extract(text) == text

    def test_escaped_quote_in_string_handled(self, extract):
        text = r'{"msg": "say \"hi\"", "n": 2}'
        assert extract(text) == text

    def test_unbalanced_returns_none(self, extract):
        assert extract('{"a": 1') is None

    def test_no_brackets_returns_none(self, extract):
        assert extract('hello world no json here') is None

    def test_array_top_level(self, extract):
        assert extract('some prose [1, 2, {"a": 3}] then more') == '[1, 2, {"a": 3}]'


class TestPersonas:
    """The contrarian persona module ships canonical templates a caller
    can drop into agent_specs to get a two-sided prediction market book."""

    def test_balanced_lineup_includes_contrarian(self):
        # Skip if app stack can't be loaded; rare in dev, common in stub CI.
        sys.path.insert(0, str(_BACKEND))
        spec = importlib.util.spec_from_file_location(
            "gtb_llm_personas", _BACKEND / "app/services/gtb_llm_personas.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        names = {p["persona"]["name"] for p in mod.BALANCED_LLM_LINEUP}
        assert "Dax" in names, "BALANCED_LLM_LINEUP must include the contrarian"
        # Every entry has the shape the service expects.
        for p in mod.BALANCED_LLM_LINEUP:
            assert p["policy"] == "llm_batched"
            assert p["count"] == 1
            assert "persona" in p
            assert {"name", "personality", "objective"} <= set(p["persona"])
