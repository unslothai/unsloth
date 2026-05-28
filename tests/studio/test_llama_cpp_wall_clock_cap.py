"""
Tests for Studio's local llama-server timeout policy.

Studio should not cut off a response that is still producing tokens after
10 minutes. The user-visible Stop control and max_tokens cap are the
front-line limits. The only fixed watchdog here is the first-token/prefill
timeout, which catches requests that never start streaming.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "inference"
    / "llama_cpp.py"
)
SRC = SOURCE_PATH.read_text()
TREE = ast.parse(SRC)


def _is_subscript_assign(stmt: ast.stmt, target_name: str, key: str) -> bool:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    t = stmt.targets[0]
    if not isinstance(t, ast.Subscript):
        return False
    if not (isinstance(t.value, ast.Name) and t.value.id == target_name):
        return False
    slc = t.slice
    return isinstance(slc, ast.Constant) and slc.value == key


def _collect_assignments(tree, target_name, key):
    hits = []
    for node in ast.walk(tree):
        if _is_subscript_assign(node, target_name, key):
            hits.append(node)
    return hits


def _module_constant(name: str):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == name:
                value = node.value
                assert isinstance(value, ast.Constant)
                return value.value
    raise AssertionError(f"{name} constant missing")


def test_first_token_timeout_is_at_least_ten_minutes():
    value = _module_constant("_DEFAULT_FIRST_TOKEN_TIMEOUT_S")
    assert value >= 600.0


def test_studio_chat_payloads_do_not_set_wall_clock_generation_cap():
    hits_payload = _collect_assignments(TREE, "payload", "t_max_predict_ms")
    hits_stream = _collect_assignments(TREE, "stream_payload", "t_max_predict_ms")
    assert hits_payload + hits_stream == []


def test_max_tokens_default_cap_still_applied():
    # The wall-clock cap is intentionally absent, but max_tokens must
    # still kick in when callers omit it.
    matches = 0
    for node in ast.walk(TREE):
        if not isinstance(node, ast.IfExp):
            continue
        src = ast.unparse(node)
        if "max_tokens" in src and "_DEFAULT_MAX_TOKENS_FLOOR" in src:
            matches += 1
    assert matches >= 3
