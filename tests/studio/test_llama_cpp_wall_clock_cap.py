"""Timeout policy checks for Unsloth's local llama-server path."""

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


def _module_constant(name: str):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == name:
                value = node.value
                assert isinstance(value, ast.Constant)
                return value.value
    raise AssertionError(f"{name} constant missing")


def test_first_token_timeout_is_at_least_twenty_minutes():
    value = _module_constant("_DEFAULT_FIRST_TOKEN_TIMEOUT_S")
    assert value >= 1200.0


def test_studio_chat_payloads_do_not_set_wall_clock_generation_cap():
    assert "t_max_predict_ms" not in SRC


def test_max_tokens_default_cap_still_applied():
    assert SRC.count("_DEFAULT_MAX_TOKENS_FLOOR") >= 3
