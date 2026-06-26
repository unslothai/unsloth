"""Regression checks for system prompt variable substitution."""

from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ADAPTER_SRC = (WORKSPACE / "studio/frontend/src/features/chat/api/chat-adapter.ts").read_text()


def _function_source(name: str) -> str:
    start = ADAPTER_SRC.index(f"function {name}")
    body_start = ADAPTER_SRC.index("{", start)
    depth = 0
    for index in range(body_start, len(ADAPTER_SRC)):
        if ADAPTER_SRC[index] == "{":
            depth += 1
        elif ADAPTER_SRC[index] == "}":
            depth -= 1
            if depth == 0:
                return ADAPTER_SRC[start : index + 1]
    raise AssertionError(f"Could not parse function body for {name}")


def test_prompt_variable_builtins_use_local_time_helpers():
    resolver = _function_source("resolveSystemPromptVariables")
    assert "formatLocalDate(now)" in resolver
    assert "formatLocalTime(now)" in resolver
    assert "formatTimezoneOffset(now)" in resolver
    assert "toISOString()" not in resolver


def test_prompt_variable_builtins_use_own_property_lookup():
    resolver = _function_source("resolveSystemPromptVariables")
    assert "if (hasOwn(systemVariables, key))" in resolver
    assert "key in systemVariables" not in resolver


def test_prompt_variable_nested_lookup_ignores_prototype_properties():
    nested_lookup = _function_source("getNestedValue")
    assert "if (!hasOwn(current, part))" in nested_lookup
