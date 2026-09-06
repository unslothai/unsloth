# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A call may ask `_truncate` for less room than the window earned, never more.

Filed against #10349: "Tool responses are truncated to arbitrary 16000 chars", asking
for a user- and/or AI-set parameter on the truncation threshold. `UNSLOTH_TOOL_RESULT_MAX_CHARS`
already covers the install-wide (user) side; this covers the per-call (AI) side -- a
`max_output_chars` argument on `python`/`terminal` that `_resolve_max_output_chars` turns
into a `_truncate` limit. It can only lower the cap `_tool_result_char_budget()` already
sized to the loaded window, never raise it: that ceiling is what stops a small resident
model's window from being handed a larger cap than it can afford, and a per-call override
that could exceed it would reopen that gap from inside a single tool call.
"""

from __future__ import annotations

import pytest

from core.inference import tools


@pytest.fixture(autouse = True)
def _unknown_window(monkeypatch):
    """Same default as test_tool_result_fits_window.py: no model loaded, no priced room."""
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: 0)
    tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    tools._REQUEST_RESULT_BUDGET.set(None)
    yield


class TestBothToolsAdvertiseTheArgument:
    def test_python_schema_has_an_optional_max_output_chars(self):
        props = tools.PYTHON_TOOL["function"]["parameters"]["properties"]
        assert "max_output_chars" in props
        assert "max_output_chars" not in tools.PYTHON_TOOL["function"]["parameters"]["required"]

    def test_terminal_schema_has_an_optional_max_output_chars(self):
        props = tools.TERMINAL_TOOL["function"]["parameters"]["properties"]
        assert "max_output_chars" in props
        assert "max_output_chars" not in tools.TERMINAL_TOOL["function"]["parameters"]["required"]


class TestResolveOnlyEverLowersTheCap:
    def test_none_changes_nothing(self):
        assert tools._resolve_max_output_chars(None) is None

    def test_a_request_under_the_ceiling_is_honoured(self):
        assert tools._resolve_max_output_chars(2000) == 2000

    def test_a_request_over_the_ceiling_is_clamped_to_it(self):
        ceiling = tools._tool_result_char_budget()
        assert tools._resolve_max_output_chars(ceiling * 100) == ceiling

    def test_a_request_below_the_floor_is_raised_to_it(self):
        assert tools._resolve_max_output_chars(1) == tools._MIN_RESULT_CHARS

    def test_zero_and_negative_requests_are_ignored(self):
        assert tools._resolve_max_output_chars(0) is None
        assert tools._resolve_max_output_chars(-100) is None

    def test_non_numeric_requests_are_ignored(self):
        assert tools._resolve_max_output_chars("not a number") is None
        assert tools._resolve_max_output_chars(object()) is None

    def test_a_numeric_string_is_accepted(self):
        assert tools._resolve_max_output_chars("2000") == 2000


class TestPythonAndTerminalHonourTheArgumentEndToEnd:
    def test_python_result_is_cut_to_the_requested_cap(self):
        out = tools._python_exec("print('x' * 5000)", max_output_chars = 800)
        assert "truncated to 800 chars for the model" in out
        # The notice explains the cut; the kept body itself is bounded by the cap.
        head = out.split("\n\n... (truncated")[0]
        assert len(head) <= 800

    def test_python_result_under_the_cap_is_untouched(self):
        out = tools._python_exec("print('hi')", max_output_chars = 800)
        assert out.strip() == "hi"
        assert "truncated" not in out

    def test_terminal_result_is_cut_to_the_requested_cap(self):
        out = tools._bash_exec("printf 'x%.0s' {1..5000}", max_output_chars = 800)
        assert "truncated to 800 chars for the model" in out

    def test_an_oversized_request_falls_back_to_the_default_budget(self):
        default_budget = tools._tool_result_char_budget()
        capped = tools._python_exec(
            "print('x' * (default_budget + 5000))".replace("default_budget", str(default_budget)),
            max_output_chars = default_budget * 1000,
        )
        uncapped = tools._python_exec(
            "print('x' * (default_budget + 5000))".replace("default_budget", str(default_budget))
        )
        assert len(capped) == len(uncapped)
