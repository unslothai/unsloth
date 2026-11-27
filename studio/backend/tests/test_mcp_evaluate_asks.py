# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MCP tool named evaluate runs the expression it is given (REPLs, debuggers, browsers), like eval."""

import pytest

from core.inference.tools import MCP_TOOL_PREFIX, is_high_risk_tool_call


@pytest.mark.parametrize(
    "tool",
    [
        "evaluate",
        "evaluate_expression",
        "debug_evaluate",
        "browser_evaluate",
        "evaluateJS",
        "eval",
    ],
)
def test_an_mcp_evaluate_tool_asks_in_auto_mode(tool):
    name = f"{MCP_TOOL_PREFIX}server__{tool}"
    assert is_high_risk_tool_call(name, {"expression": "1 + 1"})


@pytest.mark.parametrize("tool", ["get_issue", "list_files", "search_docs"])
def test_ordinary_mcp_reads_still_run(tool):
    assert not is_high_risk_tool_call(f"{MCP_TOOL_PREFIX}server__{tool}", {"q": "x"})
