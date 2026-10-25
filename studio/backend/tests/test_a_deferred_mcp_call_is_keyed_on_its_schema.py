# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With `mcp_tool_search` offered, a cached MCP tool is loaded only when the model calls it.
`prepare_call` loads its schema before healing, so a bare `"alpha"` and `{"query": "alpha"}` heal
to one call; a round's parallel gate keyed them without that schema, saw two distinct calls and
ran both. The key now heals against the cached schema, without loading it into the catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference import tools as tools_mod  # noqa: E402
from core.inference.mcp_client import MCP_TOOL_PREFIX  # noqa: E402
from core.inference.tool_loop_controller import (  # noqa: E402
    MCP_TOOL_SEARCH_NAME,
    ToolLoopController,
)
from test_studio_tool_loop import WEB, _tool  # noqa: E402

_LOOKUP = f"{MCP_TOOL_PREFIX}docs__lookup"
_BARE = {"function": {"name": _LOOKUP, "arguments": "alpha"}}
_OBJECT = {"function": {"name": _LOOKUP, "arguments": json.dumps({"query": "alpha"})}}
_OTHER = {"function": {"name": _LOOKUP, "arguments": json.dumps({"query": "beta"})}}


@pytest.fixture(autouse = True)
def cached_lookup(monkeypatch):
    """A cached MCP tool with one required string argument, not yet in this response's catalog."""
    spec = _tool(_LOOKUP)
    monkeypatch.setattr(tools_mod, "mcp_tool_spec", lambda name: spec if name == _LOOKUP else None)


def _controller():
    return ToolLoopController(tools = [WEB, _tool(MCP_TOOL_SEARCH_NAME)])


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


def test_a_bare_and_an_object_call_share_the_key_they_run_under():
    controller = _controller()
    before = _names(controller.active_tools())
    bare_key = controller.call_key(_BARE)
    assert bare_key == controller.call_key(_OBJECT), "two calls that heal to one were keyed apart"
    assert bare_key == _controller().prepare_call(_BARE).key
    assert bare_key == _controller().prepare_call(_OBJECT).key
    assert _names(controller.active_tools()) == before, "keying loaded the tool into the catalog"
    assert controller.healed_arguments(_BARE) == (_LOOKUP, {"query": "alpha"})


def test_distinct_calls_keep_distinct_keys():
    controller = _controller()
    assert controller.call_key(_OBJECT) != controller.call_key(_OTHER)
