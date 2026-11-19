# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unreal MCP call_tool adapter, against error texts and schemas captured from Epic's Unreal MCP plugin (UE 5.8.4)."""

import json

import pytest

from integrations.unreal import mcp_adapter

URL = "http://127.0.0.1:8000/mcp"
APP = "EditorToolset.EditorAppToolset"
SCENE = "editor_toolset.toolsets.scene.SceneTools"

# Trimmed describe_toolset answers from a live editor: CaptureViewport requires nothing; find_actors requires
# name, tag and collision_channels.
_DESCRIBE = {
    APP: {
        "name": APP,
        "tools": [
            {
                "name": f"{APP}.CaptureViewport",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "captureTransform": {"type": "object"},
                        "annotations": {"type": "object"},
                        "bShowUI": {"type": "boolean", "default": False},
                    },
                },
            },
            {"name": f"{APP}.StopPIE", "inputSchema": {"type": "object"}},
        ],
    },
    SCENE: {
        "name": SCENE,
        "tools": [
            {
                "name": f"{SCENE}.find_actors",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "root": {"type": "object", "default": None},
                        "name": {"type": "string"},
                        "tag": {"type": "string"},
                        "collision_channels": {"type": "array"},
                    },
                    "required": ["name", "tag", "collision_channels"],
                },
            }
        ],
    },
}
_LIST = f"- {APP}: Tools for querying and modifying Unreal Editor state\n- {SCENE}: Provides tools for the loaded level\n"


@pytest.fixture(autouse = True)
def _fresh_caches():
    mcp_adapter._schemas.clear()
    mcp_adapter._toolsets.clear()


class _Server:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls, self.meta = [], []

    def invoke(self, arguments):
        self.calls.append(arguments)
        return self.answers.pop(0)

    def describe(self, name, arguments):
        self.meta.append((name, arguments))
        if name == "list_toolsets":
            return _LIST
        return json.dumps(_DESCRIBE.get(arguments["toolset_name"], {"tools": []}))


def test_dotted_tool_name_is_split_into_toolset_and_tool():
    # Logged: {"tool_name": "editor_toolset.toolsets.asset.AssetTools.list_folders", "arguments": {...}} ->
    # "Error: Tool '...list_folders' not found".
    sent = {
        "tool_name": "editor_toolset.toolsets.asset.AssetTools.list_folders",
        "arguments": {"root_path": "/Game"},
    }
    assert mcp_adapter.prepare(sent) == {
        "tool_name": "list_folders",
        "toolset_name": "editor_toolset.toolsets.asset.AssetTools",
        "arguments": {"root_path": "/Game"},
    }
    with_toolset = {"toolset_name": APP, "tool_name": f"{APP}.StopPIE", "arguments": {}}
    assert mcp_adapter.prepare(with_toolset)["tool_name"] == "StopPIE"
    plain = {"toolset_name": APP, "tool_name": "StopPIE", "arguments": {}}
    assert mcp_adapter.prepare(plain) is plain


def test_omitted_optional_parameters_are_filled_and_retried_once():
    server = _Server(
        [
            'Error: Function "CaptureViewport", input param "captureTransform" needs a default value.'
            "Function input params Json -\n{}",
            '{"returnValue": {"image": {"mimeType": "image/png", "data": "iVBOR"}}}',
        ]
    )
    result = mcp_adapter.call(
        URL,
        {"toolset_name": APP, "tool_name": "CaptureViewport", "arguments": {}},
        server.invoke,
        server.describe,
    )
    assert result.startswith('{"returnValue"')
    assert server.calls[1]["arguments"] == {
        "captureTransform": None,
        "annotations": None,
        "bShowUI": False,
    }
    assert server.meta == [("describe_toolset", {"toolset_name": APP})]


def test_a_truly_required_parameter_is_left_to_the_model():
    error = (
        'Error: Function "find_actors", input param "collision_channels" is required by the function input '
        "schema Json, but is missing from the incoming function input params Json."
    )
    server = _Server([error])
    sent = {"toolset_name": SCENE, "tool_name": "find_actors", "arguments": {"name": "", "tag": ""}}
    assert mcp_adapter.call(URL, sent, server.invoke, server.describe) == error
    assert len(server.calls) == 1  # no retry that could only fail the same way


def test_a_retry_that_fails_again_is_not_repeated():
    error = 'Error: Function "CaptureViewport", input param "annotations" needs a default value.'
    server = _Server([error, "Error: Viewport not available"])
    result = mcp_adapter.call(
        URL,
        {"toolset_name": APP, "tool_name": "CaptureViewport", "arguments": {}},
        server.invoke,
        server.describe,
    )
    assert result == "Error: Viewport not available" and len(server.calls) == 2


def test_unknown_tool_lists_the_real_tools_of_that_toolset():
    server = _Server(["Error: Unknown tool create_viewport_capture"])
    result = mcp_adapter.call(
        URL,
        {"toolset_name": APP, "tool_name": "create_viewport_capture", "arguments": {}},
        server.invoke,
        server.describe,
    )
    assert (
        result
        == "Error: Unknown tool create_viewport_capture\nTools in "
        + APP
        + ": CaptureViewport, StopPIE"
    )


def test_unknown_toolset_lists_the_toolsets():
    server = _Server(["Error: Toolset 'AgentSkillToolset.AgentSkillToolset' not found"])
    result = mcp_adapter.call(
        URL,
        {
            "toolset_name": "AgentSkillToolset.AgentSkillToolset",
            "tool_name": "list_skills",
            "arguments": {},
        },
        server.invoke,
        server.describe,
    )
    assert result.endswith(f"\nToolsets on this server: {APP}, {SCENE}")


def test_schemas_are_cached_per_server_and_toolset():
    error = 'Error: Function "CaptureViewport", input param "annotations" needs a default value.'
    server = _Server([error, "ok", error, "ok"])
    sent = {"toolset_name": APP, "tool_name": "CaptureViewport", "arguments": {}}
    mcp_adapter.call(URL, sent, server.invoke, server.describe)
    mcp_adapter.call(URL, sent, server.invoke, server.describe)
    assert len(server.meta) == 1


def test_successful_calls_and_unparseable_describes_pass_through():
    server = _Server(['{"returnValue": true}'])
    sent = {"toolset_name": APP, "tool_name": "StopPIE", "arguments": {}}
    assert mcp_adapter.call(URL, sent, server.invoke, server.describe) == '{"returnValue": true}'
    assert server.meta == []
    broken = _Server(['Error: Function "X", input param "y" needs a default value.'])
    broken.describe = lambda name, arguments: "Error: MCP tool 'describe_toolset' failed"
    out = mcp_adapter.call(
        URL,
        {"toolset_name": "T", "tool_name": "X", "arguments": {}},
        broken.invoke,
        broken.describe,
    )
    assert out.startswith("Error: Function") and len(broken.calls) == 1


def test_only_the_unreal_builtins_call_tool_goes_through_the_adapter(monkeypatch):
    from unittest.mock import Mock

    from core.inference import tools
    from storage import mcp_servers_db as db

    db.create_server(
        "ue0123456789abcd", "Unreal Engine", "http://127.0.0.1:8000/mcp", builtin_id = "unreal"
    )
    db.create_server("fedcba9876543210", "Other", "http://127.0.0.1:9999/mcp")
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    call = Mock(return_value = "ok")
    monkeypatch.setattr(tools, "call_tool_sync", call)
    dotted = {"tool_name": f"{APP}.StopPIE", "arguments": {}}
    assert tools.execute_tool("mcp__unreal__call_tool", dotted) == "ok"
    assert call.call_args.kwargs["name"] == "call_tool"
    assert call.call_args.kwargs["args"] == {
        "tool_name": "StopPIE",
        "toolset_name": APP,
        "arguments": {},
    }
    tools.execute_tool(
        "mcp__fedcba9876543210__call_tool", dotted
    )  # another server's call_tool is not touched
    assert call.call_args.kwargs["args"] == dotted
    tools.execute_tool("mcp__unreal__list_toolsets", {})
    assert call.call_args.kwargs == {**call.call_args.kwargs, "name": "list_toolsets", "args": {}}
