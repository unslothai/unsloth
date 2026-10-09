# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from mcp_server import INSTRUCTIONS, create_studio_mcp

READ_ONLY = {"studio_status", "list_models", "embed", "system_one", "get_job", "list_training_runs"}
DESTRUCTIVE = {"unload_model", "cancel"}
OTHERS = {
    "load_model",
    "chat",
    "generate_image",
    "generate_audio",
    "transcribe",
    "generate_video",
    "run_recipe",
    "datasets",
    "start_training",
    "export_model",
}


def _tools():
    return {tool.name: tool for tool in asyncio.run(create_studio_mcp().list_tools())}


def test_studio_mcp_registers_control_plane_tools():
    tools = _tools()

    assert set(tools) == {
        "studio_status",
        "list_models",
        "chat",
        "embed",
        "system_one",
        "generate_image",
        "generate_audio",
        "transcribe",
        "generate_video",
        "get_job",
        "run_recipe",
        "datasets",
        "load_model",
        "unload_model",
        "start_training",
        "cancel",
        "list_training_runs",
        "export_model",
    }
    assert len(tools) == 18


def test_registers_the_final_tools_with_their_annotations_and_output_schemas():
    tools = _tools()
    assert READ_ONLY | DESTRUCTIVE | OTHERS == set(tools)
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints is not None, name
        assert hints.openWorldHint is False, name
        assert hints.readOnlyHint is (name in READ_ONLY), name
        if name in DESTRUCTIVE:
            assert hints.destructiveHint is True, name
        if name in OTHERS:
            # MCP reads an unset hint as destructive, so tools that only add work say so.
            assert hints.destructiveHint is False, name
        assert tool.output_schema is not None, name
        assert tool.output_schema.get("type") == "object", name
        assert tool.description, name


def test_instructions_point_agents_at_the_workflow():
    server = create_studio_mcp()
    assert server.instructions == INSTRUCTIONS
    for phrase in (
        "studio_status",
        "list_models",
        "load_model",
        "get_job",
        "cancel",
        "Studio computer",
        "galleries",
    ):
        assert phrase in INSTRUCTIONS
