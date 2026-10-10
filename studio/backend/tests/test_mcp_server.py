# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

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
    assert READ_ONLY | DESTRUCTIVE | OTHERS == set(tools)
    assert len(tools) == 18
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


# Schema facts beyond the annotations: (tool, path into the listed tool, the value or a check on it).
SCHEMA_FACTS = [
    ("embed", "parameters/properties/texts/minItems", 1),
    ("embed", "parameters/properties/texts/maxItems", 2048),
    ("system_one", "parameters/properties/model/default", "default"),
    ("system_one", "parameters/properties/images/anyOf/0/maxItems", 4),
    ("generate_image", "output_schema/properties/images/type", "array"),
    ("generate_image", "output_schema/additionalProperties", False),
    ("generate_audio", "output_schema/additionalProperties", False),
    ("generate_audio", "parameters/properties", lambda props: "text" in props),
    ("studio_status", "output_schema/additionalProperties", False),
    ("studio_status", "parameters", lambda params: params.get("properties", {}) == {}),
    (
        "list_models",
        "parameters/properties",
        lambda props: set(props) == {"kind", "loaded_only", "model"},
    ),
    (
        "load_model",
        "parameters/properties",
        lambda props: set(props)
        == {"model", "kind", "variant", "max_seq_length", "load_in_4bit", "hf_token"},
    ),
    ("load_model", "parameters/properties/kind/default", "llm"),
    (
        "cancel",
        "parameters/properties/kind/enum",
        lambda kinds: "audio" not in kinds and len(kinds) == 9,
    ),
    ("get_job", "parameters/properties/kind/enum", lambda kinds: {"video", "recipe"} <= set(kinds)),
]


@pytest.mark.parametrize("name,path,expected", SCHEMA_FACTS)
def test_tool_schemas_keep_their_limits_and_defaults(name, path, expected):
    attribute, *keys = path.split("/")
    value = getattr(_tools()[name], attribute)
    for key in keys:
        value = value[int(key)] if isinstance(value, list) else value[key]
    if callable(expected):
        assert expected(value), (name, path, value)
    else:
        assert value == expected and type(value) is type(expected), (name, path, value)


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
