# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from fastmcp.exceptions import ToolError

import storage.studio_db as studio_db
from mcp_server import INSTRUCTIONS, TOOL_MODULES, create_studio_mcp, tool_catalog, tool_names
from utils import mcp_access
from utils.mcp_access import MCP_DISABLED_TOOLS_SETTING_KEY, set_mcp_settings

READ_ONLY = {
    "embed",
    "get_job",
    "list_models",
    "list_media",
    "list_training_runs",
    "plan_load",
    "studio_status",
    "system_one",
}
DESTRUCTIVE = {
    "cancel",
    "unload_model",
    "voices",
}
OTHERS = {
    "analyze_audio",
    "chat",
    "datasets",
    "download_model",
    "export_model",
    "generate_audio",
    "generate_image",
    "generate_video",
    "load_model",
    "run_recipe",
    "start_training",
    "transcribe",
}


def _tools():
    return {tool.name: tool for tool in asyncio.run(create_studio_mcp().list_tools())}


def test_studio_mcp_registers_control_plane_tools():
    tools = _tools()
    assert READ_ONLY | DESTRUCTIVE | OTHERS == set(tools)
    assert len(tools) == len(READ_ONLY | DESTRUCTIVE | OTHERS)
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints is not None, name
        assert hints.openWorldHint is False, name
        assert hints.readOnlyHint is (name in READ_ONLY), name
        # MCP reads an unset hint as destructive, so tools that only add work say so.
        if name not in READ_ONLY:
            assert hints.destructiveHint is (name in DESTRUCTIVE), name
        assert tool.output_schema["type"] == "object", name
        assert tool.description, name


# (tool, path into the listed tool, the value or a check on it).
SCHEMA_FACTS = [
    ("embed", "parameters/properties/texts/minItems", 1),
    ("embed", "parameters/properties/texts/maxItems", 2048),
    ("system_one", "parameters/properties/model/default", "default"),
    ("system_one", "parameters/properties/images/anyOf/0/maxItems", 4),
    ("generate_image", "output_schema/properties/images/type", "array"),
    ("generate_image", "output_schema/additionalProperties", False),
    ("generate_audio", "output_schema/additionalProperties", False),
    ("generate_audio", "parameters/properties", lambda props: {"text", "model"} <= set(props)),
    ("studio_status", "output_schema/additionalProperties", False),
    ("studio_status", "parameters", lambda params: params.get("properties", {}) == {}),
    (
        "list_models",
        "parameters/properties",
        lambda props: {"kind", "loaded_only", "model", "workflow"} <= set(props),
    ),
    (
        "load_model",
        "parameters/properties",
        lambda props: {"model", "kind", "variant", "max_seq_length", "load_in_4bit", "hf_token"}
        <= set(props),
    ),
    ("load_model", "parameters/properties/kind/default", "llm"),
    (
        "cancel",
        "parameters/properties/kind/enum",
        lambda kinds: {
            "training",
            "training_start",
            "diffusion_training",
            "export",
            "recipe",
            "image",
            "video",
            "chat",
            "dataset_download",
        }
        <= set(kinds),
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
    assert create_studio_mcp().instructions == INSTRUCTIONS
    for phrase in ("studio_status", "list_models", "load_model", "get_job", "cancel"):
        assert phrase in INSTRUCTIONS
    assert "Studio computer" in INSTRUCTIONS and "galleries" in INSTRUCTIONS
    for keyword in (
        "chat",
        "images",
        "audio",
        "speak",
        "clone",
        "edit",
        "convert",
        "music",
        "sfx",
        "separate",
        "transcribe",
        "video",
        "datasets",
        "fine-tune",
        "export",
    ):
        assert keyword in INSTRUCTIONS, keyword
    assert "load a model, generate, then unload_model" in INSTRUCTIONS
    assert "speech-to-text models stay loaded" in INSTRUCTIONS
    assert len(INSTRUCTIONS) <= 1900
    for absent in ("wait_s", "alongside", "signed"):
        assert absent not in INSTRUCTIONS, absent


@pytest.fixture
def fresh_access():
    mcp_access._reset_cache()
    yield
    mcp_access._reset_cache()


def test_the_catalog_is_the_registered_tools_in_order(fresh_access):
    registered = [
        tool.name for tool in asyncio.run(create_studio_mcp().list_tools(run_middleware = False))
    ]
    assert [entry["name"] for entry in tool_catalog()] == registered
    assert list(tool_names()) == registered
    assert len(registered) == 23
    areas = {module.__name__.rsplit(".", 1)[-1] for module in TOOL_MODULES}
    assert {entry["area"] for entry in tool_catalog()} == areas
    assert tool_catalog()[0] == {"name": "studio_status", "area": "status"}
    assert tool_catalog()[-1] == {"name": "cancel", "area": "cancel"}


def test_by_default_the_middleware_lists_every_tool(fresh_access):
    mcp = create_studio_mcp()
    wrapped = asyncio.run(mcp.list_tools())
    bare = asyncio.run(mcp.list_tools(run_middleware = False))
    assert [tool.name for tool in wrapped] == [tool.name for tool in bare]


def test_a_turned_off_tool_is_hidden_and_refused_in_process(fresh_access):
    set_mcp_settings(disabled_tools = ["cancel", "chat"])
    mcp = create_studio_mcp()
    listed = [tool.name for tool in asyncio.run(mcp.list_tools())]
    assert listed == [name for name in tool_names() if name not in {"cancel", "chat"}]
    with pytest.raises(
        ToolError, match = r"^\[tool_disabled\] chat is turned off in Unsloth Studio\."
    ):
        asyncio.run(mcp.call_tool("chat", {"prompt": "hi"}))


@pytest.mark.parametrize("stored", ["x", {}, None, 3])
def test_an_unreadable_tool_list_turns_every_tool_off(fresh_access, stored):
    studio_db.upsert_app_settings({MCP_DISABLED_TOOLS_SETTING_KEY: stored})
    mcp_access._reset_cache()
    mcp = create_studio_mcp()
    assert asyncio.run(mcp.list_tools()) == []
    with pytest.raises(ToolError, match = r"^\[tool_disabled\] studio_status "):
        asyncio.run(mcp.call_tool("studio_status", {}))
