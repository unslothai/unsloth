# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import io
import json

import pytest
from fastapi import HTTPException
from PIL import Image

from core.inference import mcp_client
from core.inference import tools as tools_mod
from core.inference.mcp_image import (
    ATTACHED_IMAGE,
    McpImage,
    McpImageError,
    parse_mcp_image,
    public_tool,
)
from storage import mcp_servers_db

LOOKUP = {
    "name": "lookup",
    "description": "Find an anime scene.",
    "inputSchema": {
        "type": "object",
        "properties": {"image": {"type": "string"}, "cut_borders": {"type": "boolean"}},
        "required": ["image"],
    },
}


def _png_bytes(fmt = "PNG"):
    out = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(out, format = fmt)
    return out.getvalue()


def _data_url(data, mime = "image/png"):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


@pytest.fixture
def mapped_server(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(tools_mod, "_MCP_COMPACTED_WINDOWS", {})
    mcp_servers_db.create_server(
        id = "srv1",
        display_name = "Trace",
        url = "https://trace.example/mcp",
        image_input_mappings_json = json.dumps(
            [{"tool": "lookup", "field": "image", "encoding": "data_url"}]
        ),
    )
    mcp_client.cache_tools("srv1", [LOOKUP])
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return f"match for {kwargs['args']['image']}"

    monkeypatch.setattr(tools_mod, "call_tool_sync", fake_call)
    return calls


def test_parse_accepts_a_matching_image_and_rejects_the_rest():
    image = parse_mcp_image(_data_url(_png_bytes()))
    assert image.mime == "image/png" and image.data == _png_bytes()
    multi = io.BytesIO()
    Image.new("RGB", (4, 3)).save(
        multi, format = "MPO", save_all = True, append_images = [Image.new("RGB", (4, 3))]
    )
    assert parse_mcp_image(_data_url(multi.getvalue(), "image/jpeg")).mime == "image/jpeg"
    assert "data" not in repr(image)
    for bad in (
        "not a data url",
        _data_url(_png_bytes(), "image/gif"),
        _data_url(_png_bytes(), "image/jpeg"),
        "data:image/png;base64,@@@",
        _data_url(b"x" * (10 * 1024 * 1024 + 1)),
    ):
        with pytest.raises(McpImageError):
            parse_mcp_image(bad)


def test_public_tool_hides_only_a_mapped_string_field():
    server = {"image_input_mappings_json": json.dumps([{"tool": "lookup", "field": "image"}])}
    public = public_tool(server, LOOKUP)
    assert public["inputSchema"]["properties"]["image"]["enum"] == [ATTACHED_IMAGE]
    assert public["inputSchema"]["properties"]["cut_borders"] == {"type": "boolean"}
    assert LOOKUP["inputSchema"]["properties"]["image"] == {"type": "string"}
    # A field that stopped being a string (or vanished) after a server update leaves the tool as is.
    renamed = {
        "image_input_mappings_json": json.dumps([{"tool": "lookup", "field": "cut_borders"}])
    }
    assert public_tool(renamed, LOOKUP) is LOOKUP
    assert public_tool({}, LOOKUP) is LOOKUP


def test_listing_shows_the_model_the_placeholder(mapped_server):
    specs = tools_mod.cached_mcp_tools()[0]
    params = next(s for s in specs if s["function"]["name"] == "mcp__srv1__lookup")["function"]
    assert params["parameters"]["properties"]["image"]["enum"] == [ATTACHED_IMAGE]


def test_execute_tool_inserts_the_image_only_when_given_one(mapped_server):
    image = McpImage(mime = "image/png", data = _png_bytes())
    args = {"image": ATTACHED_IMAGE}

    out = tools_mod.execute_tool("mcp__srv1__lookup", args)
    assert out.startswith("Error: no approved image") and mapped_server == []

    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = image)
    assert mapped_server[0]["args"]["image"] == image.encoded("data_url")
    assert args == {"image": ATTACHED_IMAGE}
    # The server echoed its input; the bytes must not reach the model.
    assert out == "match for [attached image]"


def test_unmapped_and_literal_arguments_take_the_ordinary_path(mapped_server):
    image = McpImage(mime = "image/png", data = _png_bytes())
    tools_mod.execute_tool("mcp__srv1__lookup", {"image": "https://x/y.png"}, mcp_image = image)
    assert mapped_server[-1]["args"] == {"image": "https://x/y.png"}
    assert (
        tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": "https://x/y.png"}, image) is None
    )
    assert tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": ATTACHED_IMAGE}, None) is None
    assert tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": ATTACHED_IMAGE}, image) == {
        "server": "Trace",
        "tool": "lookup",
        "size_bytes": len(image.data),
    }


def _one_call_turns():
    turns = iter(
        [
            '<tool_call>{"name": "mcp__srv1__lookup", "arguments": {"image": "attached_image"}}</tool_call>',
            "done",
        ]
    )

    def single_turn(_messages):
        try:
            yield next(turns)
        except StopIteration:
            return

    return single_turn


@pytest.mark.parametrize("decision", ["allow", "deny"])
def test_safetensors_loop_always_asks_before_sending_the_image(mapped_server, decision):
    from core.inference.safetensors_agentic import run_safetensors_tool_loop
    from state.tool_approvals import resolve_tool_decision

    image = McpImage(mime = "image/png", data = _png_bytes())
    seen = []

    def fake_exec(name, arguments, **kwargs):
        seen.append(kwargs.get("mcp_image"))
        return "ok"

    starts = []
    for event in run_safetensors_tool_loop(
        single_turn = _one_call_turns(),
        messages = [{"role": "user", "content": "what anime is this?"}],
        tools = [{"type": "function", "function": {"name": "mcp__srv1__lookup"}}],
        execute_tool = fake_exec,
        session_id = "s",
        confirm_tool_calls = False,
        bypass_permissions = True,
        mcp_image = image,
    ):
        if event["type"] == "tool_start":
            starts.append(event)
            assert resolve_tool_decision(event["approval_id"], decision, session_id = "s")
    assert starts[0]["awaiting_confirmation"] is True
    assert starts[0]["image_disclosure"]["server"] == "Trace"
    assert seen == ([image] if decision == "allow" else [])


def test_route_requires_an_interactive_stream_for_the_image():
    from routes.inference import _request_mcp_image

    class Payload:
        mcp_image = _data_url(_png_bytes())
        stream = True

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_request_mcp_image(Payload, ui_events = False))
    assert exc.value.status_code == 400
    assert asyncio.run(_request_mcp_image(Payload, ui_events = True)).mime == "image/png"
    Payload.mcp_image = _data_url(b"not an image")
    with pytest.raises(HTTPException):
        asyncio.run(_request_mcp_image(Payload, ui_events = True))


def test_mappings_round_trip_through_the_routes(mapped_server):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes_mcp

    assert routes_mcp.list_mcp_server_tools("srv1", current_subject = "u")[0]["name"] == "lookup"
    updated = asyncio.run(
        routes_mcp.update_mcp_server(
            "srv1",
            McpServerUpdate(image_input_mappings = [{"tool": "lookup", "field": "image"}]),
            current_subject = "u",
        )
    )
    assert [m.model_dump() for m in updated.image_input_mappings] == [
        {"tool": "lookup", "field": "image", "encoding": "base64"}
    ]
    cleared = asyncio.run(
        routes_mcp.update_mcp_server(
            "srv1", McpServerUpdate(image_input_mappings = []), current_subject = "u"
        )
    )
    assert cleared.image_input_mappings == []


def test_gguf_loop_gates_and_forwards_the_image_like_the_other_loops():
    # The GGUF loop needs a live llama-server (see test_bypass_permissions), so check its source.
    import ast
    import inspect
    import textwrap

    llama_cpp = pytest.importorskip("core.inference.llama_cpp")
    src = textwrap.dedent(
        inspect.getsource(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    )
    statements = {
        ast.unparse(node) for node in ast.walk(ast.parse(src)) if isinstance(node, ast.stmt)
    }
    assert "needs_confirm = needs_confirm or image_share is not None" in statements
    assert "kwargs['mcp_image'] = mcp_image" in statements
