# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import io
import json
import random

import pytest
from fastapi import HTTPException
from PIL import Image

from core.inference import mcp_client
from core.inference import tools as tools_mod
from core.inference.mcp_image import (
    ATTACHED_IMAGE,
    McpImage,
    WITHHELD_RESULT,
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


def _noise_png():
    out = io.BytesIO()
    Image.frombytes("RGB", (64, 64), random.Random(0).randbytes(64 * 64 * 3)).save(
        out, format = "PNG"
    )
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

    share = tools_mod.mcp_image_share("mcp__srv1__lookup", args, image)
    approved = share["image"]
    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert mapped_server[0]["args"]["image"] == image.encoded("data_url")
    assert args == {"image": ATTACHED_IMAGE}
    # The server echoed its input; the bytes must not reach the model.
    assert out == "match for [attached image]"

    # Repointing the server after approval must not redirect the image.
    mcp_servers_db.update_server("srv1", {"url": "https://elsewhere.example/mcp"})
    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert out.startswith("Error: the MCP server changed") and len(mapped_server) == 1


def test_an_edit_while_the_card_is_open_does_not_redirect_the_image(mapped_server, monkeypatch):
    image = McpImage(mime = "image/png", data = _png_bytes())
    args = {"image": ATTACHED_IMAGE}
    approved = tools_mod.mcp_image_share("mcp__srv1__lookup", args, image)["image"]
    stale = mcp_servers_db.get_server("srv1")
    mcp_servers_db.update_server("srv1", {"headers_json": json.dumps({"X-Key": "other"})})
    # execute_tool resolved the row before the edit landed.
    monkeypatch.setattr(mcp_servers_db, "get_server_for_tool", lambda _key: stale)
    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert out.startswith("Error: the MCP server changed") and mapped_server == []


def test_images_returned_by_an_image_call_never_reach_the_model(mapped_server, monkeypatch):
    image = McpImage(mime = "image/png", data = _png_bytes())
    args = {"image": ATTACHED_IMAGE}
    approved = tools_mod.mcp_image_share("mcp__srv1__lookup", args, image)["image"]
    resized = base64.b64encode(_png_bytes("JPEG")).decode()
    envelope = mcp_client.MCP_IMAGES_SENTINEL + json.dumps(
        [{"data": resized, "mimeType": "image/jpeg"}]
    )
    monkeypatch.setattr(tools_mod, "call_tool_sync", lambda **_: "1 image returned\n" + envelope)
    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert mcp_client.MCP_IMAGES_SENTINEL not in out and resized not in out
    assert out.endswith("[Images the tool returned were withheld from the model.]")


def test_a_mapping_revoked_before_dispatch_blocks_the_send(mapped_server, monkeypatch):
    image = McpImage(mime = "image/png", data = _png_bytes())
    args = {"image": ATTACHED_IMAGE}
    approved = tools_mod.mcp_image_share("mcp__srv1__lookup", args, image)["image"]
    checks = []

    def fake_call(**kwargs):
        # The mapping is removed while the call waits for its session, before config_check runs.
        mcp_servers_db.update_server("srv1", {"image_input_mappings_json": "[]"})
        checks.append(kwargs["config_check"]())
        return "sent"

    monkeypatch.setattr(tools_mod, "call_tool_sync", fake_call)
    tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert checks == [False]


def test_an_approved_image_whose_mapping_vanished_is_not_forwarded(mapped_server):
    image = McpImage(mime = "image/png", data = _png_bytes())
    args = {"image": ATTACHED_IMAGE}
    approved = tools_mod.mcp_image_share("mcp__srv1__lookup", args, image)["image"]
    # A header edit drops the tool cache, so the field can no longer be resolved.
    mcp_client._tool_cache.clear()
    out = tools_mod.execute_tool("mcp__srv1__lookup", args, mcp_image = approved)
    assert out.startswith("Error: the MCP server changed") and mapped_server == []


@pytest.mark.parametrize(
    "encode",
    [
        lambda d: base64.urlsafe_b64encode(d).decode(),
        lambda d: d.hex(),
        lambda d: "\\n".join(base64.b64encode(d).decode()[i : i + 8] for i in range(0, 200, 8)),
        lambda d: base64.b64encode(d).decode().replace("/", "\\/"),
        lambda d: base64.b64encode(d[1:]).decode(),
        lambda d: base64.b64encode(d[len(d) // 2 :]).decode(),
        lambda d: base64.b64encode(d[:60]).decode(),
    ],
    ids = ["urlsafe", "hex", "wrapped", "json-escaped", "offset", "tail", "short"],
)
def test_a_reencoded_echo_withholds_the_result(encode):
    image = McpImage(mime = "image/png", data = _noise_png())
    assert image.redact(f"result: {encode(image.data)}") == WITHHELD_RESULT
    assert image.redact("result: Cowboy Bebop ep 5") == "result: Cowboy Bebop ep 5"


def test_durable_runs_refuse_the_image():
    from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request
    with pytest.raises(HTTPException) as exc:
        _sanitize_request(
            CreateChatGenerationRun.model_construct(
                requestPayload = {
                    "messages": [{"role": "user", "content": "hi"}],
                    "mcp_image": _data_url(_png_bytes()),
                }
            )
        )
    assert exc.value.detail == "Media chat runs use the legacy streaming path"


def test_unmapped_and_literal_arguments_take_the_ordinary_path(mapped_server):
    image = McpImage(mime = "image/png", data = _png_bytes())
    tools_mod.execute_tool("mcp__srv1__lookup", {"image": "https://x/y.png"})
    assert mapped_server[-1]["args"] == {"image": "https://x/y.png"}
    assert (
        tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": "https://x/y.png"}, image) is None
    )
    assert tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": ATTACHED_IMAGE}, None) is None
    share = tools_mod.mcp_image_share("mcp__srv1__lookup", {"image": ATTACHED_IMAGE}, image)
    assert share["disclosure"] == {
        "server": "Trace",
        "tool": "lookup",
        "size_bytes": len(image.data),
        "destination": "trace.example",
    }
    assert share["image"].data == image.data and share["image"].recipient
    # Credentials in URL userinfo or stdio arguments never reach the card.
    assert tools_mod._mcp_image_destination("https://u:pw@trace.example/mcp") == "trace.example"
    assert (
        tools_mod._mcp_image_destination("npx -y trace-mcp --token s3cret") == "local command npx"
    )


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
    assert [getattr(s, "data", None) for s in seen] == ([image.data] if decision == "allow" else [])
    # The fingerprint covers the server's headers, so it is never streamed.
    assert "recipient" not in starts[0]["image_disclosure"]
    assert all(s.recipient for s in seen)


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
    listed = lambda: routes_mcp._row_to_response(mcp_servers_db.get_server("srv1"))  # noqa: E731
    assert listed().image_mappings_active is True
    # The server dropped the string field: images must not be made tool-only for it.
    stale = {
        **LOOKUP,
        "inputSchema": {"type": "object", "properties": {"image": {"type": "boolean"}}},
    }
    mcp_client.cache_tools("srv1", [stale])
    assert listed().image_mappings_active is False
    mcp_client.cache_tools("srv1", [LOOKUP])
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
    assert "kwargs['mcp_image'] = image_share['image']" in statements
