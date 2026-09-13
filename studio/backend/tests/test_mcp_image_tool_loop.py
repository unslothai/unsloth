# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import base64
import copy
import io
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from core.inference import mcp_client, mcp_image_tool_loop as image_loop
from core.inference.mcp_image_disclosure import McpImageDisclosureError, validate_image_input_mappings
from state import tool_approvals
from storage import studio_db


@pytest.fixture
def image_request(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    output = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(output, format = "PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    url = "data:image/png;base64," + encoded
    studio_db.upsert_chat_thread({"id": "thread", "title": "test", "createdAt": 1, "modelType": "base", "modelId": "mock"})
    studio_db.upsert_chat_message({
        "id": "message", "threadId": "thread", "role": "user", "createdAt": 2,
        "content": [{"type": "text", "text": "Describe this image"}],
        "attachments": [{"id": "image", "type": "image", "mcpToolOnly": True,
                         "content": [{"type": "image", "image": url}]}],
    })
    feature = [True, "revision-1"]
    monkeypatch.setattr(studio_db, "get_chat_setting_with_revision", lambda key: tuple(feature))
    tools = []
    rows = {}
    for raw, field, encoding in (("inspect", "picture_blob", "base64"), ("classify", "frame_data", "data_url")):
        schema = {"type": "object", "properties": {field: {"type": "string"}, "threshold": {"type": "number"}}, "required": [field]}
        mapping = {"tool": raw, "field": field, "encoding": encoding}
        _, digest = validate_image_input_mappings([mapping], [{"name": raw, "inputSchema": schema}])
        row = {"id": raw, "is_enabled": True, "url": "https://example.test/mcp", "display_name": raw,
               "config_revision": 1, "image_input_schema_digest": digest}
        name = f"mcp__{raw}__{raw}"
        rows[name] = (row, mapping, schema, digest)
        tools.append({"type": "function", "function": {"name": name, "parameters": schema}})
    monkeypatch.setattr(image_loop, "_mapping_for_name", lambda name: rows.get(name))
    closed = []
    monkeypatch.setattr(mcp_client, "prepare_mcp_image_recipient", lambda *a, **k: "recipient")
    monkeypatch.setattr(mcp_client, "mcp_image_recipient_location", lambda identity: "https://example.test/mcp")
    monkeypatch.setattr(mcp_client, "close_mcp_image_recipient", closed.append)
    monkeypatch.setattr(mcp_client, "parse_server_headers", lambda row: {})
    payload = SimpleNamespace(
        mcp_image_attachment = SimpleNamespace(message_id = "message", attachment_id = "image"),
        messages = [{"role": "user", "content": "Describe this image"}],
        stream = True, mcp_enabled = True, thread_id = "thread", session_id = "session", cancel_id = "generation",
    )
    fixture = SimpleNamespace(payload = payload, tools = tools, rows = rows, feature = feature,
                              encoded = encoded, url = url, closed = closed, cancel = threading.Event())
    yield fixture
    from core.inference.mcp_image_disclosure import revoke_mcp_image_references
    revoke_mcp_image_references(subject = "subject")
    tool_approvals.revoke_mcp_image_disclosures(subject = "subject")


def prepare(fixture):
    return image_loop.prepare_image_tool_request(
        fixture.payload, subject = "subject", tools = fixture.tools,
        cancel_event = fixture.cancel, ui_events = True,
    )


@pytest.mark.parametrize("tool_index", [0, 1])
def test_configured_tools_share_only_after_exact_explicit_approval(image_request, tool_index):
    f = image_request
    original = copy.deepcopy(f.tools)
    run, tools = prepare(f)
    name = tools[tool_index]["function"]["name"]
    field = f.rows[name][1]["field"]
    args = {field: run.reference.reference, "threshold": 0.3}
    approval = run.prepare_call(name, args, "call-1")
    assert tools[tool_index]["function"]["parameters"]["properties"][field]["enum"] == [run.reference.reference]
    assert tools[tool_index]["function"]["parameters"]["properties"]["threshold"] == {"type": "number"}
    assert f.tools == original
    assert f.encoded not in repr(tools) + repr(approval.metadata) + repr(approval.binding)
    assert approval.metadata["previewUrl"] == "/api/chat/attachments/message/image/file"
    assert not tool_approvals.resolve_tool_decision(approval.approval_id, "allow", "session")
    assert not tool_approvals.resolve_mcp_image_disclosure(approval.approval_id, "allow", current_subject = "other", session_id = "session")
    assert tool_approvals.resolve_mcp_image_disclosure(approval.approval_id, "allow", current_subject = "subject", session_id = "session")
    assert not tool_approvals.resolve_mcp_image_disclosure(approval.approval_id, "allow", current_subject = "subject", session_id = "session")
    assert image_loop.wait_call_decision(approval, approval.slot, approval.approval_id, f.cancel) == "allow"
    wire = approval.context.prepare_wire(args)
    assert wire[field] == (f.url if tool_index else f.encoded)
    assert args[field] == run.reference.reference
    approval.context.commit_at_send("recipient")
    with pytest.raises(McpImageDisclosureError):
        approval.context.commit_at_send("recipient")
    run.close()


@pytest.mark.parametrize("change", ["feature", "server", "arguments", "cancel", "delete"])
def test_changed_binding_after_allow_never_commits(image_request, change):
    f = image_request
    run, _ = prepare(f)
    args = {"picture_blob": run.reference.reference}
    approval = run.prepare_call("mcp__inspect__inspect", args, "call")
    assert tool_approvals.resolve_mcp_image_disclosure(approval.approval_id, "allow", current_subject = "subject", session_id = "session")
    if change == "feature":
        f.feature[:] = [False, "revision-2"]
    elif change == "server":
        f.rows["mcp__inspect__inspect"][0]["config_revision"] += 1
    elif change == "arguments":
        args["threshold"] = 0.8
    elif change == "cancel":
        f.cancel.set()
    else:
        studio_db.delete_chat_attachment("message", "image")
    with pytest.raises(McpImageDisclosureError):
        approval.context.commit_at_send("recipient")
    run.close()


@pytest.mark.parametrize("change", ["conversation", "current_message", "private_payload", "feature", "no_channel"])
def test_request_validation_rejects_invalid_private_selection(image_request, change):
    f = image_request
    if change == "conversation":
        f.payload.thread_id = "other"
    elif change == "current_message":
        studio_db.upsert_chat_message({"id": "new", "threadId": "thread", "role": "user", "createdAt": 3, "content": []})
    elif change == "private_payload":
        f.payload.messages[0]["content"] = [{"type": "image_url", "image_url": {"url": f.url}}]
    elif change == "feature":
        f.feature[0] = False
    else:
        f.payload.stream = False
    with pytest.raises(McpImageDisclosureError):
        prepare(f)


def test_unrelated_vision_image_is_preserved(image_request):
    f = image_request
    f.payload.messages[0]["content"] = [{"type": "image_url", "image_url": {"url": "https://example.test/ordinary.png"}}]
    run, _ = prepare(f)
    assert f.payload.messages[0]["content"][0]["image_url"]["url"].endswith("ordinary.png")
    run.close()


@pytest.mark.parametrize("mode", ["auto", "off", "bypass"])
def test_provider_loop_cannot_bypass_image_consent(image_request, monkeypatch, mode):
    from core.inference import studio_tool_loop
    from .test_studio_tool_loop import FakeTransport, _sse, _DONE

    f = image_request
    run, tools = prepare(f)
    args = {"picture_blob": run.reference.reference}
    transport = FakeTransport([
        [_sse({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "mcp__inspect__inspect", "arguments": json.dumps(args)}}]}), _sse(finish = "tool_calls"), _DONE],
        [_sse({"content": "done"}), _sse(finish = "stop"), _DONE],
    ])
    executed = []

    def execute(name, arguments, **kwargs):
        context = kwargs["mcp_image_context"]
        context.commit_at_send("recipient")
        executed.append((name, copy.deepcopy(arguments)))
        return "ok"

    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(studio_tool_loop, "build_rag_autoinject", lambda *a, **k: None)

    async def collect():
        events = []
        async for line in studio_tool_loop.stream_with_studio_tools(
            transport,
            run = studio_tool_loop.ToolLoopRun(messages = f.payload.messages, session_id = "session", thread_id = "thread"),
            policy = studio_tool_loop.ToolLoopPolicy(tools = tools, max_calls = 2, timeout = 30, rag_scope = None, permission_mode = "auto" if mode == "bypass" else mode, bypass_permissions = mode == "bypass", confirm_calls = False),
            cancel_event = f.cancel, mcp_image_run = run,
        ):
            events.append(line)
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("image_disclosure"):
                    assert not executed
                    assert event["awaiting_confirmation"] is True
                    assert not tool_approvals.resolve_tool_decision(event["approval_id"], "allow", "session")
                    assert tool_approvals.resolve_mcp_image_disclosure(event["approval_id"], "allow", current_subject = "subject", session_id = "session")
        return events

    events = asyncio.run(collect())
    assert executed == [("mcp__inspect__inspect", args)]
    assert f.encoded not in repr(events) + repr(transport.requests)
    assert run.closed


def test_safetensors_close_at_image_card_revokes_without_execution(image_request):
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    f = image_request
    run, tools = prepare(f)
    arguments = {"picture_blob": run.reference.reference}
    call = '<tool_call>' + json.dumps({"name": "mcp__inspect__inspect", "arguments": arguments}) + '</tool_call>'
    executed = []
    generator = run_safetensors_tool_loop(
        single_turn = lambda messages: iter([call]), messages = f.payload.messages, tools = tools,
        execute_tool = lambda *a, **k: executed.append(a), cancel_event = f.cancel,
        session_id = "session", thread_id = "thread", confirm_tool_calls = False,
        bypass_permissions = True, permission_mode = "off", mcp_image_run = run,
    )
    for event in generator:
        if event.get("image_disclosure"):
            approval_id = event["approval_id"]
            assert event["awaiting_confirmation"]
            generator.close()
            break
    else:
        pytest.fail("No image disclosure card was shown")
    assert not executed
    assert run.closed
    assert not tool_approvals.resolve_mcp_image_disclosure(approval_id, "allow", current_subject = "subject", session_id = "session")


def test_gguf_route_passes_private_run_and_reference_schema(image_request, monkeypatch):
    from .test_gguf_tool_non_streaming import _ToolGgufBackend, _client
    from routes import inference

    f = image_request
    observed = {}

    class Backend(_ToolGgufBackend):
        def generate_chat_completion_with_tools(self, **kwargs):
            observed.update(kwargs)
            run = kwargs["mcp_image_run"]
            field = kwargs["tools"][0]["function"]["parameters"]["properties"]["picture_blob"]
            assert field["enum"] == [run.reference.reference]
            assert f.encoded not in repr(kwargs["messages"]) + repr(kwargs["tools"])
            yield {"type": "content", "text": "Image reference available"}

    client = _client(monkeypatch, Backend())

    async def select(*a, **k):
        return f.tools

    monkeypatch.setattr(inference, "_select_request_tools", select)
    response = client.post("/chat/completions", headers = {"X-Unsloth-Events": "1"}, json = {
        "messages": f.payload.messages, "stream": True, "enable_tools": True, "mcp_enabled": True,
        "thread_id": "thread", "session_id": "session", "cancel_id": "generation",
        "mcp_image_attachment": {"message_id": "message", "attachment_id": "image"},
    })
    assert response.status_code == 200, response.text
    assert observed["mcp_image_run"].closed
    assert f.encoded not in response.text


def test_failed_recipient_metadata_leaves_no_approval(image_request, monkeypatch):
    f = image_request
    run, _ = prepare(f)

    def broken_location(identity):
        raise RuntimeError("private transport internals")

    monkeypatch.setattr(mcp_client, "mcp_image_recipient_location", broken_location)
    with pytest.raises(McpImageDisclosureError, match = "cannot safely"):
        run.prepare_call("mcp__inspect__inspect", {"picture_blob": run.reference.reference}, "call")
    assert f.closed == ["recipient"]
    assert run.approvals == []
    assert not any(slot["binding"].generation_id == "generation" for slot in tool_approvals._mcp_image_pending.values())
    run.close()


def test_disabled_feature_without_selection_preserves_original_tools(image_request):
    f = image_request
    f.payload.mcp_image_attachment = None
    f.feature[0] = False
    run, tools = prepare(f)
    assert run is None
    assert tools is f.tools


def test_enabled_feature_without_selection_still_hides_payload_field(image_request):
    f = image_request
    f.payload.mcp_image_attachment = None
    run, tools = prepare(f)
    assert run is None
    assert tools[0]["function"]["parameters"]["properties"]["picture_blob"]["title"] == "Image attachment reference"
