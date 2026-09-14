# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import base64
import copy
import io
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from core.inference import mcp_client, mcp_image_tool_loop as image_loop
from core.inference.mcp_image_disclosure import (
    McpImageDisclosureError,
    validate_image_input_mappings,
)
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
    data_url = "data:image/png;base64," + encoded
    studio_db.upsert_chat_thread(
        {"id": "thread", "title": "test", "createdAt": 1, "modelType": "base", "modelId": "mock"}
    )
    studio_db.upsert_chat_message(
        {
            "id": "message",
            "threadId": "thread",
            "role": "user",
            "createdAt": 2,
            "content": [{"type": "text", "text": "Find this image"}],
            "attachments": [
                {
                    "id": "image",
                    "type": "image",
                    "mcpToolOnly": True,
                    "content": [{"type": "image", "image": data_url}],
                }
            ],
        }
    )
    tools, rows = [], {}
    for raw, field, encoding in (
        ("inspect", "picture_blob", "base64"),
        ("classify", "frame_data", "data_url"),
    ):
        schema = {
            "type": "object",
            "properties": {field: {"type": "string"}, "threshold": {"type": "number"}},
            "required": [field],
        }
        mapping = {"tool": raw, "field": field, "encoding": encoding}
        _, digest = validate_image_input_mappings([mapping], [{"name": raw, "inputSchema": schema}])
        name = f"mcp__{raw}__{raw}"
        rows[name] = (
            {
                "id": raw,
                "is_enabled": True,
                "allow_image_attachments": True,
                "url": "https://example.test/mcp",
                "display_name": raw,
                "config_revision": 1,
                "image_input_schema_digest": digest,
            },
            mapping,
            schema,
            digest,
        )
        tools.append({"type": "function", "function": {"name": name, "parameters": schema}})
    monkeypatch.setattr(image_loop, "_mapping_for_name", rows.get)
    monkeypatch.setattr(
        image_loop, "_image_policy_revisions", lambda: [("classify", 1), ("inspect", 1)]
    )
    recipient_events = []
    monkeypatch.setattr(
        mcp_client,
        "prepare_mcp_image_recipient",
        lambda *a, **k: recipient_events.append(k.get("cancel_event")) or "recipient",
    )
    monkeypatch.setattr(
        mcp_client, "mcp_image_recipient_location", lambda _: "https://example.test/mcp"
    )
    monkeypatch.setattr(mcp_client, "mcp_image_recipient_remaining_ms", lambda _: 300_000)
    monkeypatch.setattr(mcp_client, "close_mcp_image_recipient", lambda _: None)
    monkeypatch.setattr(mcp_client, "parse_server_headers", lambda _: {})
    fixture = SimpleNamespace(
        payload = SimpleNamespace(
            mcp_image_attachment = SimpleNamespace(message_id = "message", attachment_id = "image"),
            mcp_image_policy = SimpleNamespace(
                tool_only = True,
                servers = [
                    SimpleNamespace(server_id = "classify", config_revision = 1),
                    SimpleNamespace(server_id = "inspect", config_revision = 1),
                ],
            ),
            messages = [{"role": "user", "content": "Find this image"}],
            stream = True,
            mcp_enabled = True,
            thread_id = "thread",
            session_id = "session",
            cancel_id = "generation",
        ),
        tools = tools,
        rows = rows,
        encoded = encoded,
        data_url = data_url,
        cancel = threading.Event(),
        recipient_events = recipient_events,
    )
    yield fixture
    from core.inference.mcp_image_disclosure import revoke_mcp_image_references

    revoke_mcp_image_references(subject = "subject")
    tool_approvals.revoke_mcp_image_disclosures(subject = "subject")


def prepare(fixture):
    return image_loop.prepare_image_tool_request(
        fixture.payload,
        subject = "subject",
        tools = fixture.tools,
        cancel_event = fixture.cancel,
        ui_events = True,
    )


@pytest.mark.parametrize("tool_index", [0, 1])
def test_two_mappings_share_only_after_exact_one_use_approval(image_request, tool_index):
    f = image_request
    original = copy.deepcopy(f.tools)
    run, tools = prepare(f)
    name = tools[tool_index]["function"]["name"]
    field = f.rows[name][1]["field"]
    arguments = {field: run.reference.reference, "threshold": 0.3}
    approval = run.prepare_call(name, arguments, "call")
    assert f.recipient_events[-1] is f.cancel
    assert tools[tool_index]["function"]["parameters"]["properties"][field]["enum"] == [
        run.reference.reference
    ]
    assert f.tools == original
    assert f.encoded not in repr(tools) + repr(approval.metadata) + repr(approval.binding)
    assert approval.metadata["expiresInMs"] == 300_000
    assert not tool_approvals.resolve_tool_decision(approval.approval_id, "allow", "session")
    assert tool_approvals.resolve_mcp_image_disclosure(
        approval.approval_id, "allow", current_subject = "subject", session_id = "session"
    )
    wire = approval.context.prepare_wire(arguments)
    assert wire[field] == (f.data_url if tool_index else f.encoded)
    assert arguments[field] == run.reference.reference
    approval.context.commit_at_send("recipient")
    with pytest.raises(McpImageDisclosureError):
        approval.context.commit_at_send("recipient")
    run.close()


@pytest.mark.parametrize("change", ["server", "arguments", "cancel", "delete"])
def test_changed_approval_binding_never_commits(image_request, change):
    f = image_request
    run, _ = prepare(f)
    arguments = {"picture_blob": run.reference.reference}
    approval = run.prepare_call("mcp__inspect__inspect", arguments, "call")
    assert tool_approvals.resolve_mcp_image_disclosure(
        approval.approval_id, "allow", current_subject = "subject", session_id = "session"
    )
    if change == "server":
        f.rows["mcp__inspect__inspect"][0]["config_revision"] += 1
    elif change == "arguments":
        arguments["threshold"] = 0.8
    elif change == "cancel":
        f.cancel.set()
    else:
        studio_db.delete_chat_attachment("message", "image")
    with pytest.raises(McpImageDisclosureError):
        approval.context.commit_at_send("recipient")
    run.close()


def test_sibling_branch_and_large_unrelated_image_remain_valid(image_request):
    f = image_request
    studio_db.upsert_chat_message(
        {"id": "sibling", "threadId": "thread", "role": "user", "createdAt": 3, "content": []}
    )
    f.payload.messages[0]["content"] = "x" * (13 * 1024 * 1024)
    run, _ = prepare(f)
    run.close()


@pytest.mark.parametrize("change", ["conversation", "private_payload", "no_channel"])
def test_request_validation_rejects_invalid_private_selection(image_request, change):
    f = image_request
    if change == "conversation":
        f.payload.thread_id = "other"
    elif change == "private_payload":
        f.payload.messages[0]["content"] = [{"type": "image_url", "image_url": {"url": f.data_url}}]
    else:
        f.payload.stream = False
    with pytest.raises(McpImageDisclosureError):
        prepare(f)


def test_unselected_request_rewrites_only_configured_image_fields(image_request):
    f = image_request
    f.payload.mcp_image_attachment = None
    f.payload.mcp_image_policy = None
    run, tools = prepare(f)
    assert run is None
    assert tools[0]["function"]["parameters"]["properties"]["picture_blob"]["title"] == (
        "Image attachment reference"
    )


def test_changed_image_policy_revision_is_rejected_before_model_dispatch(image_request):
    image_request.payload.mcp_image_policy.servers[0].config_revision = 2
    with pytest.raises(McpImageDisclosureError, match = "settings changed"):
        prepare(image_request)
