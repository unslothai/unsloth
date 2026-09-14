# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import base64
import copy
import threading
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException
from core.inference import mcp_client, mcp_image_tool_loop as image_loop
from core.inference.mcp_image_disclosure import (
    McpImageDisclosureError,
    revoke_mcp_image_references,
    validate_image_input_mappings,
)
from routes import inference
from state import tool_approvals
from storage import studio_db
from studio.backend.tests.test_chat_attachments import (
    PNG_BYTES,
    PNG_DATA_URL,
    _image_attachment,
    _message,
    _reset_studio_db,
    _thread,
)

MAPPINGS = (("inspect", "picture_blob", "base64"), ("classify", "frame_data", "data_url"))


@pytest.fixture
def image_request(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    encoded, data_url = base64.b64encode(PNG_BYTES).decode("ascii"), PNG_DATA_URL
    attachment = _image_attachment("image") | {"mcpToolOnly": True}
    stored_message = _message("message", created_at = 2, attachments = [attachment], thread_id = "thread")
    stored_message["content"] = [{"type": "text", "text": "Find this image"}]
    studio_db.upsert_chat_thread(_thread("thread", title = "test"))
    studio_db.upsert_chat_message(stored_message)
    tools, rows = [], {}
    for raw, field, encoding in MAPPINGS:
        schema = {
            "type": "object",
            "properties": {field: {"type": "string"}, "threshold": {"type": "number"}},
            "required": [field],
        }
        mapping = {"tool": raw, "field": field, "encoding": encoding}
        _, digest = validate_image_input_mappings(
            [mapping], [{"name": raw, "inputSchema": schema}], server_key = raw
        )
        name = f"mcp__{raw}__{raw}"
        server = {
            "id": raw,
            "is_enabled": True,
            "allow_image_attachments": True,
            "url": "https://example.test/mcp",
            "display_name": raw,
            "config_revision": 1,
            "image_input_schema_digest": digest,
        }
        rows[name] = server, mapping, schema, digest
        tools.append({"type": "function", "function": {"name": name, "parameters": schema}})
    monkeypatch.setattr(image_loop, "_mapping_for_name", rows.get)
    monkeypatch.setattr(
        image_loop, "_image_policy_revisions", lambda: [("classify", 1), ("inspect", 1)]
    )
    events = []
    patches = {
        "prepare_mcp_image_recipient": lambda *a, **k: events.append(k.get("cancel_event"))
        or "recipient",
        "mcp_image_recipient_location": lambda _: "https://example.test/mcp",
        "mcp_image_recipient_remaining_ms": lambda _: 300_000,
        "close_mcp_image_recipient": lambda _: None,
        "parse_server_headers": lambda _: {},
    }
    for name, value in patches.items():
        monkeypatch.setattr(mcp_client, name, value)
    payload = NS(
        mcp_image_attachment = NS(message_id = "message", attachment_id = "image"),
        mcp_image_policy = NS(
            tool_only = True,
            servers = [NS(server_id = name, config_revision = 1) for name in ("classify", "inspect")],
        ),
        messages = [{"role": "user", "content": "Find this image"}],
        stream = True,
        mcp_enabled = True,
        thread_id = "thread",
        session_id = "session",
        cancel_id = "generation",
    )
    fixture = NS(
        payload = payload,
        tools = tools,
        rows = rows,
        encoded = encoded,
        data_url = data_url,
        cancel = threading.Event(),
        events = events,
    )
    yield fixture
    revoke_mcp_image_references(subject = "subject")
    tool_approvals.revoke_mcp_image_disclosures(subject = "subject")


def prepare(f):
    return image_loop.prepare_image_tool_request(
        f.payload, subject = "subject", tools = f.tools, cancel_event = f.cancel, ui_events = True
    )


def approve(a):
    return tool_approvals.resolve_mcp_image_disclosure(
        a.approval_id, "allow", current_subject = "subject", session_id = "session"
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
    assert f.events[-1] is f.cancel
    assert tools[tool_index]["function"]["parameters"]["properties"][field]["enum"] == [
        run.reference.reference
    ]
    assert f.tools == original
    assert f.encoded not in repr(tools) + repr(approval.metadata) + repr(approval.binding)
    assert approval.metadata["expiresInMs"] == 300_000
    assert not tool_approvals.resolve_tool_decision(approval.approval_id, "allow", "session")
    assert approve(approval)
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
    assert approve(approval)
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


@pytest.mark.parametrize(
    "case",
    "sibling_large bad_conversation private_payload legacy_private no_channel unselected_schema "
    "stale_message stale_legacy historical new_message new_legacy revision".split(),
)
def test_request_policy_cases(image_request, case):
    f = image_request
    payload = f.payload
    image = [{"type": "image_url", "image_url": {"url": f.data_url}}]
    if case == "sibling_large":
        studio_db.upsert_chat_message(_message("sibling", created_at = 3, thread_id = "thread"))
        payload.messages[0]["content"] = "x" * (13 * 1024 * 1024)
    elif case == "bad_conversation":
        payload.thread_id = "other"
    elif case == "private_payload":
        payload.messages[0]["content"] = image
    elif case == "legacy_private":
        payload.image_base64 = f.encoded
    elif case == "no_channel":
        payload.stream = False
    elif case in {"unselected_schema", "stale_message", "stale_legacy"}:
        payload.mcp_image_attachment = payload.mcp_image_policy = None
    else:
        payload.mcp_image_attachment = None
    if case in {"stale_message", "new_message"}:
        payload.messages = [{"role": "user", "content": image}]
    elif case == "stale_legacy":
        payload.image_base64 = f.encoded
    elif case == "historical":
        payload.messages = [
            {"role": "user", "content": image},
            {"role": "assistant", "content": "a red rectangle"},
            {"role": "user", "content": "describe it again"},
        ]
        payload.image_base64 = f.encoded
    elif case == "new_legacy":
        payload.image_base64 = base64.b64encode(b"a different image").decode()
    elif case == "revision":
        payload.mcp_image_policy.servers[0].config_revision = 2
    if case.startswith("stale_"):
        with pytest.raises(HTTPException, match = "settings must be checked") as exc:
            asyncio.run(inference._prepare_mcp_image_for_route(payload, "user", [], None, []))
        assert exc.value.status_code == 400
    elif case in {"sibling_large", "historical", "unselected_schema"}:
        run, tools = prepare(f)
        assert run is None if case != "sibling_large" else run is not None
        if case == "unselected_schema":
            assert tools[0]["function"]["parameters"]["properties"]["picture_blob"]["title"] == (
                "Image attachment reference"
            )
        if run:
            run.close()
    else:
        match = "settings changed" if case in {"new_message", "new_legacy", "revision"} else None
        with pytest.raises(McpImageDisclosureError, match = match):
            prepare(f)
