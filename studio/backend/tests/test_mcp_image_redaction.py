# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from core.inference import mcp_image_redaction as redaction
from core.inference.mcp_image_disclosure import McpImageDisclosureError
from core.inference.mcp_image_redaction import McpImageCallContext, REDACTED_IMAGE

DATA = b"\x89PNG\r\n\x1a\nunique-private-image-synthetic-\xfb\xff\xef"
ENCODED = base64.b64encode(DATA).decode()
PERCENT_ENCODED = quote(ENCODED, safe = "")
PUBLIC = {"picture": "mcp-image-ref-" + "x" * 40, "options": {"limit": 2}}
SCHEMA = {
    "type": "object",
    "properties": {
        "picture": {"type": "string"},
        "options": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    "required": ["picture"],
}


def make_context(**kwargs):
    return McpImageCallContext(
        **{
            "public_arguments": PUBLIC,
            "image": SimpleNamespace(
                data = DATA,
                size_bytes = len(DATA),
                sha256 = hashlib.sha256(DATA).hexdigest(),
                mime_type = "image/png",
            ),
            "field": "picture",
            "encoding": "base64",
            "original_schema": SCHEMA,
            "recipient": "recipient-a",
            "commit": lambda recipient: True,
            "tool_name": "inspect_picture",
            **kwargs,
        }
    )


@pytest.mark.parametrize("encoding", ["base64", "data_url"])
def test_only_ephemeral_copy_gets_encoded_image(encoding):
    context = make_context(encoding = encoding)
    wire = context.prepare_wire(PUBLIC)
    assert wire["picture"] == (
        ENCODED if encoding == "base64" else "data:image/png;base64," + ENCODED
    )
    wire["options"]["limit"] = 5
    assert PUBLIC["options"]["limit"] == 2
    assert PUBLIC["picture"].startswith("mcp-image-ref-")
    assert ENCODED not in repr(context)


def test_changed_ordinary_arguments_and_original_schema_fail_closed():
    with pytest.raises(McpImageDisclosureError):
        make_context().prepare_wire({**PUBLIC, "options": {"limit": 3}})
    restricted = {"type": "object", "properties": {"picture": {"type": "string", "maxLength": 2}}}
    with pytest.raises(McpImageDisclosureError, match = "wire arguments are invalid"):
        make_context(original_schema = restricted).prepare_wire(PUBLIC)


def test_remote_schema_references_never_resolve():
    schema = {"type": "object", "properties": {"options": {"$ref": "https://never.invalid/schema"}}}
    with pytest.raises(McpImageDisclosureError, match = "wire arguments are invalid"):
        make_context(original_schema = schema).prepare_wire(PUBLIC)


def test_commit_checks_actual_recipient_and_is_one_use_under_race():
    calls = []
    context = make_context(commit = lambda recipient: calls.append(recipient) is None)
    with pytest.raises(McpImageDisclosureError):
        context.commit_at_send("recipient-b")
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        try:
            context.commit_at_send("recipient-a")
            return True
        except McpImageDisclosureError:
            return False

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sorted(results) == [False, True]
    assert calls == ["recipient-a"]


@pytest.mark.parametrize(
    "echo",
    [
        ENCODED,
        "data:image/png;base64," + ENCODED,
        "data:IMAGE/JPEG;charset=utf-8;BASE64," + ENCODED,
        base64.urlsafe_b64encode(DATA).decode().rstrip("="),
        " \r\n".join(ENCODED),
        "Result: " + ENCODED + "; done",
        DATA.hex(),
        DATA.hex().upper(),
        base64.b32encode(DATA).decode().lower(),
        base64.b32hexencode(DATA).decode(),
        base64.a85encode(DATA).decode(),
        base64.b85encode(DATA).decode(),
    ],
)
def test_finite_echo_forms_are_withheld(echo):
    assert make_context().redact_result(echo) == REDACTED_IMAGE


def test_nested_keys_values_uris_resources_and_images_are_sanitized():
    result = {
        "content": [
            {"type": "text", "text": "safe output"},
            {"type": "image", "data": ENCODED, "mimeType": "image/png"},
            {"type": "resource", "resource": {"blob": ENCODED, "mimeType": "image/png"}},
            {"type": "resource_link", "uri": "https://example.test/" + quote(ENCODED, safe = "")},
        ],
        "structuredContent": {ENCODED: ["safe", {"value": ENCODED}]},
        "isError": True,
    }
    clean = make_context().redact_result(result)
    assert ENCODED not in repr(clean)
    assert clean["content"][0]["text"] == "safe output"
    assert clean["content"][1] == {"type": "text", "text": REDACTED_IMAGE}
    assert clean["isError"] is True
    assert result["content"][1]["data"] == ENCODED


def test_split_echoes_in_content_and_structured_data_are_withheld():
    result = {
        "content": [
            {"type": "text", "text": PERCENT_ENCODED[:53]},
            {"type": "text", "text": PERCENT_ENCODED[53:54]},
            {"type": "image", "data": "unrelated", "mimeType": "image/png"},
            {"type": "separator", "text": "safe"},
            {"type": "text", "text": PERCENT_ENCODED[54:]},
        ],
        "structuredContent": {
            "parts": [ENCODED[:17], {"separator": "|"}, ENCODED[17:]],
            "bytes": [[0, *DATA[:17]], [*DATA[17:], 0]],
            "framed": {
                "name": "prefix:" + ENCODED[:17],
                "noise": "unrelated",
                "b": ENCODED[17:] + "; done",
            },
            "percent": {"name": "prefix:" + PERCENT_ENCODED[:54], "payload": PERCENT_ENCODED[54:]},
            "hex": {
                "left": "prefix:" + DATA.hex()[:21],
                "noise": "unrelated",
                "right": DATA.hex()[21:].upper() + "; done",
            },
        },
    }
    clean = make_context().redact_result(result)
    assert clean["content"][0]["text"] == REDACTED_IMAGE
    assert clean["content"][1]["text"] == REDACTED_IMAGE
    assert clean["content"][4]["text"] == REDACTED_IMAGE
    assert clean["content"][2]["data"] == "unrelated"
    assert ENCODED[:17] not in repr(clean)
    assert ENCODED[17:] not in repr(clean)
    assert clean["structuredContent"]["bytes"] == [REDACTED_IMAGE, REDACTED_IMAGE]
    assert clean["structuredContent"]["framed"] == {
        "name": REDACTED_IMAGE,
        "noise": "unrelated",
        "b": REDACTED_IMAGE,
    }
    assert clean["structuredContent"]["percent"] == {
        "name": REDACTED_IMAGE,
        "payload": REDACTED_IMAGE,
    }
    assert clean["structuredContent"]["hex"] == {
        "left": REDACTED_IMAGE,
        "noise": "unrelated",
        "right": REDACTED_IMAGE,
    }


def test_unrelated_image_and_text_remain_unchanged():
    result = {
        "content": [{"type": "image", "data": "YW5vdGhlciBpbWFnZQ==", "mimeType": "image/png"}],
        "structuredContent": {"label": "cat", "probability": 0.9},
    }
    assert make_context().redact_result(result) == result


def test_percent_bridge_does_not_redact_an_unused_completion_slot():
    chunks = [ENCODED[:53] + "%", "2B", ENCODED[53:]]
    result = [{"type": "text", "text": chunk} for chunk in chunks]
    clean = make_context().redact_result(result)
    assert clean[0]["text"] == REDACTED_IMAGE
    assert clean[1]["text"] == "2B"
    assert clean[2]["text"] == REDACTED_IMAGE


def test_subsequence_match_uses_the_smallest_complete_path():
    chunks = ["iV", "2B", "BORw0KGgp1bmlxdWUtcHJpdmF0ZS1pbWFnZS1zeW50aGV0aWMt+//v"]
    clean = make_context().redact_result(chunks)
    assert clean == [REDACTED_IMAGE, "2B", REDACTED_IMAGE]


def test_limits_cycles_and_arbitrary_objects_fail_closed(monkeypatch):
    context = make_context()
    cyclic = []
    cyclic.append(cyclic)
    for value in (cyclic, object(), {1: "unsupported key"}):
        with pytest.raises(ValueError):
            context.redact_result(value)
    nested = []
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError):
        context.redact_result(nested)
    monkeypatch.setattr(redaction, "MAX_REDACTION_BYTES", 20)
    with pytest.raises(ValueError):
        context.redact_result("x" * 6)
    monkeypatch.setattr(redaction, "MAX_REDACTION_NODES", 3)
    with pytest.raises(ValueError):
        context.redact_result([1, 2, 3, 4])


def test_closed_context_withholds_result_and_cannot_prepare_or_commit():
    context = make_context()
    context.close()
    for operation in (
        lambda: context.redact_result("safe"),
        lambda: context.prepare_wire(PUBLIC),
        lambda: context.commit_at_send("recipient-a"),
    ):
        with pytest.raises(McpImageDisclosureError):
            operation()
