# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference.mcp_image_disclosure import (
    McpImageDisclosureError,
    ResolvedImageAttachment,
    issue_mcp_image_reference,
    model_schema_for_mapping,
    resolve_mcp_image_reference,
    revoke_mcp_image_references,
    validate_image_input_mappings,
)


def _tool(
    name: str,
    field: str,
    *,
    required: bool = True,
    alias: bool = False,
):
    schema = {
        "type": "object",
        "properties": {
            field: {"type": "string", "description": "raw payload", "examples": ["secret"]},
            "pattern": {"type": "string"},
            "patternProperties": {"type": "string"},
            "threshold": {"type": "number"},
        },
        "required": [field] if required else [],
    }
    return {"name": name, "input_schema" if alias else "inputSchema": schema}


def test_two_unrelated_mapping_names_and_schema_aliases_are_validated():
    mappings, digest = validate_image_input_mappings(
        [
            {"tool": "inspect_picture", "field": "picture_blob", "encoding": "base64"},
            {"tool": "classify_frame", "field": "frame_data", "encoding": "data_url"},
        ],
        [
            _tool("inspect_picture", "picture_blob"),
            _tool("classify_frame", "frame_data", required = False, alias = True),
        ],
    )
    assert [mapping["field"] for mapping in mappings] == ["picture_blob", "frame_data"]
    assert len(digest) == 64


def test_model_schema_replaces_only_the_payload_field_and_removes_hints():
    original = _tool("inspect_picture", "picture_blob")["inputSchema"]
    public = model_schema_for_mapping(original, "picture_blob")
    assert public["properties"]["threshold"] == {"type": "number"}
    reference = public["properties"]["picture_blob"]
    assert reference["type"] == "string"
    assert "secret" not in repr(reference)
    assert "picture_blob" in public["required"]
    assert original["properties"]["picture_blob"]["examples"] == ["secret"]


def test_model_schema_offers_only_the_issued_opaque_reference():
    original = _tool("inspect_picture", "picture_blob")["inputSchema"]
    ref = "mcp-image-ref-abcdefghijklmnopqrstuvwxyz012345"
    public = model_schema_for_mapping(original, "picture_blob", ref)
    assert public["properties"]["picture_blob"]["enum"] == [ref]
    assert "raw payload" not in repr(public["properties"]["picture_blob"])
    optional = _tool("inspect", "picture_blob", required = False)["inputSchema"]
    assert "picture_blob" not in model_schema_for_mapping(optional, "picture_blob")["required"]
    assert "picture_blob" in model_schema_for_mapping(optional, "picture_blob", ref)["required"]


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "object", "properties": {"image": {"type": "object"}}}, "top-level strings"),
        (
            {"type": "object", "properties": {"image": {"type": "string", "anyOf": []}}},
            "unsupported",
        ),
        (
            {"type": "object", "properties": {"image": {"type": ["string", "null"]}}},
            "top-level strings",
        ),
        ({"type": "object", "properties": {}}, "top-level strings"),
        (
            {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "label": {"type": "string", "pattern": "^(.+)+:$"},
                },
            },
            "regular expressions",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "options": {"$ref": "#/$defs/Options"},
                },
                "$defs": {"Options": {"type": "object"}},
            },
            "cannot use references",
        ),
    ],
)
def test_unsupported_mapping_schemas_fail_closed(schema, message):
    with pytest.raises(McpImageDisclosureError, match = message):
        validate_image_input_mappings(
            [{"tool": "inspect", "field": "image", "encoding": "base64"}],
            [{"name": "inspect", "inputSchema": schema}],
        )


def test_reference_is_conversation_bound_and_live_bytes_are_rechecked(monkeypatch):
    import core.inference.mcp_image_disclosure as disclosure

    assert {"DecompressionBombError", "DecompressionBombWarning"} <= set(
        disclosure.resolve_tool_only_image.__code__.co_names
    )

    image = ResolvedImageAttachment(
        message_id = "message-a",
        attachment_id = "attachment-a",
        mime_type = "image/png",
        size_bytes = 3,
        width = 1,
        height = 1,
        sha256 = "a" * 64,
        data = b"abc",
    )
    monkeypatch.setattr(disclosure, "resolve_tool_only_image", lambda **_kwargs: image)
    record = issue_mcp_image_reference(
        subject = "alice",
        thread_id = "thread-a",
        generation_id = "generation-a",
        message_id = image.message_id,
        attachment_id = image.attachment_id,
    )
    assert "abc" not in repr(image)
    assert (
        resolve_mcp_image_reference(
            record.reference,
            subject = "alice",
            thread_id = "thread-a",
            generation_id = "generation-a",
        )[0]
        == record
    )
    with pytest.raises(McpImageDisclosureError, match = "invalid or expired"):
        resolve_mcp_image_reference(
            record.reference,
            subject = "alice",
            thread_id = "thread-b",
            generation_id = "generation-a",
        )
    changed = ResolvedImageAttachment(**{**image.__dict__, "sha256": "b" * 64, "data": b"xyz"})
    monkeypatch.setattr(disclosure, "resolve_tool_only_image", lambda **_kwargs: changed)
    with pytest.raises(McpImageDisclosureError, match = "changed"):
        resolve_mcp_image_reference(
            record.reference,
            subject = "alice",
            thread_id = "thread-a",
            generation_id = "generation-a",
        )
    assert revoke_mcp_image_references(subject = "alice", generation_id = "generation-a") == 1
