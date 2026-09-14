# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64 as b64
import gc
import hashlib
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from core.inference import mcp_image_disclosure as d
from core.inference import mcp_image_redaction as r
from core.inference.mcp_image_redaction import McpImageCallContext, REDACTED_IMAGE as R

Error = d.McpImageDisclosureError

DATA = b"\x89PNG\r\n\x1a\nunique-private-image-synthetic-\xfb\xff\xef"
ENCODED = b64.b64encode(DATA).decode()
PERCENT = quote(ENCODED, safe = "")
PUBLIC = {"picture": "mcp-image-ref-" + "x" * 40, "options": {"limit": 2}}


def object_schema(**properties):
    return {"type": "object", "properties": properties}


SCHEMA = object_schema(
    picture = {"type": "string"},
    options = object_schema(limit = {"type": "integer"}),
) | {"required": ["picture"]}


def make_context(data = DATA, **overrides):
    values = dict(
        public_arguments = PUBLIC,
        image = SimpleNamespace(
            data = data,
            size_bytes = len(data),
            sha256 = hashlib.sha256(data).hexdigest(),
            mime_type = "image/png",
        ),
        field = "picture",
        encoding = "base64",
        original_schema = SCHEMA,
        recipient = "recipient-a",
        commit = lambda recipient: True,
        tool_name = "inspect_picture",
    )
    values.update(overrides)
    return McpImageCallContext(**values)


def sanitizer_for(data):
    fingerprint = b64.b64encode(data).decode("ascii").rstrip("=")
    return r._ImageEchoSanitizer(fingerprint, data)


def fails(
    action,
    error = Error,
    match = None,
):
    with pytest.raises(error, match = match):
        action()


def test_wire_copy_schema_and_single_use_commit():
    for encoding, expected in (
        ("base64", ENCODED),
        ("data_url", "data:image/png;base64," + ENCODED),
    ):
        context = make_context(encoding = encoding)
        wire = context.prepare_wire(PUBLIC)
        assert wire["picture"] == expected
        wire["options"]["limit"] = 5
        assert PUBLIC["options"]["limit"] == 2
        assert PUBLIC["picture"].startswith("mcp-image-ref-")
        assert ENCODED not in repr(context)

    invalid = (
        ({**PUBLIC, "options": {"limit": 3}}, SCHEMA, None),
        (PUBLIC, object_schema(picture = {"type": "string", "maxLength": 2}), "wire arguments"),
        (PUBLIC, object_schema(options = {"$ref": "https://never.invalid/schema"}), "wire arguments"),
    )
    for arguments, schema, message in invalid:
        fails(lambda: make_context(original_schema = schema).prepare_wire(arguments), match = message)

    calls = []
    context = make_context(commit = lambda recipient: calls.append(recipient) is None)
    fails(lambda: context.commit_at_send("recipient-b"))
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        try:
            context.commit_at_send("recipient-a")
            return True
        except Error:
            return False

    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(lambda _: attempt(), range(2))) == [False, True]
    assert calls == ["recipient-a"]


def test_finite_echo_forms_are_withheld():
    echoes = (
        ENCODED,
        "data:image/png;base64," + ENCODED,
        "data:IMAGE/JPEG;charset=utf-8;BASE64," + ENCODED,
        b64.urlsafe_b64encode(DATA).decode().rstrip("="),
        " \r\n".join(ENCODED),
        "Result: " + ENCODED + "; done",
        DATA.hex(),
        DATA.hex().upper(),
        b64.b32encode(DATA).decode().lower(),
        b64.b32hexencode(DATA).decode(),
        b64.a85encode(DATA).decode(),
        b64.b85encode(DATA).decode(),
    )
    for echo in echoes:
        assert make_context().redact_result(echo) == R


def test_nested_and_split_results_are_sanitized_without_false_positives():
    result = {
        "content": [
            {"type": "text", "text": "safe output"},
            {"type": "image", "data": ENCODED, "mimeType": "image/png"},
            {"type": "resource", "resource": {"blob": ENCODED, "mimeType": "image/png"}},
            {"type": "resource_link", "uri": "https://example.test/" + PERCENT},
        ],
        "structuredContent": {ENCODED: ["safe", {"value": ENCODED}]},
        "isError": True,
    }
    clean = make_context().redact_result(result)
    assert ENCODED not in repr(clean)
    assert clean["content"][0]["text"] == "safe output"
    assert clean["content"][1] == {"type": "text", "text": R}
    assert clean["isError"] is True
    assert result["content"][1]["data"] == ENCODED

    content = [
        {"type": "text", "text": PERCENT[:53]},
        {"type": "text", "text": PERCENT[53:54]},
        {"type": "image", "data": "unrelated", "mimeType": "image/png"},
        {"type": "separator", "text": "safe"},
        {"type": "text", "text": PERCENT[54:]},
    ]
    structured = {
        "parts": [ENCODED[:17], {"separator": "|"}, ENCODED[17:]],
        "bytes": [[0, *DATA[:17]], [*DATA[17:], 0]],
        "framed": dict(name = "prefix:" + ENCODED[:17], noise = "unrelated", b = ENCODED[17:] + "; done"),
        "percent": dict(name = "prefix:" + PERCENT[:54], payload = PERCENT[54:]),
        "hex": dict(
            left = "prefix:" + DATA.hex()[:21],
            noise = "unrelated",
            right = DATA.hex()[21:].upper() + "; done",
        ),
    }
    clean = make_context().redact_result({"content": content, "structuredContent": structured})
    assert [clean["content"][i]["text"] for i in (0, 1, 4)] == [R] * 3
    assert clean["content"][2]["data"] == "unrelated"
    assert all(part not in repr(clean) for part in (ENCODED[:17], ENCODED[17:]))
    assert clean["structuredContent"]["bytes"] == [R] * 2
    expected = {
        "framed": dict(name = R, noise = "unrelated", b = R),
        "percent": dict(name = R, payload = R),
        "hex": dict(left = R, noise = "unrelated", right = R),
    }
    for key in expected:
        assert clean["structuredContent"][key] == expected[key]

    unrelated = {
        "content": [{"type": "image", "data": "YW5vdGhlciBpbWFnZQ==", "mimeType": "image/png"}],
        "structuredContent": {"label": "cat", "probability": 0.9},
    }
    assert make_context().redact_result(unrelated) == unrelated


def test_fragment_paths_limits_and_closed_context_fail_closed(monkeypatch):
    cases = (
        (
            [{"type": "text", "text": part} for part in (ENCODED[:53] + "%", "2B", ENCODED[53:])],
            True,
        ),
        (["iV", "2B", "BORw0KGgp1bmlxdWUtcHJpdmF0ZS1pbWFnZS1zeW50aGV0aWMt+//v"], False),
    )
    for value, wrapped in cases:
        clean = make_context().redact_result(value)
        assert ([item["text"] for item in clean] if wrapped else clean) == [R, "2B", R]

    context = make_context()
    cyclic = []
    cyclic.append(cyclic)
    nested = []
    for _ in range(34):
        nested = [nested]
    for value in (cyclic, object(), {1: "unsupported key"}, nested):
        fails(lambda value = value: context.redact_result(value), ValueError)
    for limit, value, maximum in (
        ("MAX_REDACTION_BYTES", "x" * 6, 20),
        ("MAX_REDACTION_NODES", [1, 2, 3, 4], 3),
    ):
        monkeypatch.setattr(r, limit, maximum)
        fails(lambda value = value: context.redact_result(value), ValueError)

    context.close()
    for operation in (
        lambda: context.redact_result("safe"),
        lambda: context.prepare_wire(PUBLIC),
        lambda: context.commit_at_send("recipient-a"),
    ):
        fails(operation)


def test_pad_bits_fragment_boundaries_and_path_budget(monkeypatch):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    for data in (b"private-image-xy", b"private-image-xyz"):
        encoded = b64.b64encode(data).decode()
        unpadded = encoded.rstrip("=")
        first = alphabet.index(unpadded[-1])
        for last in alphabet[first : first + (1 << {1: 4, 2: 2}[len(data) % 3])]:
            alternate = unpadded[:-1] + last + encoded[len(unpadded) :]
            assert b64.b64decode(alternate) == data
            assert make_context(data).redact_result(alternate) == R

    data = b"\x89PNG\r\n\x1a\nprivate-synthetic-image-\xfb\xff\xef"
    encodings = (
        b64.b64encode(data).decode().rstrip("="),
        data.hex(),
        list(data),
    )
    for encoded in encodings:
        for split in range(1, len(encoded)):
            parts = [encoded[:split], "!", encoded[split:]]
            assert make_context(data).redact_result(parts) == [R, "!", R]

    data = b"private-synthetic-image-with-no-padding"
    encoded = b64.b64encode(data).decode()
    monkeypatch.setattr(r, "MAX_REDACTION_PATHS", 0)
    assert make_context(data).redact_result([encoded[:12], "!", encoded[12:]]) == [R] * 3


def test_uncertain_delivery_and_large_scan_memory_bound():
    for commit_result in (False, "raise"):
        context = make_context(b"private-synthetic-image")
        calls = []

        def commit(recipient):
            calls.append(recipient)
            if commit_result == "raise":
                raise RuntimeError("transport interrupted")
            return False

        context._commit = commit
        fails(lambda: context.commit_at_send("recipient-a"), (RuntimeError, Error))
        fails(lambda: context.commit_at_send("recipient-a"))
        assert calls == ["recipient-a"]

    data = b"\x89PNG\r\n\x1a\n" + b"x" * ((1 << 20) - 8)
    gc.collect()
    tracemalloc.start()
    try:
        found = r.contains_mcp_image_echo([{"role": "user", "content": "Inspect this image"}], data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert found is False
    assert peak < len(data) * 6
    assert sanitizer_for(data)._echo(b64.b32encode(data).decode("ascii"))


def test_chunked_fingerprints_percent_boundary_and_ascii85_zero_compression():
    data = (bytes(range(251)) * 280)[:70_003]
    for encode, block_size, fold_case in (
        (bytes.hex, 1, True),
        (b64.b32encode, 5, True),
        (b64.b32hexencode, 5, True),
        (b64.a85encode, 4, False),
        (b64.b85encode, 4, False),
    ):
        sanitizer = sanitizer_for(data)
        expected = encode(data)
        if isinstance(expected, bytes):
            expected = expected.decode("ascii")
        expected = sanitizer._normalized_text(expected)
        if fold_case:
            expected = expected.lower()
        assert (
            sanitizer._encoded_fingerprint(encode, block_size, len(expected), fold_case) == expected
        )

    encoded = "00000" * 16_381 + "00%41" + "00000" * 3
    sanitizer = sanitizer_for(b64.b85decode(encoded))
    assert sanitizer._encoded_fingerprint(
        b64.b85encode, 4, len(encoded), False
    ) == sanitizer._normalized_text(encoded)

    data = bytes(8)
    sanitizer = sanitizer_for(data)
    assert b64.a85encode(data) == b"zz" and sanitizer._echo("zz")
    clean = sanitizer_for(data).sanitize(
        [{"type": "text", "text": "z"}, {"type": "text", "text": "z"}]
    )
    assert [item["text"] for item in clean] == [R] * 2


def tool(name, field, required, alias):
    payload = {"type": "string", "description": "raw payload", "examples": ["secret"]}
    input_schema = object_schema(
        **{field: payload},
        pattern = {"type": "string"},
        patternProperties = {"type": "string"},
        threshold = {"type": "number"},
    ) | {"required": [field] if required else []}
    return {"name": name, "input_schema" if alias else "inputSchema": input_schema}


def validate(input_schema):
    mapping = {"tool": "inspect", "field": "image", "encoding": "base64"}
    return d.validate_image_input_mappings(
        [mapping], [{"name": "inspect", "inputSchema": input_schema}], server_key = "server"
    )


def test_mappings_catalog_and_public_schema_contract():
    mappings, digest = d.validate_image_input_mappings(
        [
            {"tool": "inspect_picture", "field": "picture_blob", "encoding": "base64"},
            {"tool": "classify_frame", "field": "frame_data", "encoding": "data_url"},
        ],
        [
            tool("inspect_picture", "picture_blob", True, False),
            tool("classify_frame", "frame_data", False, True),
        ],
        server_key = "server",
    )
    assert [mapping["field"] for mapping in mappings] == ["picture_blob", "frame_data"]
    assert len(digest) == 64

    for hidden in ("app_only", "invalid_name"):
        name = "inspect_picture" if hidden == "app_only" else "x" * 60
        candidate = tool(name, "image", True, False)
        if hidden == "app_only":
            candidate["_meta"] = {"ui": {"visibility": ["app"]}}
        with pytest.raises(Error, match = "not available to the model"):
            d.validate_image_input_mappings(
                [{"tool": name, "field": "image", "encoding": "base64"}],
                [candidate],
                server_key = "server",
            )

    original = tool("inspect_picture", "picture_blob", True, False)["inputSchema"]
    public = d.model_schema_for_mapping(original, "picture_blob")
    reference = public["properties"]["picture_blob"]
    assert public["properties"]["threshold"] == {"type": "number"}
    assert reference["type"] == "string"
    assert "secret" not in repr(reference)
    assert "picture_blob" in public["required"]
    assert original["properties"]["picture_blob"]["examples"] == ["secret"]
    opaque = "mcp-image-ref-abcdefghijklmnopqrstuvwxyz012345"
    issued = d.model_schema_for_mapping(original, "picture_blob", opaque)
    assert issued["properties"]["picture_blob"]["enum"] == [opaque]
    assert "raw payload" not in repr(issued["properties"]["picture_blob"])
    optional = tool("inspect", "picture_blob", False, False)["inputSchema"]
    assert "picture_blob" not in d.model_schema_for_mapping(optional, "picture_blob")["required"]
    assert (
        "picture_blob" in d.model_schema_for_mapping(optional, "picture_blob", opaque)["required"]
    )


def test_unsupported_mapping_schema_matrix_fails_closed():
    cases = (
        (object_schema(image = {"type": "object"}), "top-level strings"),
        (object_schema(image = {"type": "string", "anyOf": []}), "unsupported"),
        (object_schema(image = {"type": ["string", "null"]}), "top-level strings"),
        (object_schema(), "top-level strings"),
        (
            object_schema(
                image = {"type": "string"},
                label = {"type": "string", "pattern": "^(.+)+:$"},
            ),
            "regular expressions",
        ),
        (
            object_schema(image = {"type": "string"}, options = {"$ref": "#/$defs/Options"})
            | {"$defs": {"Options": {"type": "object"}}},
            "cannot use references",
        ),
    )
    for input_schema, message in cases:
        with pytest.raises(Error, match = message):
            validate(input_schema)


def test_reference_is_conversation_bound_and_live_bytes_are_rechecked(monkeypatch):
    assert {"DecompressionBombError", "DecompressionBombWarning"} <= set(
        d.resolve_tool_only_image.__code__.co_names
    )
    image = d.ResolvedImageAttachment(
        "message-a", "attachment-a", "image/png", 3, 1, 1, "a" * 64, b"abc"
    )
    monkeypatch.setattr(d, "resolve_tool_only_image", lambda **_kwargs: image)
    binding = dict(subject = "alice", thread_id = "thread-a", generation_id = "generation-a")
    record = d.issue_mcp_image_reference(
        **binding,
        message_id = image.message_id,
        attachment_id = image.attachment_id,
    )
    assert "abc" not in repr(image)

    def resolve(**changes):
        return d.resolve_mcp_image_reference(record.reference, **(binding | changes))

    assert resolve()[0] == record
    fails(lambda: resolve(thread_id = "thread-b"), match = "invalid or expired")
    changed = d.ResolvedImageAttachment(
        image.message_id, image.attachment_id, image.mime_type, 3, 1, 1, "b" * 64, b"xyz"
    )
    monkeypatch.setattr(d, "resolve_tool_only_image", lambda **_kwargs: changed)
    fails(resolve, match = "changed")
    assert d.revoke_mcp_image_references(subject = "alice", generation_id = "generation-a") == 1
