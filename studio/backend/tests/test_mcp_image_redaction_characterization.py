# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavior captured against PR #10871's original implementation."""

import base64
import gc
import hashlib
import tracemalloc
from types import SimpleNamespace

import pytest

from core.inference import mcp_image_redaction as redaction


def context_for(data):
    return redaction.McpImageCallContext(
        public_arguments = {"image": "mcp-image-ref-" + "x" * 40},
        image = SimpleNamespace(
            data = data,
            size_bytes = len(data),
            sha256 = hashlib.sha256(data).hexdigest(),
            mime_type = "image/png",
        ),
        field = "image",
        encoding = "base64",
        original_schema = {"type": "object", "properties": {"image": {"type": "string"}}},
        recipient = "recipient",
        commit = lambda recipient: True,
        tool_name = "inspect",
    )


@pytest.mark.parametrize("data", [b"private-image-xy", b"private-image-xyz"])
def test_every_byte_equivalent_base64_pad_bit_form_is_withheld(data):
    encoded = base64.b64encode(data).decode()
    unpadded = encoded.rstrip("=")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    unused_bits = {1: 4, 2: 2}[len(data) % 3]
    first = alphabet.index(unpadded[-1])
    for last in alphabet[first : first + (1 << unused_bits)]:
        alternate = unpadded[:-1] + last + encoded[len(unpadded) :]
        assert base64.b64decode(alternate) == data
        assert context_for(data).redact_result(alternate) == redaction.REDACTED_IMAGE


@pytest.mark.parametrize("encoding", ["base64", "hex", "bytes"])
def test_every_fragment_boundary_preserves_unrelated_slots(encoding):
    data = b"\x89PNG\r\n\x1a\nprivate-synthetic-image-\xfb\xff\xef"
    encoded = {
        "base64": base64.b64encode(data).decode().rstrip("="),
        "hex": data.hex(),
        "bytes": list(data),
    }[encoding]
    for split in range(1, len(encoded)):
        parts = [encoded[:split], "!", encoded[split:]]
        assert context_for(data).redact_result(parts) == [
            redaction.REDACTED_IMAGE,
            "!",
            redaction.REDACTED_IMAGE,
        ], (encoding, split)


def test_fragment_search_path_budget_withholds_all_candidates(monkeypatch):
    data = b"private-synthetic-image-with-no-padding"
    encoded = base64.b64encode(data).decode()
    monkeypatch.setattr(redaction, "MAX_REDACTION_PATHS", 0)
    parts = [encoded[:12], "!", encoded[12:]]
    assert context_for(data).redact_result(parts) == [redaction.REDACTED_IMAGE] * 3


@pytest.mark.parametrize("commit_result", [False, "raise"])
def test_uncertain_delivery_spends_consent_even_when_commit_fails(commit_result):
    context = context_for(b"private-synthetic-image")
    calls = []

    def commit(recipient):
        calls.append(recipient)
        if commit_result == "raise":
            raise RuntimeError("transport interrupted")
        return False

    context._commit = commit
    with pytest.raises((RuntimeError, redaction.McpImageDisclosureError)):
        context.commit_at_send("recipient")
    with pytest.raises(redaction.McpImageDisclosureError):
        context.commit_at_send("recipient")
    assert calls == ["recipient"]


def test_large_image_request_scan_has_bounded_memory():
    data = b"\x89PNG\r\n\x1a\n" + b"x" * ((1 << 20) - 8)
    gc.collect()
    tracemalloc.start()
    try:
        found = redaction.contains_mcp_image_echo(
            [{"role": "user", "content": "Inspect this image"}], data
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert found is False
    assert peak < len(data) * 6
    fingerprint = base64.b64encode(data).decode("ascii").rstrip("=")
    sanitizer = redaction._ImageEchoSanitizer(fingerprint, data)
    assert sanitizer._echo(base64.b32encode(data).decode("ascii"))


@pytest.mark.parametrize(
    ("encode", "block_size", "fold_case"),
    [
        (bytes.hex, 1, True),
        (base64.b32encode, 5, True),
        (base64.b32hexencode, 5, True),
        (base64.a85encode, 4, False),
        (base64.b85encode, 4, False),
    ],
)
def test_chunked_fingerprints_match_complete_encodings(encode, block_size, fold_case):
    data = (bytes(range(251)) * 280)[:70_003]
    fingerprint = base64.b64encode(data).decode("ascii").rstrip("=")
    sanitizer = redaction._ImageEchoSanitizer(fingerprint, data)
    expected = encode(data)
    if isinstance(expected, bytes):
        expected = expected.decode("ascii")
    expected = sanitizer._normalized_text(expected)
    if fold_case:
        expected = expected.lower()

    assert sanitizer._encoded_fingerprint(encode, block_size, len(expected), fold_case) == expected


def test_chunked_fingerprint_preserves_percent_escape_at_boundary():
    encoded = "00000" * 16_381 + "00%41" + "00000" * 3
    data = base64.b85decode(encoded)
    fingerprint = base64.b64encode(data).decode("ascii").rstrip("=")
    sanitizer = redaction._ImageEchoSanitizer(fingerprint, data)

    assert sanitizer._encoded_fingerprint(
        base64.b85encode, 4, len(encoded), False
    ) == sanitizer._normalized_text(encoded)


def test_ascii85_zero_compression_is_detected_whole_and_fragmented():
    data = bytes(8)
    fingerprint = base64.b64encode(data).decode("ascii").rstrip("=")
    sanitizer = redaction._ImageEchoSanitizer(fingerprint, data)
    assert base64.a85encode(data) == b"zz"
    assert sanitizer._echo("zz")

    clean = redaction._ImageEchoSanitizer(fingerprint, data).sanitize(
        [{"type": "text", "text": "z"}, {"type": "text", "text": "z"}]
    )
    assert [item["text"] for item in clean] == [redaction.REDACTED_IMAGE] * 2
