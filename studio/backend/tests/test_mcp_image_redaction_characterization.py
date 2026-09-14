# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavior captured against PR #10871's original implementation."""

import base64
import hashlib
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
