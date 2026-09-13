# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded, fail-closed handling of a single approved MCP image operation.

Approval policy and attachment lookup belong to the approval subsystem. These
objects must never be serialized into controller events or public tool arguments.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import threading
from types import SimpleNamespace
from urllib.parse import unquote

from .mcp_image_disclosure import MAX_IMAGE_BYTES, McpImageDisclosureError

REDACTED_IMAGE = "[shared image echo withheld]"
PRIVATE_CALL_ERROR = "Error: Private MCP image operation failed; no result was released."
PRIVATE_TRANSPORT_UNAVAILABLE = (
    "Error: This MCP transport cannot safely send private images. No image was shared."
)
MAX_REDACTION_DEPTH = 32
MAX_REDACTION_NODES = 10_000
MAX_REDACTION_BYTES = 64 * 1024 * 1024
_NORMALIZE_BASE64 = str.maketrans("-_", "+/", " \t\r\n\v\f=")


def _canonical_arguments(arguments):
    return json.dumps(arguments, sort_keys = True, separators = (",", ":"), allow_nan = False)


class McpImageCallContext:
    """Private call data plus a one-use callback owned by the consent authority.

    The callback receives the actual recipient identity and must atomically
    revalidate live policy and consume consent. Creating this object is not a
    grant. A transport must establish its write boundary before calling it.
    """

    def __init__(
        self, *, public_arguments, image, field, encoding, original_schema, recipient, commit,
        tool_name,
    ):
        if (
            not isinstance(public_arguments, dict)
            or not isinstance(field, str)
            or encoding not in {"base64", "data_url"}
            or not isinstance(recipient, str)
            or not recipient
            or not isinstance(tool_name, str)
            or not tool_name
            or not callable(commit)
            or not isinstance(image.data, bytes)
            or not 0 < len(image.data) <= MAX_IMAGE_BYTES
            or image.size_bytes != len(image.data)
            or hashlib.sha256(image.data).hexdigest() != image.sha256
        ):
            raise McpImageDisclosureError("Invalid private MCP image context")
        self._public = _canonical_arguments(public_arguments)
        self._image = image
        self._field = field
        self._encoding = encoding
        self._schema = copy.deepcopy(original_schema)
        self.recipient = recipient
        self.tool_name = tool_name
        self._commit = commit
        self._lock = threading.Lock()
        self._spent = False
        self._closed = False
        self._fingerprint = base64.b64encode(image.data).decode("ascii").rstrip("=")

    def __repr__(self):
        return "<McpImageCallContext private>"

    @property
    def spent(self):
        return self._spent

    def prepare_wire(self, arguments):
        """Copy public arguments and validate the complete original wire schema."""
        with self._lock:
            if self._closed or self._spent:
                raise McpImageDisclosureError("Private MCP image arguments changed or expired")
            wire = copy.deepcopy(arguments)
            if _canonical_arguments(wire) != self._public:
                raise McpImageDisclosureError("Private MCP image arguments changed or expired")
            if self._field not in wire:
                raise McpImageDisclosureError("Private MCP image reference is missing")
            encoded = base64.b64encode(self._image.data).decode("ascii")
            if self._encoding == "data_url":
                encoded = f"data:{self._image.mime_type};base64,{encoded}"
            wire[self._field] = encoded
            try:
                from jsonschema.validators import validator_for

                # Remote/local references are deliberately unavailable. Reject
                # them before validator construction, including ordinary fields.
                def check_local(value):
                    if isinstance(value, dict):
                        if "$ref" in value or "$dynamicRef" in value:
                            raise ValueError("references unsupported")
                        for child in value.values():
                            check_local(child)
                    elif isinstance(value, list):
                        for child in value:
                            check_local(child)

                check_local(self._schema)
                validator = validator_for(self._schema)
                validator.check_schema(self._schema)
                validator(self._schema).validate(wire)
            except Exception:
                wire.clear()
                raise McpImageDisclosureError("Private MCP image wire arguments are invalid") from None
            return wire

    def commit_at_send(self, recipient):
        with self._lock:
            if self._closed or self._spent or recipient != self.recipient:
                raise McpImageDisclosureError("Private MCP image consent is unavailable")
            # Unknown delivery is spent too. A throwing callback cannot be retried.
            self._spent = True
            if self._commit(recipient) is not True:
                raise McpImageDisclosureError("Private MCP image consent is unavailable")

    def redact_result(self, result):
        if self._closed or not self._fingerprint:
            raise McpImageDisclosureError("Private MCP image sanitizer is unavailable")
        return _ImageEchoSanitizer(self._fingerprint, self._image.data).sanitize(result)

    def close(self):
        """Release payload references after the transport and its tasks settle."""
        with self._lock:
            self._closed = True
            self._image = None
            self._fingerprint = None
            self._commit = None


class _ImageEchoSanitizer:
    def __init__(self, fingerprint, data):
        self.fingerprint = fingerprint
        self.data = data
        self.nodes = 0
        self.material = 0
        self.ancestors = set()
        # Decoders also accept nonzero unused pad bits. Enumerate only the
        # possible final character so byte-equivalent inputs remain recognized.
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        unused = {0: 0, 1: 4, 2: 2}[len(data) % 3]
        final = alphabet.index(fingerprint[-1])
        self.final_characters = frozenset(alphabet[final:final + (1 << unused)])

    def _charge(self, size):
        self.material += size
        if self.material > MAX_REDACTION_BYTES:
            raise ValueError("private result material limit")

    def _match_span(self, compact, start = 0):
        prefix = self.fingerprint[:-1]
        while True:
            position = compact.find(prefix, start)
            if position < 0:
                return None
            end = position + len(prefix)
            if end < len(compact) and compact[end] in self.final_characters:
                return position, end + 1
            start = position + 1

    def _echo(self, value, uri = False):
        # Base64 is canonical apart from alphabet, padding and ASCII whitespace.
        # A matching canonical encoding therefore represents the same bytes.
        if self._match_span(value.translate(_NORMALIZE_BASE64)) is not None:
            return True
        if uri and "%" in value:
            return self._match_span(unquote(value).translate(_NORMALIZE_BASE64)) is not None
        return False

    def sanitize(self, value, depth = 0, uri = False):
        self.nodes += 1
        if self.nodes > MAX_REDACTION_NODES or depth > MAX_REDACTION_DEPTH:
            raise ValueError("private result traversal limit")
        if value is None or type(value) in (bool, int, float):
            return value
        if type(value) is str:
            # Charge conservatively before creating normalized/unquoted strings.
            self._charge(len(value) * 4)
            return REDACTED_IMAGE if self._echo(value, uri) else value
        if type(value) is bytes:
            self._charge(len(value))
            return REDACTED_IMAGE if self.data in value else value
        if id(value) in self.ancestors:
            raise ValueError("cyclic private result")
        self.ancestors.add(id(value))
        try:
            if type(value) in (list, tuple):
                if len(value) > MAX_REDACTION_NODES - self.nodes:
                    raise ValueError("private result node limit")
                clean = [self.sanitize(child, depth + 1) for child in value]
                # Check adjacent text content as one bounded candidate before
                # flatten_result inserts newlines between the original blocks.
                run = []

                def finish_run():
                    if len(run) > 1:
                        texts = [item["text"] if isinstance(item, dict) else item.text for item in run]
                        combined_size = sum(len(text) for text in texts)
                        self._charge(combined_size * 4)
                        compact_texts = [text.translate(_NORMALIZE_BASE64) for text in texts]
                        combined = "".join(compact_texts)
                        span = self._match_span(combined)
                        while span is not None:
                            start = 0
                            for item, text in zip(run, compact_texts):
                                end = start + len(text)
                                if start < span[1] and end > span[0]:
                                    if isinstance(item, dict):
                                        item["text"] = REDACTED_IMAGE
                                    else:
                                        item.text = REDACTED_IMAGE
                                start = end
                            span = self._match_span(combined, span[1])
                    run.clear()

                for child in clean:
                    child_type = child.get("type") if isinstance(child, dict) else getattr(child, "type", None)
                    child_text = child.get("text") if isinstance(child, dict) else getattr(child, "text", None)
                    if child_type == "text" and isinstance(child_text, str):
                        run.append(child)
                    else:
                        finish_run()
                finish_run()
                return clean
            as_object = type(value) is SimpleNamespace
            if not as_object and not isinstance(value, dict):
                # Inspect only known protocol objects; never invoke arbitrary
                # __str__, properties or custom serialization before redaction.
                module = type(value).__module__
                as_object = module.startswith(("mcp.types", "fastmcp.client")) and hasattr(value, "__dict__")
            if type(value) is dict or as_object:
                fields = vars(value) if as_object else value
                if len(fields) > MAX_REDACTION_NODES - self.nodes:
                    raise ValueError("private result node limit")
                clean = {}
                for key, child in fields.items():
                    if type(key) is not str:
                        raise ValueError("unsupported private result key")
                    clean_key = self.sanitize(key, depth + 1)
                    clean[clean_key] = self.sanitize(
                        child, depth + 1, uri = key.lower() in {"uri", "url", "href"}
                    )
                if clean.get("type") == "image" and clean.get("data") == REDACTED_IMAGE:
                    clean = {"type": "text", "text": REDACTED_IMAGE}
                elif clean.get("blob") == REDACTED_IMAGE:
                    clean.pop("blob", None)
                    clean["text"] = REDACTED_IMAGE
                # Protocol dictionaries need attributes for the existing public
                # formatter; ordinary structured dictionaries remain dictionaries.
                return SimpleNamespace(**clean) if as_object else clean
            raise ValueError("unsupported private result object")
        finally:
            self.ancestors.remove(id(value))
