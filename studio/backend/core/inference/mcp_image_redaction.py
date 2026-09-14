# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Redact approved MCP image bytes before results reach public arguments or events.

Approval and attachment lookup stay in their own subsystems.
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
        self,
        *,
        public_arguments,
        image,
        field,
        encoding,
        original_schema,
        recipient,
        commit,
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
                raise McpImageDisclosureError(
                    "Private MCP image wire arguments are invalid"
                ) from None
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
    def __init__(
        self,
        fingerprint,
        data,
        *,
        string_charge_factor = 4,
    ):
        self.fingerprint = fingerprint
        self.data = data
        self.string_charge_factor = string_charge_factor
        self.nodes = 0
        self.material = 0
        self.ancestors = set()
        self.found_echo = False
        # Decoders also accept nonzero unused pad bits. Enumerate only the
        # possible final character so byte-equivalent inputs remain recognized.
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        unused = {0: 0, 1: 4, 2: 2}[len(data) % 3]
        final = alphabet.index(fingerprint[-1])
        self.final_characters = frozenset(alphabet[final : final + (1 << unused)])

    def _charge(self, size):
        self.material += size
        if self.material > MAX_REDACTION_BYTES:
            raise ValueError("private result material limit")

    def _match_span(
        self,
        compact,
        start = 0,
    ):
        prefix = self.fingerprint[:-1]
        while True:
            position = compact.find(prefix, start)
            if position < 0:
                return None
            end = position + len(prefix)
            if end < len(compact) and compact[end] in self.final_characters:
                return position, end + 1
            start = position + 1

    def _echo(
        self,
        value,
        uri = False,
    ):
        # Base64 is canonical apart from alphabet, padding and ASCII whitespace.
        # A matching canonical encoding therefore represents the same bytes.
        normalized = (
            unquote(value).translate(_NORMALIZE_BASE64)
            if "%" in value
            else value.translate(_NORMALIZE_BASE64)
        )
        if self._match_span(normalized) is not None:
            return True
        return False

    @staticmethod
    def _normalized_text(value):
        normalized = (
            unquote(value).translate(_NORMALIZE_BASE64)
            if "%" in value
            else value.translate(_NORMALIZE_BASE64)
        )
        marker = ";base64,"
        position = normalized.lower().find(marker)
        return normalized[position + len(marker) :] if position >= 0 else normalized

    @staticmethod
    def _text_slots(value):
        """Return mutable fields whose values are rendered by ``_flatten_result``."""
        slots = []
        if isinstance(value, dict):
            value_type = value.get("type")
            if isinstance(value.get("text"), str) and value["text"]:
                return [(value, "text", value["text"])]
            resource = value.get("resource")
            if (
                isinstance(resource, dict)
                and isinstance(resource.get("text"), str)
                and resource["text"]
            ):
                return [(resource, "text", resource["text"])]
            resource_text = getattr(resource, "text", None)
            if isinstance(resource_text, str) and resource_text:
                return [(resource, "text", resource_text)]
            if value_type == "resource_link" and isinstance(value.get("uri"), str) and value["uri"]:
                if isinstance(value.get("name"), str) and value["name"]:
                    slots.append((value, "name", value["name"]))
                slots.append((value, "uri", value["uri"]))
            return slots
        value_type = getattr(value, "type", None)
        text = getattr(value, "text", None)
        if isinstance(text, str) and text:
            return [(value, "text", text)]
        resource = getattr(value, "resource", None)
        resource_text = getattr(resource, "text", None)
        if isinstance(resource_text, str) and resource_text:
            return [(resource, "text", resource_text)]
        uri = getattr(value, "uri", None)
        if value_type == "resource_link" and isinstance(uri, str) and uri:
            name = getattr(value, "name", None)
            if isinstance(name, str) and name:
                slots.append((value, "name", name))
            slots.append((value, "uri", uri))
        return slots

    def _set_slot(self, slot, value):
        owner, key, _ = slot
        if isinstance(owner, (dict, list)):
            owner[key] = value
        else:
            setattr(owner, key, value)
        if value == REDACTED_IMAGE:
            self.found_echo = True

    def _redact_slots(self, slots):
        slots = [slot for slot in slots if slot[2] != REDACTED_IMAGE]
        if len(slots) < 2:
            return
        normalized = [self._normalized_text(text) for _, _, text in slots]
        combined = "".join(normalized)
        span = self._match_span(combined)
        while span is not None:
            start = 0
            for slot, text in zip(slots, normalized):
                end = start + len(text)
                if start < span[1] and end > span[0]:
                    self._set_slot(slot, REDACTED_IMAGE)
                start = end
            span = self._match_span(combined, span[1])

        # A server can place payload chunks in separate fields with unrelated
        # blocks between them. Track subsequences of slots against the exact
        # image fingerprint. If a pathological result creates too many partial
        # matches, fail closed by withholding every candidate slot.
        variants = [self.fingerprint[:-1] + final for final in self.final_characters]
        matched_slots = set()
        work = 0
        for fingerprint in variants:
            states = {0: ()}
            for index, text in enumerate(normalized):
                if not text or text == REDACTED_IMAGE:
                    continue
                advanced = dict(states)
                for position, path in states.items():
                    start = 0
                    while position < len(fingerprint) and start < len(text):
                        found = text.find(fingerprint[position], start)
                        scanned = (len(text) - start) if found < 0 else (found - start + 1)
                        # Initial searches are bounded by the existing material
                        # limit and run in native code. Charge spans only after a
                        # prior slot has begun reconstructing the fingerprint.
                        work += scanned if position else 1
                        if work > MAX_REDACTION_NODES * 32:
                            for slot in slots:
                                self._set_slot(slot, REDACTED_IMAGE)
                            return
                        if found < 0:
                            break
                        length = 0
                        limit = min(len(text) - found, len(fingerprint) - position)
                        while (
                            length < limit
                            and text[found + length] == fingerprint[position + length]
                        ):
                            length += 1
                            work += 1
                            if work > MAX_REDACTION_NODES * 32:
                                for slot in slots:
                                    self._set_slot(slot, REDACTED_IMAGE)
                                return
                        end = position + length
                        candidate = path + (index,)
                        if end == len(fingerprint):
                            matched_slots.update(candidate)
                        elif length:
                            advanced.setdefault(end, candidate)
                        start = found + 1
                if len(advanced) > 4096:
                    for slot in slots:
                        self._set_slot(slot, REDACTED_IMAGE)
                    return
                states = advanced
        for matched in matched_slots:
            self._set_slot(slots[matched], REDACTED_IMAGE)

    @staticmethod
    def _integer_image_echo(value, data):
        if not all(type(item) is int and 0 <= item <= 255 for item in value):
            return False
        return data in bytes(value)

    def _redact_integer_fragments(self, slots):
        """Redact byte arrays that reconstruct the image, allowing framing between fields."""
        states = {0: ()}
        matched_slots = set()
        work = 0
        for index, (_, _, value) in enumerate(slots):
            chunk = bytes(value)
            advanced = dict(states)
            for position, path in states.items():
                start = 0
                while position < len(self.data):
                    start = chunk.find(self.data[position : position + 1], start)
                    if start < 0:
                        break
                    length = 0
                    limit = min(len(chunk) - start, len(self.data) - position)
                    while length < limit and chunk[start + length] == self.data[position + length]:
                        length += 1
                    work += length + 1
                    if work > MAX_REDACTION_NODES * 32:
                        for slot in slots:
                            self._set_slot(slot, REDACTED_IMAGE)
                        return
                    if length:
                        end = position + length
                        candidate = path + (index,)
                        if end == len(self.data):
                            matched_slots.update(candidate)
                        else:
                            advanced.setdefault(end, candidate)
                    start += 1
            if len(advanced) > 4096:
                for slot in slots:
                    self._set_slot(slot, REDACTED_IMAGE)
                return
            states = advanced
        for matched in matched_slots:
            self._set_slot(slots[matched], REDACTED_IMAGE)

    def _redact_percent_fragments(self, slots):
        if len(slots) < 2:
            return
        combined = self._normalized_text("".join(text for _, _, text in slots))
        if self._match_span(combined) is not None:
            for slot in slots:
                self._set_slot(slot, REDACTED_IMAGE)

    def _redact_structured_fragments(self, value):
        """Check structured result fragments that are rendered through str()."""
        slots = []
        integer_slots = []
        rekeys = []
        percent_groups = []

        def collect(
            node,
            owner = None,
            key = None,
        ):
            if type(node) is str:
                if owner is not None:
                    slots.append((owner, key, node))
                return
            if isinstance(node, dict):
                direct_strings = []
                for child_key, child in node.items():
                    key_holder = [child_key]
                    slots.append((key_holder, 0, child_key))
                    rekeys.append((node, child_key, key_holder))
                    if type(child) is str:
                        direct_strings.append((node, child_key, child))
                    collect(child, node, child_key)
                percent_groups.append(direct_strings)
                return
            if isinstance(node, (list, tuple)):
                if node and all(type(item) is int and 0 <= item <= 255 for item in node):
                    if owner is not None:
                        integer_slots.append((owner, key, node))
                    return
                for index, child in enumerate(node):
                    collect(child, node, index)
                return
            if isinstance(node, SimpleNamespace):
                for child_key, child in vars(node).items():
                    collect(child, node, child_key)

        collect(value)
        self._redact_slots(slots)
        for group in percent_groups:
            self._redact_percent_fragments(group)
        self._redact_integer_fragments(integer_slots)
        for owner, original, holder in rekeys:
            if holder[0] != original and original in owner:
                child = owner.pop(original)
                owner[holder[0]] = child

    def sanitize(
        self,
        value,
        depth = 0,
        uri = False,
    ):
        self.nodes += 1
        if self.nodes > MAX_REDACTION_NODES or depth > MAX_REDACTION_DEPTH:
            raise ValueError("private result traversal limit")
        if value is None or type(value) in (bool, int, float):
            return value
        if type(value) is str:
            # Charge conservatively before creating normalized/unquoted strings.
            self._charge(len(value) * self.string_charge_factor)
            if self._echo(value, uri):
                self.found_echo = True
                return REDACTED_IMAGE
            return value
        if type(value) is bytes:
            self._charge(len(value))
            if self.data in value:
                self.found_echo = True
                return REDACTED_IMAGE
            return value
        if id(value) in self.ancestors:
            raise ValueError("cyclic private result")
        self.ancestors.add(id(value))
        try:
            if type(value) in (list, tuple):
                if len(value) > MAX_REDACTION_NODES - self.nodes:
                    raise ValueError("private result node limit")
                if self._integer_image_echo(value, self.data):
                    self.found_echo = True
                    return REDACTED_IMAGE
                clean = [self.sanitize(child, depth + 1) for child in value]
                # flatten_result joins every text/link block, even when an image or
                # another non-text block appears between them. Preserve one bounded
                # candidate across those ignored blocks before it inserts newlines.
                slots = []
                for child in clean:
                    slots.extend(self._text_slots(child))
                    if isinstance(child, (dict, SimpleNamespace)):
                        self._redact_structured_fragments(child)
                self._redact_slots(slots)
                self._redact_structured_fragments(clean)
                return clean
            as_object = type(value) is SimpleNamespace
            if not as_object and not isinstance(value, dict):
                # Inspect only known protocol objects; never invoke arbitrary
                # __str__, properties or custom serialization before redaction.
                module = type(value).__module__
                as_object = module.startswith(("mcp.types", "fastmcp.client")) and hasattr(
                    value, "__dict__"
                )
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
                if "structuredContent" in clean:
                    self._redact_structured_fragments(clean["structuredContent"])
                if "structured_content" in clean:
                    self._redact_structured_fragments(clean["structured_content"])
                # Protocol dictionaries need attributes for the existing public
                # formatter; ordinary structured dictionaries remain dictionaries.
                return SimpleNamespace(**clean) if as_object else clean
            raise ValueError("unsupported private result object")
        finally:
            self.ancestors.remove(id(value))


def contains_mcp_image_echo(value, data):
    """Return whether a public value contains the selected image, including fragments."""
    fingerprint = base64.b64encode(data).decode("ascii").rstrip("=")
    sanitizer = _ImageEchoSanitizer(fingerprint, data, string_charge_factor = 1)
    sanitizer.sanitize(value)
    return sanitizer.found_echo
