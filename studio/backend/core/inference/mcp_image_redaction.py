# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Redact approved MCP image bytes before results reach public arguments or events.

Approval and attachment lookup stay in their own subsystems.
"""

from __future__ import annotations

import base64
import codecs
import copy
import hashlib
import json
import threading
from types import SimpleNamespace
from urllib.parse import unquote, unquote_to_bytes

from .mcp_image_disclosure import MAX_IMAGE_BYTES, McpImageDisclosureError

REDACTED_IMAGE = "[shared image echo withheld]"
PRIVATE_CALL_COMPLETE = "Private MCP image operation completed; server response withheld."
PRIVATE_CALL_ERROR = "Error: Private MCP image operation failed; no result was released."
PRIVATE_TRANSPORT_UNAVAILABLE = (
    "Error: This MCP transport cannot safely send private images. No image was shared."
)
MAX_REDACTION_DEPTH = 32
MAX_REDACTION_NODES = 10_000
MAX_REDACTION_BYTES = 64 * 1024 * 1024
MAX_REDACTION_PATHS = 512
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
        self._fingerprint_lower_bounds = {}
        # Decoders also accept nonzero unused pad bits. Enumerate only the
        # possible final character so byte-equivalent inputs remain recognized.
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        unused = {0: 0, 1: 4, 2: 2}[len(data) % 3]
        final = alphabet.index(fingerprint[-1])
        self.final_characters = frozenset(alphabet[final : final + (1 << unused)])
        groups, remainder = divmod(len(data), 4)
        ascii85_min_length = groups + (remainder + 1 if remainder else 0)
        self.min_raw_text_fingerprint = min(len(fingerprint), ascii85_min_length)

    def _encoded_fingerprint(self, encode, block_size, max_length, fold_case):
        """Encode in aligned chunks and stop once the result cannot fit."""
        name = encode.__name__
        if self._fingerprint_lower_bounds.get(name, 0) > max_length:
            return None
        chunk_size = (64 * 1024 // block_size) * block_size
        pieces = []
        pending = ""
        length = 0
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for start in range(0, len(self.data), chunk_size):
            encoded = encode(self.data[start : start + chunk_size])
            pending += encoded.decode("ascii") if isinstance(encoded, bytes) else encoded
            final = start + chunk_size >= len(self.data)
            if final:
                stable, pending = pending, ""
            else:
                cut = len(pending)
                if pending.endswith("%"):
                    cut -= 1
                elif (
                    len(pending) >= 2
                    and pending[-2] == "%"
                    and pending[-1] in "0123456789abcdefABCDEF"
                ):
                    cut -= 2
                stable, pending = pending[:cut], pending[cut:]
            normalized = decoder.decode(unquote_to_bytes(stable), final = final).translate(
                _NORMALIZE_BASE64
            )
            if fold_case:
                normalized = normalized.lower()
            pieces.append(normalized)
            length += len(normalized)
            if length > max_length:
                self._fingerprint_lower_bounds[name] = max(
                    self._fingerprint_lower_bounds.get(name, 0), length
                )
                return None
        return "".join(pieces)

    def _text_fingerprints(self, max_length):
        """Yield one bounded reversible encoding at a time."""
        if len(self.fingerprint) <= max_length:
            for char in self.final_characters:
                yield self.fingerprint[:-1] + char, False
        for encode, block_size, fold_case in (
            (bytes.hex, 1, True),
            (base64.b32encode, 5, True),
            (base64.b32hexencode, 5, True),
            (base64.a85encode, 4, False),
            (base64.b85encode, 4, False),
        ):
            fingerprint = self._encoded_fingerprint(encode, block_size, max_length, fold_case)
            if fingerprint is not None:
                yield fingerprint, fold_case

    def _charge(self, size):
        self.material += size
        if self.material > MAX_REDACTION_BYTES:
            raise ValueError("private result material limit")

    def _match_span(
        self,
        compact,
        start = 0,
    ):
        spans = []
        for fingerprint, fold_case in self._text_fingerprints(len(compact) - start):
            haystack = compact.lower() if fold_case else compact
            position = haystack.find(fingerprint, start)
            if position >= 0:
                spans.append((position, position + len(fingerprint)))
        return min(
            spans,
            default = None,
        )

    def _echo(
        self,
        value,
        uri = False,
    ):
        # Base64 is canonical apart from alphabet, padding and ASCII whitespace.
        # A matching canonical encoding therefore represents the same bytes.
        if len(value) < self.min_raw_text_fingerprint:
            return False
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
        return (
            unquote(value).translate(_NORMALIZE_BASE64)
            if "%" in value
            else value.translate(_NORMALIZE_BASE64)
        )

    @staticmethod
    def _text_slots(value):
        """Return mutable fields whose values are rendered by ``_flatten_result``."""

        def get(owner, key):
            return owner.get(key) if isinstance(owner, dict) else getattr(owner, key, None)

        def text_slot(owner, key):
            text = get(owner, key)
            return [(owner, key, text)] if isinstance(text, str) and text else []

        slots = text_slot(value, "text")
        if slots:
            return slots
        resource = get(value, "resource")
        # Attribute-backed protocol blocks historically read resource attributes;
        # dictionary blocks also support an embedded resource dictionary.
        if not isinstance(value, dict) and isinstance(resource, dict):
            resource = None
        slots = text_slot(resource, "text")
        if slots:
            return slots
        uri = text_slot(value, "uri")
        return (
            text_slot(value, "name") + uri if get(value, "type") == "resource_link" and uri else []
        )

    def _set_slot(self, slot, value):
        owner, key, _ = slot
        if isinstance(owner, (dict, list)):
            owner[key] = value
        else:
            setattr(owner, key, value)
        if value == REDACTED_IMAGE:
            self.found_echo = True

    def _withhold_slots(self, slots):
        for slot in slots:
            self._set_slot(slot, REDACTED_IMAGE)

    def _redact_slots(
        self,
        slots,
        *,
        bridge_percent = True,
    ):
        slots = [slot for slot in slots if slot[2] != REDACTED_IMAGE]
        if len(slots) < 2:
            return
        raw = "".join(text for _, _, text in slots)
        if len(raw) < self.min_raw_text_fingerprint:
            return
        normalized = [self._normalized_text(text) for _, _, text in slots]
        matched_slots = set()
        combined = "".join(normalized)
        span = self._match_span(combined)
        while span is not None:
            start = 0
            for index, (slot, text) in enumerate(zip(slots, normalized)):
                end = start + len(text)
                if start < span[1] and end > span[0]:
                    matched_slots.add(index)
                    self._set_slot(slot, REDACTED_IMAGE)
                start = end
            span = self._match_span(combined, span[1])

        # Search every reversible encoding with a shared work budget. A result
        # exceeding any search bound withholds all candidate slots.
        completed_paths = set()
        work = [0]
        for fingerprint, fold_case in self._text_fingerprints(sum(map(len, normalized))):
            texts = [text.lower() for text in normalized] if fold_case else normalized
            paths = self._fragment_paths(texts, fingerprint, work, completed_paths)
            if paths is None:
                self._withhold_slots(slots)
                return
        path_sets = {path: frozenset(path) for path in completed_paths}
        minimal_paths = [
            path
            for path in completed_paths
            if not any(path_sets[other] < path_sets[path] for other in completed_paths)
        ]
        for path in minimal_paths:
            matched_slots.update(path)
        for matched in matched_slots:
            self._set_slot(slots[matched], REDACTED_IMAGE)
        if bridge_percent:
            self._redact_percent_bridges(slots, matched_slots)

    def _redact_percent_bridges(self, slots, already_matched):
        """Decode percent triplets split across one or more result fields."""
        hex_digits = "0123456789abcdefABCDEF"
        originals = tuple(text for _, _, text in slots)
        queue = [(originals, tuple((index,) for index in range(len(slots))))]
        seen = {originals}
        comparisons = 0
        candidates = 0
        while queue:
            values, aliases = queue.pop()
            for left_index, left in enumerate(values):
                if not left or any(index in already_matched for index in aliases[left_index]):
                    continue
                carry = (
                    1
                    if left.endswith("%")
                    else 2
                    if len(left) > 1 and left[-2] == "%" and left[-1] in hex_digits
                    else 0
                )
                if not carry:
                    continue
                needed = 3 - carry
                for right_index in range(left_index + 1, len(slots)):
                    comparisons += 1
                    if comparisons > MAX_REDACTION_NODES * 4:
                        self._withhold_slots(slots)
                        return
                    right = values[right_index]
                    if (
                        not right
                        or any(index in already_matched for index in aliases[right_index])
                        or any(
                            character not in hex_digits
                            for character in right[: min(needed, len(right))]
                        )
                    ):
                        continue
                    bridged = list(values)
                    bridged[left_index] = left + right
                    bridged[right_index] = ""
                    key = tuple(bridged)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates += 1
                    if candidates > 16:
                        self._withhold_slots(slots)
                        return
                    bridged_aliases = list(aliases)
                    bridged_aliases[left_index] += bridged_aliases[right_index]
                    bridged_aliases[right_index] = ()
                    probe = list(bridged)
                    synthetic = [(probe, index, text) for index, text in enumerate(probe)]
                    self._redact_slots(synthetic, bridge_percent = False)
                    if any(
                        value == REDACTED_IMAGE and len(bridged_aliases[index]) > 1
                        for index, value in enumerate(probe)
                    ):
                        for index, value in enumerate(probe):
                            if value == REDACTED_IMAGE:
                                for original in bridged_aliases[index]:
                                    already_matched.add(original)
                                    self._set_slot(slots[original], REDACTED_IMAGE)
                    queue.append((key, tuple(bridged_aliases)))

    @staticmethod
    def _integer_image_echo(value, data):
        if not all(type(item) is int and 0 <= item <= 255 for item in value):
            return False
        return data in bytes(value)

    @staticmethod
    def _fragment_paths(chunks, fingerprint, work, completed_paths):
        """Find ordered payload fragments, allowing unrelated framing and fields.

        Text searches share a budget across encodings and keep complete paths for
        minimal-path selection. Byte searches retain their original successful-
        match charging and collect every participating slot.
        """
        text_search = isinstance(fingerprint, str)
        states = {0: ()}
        for index, chunk in enumerate(chunks):
            if text_search and (not chunk or chunk == REDACTED_IMAGE):
                continue
            advanced = dict(states)
            for position, path in states.items():
                start = 0
                while position < len(fingerprint) and (not text_search or start < len(chunk)):
                    found = chunk.find(fingerprint[position : position + 1], start)
                    if text_search:
                        work[0] += 1
                        if work[0] > MAX_REDACTION_NODES * 32:
                            return None
                    if found < 0:
                        break
                    length = 0
                    limit = min(len(chunk) - found, len(fingerprint) - position)
                    while (
                        length < limit and chunk[found + length] == fingerprint[position + length]
                    ):
                        length += 1
                        if text_search:
                            work[0] += 1
                            if work[0] > MAX_REDACTION_NODES * 32:
                                return None
                    if not text_search:
                        work[0] += length + 1
                        if work[0] > MAX_REDACTION_NODES * 32:
                            return None
                    end = position + length
                    candidate = path + (index,)
                    if end == len(fingerprint):
                        if text_search:
                            completed_paths.add(candidate)
                            if len(completed_paths) > MAX_REDACTION_PATHS:
                                return None
                        else:
                            completed_paths.update(candidate)
                    elif length:
                        advanced.setdefault(end, candidate)
                    start = found + 1
            if len(advanced) > 4096:
                return None
            states = advanced
        return completed_paths

    def _redact_integer_fragments(self, slots):
        """Redact byte arrays that reconstruct the image, allowing framing between fields."""
        chunks = (bytes(value) for _, _, value in slots)
        matched = self._fragment_paths(chunks, self.data, [0], set())
        self._withhold_slots(slots if matched is None else (slots[index] for index in matched))

    def _redact_percent_fragments(self, slots):
        if len(slots) < 2:
            return
        combined = self._normalized_text("".join(text for _, _, text in slots))
        if self._match_span(combined) is not None:
            self._withhold_slots(slots)

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
