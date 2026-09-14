# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validate configured image inputs and resolve tool-only attachments.

Private bytes stay in objects that callers must not serialize into model arguments.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import secrets
import threading
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Iterable

from core.inference.mcp_client import (
    MCP_MODEL_TOOL_NAME_RE,
    mcp_model_tool_name,
    mcp_tool_model_visible,
)


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_IMAGE_MIME = frozenset({"image/png", "image/jpeg", "image/webp"})
_UNSUPPORTED_FIELD_KEYS = frozenset(
    {"$ref", "anyOf", "oneOf", "allOf", "not", "const", "enum", "contentSchema"}
)
_UNSUPPORTED_OBJECT_KEYS = frozenset(
    {"$ref", "anyOf", "oneOf", "allOf", "not", "dependentRequired", "dependentSchemas"}
)
_REGEX_SCHEMA_KEYS = frozenset({"pattern", "patternProperties"})
_REFERENCE_SCHEMA_KEYS = frozenset({"$ref", "$dynamicRef"})
_SCHEMA_MAP_KEYS = frozenset({"$defs", "definitions", "dependentSchemas", "properties"})


class McpImageDisclosureError(ValueError):
    """A fixed-detail validation failure safe to surface without private bytes."""


@dataclass(frozen = True)
class ResolvedImageAttachment:
    message_id: str
    attachment_id: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    data: bytes = dataclass_field(repr = False)


@dataclass(frozen = True)
class McpImageReference:
    reference: str
    subject: str
    thread_id: str
    generation_id: str
    message_id: str
    attachment_id: str
    mime_type: str
    size_bytes: int
    sha256: str


_reference_lock = threading.Lock()
_references: dict[str, McpImageReference] = {}


def _mapping_dict(mapping: Any) -> dict[str, str]:
    if hasattr(mapping, "model_dump"):
        mapping = mapping.model_dump()
    if not isinstance(mapping, dict):
        raise McpImageDisclosureError("Image input mappings must be objects")
    return {
        "tool": str(mapping.get("tool") or ""),
        "field": str(mapping.get("field") or ""),
        "encoding": str(mapping.get("encoding") or ""),
    }


def stored_image_input_mappings(server: dict) -> list:
    """Read optional persisted mappings for display and policy discovery.

    Approval validation uses the strict parser: corrupt configuration must not
    silently become an unmapped tool at that boundary.
    """
    try:
        mappings = json.loads(server.get("image_input_mappings_json") or "[]")
    except (TypeError, ValueError):
        return []
    return mappings if isinstance(mappings, list) else []


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        schema = tool.get("input_schema")
    return schema if isinstance(schema, dict) else {}


def _eligible_field(schema: dict[str, Any], field: str) -> dict[str, Any]:
    if schema.get("type") != "object" or any(key in schema for key in _UNSUPPORTED_OBJECT_KEYS):
        raise McpImageDisclosureError("Mapped tools must have a direct object input schema")
    properties = schema.get("properties")
    field_schema = properties.get(field) if isinstance(properties, dict) else None
    if not isinstance(field_schema, dict) or field_schema.get("type") != "string":
        raise McpImageDisclosureError("Mapped image fields must be direct top-level strings")
    if any(key in field_schema for key in _UNSUPPORTED_FIELD_KEYS):
        raise McpImageDisclosureError("Mapped image fields use unsupported schema constraints")
    for key in ("required",):
        value = schema.get(key)
        if value is not None and (
            not isinstance(value, list) or any(not isinstance(item, str) for item in value)
        ):
            raise McpImageDisclosureError("Mapped tool schema has an invalid required list")
    return field_schema


def _reject_regex_schema(schema: dict[str, Any]) -> None:
    pending = [schema]
    nodes = 0
    while pending:
        node = pending.pop()
        nodes += 1
        if nodes > 10_000:
            raise McpImageDisclosureError("Mapped tool schema is too complex")
        if isinstance(node, dict):
            if _REGEX_SCHEMA_KEYS.intersection(node):
                raise McpImageDisclosureError("Mapped tool schemas cannot use regular expressions")
            if _REFERENCE_SCHEMA_KEYS.intersection(node):
                raise McpImageDisclosureError("Mapped tool schemas cannot use references")
            for key, value in node.items():
                if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
                    pending.extend(value.values())
                else:
                    pending.append(value)
        elif isinstance(node, list):
            pending.extend(node)


def validate_image_input_mappings(
    mappings: Iterable[Any], tools: Iterable[dict[str, Any]], *, server_key: str
) -> tuple[list[dict[str, str]], str | None]:
    normalized = [_mapping_dict(mapping) for mapping in mappings]
    if not normalized:
        return [], None
    by_name = {
        str(tool.get("name") or ""): tool
        for tool in tools
        if isinstance(tool, dict) and str(tool.get("name") or "")
    }
    seen: set[str] = set()
    digest_rows: list[dict[str, Any]] = []
    for mapping in normalized:
        tool_name = mapping["tool"]
        field = mapping["field"]
        encoding = mapping["encoding"]
        if not tool_name or not field or encoding not in {"base64", "data_url"}:
            raise McpImageDisclosureError("Image input mappings are incomplete")
        if tool_name in seen:
            raise McpImageDisclosureError("Only one image input mapping is allowed per tool")
        seen.add(tool_name)
        tool = by_name.get(tool_name)
        if tool is None:
            raise McpImageDisclosureError(f"MCP tool '{tool_name}' was not discovered")
        if not mcp_tool_model_visible(tool) or not MCP_MODEL_TOOL_NAME_RE.fullmatch(
            mcp_model_tool_name(server_key, tool_name)
        ):
            raise McpImageDisclosureError(f"MCP tool '{tool_name}' is not available to the model")
        schema = _tool_schema(tool)
        _eligible_field(schema, field)
        _reject_regex_schema(schema)
        digest_rows.append({"mapping": mapping, "schema": schema})
    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()
    return normalized, digest


def model_schema_for_mapping(
    schema: dict[str, Any],
    field: str,
    attachment_ref: str | None = None,
) -> dict[str, Any]:
    """Return a public schema that accepts a selector and contains no payload hints."""
    _eligible_field(schema, field)
    public = copy.deepcopy(schema)
    properties = dict(public["properties"])
    properties[field] = {
        "type": "string",
        "title": "Image attachment reference",
        "description": (
            "Opaque Studio attachment reference. Select the supplied reference; never provide "
            "image bytes or a URL."
            + (f" Available reference: {attachment_ref}" if attachment_ref else "")
        ),
        "pattern": r"^mcp-image-ref-[A-Za-z0-9_-]{32,128}$",
    }
    if attachment_ref:
        properties[field]["enum"] = [attachment_ref]
    public["properties"] = properties
    required = [item for item in public.get("required", []) if isinstance(item, str)]
    if attachment_ref and field not in required:
        required.append(field)
    public["required"] = required
    for key in ("examples", "example", "default"):
        public.pop(key, None)
    return public


def _decode_data_url(value: str) -> tuple[str, bytes]:
    if not value.startswith("data:") or "," not in value:
        raise McpImageDisclosureError("The selected attachment is not an inline image")
    header, encoded = value.split(",", 1)
    if ";base64" not in header.lower():
        raise McpImageDisclosureError("The selected attachment is not base64 encoded")
    mime = header[5:].split(";", 1)[0].lower()
    if mime not in SUPPORTED_IMAGE_MIME:
        raise McpImageDisclosureError("Only PNG, JPEG, and WebP images can be shared")
    compact = "".join(encoded.split())
    if len(compact) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise McpImageDisclosureError("The selected image is too large")
    compact += "=" * (-len(compact) % 4)
    try:
        data = base64.b64decode(compact, altchars = b"-_", validate = True)
    except (binascii.Error, ValueError) as exc:
        raise McpImageDisclosureError("The selected image is invalid") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise McpImageDisclosureError("The selected image is empty or too large")
    return mime, data


def resolve_tool_only_image(
    *, thread_id: str, message_id: str, attachment_id: str
) -> ResolvedImageAttachment:
    from PIL import Image, UnidentifiedImageError
    from storage.studio_db import get_chat_attachment_for_thread

    attachment = get_chat_attachment_for_thread(thread_id, message_id, attachment_id)
    if attachment is None:
        raise McpImageDisclosureError("The selected image is no longer available")
    if attachment.get("mcpToolOnly") is not True or attachment.get("type") != "image":
        raise McpImageDisclosureError("The selected attachment is not a private MCP image")
    content = attachment.get("content")
    image_value = None
    if isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image"
                and isinstance(part.get("image"), str)
            ):
                image_value = part["image"]
                break
    if image_value is None:
        raise McpImageDisclosureError("The selected image has no inline data")
    mime, data = _decode_data_url(image_value)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise McpImageDisclosureError("The selected image is invalid") from exc
    expected_format = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[mime]
    if (
        image_format != expected_format
        or width <= 0
        or height <= 0
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise McpImageDisclosureError("The selected image format or dimensions are invalid")
    return ResolvedImageAttachment(
        message_id = message_id,
        attachment_id = attachment_id,
        mime_type = mime,
        size_bytes = len(data),
        width = width,
        height = height,
        sha256 = hashlib.sha256(data).hexdigest(),
        data = data,
    )


def issue_mcp_image_reference(
    *, subject: str, thread_id: str, generation_id: str, message_id: str, attachment_id: str
) -> McpImageReference:
    """Mint a selector for one verified attachment in the authenticated run."""
    if not subject or not thread_id or not generation_id:
        raise McpImageDisclosureError("A saved conversation is required to share an image")
    image = resolve_tool_only_image(
        thread_id = thread_id,
        message_id = message_id,
        attachment_id = attachment_id,
    )
    reference = f"mcp-image-ref-{secrets.token_urlsafe(32)}"
    record = McpImageReference(
        reference = reference,
        subject = subject,
        thread_id = thread_id,
        generation_id = generation_id,
        message_id = message_id,
        attachment_id = attachment_id,
        mime_type = image.mime_type,
        size_bytes = image.size_bytes,
        sha256 = image.sha256,
    )
    with _reference_lock:
        _references[reference] = record
    return record


def resolve_mcp_image_reference(
    reference: str, *, subject: str, thread_id: str, generation_id: str
) -> tuple[McpImageReference, ResolvedImageAttachment]:
    """Resolve a selector against live conversation storage and immutable bytes."""
    with _reference_lock:
        record = _references.get(reference)
    if record is None or (
        record.subject != subject
        or record.thread_id != thread_id
        or record.generation_id != generation_id
    ):
        raise McpImageDisclosureError("The image attachment reference is invalid or expired")
    image = resolve_tool_only_image(
        thread_id = record.thread_id,
        message_id = record.message_id,
        attachment_id = record.attachment_id,
    )
    if (
        image.sha256 != record.sha256
        or image.size_bytes != record.size_bytes
        or image.mime_type != record.mime_type
    ):
        raise McpImageDisclosureError("The selected image changed after it was attached")
    return record, image


def revoke_mcp_image_references(
    *,
    subject: str | None = None,
    thread_id: str | None = None,
    generation_id: str | None = None,
    message_id: str | None = None,
    attachment_id: str | None = None,
) -> int:
    selectors = {
        "subject": subject,
        "thread_id": thread_id,
        "generation_id": generation_id,
        "message_id": message_id,
        "attachment_id": attachment_id,
    }
    removed = 0
    with _reference_lock:
        for reference, record in list(_references.items()):
            if any(
                value is not None and getattr(record, name) != value
                for name, value in selectors.items()
            ):
                continue
            _references.pop(reference, None)
            removed += 1
    return removed


def canonical_arguments_digest(arguments: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(
            arguments,
            sort_keys = True,
            separators = (",", ":"),
            ensure_ascii = False,
            allow_nan = False,
        )
    except (TypeError, ValueError) as exc:
        raise McpImageDisclosureError("Tool arguments cannot be bound for image sharing") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
