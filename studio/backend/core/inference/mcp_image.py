# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Send a user's attached image to a mapped MCP tool field without the model seeing its bytes.

A server opts in by mapping one top-level string field of a tool. The model only ever sees the
placeholder ``ATTACHED_IMAGE`` in that field; the tool loop asks the user before every call that
would carry the image, and ``execute_tool`` swaps the bytes in after that approval.
"""

from __future__ import annotations

import base64
import binascii
import copy
import io
import json
import re
from dataclasses import dataclass, field, replace
from typing import Optional
from urllib.parse import unquote

ATTACHED_IMAGE = "attached_image"
WITHHELD_RESULT = (
    "[The tool's reply contained the attached image, so it was withheld from the model.]"
)
_PROBE_BYTES = 48
# Any inline image from a call carrying the user's image may be a resized copy of it.
_B64_RUN = r"(?:[A-Za-z0-9+/_=-]|\\/)+"
_IMAGE_DATA_URL = re.compile(
    r"data:image\\?/[\w.+-]+(?:;[\w.+-]+=[^;,\s\"']*)*;base64,"
    + _B64_RUN
    + r"(?:(?:\\[rn]|\r?\n)"
    + _B64_RUN
    + r")*",
    re.IGNORECASE,
)
_LONG_B64 = re.compile(_B64_RUN + r"(?:(?:\\[rn]|\r?\n)" + _B64_RUN + r")*")
_IMAGE_MAGIC = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"BM",
    b"II*\x00",
    b"MM\x00*",
    b"\x00\x00\x01\x00",
)
# A blob this long that decodes cleanly is binary the model cannot use; it may be a copy in a format not listed above.
_OPAQUE_B64_CHARS = 1024


def _decode_b64(text: str) -> "bytes | None":
    for alphabet in (text, text.replace("-", "+").replace("_", "/")):
        try:
            return base64.b64decode(alphabet + "=" * (-len(alphabet) % 4), validate = True)
        except (binascii.Error, ValueError):
            continue
    return None


def _is_image_b64(run: str) -> bool:
    if len(run) < 64:
        return False
    compact = re.sub(r"\\[rn]|\s", "", run).replace("\\/", "/")
    data = _decode_b64(compact[:32])
    if data and (data.startswith(_IMAGE_MAGIC) or data[8:12] == b"WEBP" or data[4:8] == b"ftyp"):
        return True
    lines = [line for line in re.split(r"\\[rn]|\s+", run.split("=", 1)[0]) if line]
    return (
        len(compact) >= _OPAQUE_B64_CHARS
        and all(len(line) >= 60 for line in lines[:-2])
        and _decode_b64(compact.split("=", 1)[0]) is not None
    )


MAX_IMAGE_BYTES = 10 * 1024 * 1024
# Pillow reports a JPEG carrying extra pictures (phone HDR / portrait shots) as MPO.
_FORMATS = {"image/png": ("PNG",), "image/jpeg": ("JPEG", "MPO"), "image/webp": ("WEBP",)}


class McpImageError(ValueError):
    pass


@dataclass(frozen = True)
class McpImage:
    mime: str
    data: bytes = field(repr = False)
    # The server configuration the user approved; execute_tool sends nowhere else.
    recipient: Optional[str] = None

    def approved_for(self, recipient: str) -> "McpImage":
        return replace(self, recipient = recipient)

    def encoded(self, encoding: str) -> str:
        text = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime};base64,{text}" if encoding == "data_url" else text

    def redact(self, text: str) -> str:
        """``text`` without the image: exact echoes replaced, re-encoded ones withhold it all."""
        head, *tails = text.split(self.encoded("base64").rstrip("="))
        for tail in tails:
            head = head.removesuffix(f"data:{self.mime};base64,") + "[attached image]"
            head += tail.removeprefix("==").removeprefix("=")
        head = _IMAGE_DATA_URL.sub("[image withheld]", head)
        head = _LONG_B64.sub(
            lambda m: "[image withheld]" if _is_image_b64(m.group()) else m.group(), head
        )
        return WITHHELD_RESULT if self._reencoded_in(head) else head

    def _reencoded_in(self, text: str) -> bool:
        compact = "".join(text.split()).replace("\\n", "").replace("\\r", "").replace("\\/", "/")
        if "%" in compact:
            compact = unquote(compact)
        lowered = compact.lower()
        for probe in self._probes():
            b64 = base64.b64encode(probe).decode("ascii")
            if (
                b64 in compact
                or b64.replace("+", "-").replace("/", "_") in compact
                or probe.hex() in lowered
            ):
                return True
        return False

    def _probes(self):
        last = len(self.data) - _PROBE_BYTES
        if last < 0:
            yield self.data[: len(self.data) // 3 * 3]
            return
        for start in sorted({last * i // 7 for i in range(8)}):
            for shift in range(3):
                if start + shift <= last:
                    yield self.data[start + shift : start + shift + _PROBE_BYTES]


def parse_mcp_image(data_url: str) -> McpImage:
    """Decode and verify the data URL a Studio client sends as ``mcp_image``."""
    header, sep, payload = data_url.partition(",")
    mime = header[5:].split(";", 1)[0].lower() if header.startswith("data:") else ""
    if not sep or mime not in _FORMATS or ";base64" not in header.lower():
        raise McpImageError("The tool image must be a PNG, JPEG or WebP data URL.")
    try:
        data = base64.b64decode("".join(payload.split()), validate = True)
    except (binascii.Error, ValueError):
        raise McpImageError("The tool image is not valid base64.") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise McpImageError("The tool image must be at most 10 MiB.")
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            actual = image.format
    except Exception:
        raise McpImageError("The tool image could not be decoded.") from None
    if actual not in _FORMATS[mime]:
        raise McpImageError("The tool image content does not match its type.")
    return McpImage(mime = mime, data = data)


def image_input_mappings(server: dict) -> list[dict]:
    raw = server.get("image_input_mappings_json")
    if not raw:
        return []
    try:
        mappings = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [m for m in mappings if isinstance(m, dict)] if isinstance(mappings, list) else []


def _input_schema(tool: dict) -> dict:
    schema = tool.get("inputSchema") or tool.get("input_schema")
    return schema if isinstance(schema, dict) else {}


def image_mapping(server: dict, tool: Optional[dict]) -> Optional[dict]:
    """return the server mapping only while its field remains a top-level string."""
    if not tool:
        return None
    for mapping in image_input_mappings(server):
        if mapping.get("tool") != tool.get("name"):
            continue
        properties = _input_schema(tool).get("properties")
        prop = properties.get(mapping.get("field")) if isinstance(properties, dict) else None
        if isinstance(prop, dict) and prop.get("type") == "string":
            return mapping
    return None


def _loose(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip(" \"'`<>[]{}").lower())


def _names_the_image(value) -> bool:
    """match only the placeholder or a path or URL ending in it, never a mention."""
    return (
        isinstance(value, str)
        and _loose(re.split(r"[/\\]", value.strip().rstrip("/\\"))[-1]) == ATTACHED_IMAGE
    )


def settle_image_call(
    arguments: dict,
    field: str,
    required = (),
) -> bool:
    """normalize small-model image arguments in place so approval shows the exact outgoing call."""
    value = arguments.get(field)
    if not (
        value is None or value == "" or isinstance(value, str) and _loose(value) == ATTACHED_IMAGE
    ):
        return False
    for key in [
        key
        for key, other in arguments.items()
        if key != field and key not in required and _names_the_image(other)
    ]:
        del arguments[key]
    arguments[field] = ATTACHED_IMAGE
    return True


IMAGE_NOTE_PREFIX = "[The user attached an image to this message."
_IMAGE_NOTE_INSTRUCTION = " You cannot see it. To use it, call "


def _is_attached_image_note(text: str) -> bool:
    return text.startswith(IMAGE_NOTE_PREFIX + _IMAGE_NOTE_INSTRUCTION) and text.endswith(".]")


def strip_attached_image_note(text: str) -> str:
    """remove the synthetic image note before exposing user-authored text."""
    if _is_attached_image_note(text):
        return ""
    head, separator, tail = text.rpartition(f"\n\n{IMAGE_NOTE_PREFIX}")
    note = IMAGE_NOTE_PREFIX + tail
    return head if separator and _is_attached_image_note(note) else text


def note_attached_image(messages: list, targets: list[tuple[str, str]]) -> list:
    """tell the model which fields accept the attached image without exposing its bytes."""
    if not targets:
        return messages
    calls = " or ".join(
        f"{name} with {json.dumps({field: ATTACHED_IMAGE})}" for name, field in targets
    )
    note = f"{IMAGE_NOTE_PREFIX}{_IMAGE_NOTE_INSTRUCTION}{calls}.]"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = [*content, {"type": "text", "text": note}]
        else:
            content = f"{content}\n\n{note}" if content else note
        return [*messages[:index], {**message, "content": content}, *messages[index + 1 :]]
    return messages


def public_tool(server: dict, tool: dict) -> dict:
    """expose a mapped field to the model as a placeholder-only string."""
    mapping = image_mapping(server, tool)
    if mapping is None:
        return tool
    schema = copy.deepcopy(_input_schema(tool))
    schema["properties"][mapping["field"]] = {
        "type": "string",
        "enum": [ATTACHED_IMAGE],
        "description": (
            f'Pass "{ATTACHED_IMAGE}" to send the image the user attached to their latest '
            "message. Unsloth Studio inserts it after the user approves; never pass image data, a "
            "path or a URL."
        ),
    }
    public = {k: v for k, v in tool.items() if k not in ("inputSchema", "input_schema")}
    public["inputSchema"] = schema
    return public
