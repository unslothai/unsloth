# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How generated media reaches the agent: inline when small, otherwise a link to the Studio gallery. Links are built from the address the agent used to reach /mcp, never from a forwarded answer, which only knows the in-process ``unsloth-mcp.invalid`` host."""

from __future__ import annotations

import base64
import json
from typing import Any, Optional, Union
from urllib.parse import urlsplit

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import AudioContent, ImageContent, ResourceLink, TextContent

from studio_mcp.caller import Caller
from studio_mcp.outputs import ToolOutput

# Agents put inline media straight into the model's context; past this a link is kinder.
INLINE_CAP = 1536 * 1024
MEDIA_PREFIXES = ("/v1/", "/api/inference/")

Content = Union[ImageContent, AudioContent, ResourceLink]


def image_content(data: bytes, mime: str) -> ImageContent:
    return ImageContent(type = "image", data = base64.b64encode(data).decode("ascii"), mimeType = mime)


def audio_content(data: bytes, mime: str) -> AudioContent:
    return AudioContent(type = "audio", data = base64.b64encode(data).decode("ascii"), mimeType = mime)


def resource_link(
    url: str,
    name: str,
    mime: Optional[str] = None,
) -> ResourceLink:
    return ResourceLink(type = "resource_link", uri = url, name = name, mimeType = mime)


def _base(caller: Caller) -> str:
    """The outer request's base, with the tunnel's scheme when it came through cloudflared, which reaches Studio over plain http."""
    base = urlsplit(caller.public_base)
    tunnel = getattr(getattr(caller.studio_app, "state", None), "cloudflare_url", None)
    if isinstance(tunnel, str) and tunnel:
        published = urlsplit(tunnel)
        if (
            published.scheme in ("http", "https")
            and published.netloc.lower() == base.netloc.lower()
        ):
            return f"{published.scheme}://{base.netloc}{base.path}"
    return f"{base.scheme}://{base.netloc}{base.path}"


def public_url(caller: Caller, path: str) -> str:
    """An absolute Studio URL for a media route, from a path or from a URL a route built for itself."""
    parts = urlsplit(path)
    if not parts.path.startswith(MEDIA_PREFIXES):
        raise ToolError("Studio returned a media link this tool cannot share")
    query = f"?{parts.query}" if parts.query else ""
    return f"{_base(caller)}{parts.path}{query}"


def inline_or_link(
    data: Optional[bytes], mime: str, *, url: str, name: str, kind: str
) -> list[Content]:
    """Inline content for media under the cap, else only a link."""
    if data is not None and len(data) <= INLINE_CAP:
        return [image_content(data, mime) if kind == "image" else audio_content(data, mime)]
    return [resource_link(url, name, mime)]


def media_result(contents: list[Any], output: ToolOutput) -> ToolResult:
    """Media plus the typed output, which also goes first as JSON text for clients that ignore structuredContent."""
    structured = output.model_dump(mode = "json")
    text = TextContent(type = "text", text = json.dumps(structured, ensure_ascii = False))
    return ToolResult(content = [text, *contents], structured_content = structured)
