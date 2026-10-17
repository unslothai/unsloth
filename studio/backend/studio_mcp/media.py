# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How generated media reaches the agent: inline when small, otherwise a link to the Unsloth Studio gallery. Links are built from the address the agent used to reach /mcp, never from a forwarded answer, which only knows the in-process ``unsloth-mcp.invalid`` host."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Literal, Optional, Union
from urllib.parse import parse_qsl, quote, urlsplit

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import AudioContent, ImageContent, ResourceLink, TextContent

from studio_mcp.caller import Caller
from studio_mcp.forward import forward
from studio_mcp.outputs import ToolOutput

# Agents put inline media straight into the model's context; past this a link is kinder.
INLINE_CAP = 1536 * 1024
MEDIA_PREFIXES = ("/v1/", "/api/inference/")
THUMB_SIDE = 1024
# inline: the media itself when small; preview: a small image plus a link; link: only the link.
ReturnMedia = Literal["inline", "preview", "link"]
# Every link the MCP mints lasts about an hour, whatever the UI's own links last.
MCP_LINK_TTL_S = 3600
_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,512}$")
SIGNED_BASES = {
    "image": "/api/inference/images/gallery",
    "audio": "/v1/audio/gallery",
    "video": "/api/inference/video/gallery",
}


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
    """The outer request's base, with the tunnel's scheme when it came through cloudflared, which reaches Unsloth Studio over plain http."""
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
    """An absolute Unsloth Studio URL for a media route, from a path or from a URL a route built for itself."""
    parts = urlsplit(path)
    if not parts.path.startswith(MEDIA_PREFIXES):
        raise ToolError("Unsloth Studio returned a media link this tool cannot share")
    query = f"?{parts.query}" if parts.query else ""
    return f"{_base(caller)}{parts.path}{query}"


def image_gallery_path(image_id: str, thumb: Union[bool, int] = False) -> str:
    """The gallery PNG, or with ``thumb`` a WebP thumbnail that many pixels on its long side (True: 1024)."""
    path = f"/api/inference/images/gallery/{quote(image_id, safe = '')}/file"
    if thumb is False:
        return path
    return f"{path}?thumb={THUMB_SIDE if thumb is True else int(thumb)}"


def audio_gallery_path(clip_id: str) -> str:
    return f"/v1/audio/gallery/{quote(clip_id, safe = '')}/file"


def video_gallery_path(video_id: str, suffix: str = "/file") -> str:
    return f"{SIGNED_BASES['video']}/{quote(video_id, safe = '')}{suffix}"


def video_frame_path(video_id: str) -> str:
    return video_gallery_path(video_id, "/frame")


def voice_path(voice_id: str = "", suffix: str = "") -> str:
    return (
        f"/v1/audio/voices/{quote(voice_id, safe = '')}{suffix}" if voice_id else "/v1/audio/voices"
    )


async def signed_url(caller: Caller, kind: str, media_id: str) -> Optional[str]:
    """A link that works without the API key for about an hour, for this caller's account only, or None when Unsloth Studio would not mint one. Only the token is taken from the answer; the path is rebuilt here."""
    base = f"{SIGNED_BASES[kind]}/{quote(media_id, safe = '')}"
    params = {"ttl_s": MCP_LINK_TTL_S} if kind == "video" else None
    try:
        response = await forward(caller, "GET", f"{base}/signed-url", params = params)
        url = response.json().get("url") if response.status_code == 200 else None
        token = dict(parse_qsl(urlsplit(url).query)).get("token") if isinstance(url, str) else None
    except Exception:
        return None
    if not isinstance(token, str) or not _TOKEN.match(token):
        return None
    return public_url(caller, f"{base}/file-signed?token={token}")


def media_result(contents: list[Any], output: ToolOutput) -> ToolResult:
    """Media plus the typed output, which also goes first as JSON text for clients that ignore structuredContent."""
    structured = output.model_dump(mode = "json")
    text = TextContent(
        type = "text", text = json.dumps(structured, ensure_ascii = False, separators = (",", ":"))
    )
    return ToolResult(content = [text, *contents], structured_content = structured)
