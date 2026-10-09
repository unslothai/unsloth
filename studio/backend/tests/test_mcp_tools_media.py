# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from studio_mcp import media
from studio_mcp.caller import Caller
from studio_mcp.outputs import ToolOutput


def caller(
    public_base = "http://127.0.0.1:8888",
    cloudflare_url = None,
    direct_local = True,
):
    return Caller(
        token = "sk-unsloth-test",
        account_id = "owner",
        direct_local = direct_local,
        public_base = public_base,
        studio_app = SimpleNamespace(state = SimpleNamespace(cloudflare_url = cloudflare_url)),
    )


def test_media_under_the_cap_is_inline():
    data = b"\x89PNG" + b"x" * 100
    (content,) = media.inline_or_link(
        data, "image/png", url = "http://h/v1/x", name = "x.png", kind = "image"
    )
    assert content.type == "image"
    assert base64.b64decode(content.data) == data
    assert content.mimeType == "image/png"

    (content,) = media.inline_or_link(
        b"RIFF", "audio/wav", url = "http://h/v1/x", name = "x.wav", kind = "audio"
    )
    assert content.type == "audio"


@pytest.mark.parametrize("data", [b"x" * (media.INLINE_CAP + 1), None])
def test_media_over_the_cap_is_only_a_link(data):
    contents = media.inline_or_link(
        data, "audio/wav", url = "http://h/v1/a.wav", name = "a.wav", kind = "audio"
    )
    assert [c.type for c in contents] == ["resource_link"]
    assert str(contents[0].uri) == "http://h/v1/a.wav"
    assert contents[0].name == "a.wav"


def test_the_cap_is_inclusive():
    contents = media.inline_or_link(
        b"x" * media.INLINE_CAP, "image/png", url = "http://h/v1/x", name = "x", kind = "image"
    )
    assert contents[0].type == "image"


@pytest.mark.parametrize(
    "route_url",
    [
        "/v1/audio/gallery/clip-1/file",
        "http://unsloth-mcp.invalid/v1/audio/gallery/clip-1/file",
        "http://unsloth-mcp.invalid:80/v1/audio/gallery/clip-1/file",
    ],
)
def test_urls_come_from_the_outer_request_never_the_forwarded_host(route_url):
    url = media.public_url(caller("http://192.168.1.20:8888"), route_url)
    assert url == "http://192.168.1.20:8888/v1/audio/gallery/clip-1/file"
    assert "unsloth-mcp.invalid" not in url


def test_the_query_survives():
    url = media.public_url(caller(), "/api/inference/images/gallery/img-1/file?thumb=1024")
    assert url == "http://127.0.0.1:8888/api/inference/images/gallery/img-1/file?thumb=1024"


def test_a_tunnel_request_takes_the_tunnels_scheme():
    tunnel = caller(
        "http://abc-def.trycloudflare.com", cloudflare_url = "https://abc-def.trycloudflare.com"
    )
    assert (
        media.public_url(tunnel, "/v1/videos/v1")
        == "https://abc-def.trycloudflare.com/v1/videos/v1"
    )
    # A different host keeps its own scheme, even while a tunnel is up.
    local = caller("http://127.0.0.1:8888", cloudflare_url = "https://abc-def.trycloudflare.com")
    assert media.public_url(local, "/v1/videos/v1") == "http://127.0.0.1:8888/v1/videos/v1"


@pytest.mark.parametrize(
    "route_url", ["/api/auth/api-keys", "/srv/gallery/x.png", "file:///srv/x.png"]
)
def test_non_media_links_are_refused(route_url):
    with pytest.raises(ToolError):
        media.public_url(caller(), route_url)


def test_a_media_result_carries_the_output_as_json_text_first():
    class Out(ToolOutput):
        id: str

    result = media.media_result([media.image_content(b"png", "image/png")], Out(id = "a"))
    assert [c.type for c in result.content] == ["text", "image"]
    assert json.loads(result.content[0].text) == {"id": "a"}
    assert result.structured_content == {"id": "a"}
