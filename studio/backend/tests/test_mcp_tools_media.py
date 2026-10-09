# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from studio_mcp import inputs, media
from studio_mcp.inputs import ImageInput
from studio_mcp.outputs import ToolOutput

from .mcp_harness import PNG, file_spy, make_caller  # noqa: F401  (file_spy is a fixture)


@pytest.mark.parametrize(
    "route_url",
    [
        "/v1/audio/gallery/clip-1/file",
        "http://unsloth-mcp.invalid/v1/audio/gallery/clip-1/file",
        "http://unsloth-mcp.invalid:80/v1/audio/gallery/clip-1/file",
    ],
)
def test_urls_come_from_the_outer_request_never_the_forwarded_host(route_url):
    url = media.public_url(make_caller(public_base = "http://192.168.1.20:8888"), route_url)
    assert url == "http://192.168.1.20:8888/v1/audio/gallery/clip-1/file"
    assert "unsloth-mcp.invalid" not in url


def test_the_query_survives():
    url = media.public_url(make_caller(), "/api/inference/images/gallery/img-1/file?thumb=1024")
    assert url == "http://127.0.0.1:8888/api/inference/images/gallery/img-1/file?thumb=1024"


def test_a_tunnel_request_takes_the_tunnels_scheme():
    tunnel = make_caller(
        public_base = "http://abc-def.trycloudflare.com",
        cloudflare_url = "https://abc-def.trycloudflare.com",
    )
    assert (
        media.public_url(tunnel, "/v1/videos/v1")
        == "https://abc-def.trycloudflare.com/v1/videos/v1"
    )
    # A different host keeps its own scheme, even while a tunnel is up.
    local = make_caller(cloudflare_url = "https://abc-def.trycloudflare.com")
    assert media.public_url(local, "/v1/videos/v1") == "http://127.0.0.1:8888/v1/videos/v1"


@pytest.mark.parametrize(
    "route_url", ["/api/auth/api-keys", "/srv/gallery/x.png", "file:///srv/x.png"]
)
def test_non_media_links_are_refused(route_url):
    with pytest.raises(ToolError):
        media.public_url(make_caller(), route_url)


def test_a_media_result_carries_the_output_as_json_text_first():
    class Out(ToolOutput):
        id: str

    result = media.media_result([media.image_content(b"png", "image/png")], Out(id = "a"))
    assert [c.type for c in result.content] == ["text", "image"]
    assert json.loads(result.content[0].text) == {"id": "a"}
    assert result.structured_content == {"id": "a"}


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"path": "/a", "data_url": "data:image/png;base64,AA=="},
        {"gallery_id": "a", "path": "/a"},
    ],
)
def test_an_image_input_takes_exactly_one_source(fields):
    with pytest.raises(ValidationError):
        ImageInput(**fields)


def test_an_oversized_file_is_refused_before_it_is_read(file_spy, tmp_path):
    path = tmp_path / "mcp-input-big.png"
    path.write_bytes(PNG + b"\x00" * 2048)
    with pytest.raises(ToolError, match = "larger than"):
        inputs.read_local(make_caller(), str(path), max_bytes = 1024)
    assert {kind for kind, _ in file_spy} == {"stat"}


def test_oversized_inline_data_is_refused_before_decoding(monkeypatch):
    decoded = []
    monkeypatch.setattr(inputs.base64, "b64decode", lambda *a, **k: decoded.append(a) or b"")
    with pytest.raises(ToolError, match = "larger than"):
        inputs.decode_base64("A" * 2000, 1024, "data_url")
    assert decoded == []


def test_media_urls_keep_a_root_path_prefix():
    prefixed = make_caller(public_base = "http://127.0.0.1:8888/studio")
    assert (
        media.public_url(prefixed, "/v1/audio/gallery/c1/file")
        == "http://127.0.0.1:8888/studio/v1/audio/gallery/c1/file"
    )


def test_gallery_paths_quote_the_id():
    assert media.image_gallery_path("a/b") == "/api/inference/images/gallery/a%2Fb/file"
    assert (
        media.image_gallery_path("a", thumb = True)
        == "/api/inference/images/gallery/a/file?thumb=1024"
    )
    assert media.audio_gallery_path("c 1") == "/v1/audio/gallery/c%201/file"
