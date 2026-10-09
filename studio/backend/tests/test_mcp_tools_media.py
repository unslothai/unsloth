# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from mcp_server import create_studio_mcp
from studio_mcp import inputs, media
from studio_mcp.caller import Caller
from studio_mcp.inputs import ImageInput
from studio_mcp.outputs import ToolOutput

from .mcp_harness import call_tool, fake_studio, served


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


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
LOCAL = {"base_url": "http://127.0.0.1:8888", "client": ("127.0.0.1", 50000)}
REMOTE = {"base_url": "http://192.168.1.20:8888", "client": ("192.0.2.7", 50000)}
DECISION = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}}
COMPLETION = {"model": "m", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


@pytest.fixture
def file_spy(monkeypatch):
    """Records every way the tool could open or inspect a file."""
    touched = []
    real_open, real_stat, real_read = builtins.open, Path.stat, Path.read_bytes

    def spy_open(file, *args, **kwargs):
        if "mcp-input" in str(file):
            touched.append(("open", str(file)))
        return real_open(file, *args, **kwargs)

    def spy_stat(self, *args, **kwargs):
        if "mcp-input" in str(self):
            touched.append(("stat", str(self)))
        return real_stat(self, *args, **kwargs)

    def spy_read(self):
        if "mcp-input" in str(self):
            touched.append(("read", str(self)))
        return real_read(self)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "stat", spy_stat)
    monkeypatch.setattr(Path, "read_bytes", spy_read)
    return touched


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "mcp-input.png"
    path.write_bytes(PNG)
    return path


PATH_CALLS = [
    (
        "chat",
        lambda path: {"prompt": "What is this?", "images": [{"path": path}]},
        "/v1/chat/completions",
    ),
    (
        "system_one",
        lambda path: {
            "state": "x",
            "questions": {"q": {"type": "noul"}},
            "images": [{"path": path}],
        },
        "/v1/systemone",
    ),
]


def _studio():
    return fake_studio(
        {
            ("POST", "/v1/chat/completions"): lambda request, body: COMPLETION,
            ("POST", "/v1/systemone"): lambda request, body: DECISION,
        }
    )


@pytest.mark.parametrize("tool,args,route", PATH_CALLS)
def test_a_local_agent_may_send_a_path(monkeypatch, file_spy, image_file, tool, args, route):
    studio = _studio()
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **LOCAL) as http:
        result = call_tool(http, tool, args(str(image_file)))
    assert result["isError"] is False, result
    assert ("read", str(image_file)) in file_spy
    assert [c[1] for c in studio.state.calls] == [route]


@pytest.mark.parametrize("tool,args,route", PATH_CALLS)
def test_a_remote_agent_may_not_and_the_file_is_never_opened(
    monkeypatch, file_spy, image_file, tool, args, route
):
    studio = _studio()
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **REMOTE) as http:
        result = call_tool(http, tool, args(str(image_file)))
    assert result["isError"] is True
    assert result["content"][0]["text"] == inputs.PATH_REMOTE
    assert file_spy == []
    assert studio.state.calls == []


def test_a_proxied_loopback_request_counts_as_remote(monkeypatch, file_spy, image_file):
    studio = _studio()
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **LOCAL) as http:
        result = call_tool(
            http,
            "chat",
            PATH_CALLS[0][1](str(image_file)),
            headers = {"X-Forwarded-For": "203.0.113.9"},
        )
    assert result["content"][0]["text"] == inputs.PATH_REMOTE
    assert file_spy == []


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
        inputs.read_local(caller(), str(path), max_bytes = 1024)
    assert {kind for kind, _ in file_spy} == {"stat"}


def test_oversized_inline_data_is_refused_before_decoding(monkeypatch):
    decoded = []
    monkeypatch.setattr(inputs.base64, "b64decode", lambda *a, **k: decoded.append(a) or b"")
    with pytest.raises(ToolError, match = "larger than"):
        inputs.decode_base64("A" * 2000, 1024, "data_url")
    assert decoded == []


def test_media_urls_keep_a_root_path_prefix():
    prefixed = caller("http://127.0.0.1:8888/studio")
    assert (
        media.public_url(prefixed, "/v1/audio/gallery/c1/file")
        == "http://127.0.0.1:8888/studio/v1/audio/gallery/c1/file"
    )
