# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import json

from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from studio_mcp.inputs import PATH_REMOTE

from .mcp_harness import call_tool, fake_studio, served

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def _job(
    video_id = "video-1",
    status = "queued",
    progress = 0,
    error = None,
):
    return {
        "id": video_id,
        "object": "video",
        "model": "Lightricks/LTX-Video",
        "status": status,
        "progress": progress,
        "created_at": 1,
        "prompt": "waves",
        "size": "704x480",
        "seconds": "4",
        "error": error,
    }


PAYLOADS = {
    ("POST", "/v1/videos"): _job(),
    ("GET", "/v1/videos"): {
        "object": "list",
        "data": [_job("video-2", "in_progress", 40), _job("video-1", "completed", 100)],
    },
    ("GET", "/v1/videos/video-1"): _job("video-1", "completed", 100),
}


def _thumbnail(request, body):
    variant = request.query_params.get("variant")
    assert variant == "thumbnail", f"the tool asked for variant={variant!r}"
    return Response(WEBP, media_type = "image/webp")


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes[("GET", "/v1/videos/video-1/content")] = _thumbnail
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, name, args, **client):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **client) as http:
        return call_tool(http, name, args)


def test_generate_video_starts_a_job_with_string_fields(monkeypatch):
    studio = _studio()
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        result = call_tool(
            http,
            "generate_video",
            {
                "prompt": "waves",
                "seconds": "4",
                "size": "704x480",
                "first_frame": {
                    "data_url": "data:image/png;base64," + base64.b64encode(PNG).decode()
                },
            },
            headers = {"X-Unsloth-HF-Token": "hf_h"},
        )
    assert result["structuredContent"] == {
        "id": "video-1",
        "status": "queued",
        "progress": 0,
        "model": "Lightricks/LTX-Video",
        "seconds": "4",
        "size": "704x480",
    }
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1] == "/v1/videos")
    assert json.loads(body) == {
        "prompt": "waves",
        "seconds": "4",
        "size": "704x480",
        "input_reference": "data:image/png;base64," + base64.b64encode(PNG).decode(),
    }
    assert headers["x-unsloth-hf-token"] == "hf_h"


def test_no_video_model_says_to_load_one(monkeypatch):
    def none(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "No video model is loaded.",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            },
            status_code = 503,
        )

    studio = _studio({("POST", "/v1/videos"): none})
    result = _call(monkeypatch, studio, "generate_video", {"prompt": "waves"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "No video model is loaded. (HTTP 503) Load a video model with load_model(kind='video') first."
    )


def test_a_completed_job_returns_its_thumbnail_and_a_link_never_the_mp4(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        "get_job",
        {"kind": "video", "id": "video-1"},
        base_url = "http://192.168.1.20:8888",
    )
    out = result["structuredContent"]
    assert out == {
        "kind": "video",
        "id": "video-1",
        "status": "completed",
        "progress_percent": 100.0,
        "error": None,
        "video": {
            "url": "http://192.168.1.20:8888/v1/videos/video-1/content",
            "thumbnail_inline": True,
        },
        "jobs": None,
    }
    content = result["content"]
    assert [c["type"] for c in content[1:]] == ["image", "resource_link"]
    assert base64.b64decode(content[1]["data"]) == WEBP
    assert content[2]["uri"] == "http://192.168.1.20:8888/v1/videos/video-1/content"
    assert content[2]["mimeType"] == "video/mp4"


def test_the_forwarder_never_asks_for_the_video_or_a_gallery_file(monkeypatch):
    seen = []

    def guard(request, body):
        seen.append(dict(request.query_params))
        return _thumbnail(request, body)

    studio = _studio({("GET", "/v1/videos/video-1/content"): guard})
    _call(monkeypatch, studio, "get_job", {"kind": "video", "id": "video-1"})
    _call(monkeypatch, studio, "get_job", {"kind": "video"})
    assert seen == [{"variant": "thumbnail"}]
    assert not [c for c in studio.state.calls if "/gallery/" in c[1]]


def test_an_unavailable_thumbnail_leaves_only_the_link(monkeypatch):
    def unavailable(request, body):
        return JSONResponse(
            {
                "error": {
                    "message": "ffmpeg is not installed",
                    "type": "api_error",
                    "param": None,
                    "code": "video_thumbnail_unavailable",
                }
            },
            status_code = 501,
        )

    studio = _studio({("GET", "/v1/videos/video-1/content"): unavailable})
    result = _call(monkeypatch, studio, "get_job", {"kind": "video", "id": "video-1"})
    assert result["isError"] is False
    assert [c["type"] for c in result["content"][1:]] == ["resource_link"]
    assert result["structuredContent"]["video"]["thumbnail_inline"] is False


def test_a_running_job_reports_progress_without_media(monkeypatch):
    studio = _studio(
        {("GET", "/v1/videos/video-2"): lambda request, body: _job("video-2", "in_progress", 40)}
    )
    result = _call(monkeypatch, studio, "get_job", {"kind": "video", "id": "video-2"})
    assert result["structuredContent"]["status"] == "in_progress"
    assert result["structuredContent"]["progress_percent"] == 40.0
    assert result["structuredContent"]["video"] is None
    assert [c["type"] for c in result["content"]] == ["text"]
    assert [c[1] for c in studio.state.calls] == ["/v1/videos/video-2"]


def test_a_failed_job_carries_its_error(monkeypatch):
    failed = _job(
        "video-3", "failed", 10, {"code": "video_generation_failed", "message": "Out of memory"}
    )
    studio = _studio({("GET", "/v1/videos/video-3"): lambda request, body: failed})
    result = _call(monkeypatch, studio, "get_job", {"kind": "video", "id": "video-3"})
    assert result["structuredContent"]["error"] == "Out of memory"


def test_without_an_id_recent_jobs_are_listed(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "get_job", {"kind": "video"})
    assert result["structuredContent"]["jobs"] == [
        {"id": "video-2", "status": "in_progress", "progress_percent": 40.0},
        {"id": "video-1", "status": "completed", "progress_percent": 100.0},
    ]
    assert [c[1] for c in studio.state.calls] == ["/v1/videos"]


def test_a_remote_agent_cannot_send_a_first_frame_path(monkeypatch, tmp_path):
    frame = tmp_path / "mcp-input.png"
    frame.write_bytes(PNG)
    opened = []
    monkeypatch.setattr(type(frame), "read_bytes", lambda self: opened.append(self) or PNG)
    studio = _studio()
    remote = _call(
        monkeypatch,
        studio,
        "generate_video",
        {"prompt": "waves", "first_frame": {"path": str(frame)}},
        base_url = "http://192.168.1.20:8888",
        client = ("192.0.2.7", 50000),
    )
    assert remote["content"][0]["text"] == PATH_REMOTE
    assert opened == [] and studio.state.calls == []
    local = _call(
        monkeypatch,
        studio,
        "generate_video",
        {"prompt": "waves", "first_frame": {"path": str(frame)}},
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
    )
    assert local["isError"] is False
    assert opened == [frame]


def test_video_annotations():
    tools = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}
    generate, job = tools["generate_video"], tools["get_job"]
    assert generate.annotations.readOnlyHint is False
    assert generate.annotations.destructiveHint is False
    assert job.annotations.readOnlyHint is True
    for tool in (generate, job):
        assert tool.annotations.openWorldHint is False
        assert tool.output_schema is not None
    kind = job.parameters["properties"]["kind"]
    assert kind.get("enum", [kind.get("const")]) == ["video"]
