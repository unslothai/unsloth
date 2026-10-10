# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64

from fastapi.responses import Response

from .mcp_harness import (
    PNG,
    PNG_URL,
    WEBP,
    bodies,
    call_to,
    fake_studio,
    openai_error,
    queries,
    run_tool,
)


def _job(video_id, status, progress, **fields):
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
        "error": None,
        **fields,
    }


PAYLOADS = {
    ("POST", "/v1/videos"): _job("video-1", "queued", 0),
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
    return fake_studio(
        {**PAYLOADS, ("GET", "/v1/videos/video-1/content"): _thumbnail, **(overrides or {})}
    )


def _get_job(
    monkeypatch,
    args,
    overrides = None,
    **client,
):
    return run_tool(monkeypatch, _studio(overrides), "get_job", {"kind": "video", **args}, **client)


def test_generate_video_starts_a_job_with_string_fields(monkeypatch):
    args = {"prompt": "waves", "seconds": "4", "size": "704x480", "first_frame": PNG_URL}
    result, studio = run_tool(
        monkeypatch, _studio(), "generate_video", args, headers = {"X-Unsloth-HF-Token": "hf_h"}
    )
    assert result["structuredContent"] == {
        "id": "video-1",
        "status": "queued",
        "progress": 0,
        "model": "Lightricks/LTX-Video",
        "seconds": "4",
        "size": "704x480",
    }
    assert bodies(studio, "/v1/videos") == [
        {
            "prompt": "waves",
            "seconds": "4",
            "size": "704x480",
            "input_reference": "data:image/png;base64," + base64.b64encode(PNG).decode(),
        }
    ]
    assert call_to(studio, "/v1/videos")[2]["x-unsloth-hf-token"] == "hf_h"


def test_a_completed_job_returns_its_thumbnail_and_a_link_never_the_mp4(monkeypatch):
    result, _studio = _get_job(monkeypatch, {"id": "video-1"}, base_url = "http://192.168.1.20:8888")
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
        "recipe": None,
        "export": None,
        "jobs": None,
    }
    content = result["content"]
    assert [c["type"] for c in content[1:]] == ["image", "resource_link"]
    assert base64.b64decode(content[1]["data"]) == WEBP
    assert content[2]["uri"] == "http://192.168.1.20:8888/v1/videos/video-1/content"
    assert content[2]["mimeType"] == "video/mp4"


def test_the_forwarder_never_asks_for_the_video_or_a_gallery_file(monkeypatch):
    _result, studio = _get_job(monkeypatch, {"id": "video-1"})
    run_tool(monkeypatch, studio, "get_job", {"kind": "video"})
    assert queries(studio, "/v1/videos/video-1/content") == [{"variant": "thumbnail"}]
    assert not [c for c in studio.state.calls if "/gallery/" in c[1]]


def test_an_unavailable_thumbnail_leaves_only_the_link(monkeypatch):
    unavailable = openai_error(
        "ffmpeg is not installed", 501, type = "api_error", code = "video_thumbnail_unavailable"
    )
    content = {("GET", "/v1/videos/video-1/content"): unavailable}
    result, _studio = _get_job(monkeypatch, {"id": "video-1"}, content)
    assert result["isError"] is False
    assert [c["type"] for c in result["content"][1:]] == ["resource_link"]
    assert result["structuredContent"]["video"]["thumbnail_inline"] is False


def test_a_running_job_reports_progress_without_media(monkeypatch):
    running = {("GET", "/v1/videos/video-2"): _job("video-2", "in_progress", 40)}
    result, studio = _get_job(monkeypatch, {"id": "video-2"}, running)
    assert result["structuredContent"]["status"] == "in_progress"
    assert result["structuredContent"]["progress_percent"] == 40.0
    assert result["structuredContent"]["video"] is None
    assert [c["type"] for c in result["content"]] == ["text"]
    assert [c[1] for c in studio.state.calls] == ["/v1/videos/video-2"]


def test_a_failed_job_carries_its_error(monkeypatch):
    error = {"code": "video_generation_failed", "message": "Out of memory"}
    failed = _job("video-3", "failed", 10, error = error)
    result, _studio = _get_job(
        monkeypatch, {"id": "video-3"}, {("GET", "/v1/videos/video-3"): failed}
    )
    assert result["structuredContent"]["error"] == "Out of memory"


def test_without_an_id_recent_jobs_are_listed(monkeypatch):
    result, studio = _get_job(monkeypatch, {})
    assert result["structuredContent"]["jobs"] == [
        {"id": "video-2", "status": "in_progress", "progress_percent": 40.0},
        {"id": "video-1", "status": "completed", "progress_percent": 100.0},
    ]
    assert [c[1] for c in studio.state.calls] == ["/v1/videos"]
