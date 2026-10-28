# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""return_media on generate_image, generate_audio and get_job; key-free download links; transcript ids, clip lineage and gallery videos as inputs."""

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse, Response

from studio_mcp import media

from .mcp_harness import PNG, WAV, WEBP, bodies, fake_studio, make_caller, queries, run_tool
from .test_mcp_tools_audio import COMPLETE, _from_source

IMAGES = "/api/inference/images/gallery"
VIDEOS = "/api/inference/video/gallery"
CLIPS = "/v1/audio/gallery"
BASE = "http://testserver"


def _minted(base, media_id):
    return {
        ("GET", f"{base}/{media_id}/signed-url"): {
            "url": f"{base}/{media_id}/file-signed?token=t.1"
        }
    }


def _files(studio, needle = ""):
    return [c[1] for c in studio.state.calls if needle in c[1] and not c[1].endswith("/signed-url")]


IMAGE_ROUTES = {
    ("POST", "/api/inference/images/generate"): {
        "images": [{"id": "img-1", "width": 512, "height": 512}]
    },
    ("GET", f"{IMAGES}/img-1/file"): lambda request, body: Response(
        WEBP if request.query_params.get("thumb") else PNG
    ),
    **_minted(IMAGES, "img-1"),
}


# (return_media, preview_px, the content types after the JSON, the file queries made).
@pytest.mark.parametrize(
    "mode,px,types,fetched",
    [
        ("inline", None, ["image"], [{}]),
        ("preview", 256, ["image", "resource_link"], [{"thumb": "256"}]),
        ("preview", None, ["image", "resource_link"], [{"thumb": "1024"}]),
        ("link", 256, ["resource_link"], []),
    ],
)
def test_image_return_media_modes(monkeypatch, mode, px, types, fetched):
    args = {"prompt": "x", "return_media": mode, **({"preview_px": px} if px else {})}
    result, studio = run_tool(monkeypatch, IMAGE_ROUTES, "generate_image", args)
    content = result["content"]
    assert [c["type"] for c in content[1:]] == types
    assert queries(studio, f"{IMAGES}/img-1/file") == fetched
    download = f"{BASE}{IMAGES}/img-1/file-signed?token=t.1"
    assert result["structuredContent"]["images"][0]["download_url"] == download
    assert all(c["uri"] == download for c in content if c["type"] == "resource_link")


@pytest.mark.parametrize("px", [31, 1025])
def test_preview_px_out_of_range_is_refused_without_a_call(monkeypatch, px):
    for tool, args in (
        ("generate_image", {"prompt": "x"}),
        ("get_job", {"kind": "video", "id": "v"}),
    ):
        result, studio = run_tool(monkeypatch, {}, tool, {**args, "preview_px": px}, fallback = {})
        assert result["isError"] is True and studio.state.calls == []


SECOND = 48_000
AUDIO_ROUTES = {
    ("POST", "/v1/audio/run"): {
        "clips": [
            {
                "id": "clip-1",
                "duration_s": 1.0,
                "sample_rate": 24000,
                "workflow": "speak",
                "seed": 5,
            }
        ],
        "model": "m",
    },
    ("GET", f"{CLIPS}/clip-1/file"): lambda request, body: Response(
        WAV + b"\x00" * SECOND, media_type = "audio/wav"
    ),
    **_minted(CLIPS, "clip-1"),
}


@pytest.mark.parametrize("mode", [None, "preview", "link", "inline"])
def test_audio_is_a_link_unless_asked_inline(monkeypatch, mode):
    args = {"workflow": "speak", "text": "Hi", **({"return_media": mode} if mode else {})}
    result, studio = run_tool(monkeypatch, AUDIO_ROUTES, "generate_audio", args)
    size = len(json.dumps(result))
    clip = result["structuredContent"]["clips"][0]
    assert clip["download_url"] == f"{BASE}{CLIPS}/clip-1/file-signed?token=t.1"
    assert (clip["workflow"], clip["seed"], clip["voice_id"]) == ("speak", 5, None)
    if mode == "inline":
        assert [c["type"] for c in result["content"][1:]] == ["audio"] and size > SECOND
    else:
        assert [c["type"] for c in result["content"][1:]] == ["resource_link"]
        assert result["content"][1]["uri"] == clip["download_url"]
        assert _files(studio, "/gallery/") == [] and size < 2048


VIDEO_ROUTES = {
    ("GET", "/v1/videos/video-1"): {"id": "video-1", "status": "completed", "progress": 100},
    ("GET", "/v1/videos/video-1/content"): Response(WEBP, media_type = "image/webp"),
    ("GET", f"{VIDEOS}/video-1/frame"): Response(PNG, media_type = "image/png"),
    **_minted(VIDEOS, "video-1"),
}


@pytest.mark.parametrize(
    "mode,px,mime,fetched",
    [
        ("inline", None, "image/webp", ["/v1/videos/video-1/content"]),
        ("preview", 320, "image/png", [f"{VIDEOS}/video-1/frame"]),
        ("link", None, None, []),
    ],
)
def test_video_return_media_modes(monkeypatch, mode, px, mime, fetched):
    args = {
        "kind": "video",
        "id": "video-1",
        "return_media": mode,
        **({"preview_px": px} if px else {}),
    }
    result, studio = run_tool(monkeypatch, VIDEO_ROUTES, "get_job", args)
    images = [c for c in result["content"] if c["type"] == "image"]
    assert [c["mimeType"] for c in images] == ([mime] if mime else [])
    assert _files(studio, "/video") == ["/v1/videos/video-1", *fetched]
    if px:
        assert queries(studio, f"{VIDEOS}/video-1/frame") == [{"at": "first", "max_px": "320"}]
    video = result["structuredContent"]["video"]
    assert video["download_url"] == f"{BASE}{VIDEOS}/video-1/file-signed?token=t.1"
    assert video["thumbnail_inline"] is bool(mime)


def _mint(answer, kind = "image"):
    studio = fake_studio({("GET", f"{media.SIGNED_BASES[kind]}/a/signed-url"): answer})
    return asyncio.run(media.signed_url(make_caller(studio_app = studio), kind, "a")), studio


def test_signed_url_keeps_only_the_token_and_rebuilds_the_path():
    url, _studio = _mint({"url": "/elsewhere/x/file-signed?token=acct:a.1760000000.f00d"})
    assert (
        url
        == "http://127.0.0.1:8888/api/inference/images/gallery/a/file-signed?token=acct:a.1760000000.f00d"
    )
    url, studio = _mint({"url": f"{VIDEOS}/a/file-signed?token=t.1"}, "video")
    assert url.endswith(f"{VIDEOS}/a/file-signed?token=t.1")
    assert queries(studio, f"{VIDEOS}/a/signed-url") == [{"ttl_s": str(media.MCP_LINK_TTL_S)}]


@pytest.mark.parametrize(
    "answer",
    [
        JSONResponse(
            {"detail": "Image links can only be created from the Unsloth UI or with an API key."},
            403,
        ),
        {"url": "/x/file-signed?token=a/../b"},
        {"url": "/x/file-signed"},
        {"url": 5},
        Response(b"not json"),
    ],
)
def test_a_mint_that_fails_or_looks_wrong_gives_no_link(answer):
    assert _mint(answer)[0] is None


def test_a_saved_transcript_returns_its_id(monkeypatch):
    result, _studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}})
    assert result["structuredContent"]["transcript_id"] == "tr-1"
    unsaved = {**COMPLETE, "record": None}
    result, _studio = _from_source(monkeypatch, {"audio": {"clip_id": "clip-7"}}, [unsaved])
    assert result["structuredContent"]["transcript_id"] is None


@pytest.mark.parametrize("frame,at", [(None, "first"), ("last", "last"), (2.5, "2.5")])
def test_a_video_frame_is_an_image_input(monkeypatch, frame, at):
    source = {"video_id": "video-1", **({"frame": frame} if frame is not None else {})}
    args = {"prompt": "next shot", "init_image": source, "return_media": "link"}
    result, studio = run_tool(monkeypatch, {**IMAGE_ROUTES, **VIDEO_ROUTES}, "generate_image", args)
    assert result["isError"] is False, result
    assert queries(studio, f"{VIDEOS}/video-1/frame") == [{"at": at}]
    (body,) = bodies(studio, "/api/inference/images/generate")
    assert body["init_image"].startswith("data:image/png;base64,")


def test_a_video_sound_is_an_audio_input(monkeypatch):
    routes = {**AUDIO_ROUTES, ("POST", f"{VIDEOS}/video-1/audio-input"): {"id": "in-5"}}
    source = {"video_id": "video-1", "start_s": 1, "end_s": 3.5}
    args = {"workflow": "clone", "text": "Hi", "reference": source}
    result, studio = run_tool(monkeypatch, routes, "generate_audio", args)
    assert result["isError"] is False, result
    assert bodies(studio, f"{VIDEOS}/video-1/audio-input") == [{"start_s": 1.0, "end_s": 3.5}]
    assert bodies(studio, "/v1/audio/run")[0]["inputs"]["reference"] == {"input_id": "in-5"}


def test_another_accounts_video_is_the_routes_404(monkeypatch):
    missing = {
        ("POST", f"{VIDEOS}/video-9/audio-input"): JSONResponse({"detail": "Video not found."}, 404)
    }
    result, _studio = run_tool(
        monkeypatch, missing, "transcribe", {"audio": {"video_id": "video-9"}}
    )
    assert result["isError"] is True and result["content"][0]["text"].startswith("Video not found.")


@pytest.mark.parametrize(
    "tool,args",
    [
        ("generate_image", {"prompt": "x", "init_image": {"gallery_id": "g", "frame": "last"}}),
        ("generate_image", {"prompt": "x", "init_image": {"video_id": "v", "frame": -1}}),
        ("generate_image", {"prompt": "x", "init_image": {"video_id": "../v"}}),
        ("transcribe", {"audio": {"clip_id": "c", "start_s": 1}}),
        ("transcribe", {"audio": {"video_id": "v", "start_s": 3, "end_s": 2}}),
        ("transcribe", {"audio": {"video_id": "v", "clip_id": "c"}}),
    ],
)
def test_a_bad_video_input_is_refused_without_a_call(monkeypatch, tool, args):
    result, studio = run_tool(monkeypatch, {}, tool, args, fallback = {})
    assert result["isError"] is True and studio.state.calls == []
