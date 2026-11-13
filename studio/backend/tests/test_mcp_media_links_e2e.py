# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MCP tools against the real Unsloth Studio gallery routes: links an agent can fetch with
nothing else, a library that only shows the caller's own work, and a gallery video reused as the
next shot's first frame and as a saved voice."""

from __future__ import annotations

import base64
import io
import json
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI, Request
from PIL import Image

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import image_gallery, video_gallery
from routes import inference, video
from utils.account_context import bind_account, reset_account, run_as

from .mcp_harness import run_tool
from .test_account_media_isolation import ALICE, BOB, isolated  # noqa: F401
from .test_media_signed_links import _app
from .test_video_gallery import _real_mp4_bytes, _real_mp4_with_audio

pytest.importorskip("av")


def _studio(account) -> FastAPI:
    """The real gallery, audio and video routes as ``account``, plus stand-ins for the GPU work."""
    app = FastAPI()
    app.state.video_bodies = []

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    @app.post("/api/inference/images/generate")
    async def generate(request: Request):
        meta = {"prompt": "hero", "width": 8, "height": 8, "steps": 1, "guidance": 1.0}
        meta.update(seed = 3, created_at = 100.0)
        row = run_as(account, image_gallery.save, Image.new("RGB", (8, 8), (200, 10, 10)), meta)
        return {"images": [{"id": row["id"], "width": 8, "height": 8, "seed": 3}]}

    @app.post("/v1/videos")
    async def create_video(request: Request):
        app.state.video_bodies.append(json.loads(await request.body()))
        return {"id": "video_next", "status": "queued"}

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    for prefix in ("/api/inference", "/v1"):
        app.include_router(inference.router, prefix = prefix)
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    return app


def _clip(
    account,
    data: bytes,
    prompt: str = "clip A",
) -> str:
    meta = {
        "prompt": prompt,
        "width": 64,
        "height": 48,
        "num_frames": 8,
        "fps": 8,
        "duration_s": 1.0,
        "steps": 1,
        "guidance": 1.0,
        "seed": 7,
        "model": "test/none",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return run_as(account, video_gallery.save, data, meta)["id"]


def _fetch_bare(url: str):
    """GET a link the way a stock client would: no Authorization, no cookies, no overrides."""
    parts = urlsplit(url)
    with _app() as bare:
        return bare.get(f"{parts.path}?{parts.query}")


def test_generated_and_listed_media_download_without_the_key(monkeypatch):
    studio = _studio(ALICE)
    made, _ = run_tool(
        monkeypatch, studio, "generate_image", {"prompt": "hero", "return_media": "link"}
    )
    image = made["structuredContent"]["images"][0]
    assert [c["type"] for c in made["content"]] == ["text", "resource_link"]
    assert made["content"][1]["uri"] == image["download_url"]
    fetched = _fetch_bare(image["download_url"])
    assert fetched.status_code == 200, fetched.text
    assert Image.open(io.BytesIO(fetched.content)).getpixel((0, 0)) == (200, 10, 10)
    assert _fetch_bare(image["url"]).status_code in (401, 403)

    listed, _ = run_tool(monkeypatch, studio, "list_media", {"kind": "image"})
    rows = listed["structuredContent"]["images"]
    assert [row["id"] for row in rows] == [image["id"]]
    assert _fetch_bare(rows[0]["download_url"]).content == fetched.content
    assert "/tmp" not in json.dumps(listed) and str(image_gallery.gallery_dir()) not in json.dumps(
        listed
    )
    other, _ = run_tool(monkeypatch, _studio(BOB), "list_media", {"kind": "image"})
    assert other["structuredContent"]["images"] == []


def test_the_last_frame_of_one_clip_starts_the_next(monkeypatch):
    # Frame i is grey (i * 30): the last of 8 frames is 210.
    clip_a = _clip(ALICE, _real_mp4_bytes(frames = 8, size = (64, 48), rate = 8))
    studio = _studio(ALICE)
    result, _ = run_tool(
        monkeypatch,
        studio,
        "generate_video",
        {"prompt": "shot B", "first_frame": {"video_id": clip_a, "frame": "last"}},
    )
    assert not result.get("isError"), result
    sent = studio.state.video_bodies[0]["input_reference"]
    assert sent.startswith("data:image/png;base64,")
    frame = Image.open(io.BytesIO(base64.b64decode(sent.split(",", 1)[1]))).convert("L")
    assert abs(frame.getpixel((32, 24)) - 210) <= 12
    refused, bob = run_tool(
        monkeypatch,
        _studio(BOB),
        "generate_video",
        {"prompt": "shot B", "first_frame": {"video_id": clip_a}},
    )
    assert refused.get("isError") and bob.state.video_bodies == []


def test_a_clips_sound_becomes_a_saved_voice(monkeypatch):
    clip = _clip(ALICE, _real_mp4_with_audio(seconds = 1), prompt = "narrator take")
    studio = _studio(ALICE)
    saved, _ = run_tool(
        monkeypatch,
        studio,
        "voices",
        {"action": "save", "name": "Narrator", "source": {"video_id": clip, "end_s": 0.8}},
    )
    assert not saved.get("isError"), saved
    voice = saved["structuredContent"]["voice"]
    assert voice["name"] == "Narrator" and 0.7 <= voice["duration_s"] <= 0.9
    listed, _ = run_tool(monkeypatch, studio, "list_media", {"kind": "voice"})
    assert [row["id"] for row in listed["structuredContent"]["voices"]] == [voice["id"]]
    other, _ = run_tool(monkeypatch, _studio(BOB), "list_media", {"kind": "voice"})
    assert other["structuredContent"]["voices"] == []
