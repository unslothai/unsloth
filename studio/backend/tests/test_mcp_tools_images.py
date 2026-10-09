# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import json

from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from studio_mcp import loading
from studio_mcp.media import INLINE_CAP

from .mcp_harness import call_tool, fake_studio, served

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def _image(
    image_id,
    side,
    seed = 7,
):
    return {
        "id": image_id,
        "url": f"/api/inference/images/gallery/{image_id}/file",
        "prompt": "a red fox",
        "width": side,
        "height": side,
        "steps": 4,
        "guidance": 0.0,
        "seed": seed,
        "model": "black-forest-labs/FLUX.1-schnell",
    }


GENERATED = {"images": [_image("img-small", 512, 1), _image("img-large", 1024, 2)]}


def _png(request, body):
    return Response(PNG, media_type = "image/png")


def _webp(request, body):
    assert request.query_params.get("thumb") == "1024"
    return Response(WEBP, media_type = "image/webp")


PAYLOADS = {
    ("POST", "/api/inference/images/generate"): GENERATED,
    ("GET", "/api/inference/images/generate-progress"): {
        "active": True,
        "step": 2,
        "total_steps": 4,
        "fraction": 0.5,
    },
}


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes[("GET", "/api/inference/images/gallery/img-small/file")] = _png
    routes[("GET", "/api/inference/images/gallery/img-large/file")] = _webp
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, args, **client):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **client) as http:
        return call_tool(http, "generate_image", args)


def test_txt2img_returns_ids_urls_and_inline_images(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        {"prompt": "a red fox", "width": 512, "steps": 4},
        base_url = "http://192.168.1.20:8888",
    )
    assert result["structuredContent"] == {
        "images": [
            {
                "id": "img-small",
                "url": "http://192.168.1.20:8888/api/inference/images/gallery/img-small/file",
                "width": 512,
                "height": 512,
                "seed": 1,
            },
            {
                "id": "img-large",
                "url": "http://192.168.1.20:8888/api/inference/images/gallery/img-large/file",
                "width": 1024,
                "height": 1024,
                "seed": 2,
            },
        ]
    }
    content = result["content"]
    assert json.loads(content[0]["text"]) == result["structuredContent"]
    assert [c["type"] for c in content[1:]] == ["image", "image", "resource_link"]
    assert base64.b64decode(content[1]["data"]) == PNG
    assert content[1]["mimeType"] == "image/png"
    assert base64.b64decode(content[2]["data"]) == WEBP
    assert content[2]["mimeType"] == "image/webp"
    assert (
        content[3]["uri"] == "http://192.168.1.20:8888/api/inference/images/gallery/img-large/file"
    )
    (_m, _p, _h, body) = next(
        c for c in studio.state.calls if c[1] == "/api/inference/images/generate"
    )
    assert json.loads(body) == {"prompt": "a red fox", "width": 512, "steps": 4}


def test_a_large_image_is_never_fetched_in_full(monkeypatch):
    queries = []

    def large(request, body):
        queries.append(str(request.url.query))
        return Response(WEBP, media_type = "image/webp")

    studio = _studio({("GET", "/api/inference/images/gallery/img-large/file"): large})
    _call(monkeypatch, studio, {"prompt": "x"})
    assert queries == ["thumb=1024"]


def test_a_thumbnail_studio_could_not_make_keeps_its_png_type(monkeypatch):
    # The gallery route answers a failed thumbnail with the original PNG.
    def fallback(request, body):
        return Response(PNG, media_type = "image/png")

    studio = _studio({("GET", "/api/inference/images/gallery/img-large/file"): fallback})
    result = _call(monkeypatch, studio, {"prompt": "x"})
    previews = [c for c in result["content"] if c["type"] == "image"]
    assert previews[-1]["mimeType"] == "image/png"


def test_a_full_image_over_the_cap_falls_back_to_the_thumbnail(monkeypatch):
    def huge(request, body):
        if request.query_params.get("thumb"):
            return Response(WEBP, media_type = "image/webp")
        return Response(PNG + b"\x00" * INLINE_CAP, media_type = "image/png")

    studio = _studio(
        {
            ("POST", "/api/inference/images/generate"): lambda request, body: {
                "images": [_image("img-small", 512)]
            },
            ("GET", "/api/inference/images/gallery/img-small/file"): huge,
        }
    )
    result = _call(monkeypatch, studio, {"prompt": "x"})
    assert [c["type"] for c in result["content"][1:]] == ["image", "resource_link"]
    assert result["content"][1]["mimeType"] == "image/webp"


def test_gallery_files_are_fetched_as_the_caller(monkeypatch):
    studio = _studio()
    _call(monkeypatch, studio, {"prompt": "x"})
    fetches = [c for c in studio.state.calls if "/gallery/" in c[1]]
    assert len(fetches) == 2
    for _m, _p, headers, _b in fetches:
        assert headers["authorization"] == "Bearer sk-unsloth-test"
        assert headers["host"] == "unsloth-mcp.invalid"


def test_no_loaded_model_says_to_load_one(monkeypatch):
    def not_loaded(request, body):
        return JSONResponse({"detail": "No diffusion model is loaded."}, status_code = 409)

    studio = _studio({("POST", "/api/inference/images/generate"): not_loaded})
    result = _call(monkeypatch, studio, {"prompt": "x"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "No diffusion model is loaded. (HTTP 409) Load an image model with load_model(kind='image') first."
    )


def test_a_cancelled_run_is_not_told_to_load(monkeypatch):
    def cancelled(request, body):
        return JSONResponse({"detail": "Diffusion generation was cancelled."}, status_code = 409)

    studio = _studio({("POST", "/api/inference/images/generate"): cancelled})
    result = _call(monkeypatch, studio, {"prompt": "x"})
    assert result["content"][0]["text"] == "Diffusion generation was cancelled. (HTTP 409)"


def test_progress_is_reported_while_generating(monkeypatch):
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.02)

    async def slow(request, body):
        await asyncio.sleep(0.2)
        return GENERATED

    studio = _studio({("POST", "/api/inference/images/generate"): slow})
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "generate_image",
            "arguments": {"prompt": "x"},
            "_meta": {"progressToken": "g"},
        },
    }
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        response = http.post(
            "/mcp/",
            json = request,
            headers = {
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer sk-unsloth-test",
            },
        )
    messages = [
        json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")
    ]
    progress = [m["params"] for m in messages if m.get("method") == "notifications/progress"]
    assert progress and progress[0]["progress"] == 0.5
    assert progress[0]["message"] == "Step 2 of 4"


def test_generate_image_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["generate_image"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema["properties"]["images"]["type"] == "array"
    assert tool.output_schema["additionalProperties"] is False


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def _data_url(data, mime):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _sent_body(studio):
    return json.loads(
        next(c for c in studio.state.calls if c[1] == "/api/inference/images/generate")[3]
    )


def test_edit_inputs_land_in_their_body_fields(monkeypatch):
    studio = _studio({("GET", "/api/inference/images/gallery/src-1/file"): _png})
    args = {
        "prompt": "make it night",
        "init_image": {"gallery_id": "src-1"},
        "mask_image": {"data_url": _data_url(PNG, "image/png")},
        "reference_images": [{"data_url": _data_url(JPEG, "image/jpeg")}],
        "workflow": "edit",
        "strength": 0.6,
        "allow_oversized": True,
    }
    result = _call(monkeypatch, studio, args)
    assert result["isError"] is False
    assert _sent_body(studio) == {
        "prompt": "make it night",
        "init_image": _data_url(PNG, "image/png"),
        "mask_image": _data_url(PNG, "image/png"),
        "reference_images": [_data_url(JPEG, "image/jpeg")],
        "workflow": "edit",
        "strength": 0.6,
        "allow_oversized": True,
    }


def test_upscale_sends_the_factor_with_its_source(monkeypatch):
    studio = _studio()
    _call(
        monkeypatch,
        studio,
        {
            "prompt": "sharper",
            "init_image": {"data_url": _data_url(PNG, "image/png")},
            "upscale": 2,
        },
    )
    body = _sent_body(studio)
    assert body["upscale"] == 2
    assert body["init_image"] == _data_url(PNG, "image/png")
    assert "allow_oversized" not in body


def test_a_mask_without_a_source_is_refused(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        {"prompt": "x", "mask_image": {"data_url": _data_url(PNG, "image/png")}},
    )
    assert result["isError"] is True
    assert "mask_image needs init_image" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_upscale_without_a_source_is_refused(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, {"prompt": "x", "upscale": 2})
    assert result["isError"] is True
    assert studio.state.calls == []


def test_more_than_nine_references_are_refused(monkeypatch):
    studio = _studio()
    refs = [{"data_url": _data_url(PNG, "image/png")}] * 10
    result = _call(
        monkeypatch, studio, {"prompt": "x", "init_image": refs[0], "reference_images": refs}
    )
    assert result["isError"] is True
    assert studio.state.calls == []


def test_a_memory_refusal_says_how_to_override(monkeypatch):
    def refused(request, body):
        return JSONResponse(
            {"detail": "2048x2048 needs about 30 GB; 12 GB is free."},
            status_code = 400,
            headers = {"X-Unsloth-Refusal": "memory-estimate"},
        )

    studio = _studio({("POST", "/api/inference/images/generate"): refused})
    result = _call(monkeypatch, studio, {"prompt": "x", "width": 2048, "height": 2048})
    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "2048x2048 needs about 30 GB; 12 GB is free. Pass allow_oversized=true to try anyway."
    )


def test_a_remote_agent_cannot_send_an_image_path(monkeypatch, tmp_path):
    from studio_mcp.inputs import PATH_REMOTE

    source = tmp_path / "mcp-input.png"
    source.write_bytes(PNG)
    opened = []
    monkeypatch.setattr(type(source), "read_bytes", lambda self: opened.append(self) or PNG)
    studio = _studio()
    remote = _call(
        monkeypatch,
        studio,
        {"prompt": "x", "init_image": {"path": str(source)}},
        base_url = "http://192.168.1.20:8888",
        client = ("192.0.2.7", 50000),
    )
    assert remote["content"][0]["text"] == PATH_REMOTE
    assert opened == []
    assert studio.state.calls == []
    local = _call(
        monkeypatch,
        studio,
        {"prompt": "x", "init_image": {"path": str(source)}},
        base_url = "http://127.0.0.1:8888",
        client = ("127.0.0.1", 50000),
    )
    assert local["isError"] is False
    assert opened == [source]
