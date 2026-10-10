# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json

from fastapi.responses import Response

from studio_mcp.media import INLINE_CAP

from .mcp_harness import (
    JPEG,
    PNG,
    PNG_URL,
    REMOTE,
    WEBP,
    bodies,
    call_with_progress,
    data_url,
    fake_studio,
    fast_polls,  # noqa: F401  (fixture)
    queries,
    run_tool,
    slow,
)


def _image(image_id, side, seed):
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


GALLERY = "/api/inference/images/gallery"
GENERATE = ("POST", "/api/inference/images/generate")


def _studio(overrides = None):
    return fake_studio(
        {
            **PAYLOADS,
            ("GET", f"{GALLERY}/img-small/file"): _png,
            ("GET", f"{GALLERY}/img-large/file"): _webp,
            **(overrides or {}),
        }
    )


def _generate(
    monkeypatch,
    args,
    overrides = None,
    **client,
):
    return run_tool(monkeypatch, _studio(overrides), "generate_image", args, **client)


def test_txt2img_returns_ids_urls_and_inline_images(monkeypatch):
    result, studio = _generate(
        monkeypatch, {"prompt": "a red fox", "width": 512, "steps": 4}, **REMOTE
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
    assert bodies(studio, "/api/inference/images/generate") == [
        {"prompt": "a red fox", "width": 512, "steps": 4}
    ]


def test_a_large_image_is_never_fetched_in_full(monkeypatch):
    _result, studio = _generate(monkeypatch, {"prompt": "x"})
    assert queries(studio, f"{GALLERY}/img-large/file") == [{"thumb": "1024"}]


def test_a_thumbnail_studio_could_not_make_keeps_its_png_type(monkeypatch):
    # The gallery route answers a failed thumbnail with the original PNG.
    result, _studio = _generate(
        monkeypatch, {"prompt": "x"}, {("GET", f"{GALLERY}/img-large/file"): _png}
    )
    previews = [c for c in result["content"] if c["type"] == "image"]
    assert previews[-1]["mimeType"] == "image/png"


def test_a_full_image_over_the_cap_falls_back_to_the_thumbnail(monkeypatch):
    def huge(request, body):
        if request.query_params.get("thumb"):
            return Response(WEBP, media_type = "image/webp")
        return Response(PNG + b"\x00" * INLINE_CAP, media_type = "image/png")

    overrides = {
        GENERATE: {"images": [_image("img-small", 512, 7)]},
        ("GET", f"{GALLERY}/img-small/file"): huge,
    }
    result, _studio = _generate(monkeypatch, {"prompt": "x"}, overrides)
    assert [c["type"] for c in result["content"][1:]] == ["image", "resource_link"]
    assert result["content"][1]["mimeType"] == "image/webp"


def test_gallery_files_are_fetched_as_the_caller(monkeypatch):
    _result, studio = _generate(monkeypatch, {"prompt": "x"})
    fetches = [c for c in studio.state.calls if "/gallery/" in c[1]]
    assert len(fetches) == 2
    for _m, _p, headers, _b in fetches:
        assert headers["authorization"] == "Bearer sk-unsloth-test"
        assert headers["host"] == "unsloth-mcp.invalid"


def test_progress_is_reported_while_generating(monkeypatch, fast_polls):
    studio = _studio({GENERATE: slow(GENERATED, 0.2)})
    progress, _result = call_with_progress(
        monkeypatch, studio, "generate_image", {"prompt": "x"}, "g"
    )
    assert progress and progress[0]["progress"] == 0.5
    assert progress[0]["message"] == "Step 2 of 4"


def test_edit_inputs_land_in_their_body_fields(monkeypatch):
    args = {
        "prompt": "make it night",
        "init_image": {"gallery_id": "src-1"},
        "mask_image": PNG_URL,
        "reference_images": [{"data_url": data_url(JPEG, "image/jpeg")}],
        "workflow": "edit",
        "strength": 0.6,
        "allow_oversized": True,
    }
    result, studio = _generate(monkeypatch, args, {("GET", f"{GALLERY}/src-1/file"): _png})
    assert result["isError"] is False
    (body,) = bodies(studio, "/api/inference/images/generate")
    assert body == {
        "prompt": "make it night",
        "init_image": data_url(PNG, "image/png"),
        "mask_image": data_url(PNG, "image/png"),
        "reference_images": [data_url(JPEG, "image/jpeg")],
        "workflow": "edit",
        "strength": 0.6,
        "allow_oversized": True,
    }


def test_upscale_sends_the_factor_with_its_source(monkeypatch):
    args = {"prompt": "sharper", "init_image": PNG_URL, "upscale": 2}
    _result, studio = _generate(monkeypatch, args)
    (body,) = bodies(studio, "/api/inference/images/generate")
    assert body["upscale"] == 2
    assert body["init_image"] == data_url(PNG, "image/png")
    assert "allow_oversized" not in body
