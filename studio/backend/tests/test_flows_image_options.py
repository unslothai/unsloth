# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from core.flows import compiler
from core.flows.nodes import _common
from core.flows.nodes import image as image_nodes
from core.flows.registry import FlowNodeError
from core.flows.specs import SPECS_BY_TYPE


def _png(
    size,
    color = (200, 30, 30),
    mode = "RGB",
) -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, "PNG")
    return out.getvalue()


class FakeCtx:
    """Records the image route body; gallery media resolve to the PNG bytes in `files`."""

    def __init__(
        self,
        files = None,
        segment_mask = None,
    ):
        self.files = files or {}
        self.segment_mask = segment_mask
        self.bodies = []
        self.logs = []

    async def call(
        self,
        method,
        path,
        json = None,
    ):
        if method == "GET" and path.startswith("/api/inference/images/gallery/"):
            return self.files[path.split("/")[-2]]
        if path == "/api/inference/images/generate":
            self.bodies.append(json)
            return {"images": [{"id": "out1"}]}
        if path == "/api/inference/images/segment":
            return {"mask": "data:image/png;base64," + base64.b64encode(self.segment_mask).decode()}
        return {"active": False}

    def on_cancel(self, fn):
        pass

    def progress(self, *a, **k):
        pass

    def log(
        self,
        message,
        level = "info",
    ):
        self.logs.append((level, message))


@pytest.fixture(autouse = True)
def _no_model_loads(monkeypatch):
    async def ensure(ctx, kind, choice):
        return None

    monkeypatch.setattr(_common, "ensure_model", ensure)


def _media(media_id):
    return {"kind": "image", "ref": f"image:{media_id}"}


def _node_defaults() -> dict:
    params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["image.generate"]["params"]}
    return {k: params[k] for k in image_nodes.OPTION_PARAMS}


def _run(ctx, **kwargs):
    return asyncio.run(_common.generate_image(ctx, model = None, prompt = "a cat", seed = 3, **kwargs))


def test_unset_options_post_the_same_body_as_before():
    before, after = FakeCtx(), FakeCtx()
    _run(before)
    _run(after, mask = None, control = None, options = _node_defaults())
    assert after.bodies == before.bodies
    assert set(before.bodies[0]) == {"prompt", "width", "height", "seed"}


def test_set_options_pass_through_to_the_route():
    ctx = FakeCtx()
    _run(
        ctx,
        options = {
            **_node_defaults(),
            "loras": [{"id": " a/lora ", "weight": 3}, {"id": "", "weight": 1}, {"id": "b"}],
            "sampler": "dpmpp_2m",
            "schedule_type": "karras",
            "clip_skip": 2,
            "cfg_rescale": 0.7,
            "shift": 3.0,
            "hires_scale": 2,
            "hires_denoise": 0.35,
            "hires_steps": 12,
            "hires_upscaler": "lanczos",
        },
    )
    body = ctx.bodies[0]
    assert body["loras"] == [{"id": "a/lora", "weight": 2.0}, {"id": "b", "weight": 1.0}]
    assert body["sampler"] == "dpmpp_2m" and body["schedule_type"] == "karras"
    assert body["clip_skip"] == 2 and body["cfg_rescale"] == 0.7 and body["shift"] == 3.0
    assert body["hires"] == {"scale": 2.0, "denoise": 0.35, "steps": 12, "upscaler": "lanczos"}


def test_mask_is_resized_to_the_source_with_a_warning():
    ctx = FakeCtx(files = {"src": _png((64, 48)), "m": _png((32, 24), 255, "L")})
    _run(
        ctx,
        init_image = _media("src"),
        mask = _media("m"),
        options = {"mask_blur": 4, "only_masked": True, "inpaint_padding": 16},
    )
    body = ctx.bodies[0]
    mask = Image.open(io.BytesIO(base64.b64decode(body["mask_image"].split(",", 1)[1])))
    assert mask.size == (64, 48) and mask.mode == "L"
    assert (
        body["mask_blur"] == 4
        and body["inpaint_only_masked"] is True
        and body["inpaint_padding"] == 16
    )
    assert any(level == "warning" and "resized" in msg for level, msg in ctx.logs)


def test_matching_mask_is_not_warned_about():
    ctx = FakeCtx(files = {"src": _png((64, 48)), "m": _png((64, 48), 255, "L")})
    _run(ctx, init_image = _media("src"), mask = _media("m"))
    assert not ctx.logs


def test_mask_without_a_source_image_is_refused():
    with pytest.raises(FlowNodeError, match = "source image"):
        _run(FakeCtx(files = {"m": _png((8, 8), 255, "L")}), mask = _media("m"))


def test_controlnet_body():
    ctx = FakeCtx(files = {"c": _png((16, 16))})
    _run(
        ctx,
        control = {
            "id": "flux-union-pro",
            "image": _media("c"),
            "control_type": "canny",
            "strength": 0.7,
            "guidance_start": 0.0,
            "guidance_end": 0.8,
        },
    )
    cn = ctx.bodies[0]["controlnet"]
    assert cn["id"] == "flux-union-pro" and cn["control_type"] == "canny"
    assert (
        cn["strength"] == 0.7
        and cn["guidance_end"] == 0.8
        and cn["image"].startswith("data:image/png")
    )


def test_control_spec_needs_both_halves():
    ctx = SimpleNamespace(inputs = {}, params = {"controlnet": "x"})
    with pytest.raises(FlowNodeError, match = "control image"):
        image_nodes.control_spec(ctx)
    ctx = SimpleNamespace(inputs = {"control_image": _media("c")}, params = {})
    with pytest.raises(FlowNodeError, match = "Pick a ControlNet"):
        image_nodes.control_spec(ctx)
    ctx = SimpleNamespace(inputs = {}, params = {})
    assert image_nodes.control_spec(ctx) is None


def test_pad_geometry_and_mask():
    image_png, mask_png = image_nodes.pad_canvas(_png((64, 48)), 10, 0, 20, 5, 0)
    im, mask = Image.open(io.BytesIO(image_png)), Image.open(io.BytesIO(mask_png))
    assert im.size == mask.size == (94, 53)
    assert mask.getpixel((0, 0)) == 255  # new border is painted
    assert mask.getpixel((10, 0)) == 0 and mask.getpixel((73, 47)) == 0  # the original is kept
    assert mask.getpixel((74, 10)) == 255 and mask.getpixel((40, 52)) == 255
    assert im.getpixel((40, 20))[:3] == (200, 30, 30)


def test_pad_feather_repaints_the_old_edge():
    _, mask_png = image_nodes.pad_canvas(_png((64, 48)), 10, 10, 10, 10, 4)
    mask = Image.open(io.BytesIO(mask_png))
    assert mask.getpixel((11, 11)) == 255  # inside the old edge band
    assert mask.getpixel((42, 34)) == 0  # the middle stays


def test_segment_returns_a_source_sized_mask():
    ctx = FakeCtx(files = {"src": _png((40, 30))}, segment_mask = _png((20, 15), 255, "L"))
    png = asyncio.run(_common.segment(ctx, _media("src"), "the face"))
    mask = Image.open(io.BytesIO(png))
    assert mask.size == (40, 30) and mask.mode == "L"


def test_segment_node_grow_and_invert():
    half = Image.new("L", (20, 20), 0)
    half.paste(255, (0, 0, 10, 20))
    out = io.BytesIO()
    half.save(out, "PNG")
    saved = {}

    async def segment(image, subject):
        return out.getvalue()

    async def save_asset(data, mime, kind):
        saved["png"] = data
        return {"kind": kind, "ref": "sha256:x"}

    ctx = SimpleNamespace(
        inputs = {"image": _media("src")},
        params = {"subject": "face", "grow": 2, "invert": True},
        segment = segment,
        save_asset = save_asset,
    )
    result = asyncio.run(image_nodes.segment(ctx))
    mask = Image.open(io.BytesIO(saved["png"]))
    assert result["mask"]["ref"] == "sha256:x"
    assert mask.getpixel((11, 5)) == 0  # grown by 2 px, then inverted
    assert mask.getpixel((13, 5)) == 255


def test_mask_and_image_ports_connect_both_ways():
    assert compiler.port_compatible({"type": "mask"}, {"type": "image"})
    assert compiler.port_compatible({"type": "image"}, {"type": "mask"})
    assert not compiler.port_compatible({"type": "mask"}, {"type": "audio"})


def test_grid_labels_add_headers_only_when_given():
    frames = [_png((32, 32)) for _ in range(4)]
    kw = dict(columns = 2, gap = 8, background = "#ffffff", caption_style = "none", cell = 64)
    plain = Image.open(io.BytesIO(image_nodes.compose_grid(frames, [], **kw)))
    labelled = Image.open(
        io.BytesIO(
            image_nodes.compose_grid(
                frames, [], row_labels = ["euler", "dpmpp"], column_labels = ["cfg 3", "cfg 7"], **kw
            )
        )
    )
    assert plain.size == (2 * 64 + 3 * 8, 2 * 64 + 3 * 8)
    assert labelled.width > plain.width and labelled.height > plain.height


def _clip(size, frames, color) -> bytes:
    import av
    import numpy as np

    out = io.BytesIO()
    with av.open(out, mode = "w", format = "mp4") as dst:
        stream = dst.add_stream("libx264", rate = 12)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for _ in range(frames):
            arr = np.full((size[1], size[0], 3), color, dtype = np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format = "rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    return out.getvalue()


def test_join_videos_keeps_order_size_and_every_frame():
    import av

    from core.flows.nodes.video import concat_clips

    joined = concat_clips([_clip((64, 48), 5, (250, 0, 0)), _clip((32, 32), 4, (0, 0, 250))])
    with av.open(io.BytesIO(joined)) as src:
        frames = [f.to_ndarray(format = "rgb24") for f in src.decode(src.streams.video[0])]
    assert len(frames) == 9
    assert frames[0].shape == frames[-1].shape == (48, 64, 3)
    assert frames[0][24, 32, 0] > 200 and frames[-1][24, 32, 2] > 200
