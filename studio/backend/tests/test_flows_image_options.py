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


def test_pad_feather_leaves_sides_that_did_not_grow():
    # Only the left and right grew: the top and bottom rows of the original must stay untouched.
    _, mask_png = image_nodes.pad_canvas(_png((64, 48)), 20, 0, 20, 0, 6)
    mask = Image.open(io.BytesIO(mask_png))
    assert mask.size == (104, 48)
    assert mask.getpixel((52, 0)) == 0 and mask.getpixel((52, 47)) == 0
    assert mask.getpixel((22, 24)) == 255  # the left seam band is repainted
    assert mask.getpixel((52, 24)) == 0


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


def test_grid_row_labels_wrap_between_words():
    font = image_nodes._font(max(14, 384 // 18))
    assert image_nodes._wrap_label("watercolor", 128, font) == "watercolor"
    text = "film photo of a red fox"
    assert image_nodes._wrap_label(text, 128, font).split() == text.split()


def test_grow_mask_matches_max_filter_and_stays_fast():
    import time

    from PIL import ImageFilter

    noise = Image.effect_noise((61, 47), 80).convert("L").point(lambda v: 255 if v > 235 else 0)
    for radius in (1, 2, 5, 12, 64):
        expected = noise.filter(ImageFilter.MaxFilter(2 * radius + 1))
        assert image_nodes.grow_mask(noise, radius).tobytes() == expected.tobytes()
    started = time.monotonic()
    image_nodes.grow_mask(Image.new("L", (2048, 2048), 0), 64)
    assert time.monotonic() - started < 3  # a 129 px MaxFilter here takes minutes


def test_video_rejects_references_with_a_last_frame_before_calling_the_route():
    from core.flows.nodes import video as video_nodes

    calls = []

    async def ensure_model(*args, **kwargs):
        return None

    async def call(*args, **kwargs):
        calls.append(args)
        return {}

    ctx = SimpleNamespace(
        inputs = {"reference": [_media("a")], "last_frame": _media("b")},
        params = {"prompt": "a fox"},
        flow_params = {},
        seed = 1,
        ensure_model = ensure_model,
        call = call,
    )
    with pytest.raises(FlowNodeError, match = "keyframes"):
        asyncio.run(video_nodes.generate(ctx))
    assert calls == []


def _phone_jpeg(size = (60, 40)) -> bytes:
    # Stored landscape, EXIF orientation 6: shown (and processed by Studio's routes) as portrait.
    exif = Image.Exif()
    exif[0x0112] = 6
    out = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(out, "JPEG", exif = exif.tobytes())
    return out.getvalue()


def test_masks_and_canvas_follow_the_exif_upright_image():
    photo = _phone_jpeg()
    assert _common.upright_size(photo) == (40, 60)
    ctx = FakeCtx(files = {"src": photo}, segment_mask = _png((40, 60), 255, "L"))
    mask = Image.open(io.BytesIO(asyncio.run(_common.segment(ctx, _media("src"), "the face"))))
    assert mask.size == (40, 60)
    image_png, _ = image_nodes.pad_canvas(photo, 10, 0, 0, 0, 0)
    assert Image.open(io.BytesIO(image_png)).size == (50, 60)


def test_find_mask_downloads_the_segmenter_on_first_use(monkeypatch):
    monkeypatch.setattr(_common.asyncio, "sleep", _no_sleep)
    state = {"downloaded": False, "downloading": False}
    calls = []

    class SegCtx(FakeCtx):
        async def call(
            self,
            method,
            path,
            json = None,
        ):
            calls.append((method, path))
            if path.endswith("/segment/status"):
                body = {"supported": True, "downloaded": state["downloaded"]}
                if state["downloading"]:
                    # The second poll sees the finished download.
                    state.update(downloaded = True, downloading = False)
                    body["download"] = {"downloading": True, "bytes_done": 5, "bytes_total": 10}
                return body
            if path.endswith("/segment/download"):
                state["downloading"] = True
                return {}
            if path == "/api/inference/images/segment" and not state["downloaded"]:
                raise FlowNodeError("The segmenter is not downloaded")
            return await super().call(method, path, json)

    ctx = SegCtx(files = {"src": _png((40, 30))}, segment_mask = _png((40, 30), 255, "L"))
    asyncio.run(_common.segment(ctx, _media("src"), "the face"))
    assert ("POST", "/api/inference/images/segment/download") in calls
    assert calls[-1] == ("POST", "/api/inference/images/segment")


def test_find_mask_explains_an_install_without_sam3():
    class NoSam(FakeCtx):
        async def call(
            self,
            method,
            path,
            json = None,
        ):
            if path.endswith("/segment/status"):
                return {
                    "supported": False,
                    "downloaded": False,
                    "reason": "Mask from text needs transformers 5",
                }
            return await super().call(method, path, json)

    ctx = NoSam(files = {"src": _png((40, 30))}, segment_mask = _png((40, 30), 255, "L"))
    with pytest.raises(FlowNodeError, match = "needs transformers 5"):
        asyncio.run(_common.segment(ctx, _media("src"), "the face"))


async def _no_sleep(*_a, **_k):
    return None


def test_generate_refuses_a_mask_with_reference_images_before_loading():
    called = []

    async def generate_image(**kwargs):
        called.append(kwargs)
        return [{"kind": "image", "ref": "x"}]

    ctx = SimpleNamespace(
        inputs = {"reference": [_media("a")], "mask": _media("m")},
        params = {"prompt": "a fox"},
        flow_params = {},
        seed = 1,
        generate_image = generate_image,
    )
    with pytest.raises(FlowNodeError, match = "mask or reference"):
        asyncio.run(image_nodes.generate(ctx))
    assert called == []


def test_a_mask_without_a_source_fails_before_the_model_loads(monkeypatch):
    loads = []

    async def ensure(ctx, kind, choice):
        loads.append(kind)

    monkeypatch.setattr(_common, "ensure_model", ensure)
    with pytest.raises(FlowNodeError, match = "needs a source image"):
        asyncio.run(_common.generate_image(FakeCtx(), model = None, prompt = "a fox", mask = _media("m")))
    assert loads == []


def test_find_mask_tells_a_managed_account_who_can_download_the_segmenter():
    class Managed(FakeCtx):
        async def call(
            self,
            method,
            path,
            json = None,
        ):
            if path.endswith("/segment/status"):
                return {"supported": True, "downloaded": False}
            if path.endswith("/segment/download"):
                raise FlowNodeError("Only the installation owner can do this")
            return await super().call(method, path, json)

    ctx = Managed(files = {"src": _png((40, 30))}, segment_mask = _png((40, 30), 255, "L"))
    with pytest.raises(FlowNodeError, match = "Ask them to run Find Mask once"):
        asyncio.run(_common.segment(ctx, _media("src"), "the face"))


def _control(media_id = "c"):
    return {"id": "flux-union-pro", "image": _media(media_id), "strength": 0.7}


class _StatusCtx(FakeCtx):
    def __init__(self, status, **kw):
        super().__init__(**kw)
        self.status = status

    async def call(
        self,
        method,
        path,
        json = None,
    ):
        if path == _common.IMAGE_STATUS:
            return self.status
        return await super().call(method, path, json)


def _media_loads(monkeypatch):
    from studio_mcp import loading

    loads = []

    async def fake_load_media(caller, ctx, **kwargs):
        loads.append(kwargs)

    monkeypatch.setattr(loading, "load_media", fake_load_media)
    monkeypatch.setattr(_common, "_caller", lambda ctx: None)
    return loads


FLUX1 = {"repo": "black-forest-labs/FLUX.1-dev"}


def test_controlnet_loads_the_picked_model_unquantized(monkeypatch):
    loads = _media_loads(monkeypatch)
    # Resident at an auto-picked int8: ControlNet cannot run on it, so it is reloaded in bf16.
    status = {"loaded": True, "repo_id": FLUX1["repo"], "supports_controlnet": False}
    ctx = _StatusCtx(status, files = {"c": _png((16, 16))})
    asyncio.run(_common.generate_image(ctx, model = FLUX1, prompt = "a fox", control = _control()))
    assert len(loads) == 1 and loads[0]["knobs"] == {"transformer_quant": "off"}
    assert loads[0]["model"] == FLUX1["repo"] and loads[0]["kind"] == "image"
    # Not loaded at all: the first load is already the unquantized one.
    ctx = _StatusCtx({"loaded": False}, files = {"c": _png((16, 16))})
    asyncio.run(_common.generate_image(ctx, model = FLUX1, prompt = "a fox", control = _control()))
    assert len(loads) == 2 and loads[1]["knobs"] == {"transformer_quant": "off"}


def test_controlnet_keeps_a_model_that_already_runs_it(monkeypatch):
    loads = _media_loads(monkeypatch)
    status = {"loaded": True, "repo_id": FLUX1["repo"], "supports_controlnet": True}
    ctx = _StatusCtx(status, files = {"c": _png((16, 16))})
    asyncio.run(_common.generate_image(ctx, model = FLUX1, prompt = "a fox", control = _control()))
    # Without a control image nothing is checked or reloaded either.
    asyncio.run(_common.generate_image(ctx, model = FLUX1, prompt = "a fox"))
    assert loads == []


def test_controlnet_on_an_unpicked_quantized_model_says_how_to_fix_it(monkeypatch):
    loads = _media_loads(monkeypatch)
    ctx = _StatusCtx({"loaded": True, "repo_id": "x/y", "supports_controlnet": False})
    with pytest.raises(FlowNodeError, match = "load it in bf16 on the Images page"):
        asyncio.run(_common.generate_image(ctx, model = None, prompt = "a fox", control = _control()))
    assert loads == []


def test_controlnet_at_zero_strength_or_with_an_init_image_never_reloads(monkeypatch):
    loads = _media_loads(monkeypatch)
    status = {"loaded": True, "repo_id": FLUX1["repo"], "supports_controlnet": False}
    ctx = _StatusCtx(status, files = {"c": _png((16, 16)), "src": _png((16, 16))})
    # Strength 0 is ControlNet off: no bf16 reload, and the request carries no control.
    asyncio.run(
        _common.generate_image(
            ctx, model = FLUX1, prompt = "a fox", control = {**_control(), "strength": 0.0}
        )
    )
    # An Init image with ControlNet is refused before anything loads.
    with pytest.raises(FlowNodeError, match = "plain text-to-image"):
        asyncio.run(
            _common.generate_image(
                ctx, model = FLUX1, prompt = "a fox", init_image = _media("src"), control = _control()
            )
        )
    assert loads == []


KREA = {"repo": "krea/Krea-2-Turbo"}
WATERCOLOR = [{"id": "krea/Krea-2-LoRA-softwatercolor", "weight": 1.0}]


def test_loras_on_a_quantized_build_reload_it_with_them_baked_in(monkeypatch):
    loads = _media_loads(monkeypatch)
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    status = {
        "loaded": True,
        "repo_id": KREA["repo"],
        "resolved": {"transformer_quant": {"value": "int8"}},
        "baked_loras": [],
    }
    ctx = _StatusCtx(status)
    # A plain render keeps the build as it is.
    asyncio.run(_common.generate_image(ctx, model = KREA, prompt = "a lighthouse"))
    assert loads == []
    # Studio refuses new adapters on an int8 build at generate time, so the node reloads with them baked in, once.
    asyncio.run(
        _common.generate_image(
            ctx, model = KREA, prompt = "a lighthouse", options = {"loras": WATERCOLOR}
        )
    )
    asyncio.run(
        _common.generate_image(
            ctx, model = KREA, prompt = "a lighthouse", options = {"loras": WATERCOLOR}
        )
    )
    assert len(loads) == 1
    assert loads[0]["model"] == KREA["repo"] and loads[0]["knobs"] == {"loras": WATERCOLOR}
    # Another set needs another bake.
    other = [{"id": "krea/Krea-2-LoRA-ink", "weight": 0.8}]
    asyncio.run(
        _common.generate_image(ctx, model = KREA, prompt = "a lighthouse", options = {"loras": other})
    )
    assert len(loads) == 2 and loads[1]["knobs"] == {"loras": other}


def test_loras_on_a_bf16_build_apply_at_generate_time(monkeypatch):
    loads = _media_loads(monkeypatch)
    monkeypatch.setattr(_common, "_BAKED_LORAS", {})
    status = {
        "loaded": True,
        "repo_id": KREA["repo"],
        "resolved": {"transformer_quant": {"value": "off"}},
    }
    ctx = _StatusCtx(status)
    asyncio.run(
        _common.generate_image(
            ctx, model = KREA, prompt = "a lighthouse", options = {"loras": WATERCOLOR}
        )
    )
    assert loads == []


def test_video_stop_cancels_only_its_own_render():
    from core.flows.nodes import video as video_nodes

    calls = []
    hooks = []

    async def ensure_model(*args, **kwargs):
        return None

    async def call(
        method,
        path,
        json = None,
    ):
        calls.append((method, path, json))
        if path == "/v1/videos":
            return {"id": "vid1"}
        if method == "GET":
            # The first poll: Stop is pressed while it renders.
            await hooks[0]()
            return {"status": "completed"}
        return {}

    ctx = SimpleNamespace(
        inputs = {},
        params = {"prompt": "a fox"},
        flow_params = {},
        seed = 1,
        ensure_model = ensure_model,
        call = call,
        on_cancel = hooks.append,
        progress = lambda *a, **k: None,
    )
    video_nodes.POLL_S, poll = 0, video_nodes.POLL_S
    try:
        asyncio.run(video_nodes.generate(ctx))
    finally:
        video_nodes.POLL_S = poll
    cancel = [c for c in calls if c[1] == "/api/inference/video/generate/cancel"]
    assert cancel == [("POST", "/api/inference/video/generate/cancel", {"video_id": "vid1"})]
