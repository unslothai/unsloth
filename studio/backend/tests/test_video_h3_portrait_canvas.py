# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 renders the canvas a request names, portrait included: the keyframe is fitted onto it without being
squashed, and a tiny random release decodes and exports a clip of exactly that size. Landscape and the omitted-size
default stay as they were."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.inference import video_families as vf  # noqa: E402
from core.inference.video import VideoBackend  # noqa: E402

H3 = next(f for f in vf._FAMILIES if f.name == "minimax-h3")


def _picture(width: int, height: int) -> str:
    """A data URL whose top third is red and the rest green, so a squash or a turn shows in the band's extent."""
    from PIL import Image

    img = Image.new("RGB", (width, height), (30, 160, 60))
    img.paste((220, 30, 30), (0, 0, width, height // 3))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _red_band_rows(image) -> float:
    """Fraction of rows whose centre pixel is red."""
    width, height = image.size
    red = sum(1 for y in range(height) if image.getpixel((width // 2, y))[0] > 128)
    return red / height


@pytest.mark.parametrize(
    "size, source",
    [
        ((544, 960), (720, 1280)),  # portrait trained size, 9:16 start image
        ((768, 1344), (1080, 1920)),
        ((960, 544), (1280, 720)),  # landscape
        ((1024, 1024), (1000, 1000)),  # square
    ],
)
def test_i2v_keeps_the_requested_canvas_and_the_picture_upright(size, source):
    vf.validate_video_request_shape(H3, *size, num_frames = 124)
    first, last, width, height, conditioning = VideoBackend._resolve_keyframes(
        H3, None, _picture(*source), None, *size
    )
    assert (width, height) == size
    assert first.size == size
    assert last is None and conditioning == "i2va"
    # Same aspect in and out: the red third stays a third of the rows (a squash or a turn would move it).
    assert _red_band_rows(first) == pytest.approx(1 / 3, abs = 0.01)


@pytest.mark.parametrize("size", [(544, 960), (480, 864), (960, 544), (1024, 1024)])
def test_t2v_keeps_the_requested_canvas(size):
    vf.validate_video_request_shape(H3, *size, num_frames = 124)
    first, last, width, height, conditioning = VideoBackend._resolve_keyframes(
        H3, None, None, None, *size
    )
    assert (first, last) == (None, None)
    assert (width, height) == size


def test_omitted_size_keeps_the_reference_default_canvas():
    # Unchanged contract: no size means the reference node's 1344x768, whatever the keyframe.
    first, _, width, height, _ = VideoBackend._resolve_keyframes(
        H3, None, _picture(1920, 1080), None, None, None
    )
    assert (width, height) == (1344, 768) == first.size


@pytest.fixture(scope = "module")
def tiny_pipe(tmp_path_factory):
    from test_h3_train_trim import _needs_tiny_stack

    _needs_tiny_stack()
    import torch
    from _minimax_h3_video_tiny import make_tiny_h3_video_release
    from diffusers import ModularPipeline

    try:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Scheduler
    except ImportError:
        pytest.skip("diffusers has no MiniMax-H3 modular pipeline")
    root = make_tiny_h3_video_release(tmp_path_factory.mktemp("h3_portrait") / "release")
    pipe = ModularPipeline.from_pretrained(str(root))
    pipe.load_components(torch_dtype = torch.float32)
    pipe.update_components(scheduler = MiniMaxH3Scheduler(), audio_scheduler = MiniMaxH3Scheduler())
    return pipe


def test_portrait_i2v_renders_and_exports_a_portrait_clip(tiny_pipe):
    """The whole Diffusers render on CPU: latent canvas, decode and MP4 export all stay 544 wide, 960 tall."""
    import av
    from core.inference import video as video_module

    backend = VideoBackend()
    backend._state = video_module._VideoLoadState(
        pipe = tiny_pipe,
        family = H3,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base_repo = H3.base_repo,
        device = "cpu",
        dtype = "float32",
        kind = "pipeline",
        h3_task = "fl2va",
    )
    result = backend.generate(
        prompt = "a cat",
        width = 544,
        height = 960,
        num_frames = 124,
        steps = 1,
        seed = 1,
        first_frame = _picture(720, 1280),
        live_preview = False,
    )
    assert (result["width"], result["height"]) == (544, 960)
    assert result["conditioning"] == "i2va"
    with av.open(io.BytesIO(result["mp4_bytes"]), format = "mp4") as clip:
        stream = clip.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (544, 960)
