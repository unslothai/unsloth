# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 first/last-frame (image-to-video) conditioning.

Three seams, none of which need torch, diffusers, weights or a GPU:

* the registry: which families declare keyframe conditioning, and the canvas a keyframe
  resolves to (H3's own arithmetic -- short edge 768, area capped at 768*1344, both axes
  rounded to 32);
* the backend's request handling: decode, refusal for a family that cannot condition,
  the canvas override, and what reaches the pipeline call;
* the sd-cli argv: --init-img / --end-img, present exactly when a frame was sent.
"""

from __future__ import annotations

import base64
import io

import pytest

from core.inference.sd_cpp_args import SdCppModelFiles, SdCppVideoGenParams, build_sd_cpp_video_command
from core.inference.video_families import (
    detect_video_family,
    resolve_keyframe_canvas,
    video_family_supports_keyframes,
)

PIL = pytest.importorskip("PIL.Image", reason = "keyframes are decoded through Pillow")


def _png(width: int, height: int, colour = (32, 96, 160)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format = "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _h3():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.name == "minimax-h3"
    return fam


# ── registry ────────────────────────────────────────────────────────────────


def test_only_minimax_h3_declares_keyframes():
    """Wan / LTX / HunyuanVideo must not grow a control that does nothing."""
    assert video_family_supports_keyframes(_h3()) is True
    for repo in (
        "Lightricks/LTX-2",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    ):
        fam = detect_video_family(repo)
        assert fam is not None, repo
        assert video_family_supports_keyframes(fam) is False, repo


def test_supports_keyframes_is_false_for_unknown_objects():
    """The flag is read through getattr, so a family object without the field says no."""
    assert video_family_supports_keyframes(None) is False
    assert video_family_supports_keyframes(object()) is False


def test_keyframe_workflow_is_the_component_set_not_the_run_workflow():
    """`modular_workflow` still names the text-only workflow: passing `fl2va` to
    `from_pretrained` prunes the block graph statically and breaks text-only generation, so the
    keyframe name is carried separately and only ever bounds `load_components`."""
    fam = _h3()
    assert fam.modular_workflow == "t2va"
    assert fam.keyframe_workflow == "fl2va"


# ── canvas ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "size, expected",
    [
        # 16:9 lands on the released 1344x768 canvas: the area budget IS 768*1344.
        ((1920, 1080), (1344, 768)),
        ((3840, 2160), (1344, 768)),
        # Square: short edge 768 both ways, well under the budget, so no downscale.
        ((1000, 1000), (768, 768)),
        ((512, 512), (768, 768)),
        # Portrait 9:16 is the landscape case transposed.
        ((1080, 1920), (768, 1344)),
        # 4:3 keeps the 768 short edge (768*1024 is under the cap).
        ((1024, 768), (1024, 768)),
    ],
)
def test_keyframe_canvas_matches_the_released_arithmetic(size, expected):
    assert resolve_keyframe_canvas(_h3(), *size) == expected


def test_keyframe_canvas_is_always_a_multiple_of_32_and_within_budget():
    """The rule that matters at runtime: an off-multiple canvas produces a garbled clip rather
    than an error, so every ratio the model accepts has to land on the lattice."""
    fam = _h3()
    for width in range(64, 2049, 37):
        for height in (64, 199, 512, 1080, 2048):
            ratio = width / height
            if not fam.min_aspect_ratio <= ratio <= fam.max_aspect_ratio:
                continue
            canvas_w, canvas_h = resolve_keyframe_canvas(fam, width, height)
            assert canvas_w % 32 == 0 and canvas_h % 32 == 0, (width, height)
            # Rounding happens after the cap, so the budget is honoured to within one step per axis.
            assert canvas_w * canvas_h <= fam.canvas_max_pixels + 32 * (canvas_w + canvas_h)


def test_keyframe_canvas_agrees_with_the_released_implementation():
    """The canvas rule is a checkpoint contract, so it is checked against the pipeline's own
    resolver rather than only against hand-written expectations. Skipped where diffusers does
    not ship MiniMax-H3, which is most CI runners."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as modular
    except Exception as exc:  # noqa: BLE001 -- any import failure means it is not available here
        pytest.skip(f"needs a diffusers build with MiniMax-H3: {exc}")
    fam = _h3()
    compared = 0
    for width in range(64, 3841, 149):
        for height in range(64, 2161, 101):
            if not fam.min_aspect_ratio <= width / height <= fam.max_aspect_ratio:
                continue
            reference_h, reference_w = modular.resolve_canvas_size(
                width, height, fam.resolution_multiple, fam.canvas_short_edge, fam.canvas_max_pixels
            )
            assert resolve_keyframe_canvas(fam, width, height) == (reference_w, reference_h), (
                width,
                height,
            )
            compared += 1
    assert compared > 100


def test_keyframe_canvas_refuses_ratios_outside_the_trained_range():
    fam = _h3()
    with pytest.raises(ValueError, match = "aspect ratios"):
        resolve_keyframe_canvas(fam, 2000, 100)
    with pytest.raises(ValueError, match = "aspect ratios"):
        resolve_keyframe_canvas(fam, 100, 2000)


def test_keyframe_canvas_refuses_a_degenerate_image():
    with pytest.raises(ValueError, match = "no area"):
        resolve_keyframe_canvas(_h3(), 0, 512)


# ── backend request handling ────────────────────────────────────────────────


class _FakeState:
    def __init__(self, family, engine = "diffusers"):
        self.family = family
        self.engine = engine


def _backend():
    from core.inference.video import VideoBackend

    return VideoBackend()


def test_decode_keyframes_is_a_no_op_for_a_text_only_request():
    """A text-only request must not pay for the keyframe path at all, whatever the family."""
    assert _backend()._decode_keyframes(_h3(), None, None) == (None, None)
    assert _backend()._decode_keyframes(None, None, None) == (None, None)


def test_decode_keyframes_returns_pil_images_in_first_last_order():
    first, last = _backend()._decode_keyframes(_h3(), _png(64, 32), _png(96, 48))
    assert (first.size, last.size) == ((64, 32), (96, 48))
    # The keyframe encoder wants RGB; the decoder is the image backend's, which converts.
    assert first.mode == "RGB"


def test_decode_keyframes_accepts_a_data_url():
    (first, _) = _backend()._decode_keyframes(
        _h3(), "data:image/png;base64," + _png(64, 32), None
    )
    assert first.size == (64, 32)


def test_decode_keyframes_refuses_a_family_without_keyframe_support():
    ltx = detect_video_family("Lightricks/LTX-2")
    with pytest.raises(ValueError, match = "cannot start or end on a reference frame"):
        _backend()._decode_keyframes(ltx, _png(64, 32), None)


def test_decode_keyframes_refuses_frames_when_nothing_is_loaded():
    with pytest.raises(ValueError, match = "cannot start or end on a reference frame"):
        _backend()._decode_keyframes(None, _png(64, 32), None)


def test_decode_keyframes_names_the_frame_that_would_not_decode():
    backend = _backend()
    with pytest.raises(ValueError, match = "last reference frame"):
        backend._decode_keyframes(_h3(), _png(64, 32), "not base64 at all !!!")


def test_keyframe_anchors_records_only_the_ends_that_were_pinned():
    from core.inference.video import _keyframe_anchors

    sentinel = object()
    assert _keyframe_anchors(None, None) is None
    assert _keyframe_anchors(sentinel, None) == ["first"]
    assert _keyframe_anchors(None, sentinel) == ["last"]
    assert _keyframe_anchors(sentinel, sentinel) == ["first", "last"]


# ── generate(): canvas override + what reaches the pipeline ─────────────────


class _Scheduler:
    """generate() wraps scheduler.step for progress on a pipeline with no step callback."""

    def step(self, *args, **kwargs):
        return None


class _RecordingPipe:
    """Stands in for the modular pipeline: records the call and returns a minimal output."""

    def __init__(self):
        self.calls: list[dict] = []
        self.scheduler = _Scheduler()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = kwargs["num_frames"]
        height, width = kwargs["height"], kwargs["width"]
        return {
            "videos": [[[[0.0] * width] * height] * frames],
            "audio": None,
            "sampling_rate": None,
        }


def _loaded_backend(monkeypatch, engine = "diffusers"):
    """A backend with a fake H3 state committed, so generate() runs its own logic."""
    import core.inference.video as video_module

    backend = _backend()
    pipe = _RecordingPipe()
    state = video_module._VideoLoadState(
        pipe = pipe,
        family = _h3(),
        repo_id = "MiniMaxAI/MiniMax-H3",
        base_repo = "MiniMaxAI/MiniMax-H3",
        gguf_filename = None,
        kind = "pipeline",
        device = "cpu",
        dtype = "bfloat16",
        engine = engine,
    )
    backend._state = state
    monkeypatch.setattr(video_module.VideoBackend, "_encode_mp4", staticmethod(lambda *a, **k: b"mp4"))
    return backend, pipe


def test_generate_keyframe_overrides_the_requested_resolution(monkeypatch):
    """A 4:3 frame generates on a 4:3 canvas even though the request asked for 1344x768;
    stretching it onto the preset silently degrades the clip."""
    backend, pipe = _loaded_backend(monkeypatch)
    from PIL import Image

    result = backend.generate(
        prompt = "a cat",
        width = 1344,
        height = 768,
        steps = 1,
        seed = 0,
        image = Image.new("RGB", (1024, 768)),
    )
    assert (pipe.calls[0]["width"], pipe.calls[0]["height"]) == (1024, 768)
    assert (result["width"], result["height"]) == (1024, 768)
    assert result["keyframe_anchors"] == ["first"]


def test_generate_without_keyframes_still_honours_the_requested_resolution(monkeypatch):
    backend, pipe = _loaded_backend(monkeypatch)
    backend.generate(prompt = "a cat", width = 1344, height = 768, steps = 1, seed = 0)
    assert (pipe.calls[0]["width"], pipe.calls[0]["height"]) == (1344, 768)
    assert "image" not in pipe.calls[0] and "last_image" not in pipe.calls[0]


def test_generate_omits_the_keyframe_kwargs_rather_than_passing_none(monkeypatch):
    """The auto block graph selects its keyframe branches on the PRESENCE of these inputs, so a
    text-only request must not carry them at all."""
    backend, pipe = _loaded_backend(monkeypatch)
    from PIL import Image

    backend.generate(
        prompt = "a cat", steps = 1, seed = 0, last_image = Image.new("RGB", (768, 768))
    )
    assert "image" not in pipe.calls[0]
    assert pipe.calls[0]["last_image"].size == (768, 768)


def test_generate_refuses_keyframes_for_a_family_without_them(monkeypatch):
    import core.inference.video as video_module
    from PIL import Image

    backend, _ = _loaded_backend(monkeypatch)
    object.__setattr__(backend._state, "family", detect_video_family("Lightricks/LTX-2"))
    assert video_module is not None
    with pytest.raises(ValueError, match = "cannot start or end on a reference frame"):
        backend.generate(prompt = "a cat", steps = 1, seed = 0, image = Image.new("RGB", (768, 768)))


# ── sd-cli argv ─────────────────────────────────────────────────────────────


def _files():
    return SdCppModelFiles(
        diffusion_model = "/models/h3.gguf",
        vae = "/models/vae.gguf",
        llm = "/models/qwen3vl.gguf",
    )


def _video_command(**params):
    return build_sd_cpp_video_command(
        "/bin/sd-cli",
        _files(),
        SdCppVideoGenParams(
            prompt = "a cat",
            width = 1344,
            height = 768,
            num_frames = 124,
            **params,
        ),
        output_path = "/tmp/out.webm",
    )


def test_sd_cpp_video_command_has_no_keyframe_flags_for_a_text_only_run():
    cmd = _video_command()
    assert "--init-img" not in cmd and "--end-img" not in cmd


def test_sd_cpp_video_command_passes_the_first_frame():
    cmd = _video_command(init_image_path = "/tmp/first.png")
    assert cmd[cmd.index("--init-img") + 1] == "/tmp/first.png"
    assert "--end-img" not in cmd


def test_sd_cpp_video_command_passes_the_last_frame_on_its_own():
    cmd = _video_command(end_image_path = "/tmp/last.png")
    assert cmd[cmd.index("--end-img") + 1] == "/tmp/last.png"
    assert "--init-img" not in cmd


def test_sd_cpp_video_command_passes_both_frames():
    cmd = _video_command(init_image_path = "/tmp/first.png", end_image_path = "/tmp/last.png")
    assert cmd[cmd.index("--init-img") + 1] == "/tmp/first.png"
    assert cmd[cmd.index("--end-img") + 1] == "/tmp/last.png"
    # The canvas still reaches sd-cli explicitly: it resizes a loaded frame to --width/--height,
    # so dropping them would let the frame's own size decide the clip's.
    assert cmd[cmd.index("--width") + 1] == "1344"
    assert cmd[cmd.index("--height") + 1] == "768"
