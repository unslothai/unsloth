# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every video family refuses a NaN / inf decode with NonFiniteFramesError, before a PIL / uint8 cast or a clamp hides
it; a finite clip is untouched."""

import contextlib
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.autoencoders.vae import DecoderOutput  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402

from core.inference import video_frames as VF  # noqa: E402
from core.inference.video_frames import (  # noqa: E402
    NON_FINITE_FRAMES_MSG,
    NonFiniteFramesError,
    uint8_video_frames,
)

# Absent before this change: a null context, so the tests fail on behaviour rather than on import.
_finite_video_decode = getattr(VF, "finite_video_decode", lambda pipe: contextlib.nullcontext())
_ENV = "UNSLOTH_VIDEO_NONFINITE_CHECK"


class _VAE(torch.nn.Module):
    def __init__(
        self,
        pixels,
        style = "tuple",
    ):
        super().__init__()
        self.pixels = pixels
        self.style = style

    def decode(
        self,
        z,
        *args,
        return_dict = True,
        **kwargs,
    ):
        if self.style == "output":  # Cosmos3: self.vae.decode(z).sample
            return DecoderOutput(sample = self.pixels)
        return (self.pixels,)


def _pixels(
    bad = None,
    frames = 4,
    where = (0, 0, 1, 2, 3),
):
    g = torch.Generator().manual_seed(0)
    video = torch.rand((1, 3, frames, 16, 24), generator = g) * 2 - 1
    if bad is not None:
        video[where] = bad
    return video


def _render(pipe, family):
    if family == "wan":  # WanPipeline / HunyuanVideo15Pipeline: output_type="np"
        return pipe.video_processor.postprocess_video(
            pipe.vae.decode(None, return_dict = False)[0], output_type = "np"
        )
    if family == "ltx2":  # LTX2Pipeline (LTX-2, LTX-2.3, LTX-2.5 stage 2): output_type="pil"
        return pipe.video_processor.postprocess_video(
            pipe.vae.decode(None, None, return_dict = False)[0], output_type = "pil"
        )
    if family == "cosmos3":  # Cosmos3OmniPipeline: output_type="pil"
        decoded = pipe.vae.decode(None).sample
        return pipe.video_processor.postprocess_video(decoded, output_type = "pil")[0]
    if (
        family == "h3"
    ):  # MiniMaxH3 modular decode step: denormalise + clamp(0, 1) BEFORE postprocess_video
        video = pipe.vae.decode(None, return_dict = False)[0]
        video = (video.float() * 0.5 + 0.5).clamp(0, 1)
        return pipe.video_processor.postprocess_video(video, output_type = "pil")
    raise AssertionError(family)


def _pipe(pixels, family):
    style = "output" if family == "cosmos3" else "tuple"
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = family != "h3")
    return types.SimpleNamespace(vae = _VAE(pixels, style), video_processor = vp)


def _run(pipe, family):
    with _finite_video_decode(pipe), uint8_video_frames(pipe):
        return _render(pipe, family)


FAMILIES = ["wan", "ltx2", "cosmos3", "h3"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_every_family_refuses_a_non_finite_decode(family, bad):
    pipe = _pipe(_pixels(bad), family)
    with pytest.raises(NonFiniteFramesError) as info:
        _run(pipe, family)
    assert str(info.value) == NON_FINITE_FRAMES_MSG


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_postprocess_refuses_before_the_cast(output_type, bad):
    # PIL: NaN became uint8 0 (a black pixel); np: +inf was clamped to 255. Both used to pass silently.
    vp = VideoProcessor(vae_scale_factor = 8)
    with uint8_video_frames(types.SimpleNamespace(video_processor = vp)):
        with pytest.raises(NonFiniteFramesError):
            vp.postprocess_video(_pixels(bad), output_type = output_type)
    assert "postprocess_video" not in vars(vp)


def test_a_late_frame_in_a_sliced_check_is_seen(monkeypatch):
    monkeypatch.setattr(VF, "_FINITE_SLICE_ELEMS", 16 * 24 * 3, raising = False)
    pipe = _pipe(_pixels(float("inf"), frames = 9, where = (0, 2, 8, 15, 23)), "ltx2")
    with pytest.raises(NonFiniteFramesError):
        _run(pipe, "ltx2")


@pytest.mark.parametrize("family", FAMILIES)
def test_a_finite_clip_is_bit_identical(family, monkeypatch):
    pixels = _pixels()
    got = _run(_pipe(pixels, family), family)
    monkeypatch.setenv(_ENV, "0")
    expected = _run(_pipe(pixels.clone(), family), family)
    if family == "wan":
        assert (
            isinstance(got, torch.Tensor)
            and got.dtype == torch.uint8
            and torch.equal(got, expected)
        )
        return
    exp = expected[0] if family != "cosmos3" else expected
    out = got[0] if family != "cosmos3" else got
    assert len(out) == len(exp) == pixels.shape[2]
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out, exp))


def test_decode_output_is_returned_untouched_and_the_slot_restored():
    pixels = _pixels()
    vae = _VAE(pixels)
    pipe = types.SimpleNamespace(vae = vae)
    assert "decode" not in vars(vae)
    with _finite_video_decode(pipe):
        assert "decode" in vars(vae)
        assert vae.decode.__wrapped__ is not None
        assert vae.decode(None)[0] is pixels
    assert "decode" not in vars(vae)


def test_an_own_decode_is_put_back():
    vae = _VAE(_pixels())
    own = vae.decode
    vae.decode = own
    with _finite_video_decode(types.SimpleNamespace(vae = vae)):
        assert vars(vae)["decode"] is not own
    assert vars(vae)["decode"] is own


@pytest.mark.parametrize("family", FAMILIES)
def test_kill_switch_restores_the_previous_behaviour(family, monkeypatch):
    monkeypatch.setenv(_ENV, "0")
    pipe = _pipe(_pixels(float("inf")), family)
    vae = pipe.vae
    with _finite_video_decode(pipe):
        assert "decode" not in vars(vae)
    _run(pipe, family)


def test_the_render_stack_enters_the_decode_check():
    source = (Path(VF.__file__).with_name("video.py")).read_text(encoding = "utf-8")
    assert "stack.enter_context(finite_video_decode(pipe))" in source


class _ScopedVAE(torch.nn.Module):
    """A VAE whose public decode calls ``_decode`` (stock diffusers shape), so the OOM ladder and scope install."""

    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels
        self.use_tiling = False

    def enable_tiling(self):
        self.use_tiling = True

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        return (self.pixels,)

    def decode(
        self,
        z,
        return_dict = True,
    ):
        return self._decode(z, return_dict = return_dict)


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_the_oom_scope_stays_intact_and_is_put_back(bad):
    from core.inference import decode_layers
    from core.inference import diffusion_vae_oom as O

    pixels = _pixels(bad)
    vae = _ScopedVAE(pixels)
    assert O.install(vae)
    scope = vars(vae)["decode"]
    assert O._scope_intact(vae)
    with _finite_video_decode(types.SimpleNamespace(vae = vae)):
        assert vars(vae)["decode"] is not scope
        assert O._scope_intact(vae)  # the finiteness layer sits over the scope without breaking it
        assert decode_layers.find(vae, "decode", "video_finite_decode") is not None
        if bad is None:
            assert vae.decode(None)[0] is pixels
        else:
            with pytest.raises(NonFiniteFramesError):
                vae.decode(None)
    assert vars(vae)["decode"] is scope and O._scope_intact(vae)
    assert decode_layers.find(vae, "decode", "video_finite_decode") is None
