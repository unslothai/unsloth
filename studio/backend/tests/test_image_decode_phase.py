# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image bar reports the decode, which runs inside pipe() after the last step callback."""

from __future__ import annotations

import pytest

from core.inference.media_decode_phase import decode_phase


class _Vae:
    def __init__(self):
        self.calls = 0

    def decode(self, latents):
        self.calls += 1
        return f"decoded:{latents}"


class _Pipe:
    def __init__(self):
        self.vae = _Vae()


def test_the_hook_fires_on_the_first_decode_and_not_again():
    pipe, seen = _Pipe(), []
    with decode_phase(pipe, lambda: seen.append("decode")):
        assert pipe.vae.decode("a") == "decoded:a"
        assert pipe.vae.decode("b") == "decoded:b"
    assert seen == ["decode"]
    assert pipe.vae.calls == 2


def test_the_wrapper_is_removed_even_when_the_decode_raises():
    """A leftover wrapper would fire a finished job's callback on the next generation."""
    pipe = _Pipe()
    boom = RuntimeError("vae oom")

    def _raise(_latents):
        raise boom

    pipe.vae.decode = _raise
    with pytest.raises(RuntimeError):
        with decode_phase(pipe, lambda: None):
            pipe.vae.decode("a")
    assert pipe.vae.decode is _raise


def test_a_pipe_with_no_decoder_is_a_no_op():
    class _Bare:
        pass

    with decode_phase(_Bare(), lambda: pytest.fail("fired with no decoder")):
        pass


def test_generate_progress_reports_the_phase():
    from core.inference.diffusion import _GenState

    gen = _GenState(total_steps = 40)
    assert gen.phase == "encode"

    class _Backend:
        _gen = gen

    from core.inference.diffusion import DiffusionBackend

    progress = DiffusionBackend.generate_progress(_Backend())
    assert progress["phase"] == "encode"

    gen.step, gen.phase = 40, "decode"
    progress = DiffusionBackend.generate_progress(_Backend())
    assert progress["phase"] == "decode"
    # Fields missing from the response model are dropped silently.
    from models.inference import DiffusionGenerateProgressResponse

    assert DiffusionGenerateProgressResponse(**progress).phase == "decode"
    assert DiffusionGenerateProgressResponse(active = False).phase is None

    _Backend._gen = None
    assert DiffusionBackend.generate_progress(_Backend())["phase"] == "denoise"


def test_denoise_phase_fires_when_the_loop_opens_its_progress_bar_and_is_removed():
    from core.inference.media_decode_phase import denoise_phase

    class _Bar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _LoopPipe:
        def progress_bar(self, total = None):
            return _Bar()

    pipe, seen = _LoopPipe(), []
    with denoise_phase(pipe, lambda: seen.append("denoise")):
        with pipe.progress_bar(total = 4):
            pass
        with pipe.progress_bar(total = 4):
            pass
    assert seen == ["denoise"]
    assert "progress_bar" not in pipe.__dict__


def test_generate_progress_carries_the_live_preview_through_the_route_model():
    from core.inference.diffusion import DiffusionBackend, _GenState
    from models.inference import DiffusionGenerateProgressResponse

    class _Backend:
        _gen = _GenState(total_steps = 8, step = 2, phase = "encode")

    _Backend._gen.preview, _Backend._gen.preview_seq = "data:image/jpeg;base64,AA==", 3
    resp = DiffusionGenerateProgressResponse(**DiffusionBackend.generate_progress(_Backend()))
    assert resp.phase == "encode"
    assert resp.preview == "data:image/jpeg;base64,AA==" and resp.preview_seq == 3
