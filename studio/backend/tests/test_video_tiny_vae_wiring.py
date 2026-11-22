# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wiring of the tiny Wan decoder through code that exists without it: every Wan family offers live previews on
CUDA, and the video request / generate path carries an opt-in draft decode."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_preview as DP


def test_every_wan_family_offers_live_previews_on_cuda(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_VIDEO_TINY_VAE", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for fam in ("wan2.1-t2v-1.3b", "wan2.1-t2v-14b", "wan2.2-t2v-a14b", "wan2.2-ti2v-5b"):
        assert DP.preview_available(fam, "cuda")
    assert not DP.preview_available("wan2.1-t2v-1.3b", "cpu")
    monkeypatch.setenv("UNSLOTH_VIDEO_TINY_VAE", "0")
    assert not DP.preview_available("wan2.1-t2v-1.3b", "cuda")
    assert DP.preview_available("wan2.2-ti2v-5b", "cuda")  # its linear map stays


def test_generate_and_request_carry_draft_decode_off_by_default():
    from core.inference.video import VideoBackend
    from models.inference import VideoGenerateRequest

    for fn in (VideoBackend.begin_generate, VideoBackend.generate):
        assert inspect.signature(fn).parameters["draft_decode"].default is None
    assert VideoGenerateRequest.model_fields["draft_decode"].default is None
    assert VideoGenerateRequest(prompt = "a cat").draft_decode is None
