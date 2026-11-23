# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portable whole-map VAE decode (diffusion_vae_strips.full_map_decode, torch ops): a non-OOM failure inside the
upsample fold marks the decoder, so this decode retries once without the fold and later decodes skip it, as the
kernels path does after a strip failure. An OOM still goes to the tiles unmarked. CPU, tiny random VAE."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_strips as strips  # noqa: E402


def _vae():
    torch.manual_seed(0)
    return diffusers.AutoencoderKL(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        latent_channels = 4,
        norm_num_groups = 8,
        sample_size = 64,
    ).eval()


def _latent():
    return torch.randn(1, 4, 24, 24, generator = torch.Generator().manual_seed(1))


@pytest.fixture
def failing_fold(monkeypatch):
    """The fold forced on (CPU) and raising ``error``; returns the attempt log."""
    monkeypatch.setenv(strips.UPSAMPLE_FOLD_ENV, "1")
    attempts = []
    state = {"error": RuntimeError("injected fold failure")}

    def boom(*args, **kwargs):
        attempts.append(1)
        raise state["error"]

    monkeypatch.setattr(strips, "_fold_upsample", boom)
    return attempts, state


def _reference(vae, z, monkeypatch):
    monkeypatch.setenv(strips.UPSAMPLE_FOLD_ENV, "0")
    try:
        return strips.full_map_decode(vae, z)
    finally:
        monkeypatch.setenv(strips.UPSAMPLE_FOLD_ENV, "1")


def test_a_failed_portable_fold_is_marked_and_not_retried(failing_fold, monkeypatch):
    attempts, _ = failing_fold
    vae, z = _vae(), _latent()
    ref = _reference(vae, z, monkeypatch)
    assert not attempts
    for _ in range(3):
        out = strips.full_map_decode(vae, z)
        assert torch.allclose(out, ref, atol = 1e-6, rtol = 0)
    # one fold attempt in all: this decode retried without it, the later ones skipped it
    assert len(attempts) == 1
    assert vae.decoder.__dict__.get("_unsloth_vae_fold_failed") is True
    # the portable path never marks the kernels as failed
    assert "_unsloth_vae_strips_failed" not in vae.decoder.__dict__


def test_an_oom_in_the_portable_fold_still_goes_to_the_tiles_unmarked(failing_fold):
    attempts, state = failing_fold
    state["error"] = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    vae = _vae()
    for _ in range(2):
        with pytest.raises(RuntimeError, match = "out of memory"):
            strips.full_map_decode(vae, _latent())
    assert len(attempts) == 2
    assert "_unsloth_vae_fold_failed" not in vae.decoder.__dict__


def test_a_portable_failure_outside_the_fold_still_raises(monkeypatch):
    monkeypatch.setenv(strips.UPSAMPLE_FOLD_ENV, "0")
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(strips, "_decode_one", boom)
    vae = _vae()
    with pytest.raises(RuntimeError, match = "injected decode failure"):
        strips.full_map_decode(vae, _latent())
    assert len(calls) == 1
    assert "_unsloth_vae_fold_failed" not in vae.decoder.__dict__
