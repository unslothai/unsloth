# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resident video VAE decodes untiled when it fits, tiled otherwise, and falls back to tiled on OOM."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_vae_untiled as U


class FakeVAE:
    def __init__(
        self,
        oom_untiled = False,
        dtype = torch.float16,
        error = None,
    ):
        self.error = error
        self.decoder = torch.nn.Conv3d(1, 1, 1).to(dtype)
        self.use_tiling = True
        self.calls = []
        self.oom_untiled = oom_untiled

    def decode(
        self,
        z,
        return_dict = False,
    ):
        self.calls.append(self.use_tiling)
        if not self.use_tiling and self.error is not None:
            raise self.error
        if not self.use_tiling and self.oom_untiled:
            raise torch.cuda.OutOfMemoryError("fake")
        return ("tiled" if self.use_tiling else "untiled",)


def _pipe(vae):
    return types.SimpleNamespace(vae = vae)


WAN = "wan2.2-ti2v-5b"
Z = torch.zeros(1, 48, 31, 44, 80)


def test_untiled_when_it_fits(monkeypatch):
    vae = FakeVAE()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    assert U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z, return_dict = False) == ("untiled",)
    assert vae.use_tiling is True
    assert vae.decode._unsloth_untiled_stats == {"untiled": 1, "tiled": 0, "oom_fallback": 0}


def test_tiled_when_it_does_not_fit(monkeypatch):
    vae = FakeVAE()
    need = U.untiled_decode_bytes(WAN, tuple(Z.shape))
    monkeypatch.setattr(U, "_free_bytes", lambda device: need)
    assert U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z) == ("tiled",)
    assert vae.calls == [True]


def test_tiled_when_memory_unreadable(monkeypatch):
    vae = FakeVAE()
    monkeypatch.setattr(U, "_free_bytes", lambda device: None)
    U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z) == ("tiled",)


def test_oom_falls_back_to_tiled(monkeypatch):
    vae = FakeVAE(oom_untiled = True)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z) == ("tiled",)
    assert vae.calls == [False, True]
    assert vae.use_tiling is True
    assert vae.decode._unsloth_untiled_stats["oom_fallback"] == 1


def test_not_installed_for_unmeasured_family_or_untiled_vae_or_env(monkeypatch):
    assert not U.install_untiled_decode(_pipe(FakeVAE()), "hunyuanvideo-1.5")
    vae = FakeVAE()
    vae.use_tiling = False
    assert not U.install_untiled_decode(_pipe(vae), WAN)
    monkeypatch.setenv(U.UNTILED_ENV, "0")
    assert not U.install_untiled_decode(_pipe(FakeVAE()), WAN)


def test_install_is_idempotent(monkeypatch):
    vae = FakeVAE()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    assert U.install_untiled_decode(_pipe(vae), WAN)
    first = vae.decode
    assert U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode is first


def test_estimate_scales_with_latent_area_and_output_frames():
    a = U.untiled_decode_bytes(WAN, (1, 48, 31, 44, 80))
    b = U.untiled_decode_bytes(WAN, (1, 48, 5, 44, 80))
    c = U.untiled_decode_bytes(WAN, (1, 48, 31, 88, 80))
    assert c == 2 * a
    # 121 -> 17 frames: only the RGB output (held twice) shrinks.
    assert a - b == 2 * 3 * (121 - 17) * 704 * 1280 * 2
    long = U.untiled_decode_bytes(WAN, (1, 48, 256, 44, 80))
    assert long - b >= 10 * 2**30 - 2 * 3 * 17 * 704 * 1280 * 2
    assert U.untiled_decode_bytes("nope", (1, 48, 31, 44, 80)) is None


def test_fp32_decoder_needs_twice_the_fp16_estimate(monkeypatch):
    # Wan's untiled fp32 decode peaks ~2x fp16, so an fp16-sized gate must not go untiled.
    fp16_gate = U.untiled_decode_bytes(WAN, tuple(Z.shape)) * U._MARGIN + U._MARGIN_BYTES
    monkeypatch.setattr(U, "_free_bytes", lambda device: int(fp16_gate) + 1)
    half, full = FakeVAE(), FakeVAE(dtype = torch.float32)
    U.install_untiled_decode(_pipe(half), WAN)
    U.install_untiled_decode(_pipe(full), WAN)
    assert half.decode(Z) == ("untiled",)
    assert full.decode(Z) == ("tiled",)
    assert U.untiled_decode_bytes(WAN, tuple(Z.shape), itemsize = 4) == 2 * U.untiled_decode_bytes(
        WAN, tuple(Z.shape)
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("compile failed"),
    ],
)
def test_wrapped_or_backend_oom_falls_back_other_errors_raise(monkeypatch, error):
    if error.args[0] == "compile failed":
        error.__cause__ = torch.cuda.OutOfMemoryError("inner")
    vae = FakeVAE(error = error)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z) == ("tiled",)
    assert vae.use_tiling is True


def test_non_oom_error_raises_and_restores_tiling(monkeypatch):
    vae = FakeVAE(error = ValueError("bad latent"))
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    U.install_untiled_decode(_pipe(vae), WAN)
    with pytest.raises(ValueError):
        vae.decode(Z)
    assert vae.use_tiling is True


def test_estimate_that_ran_out_of_memory_is_not_retried_untiled(monkeypatch):
    vae = FakeVAE(oom_untiled = True)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    U.install_untiled_decode(_pipe(vae), WAN)
    assert vae.decode(Z) == ("tiled",)
    assert vae.decode(Z) == ("tiled",)
    assert vae.calls == [False, True, True]
    assert vae.decode._unsloth_untiled_stats["oom_fallback"] == 1
    vae.decode(torch.zeros(1, 48, 2, 44, 80))
    assert vae.calls[-2:] == [False, True]


def test_wrapper_survives_a_compile_fallback_replacing_the_slot(monkeypatch):
    # _guard_compiled_decode writes its eager decode back into vae.decode when the compile fails mid-call.
    vae = FakeVAE()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    eager = vae.decode

    def compiled_then_fails(z, *args, **kwargs):
        vae.decode = eager
        return eager(z, *args, **kwargs)

    vae.decode = compiled_then_fails
    U.install_untiled_decode(_pipe(vae), WAN)
    wrapper = vae.decode
    assert wrapper(Z) == ("untiled",)
    assert vae.decode is wrapper
    assert vae.decode(Z) == ("untiled",)
