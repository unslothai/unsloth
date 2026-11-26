# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MPS zero-buffer causal pad on Qwen-Image VAE convs (one-frame, no-cache calls only), kept under the single-frame
pass. Runs on CPU; the MPS test runs only where MPS is available."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_device as D  # noqa: E402
from core.inference import diffusion_vae_single_frame as SF  # noqa: E402

MPS = types.SimpleNamespace(device = "mps")


def _vae():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLQwenImage(
        base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 2],
        num_res_blocks = 1,
        attn_scales = [],
        temperal_downsample = [True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    ).eval()


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(SF.SINGLE_FRAME_ENV, raising = False)


@pytest.fixture
def counted(monkeypatch):
    calls = {"n": 0}
    real = D._zero_buffer_causal_conv

    def wrapped(
        module,
        x,
        cache_x = None,
    ):
        calls["n"] += 1
        return real(module, x, cache_x)

    monkeypatch.setattr(D, "_zero_buffer_causal_conv", wrapped)
    return calls


def _convs(vae):
    return [m for m in vae.modules() if type(m).__name__ == "QwenImageCausalConv3d"]


def test_qwen_causal_convs_pad_into_a_zero_buffer_bit_identically(counted):
    vae = _vae()
    z = torch.randn(1, 4, 3, 6, 6)
    with torch.no_grad():
        ref = vae.decode(z).sample
    patched = D.install_frame_pad_fix(types.SimpleNamespace(components = {"vae": vae}), MPS)
    assert patched == len(_convs(vae)) > 0
    with torch.no_grad():
        out = vae.decode(z).sample
    assert counted["n"] > 0
    assert torch.equal(ref, out)


def test_single_frame_pass_keeps_the_zero_buffer_forward_for_other_calls(counted):
    vae = _vae()
    z1 = torch.randn(1, 4, 1, 6, 6)
    z3 = torch.randn(1, 4, 3, 6, 6)
    with torch.no_grad():
        ref3 = vae.decode(z3).sample
    plain = _vae()
    assert SF.install(plain)
    with torch.no_grad():
        ref1 = plain.decode(z1).sample
    # Studio's order: the MPS pad fix at load, then the speed passes.
    D.install_frame_pad_fix(types.SimpleNamespace(components = {"vae": vae}), MPS)
    assert SF.install(vae)
    with torch.no_grad():
        out1 = vae.decode(z1).sample
        single = counted["n"]
        out3 = vae.decode(z3).sample
    assert torch.equal(ref1, out1)
    assert torch.equal(ref3, out3)
    assert counted["n"] > single


def test_single_frame_alone_still_falls_back_to_stock():
    vae = _vae()
    z3 = torch.randn(1, 4, 3, 6, 6)
    with torch.no_grad():
        ref3 = vae.decode(z3).sample
    assert SF.install(vae)
    assert all(m._unsloth_sf_inner is None for m in _convs(vae))
    with torch.no_grad():
        assert torch.equal(ref3, vae.decode(z3).sample)


def test_other_devices_are_untouched():
    vae = _vae()
    assert (
        D.install_frame_pad_fix(
            types.SimpleNamespace(components = {"vae": vae}), types.SimpleNamespace(device = "cuda")
        )
        == 0
    )
    assert all("forward" not in m.__dict__ for m in _convs(vae))


@pytest.fixture
def stock_calls(monkeypatch):
    cls = diffusers.models.autoencoders.autoencoder_kl_qwenimage.QwenImageCausalConv3d
    calls = []
    real = cls.forward

    def wrapped(
        self,
        x,
        cache_x = None,
    ):
        calls.append((int(x.shape[2]), cache_x is not None))
        return real(self, x, cache_x)

    monkeypatch.setattr(cls, "forward", wrapped)
    return calls


def test_one_frame_decode_takes_the_zero_buffer_on_every_call(counted, stock_calls):
    vae = _vae()
    z1 = torch.randn(1, 4, 1, 6, 6)
    with torch.no_grad():
        ref = vae.decode(z1).sample
    D.install_frame_pad_fix(types.SimpleNamespace(components = {"vae": vae}), MPS)
    stock_calls.clear()
    with torch.no_grad():
        out = vae.decode(z1).sample
    assert torch.equal(ref, out)
    assert counted["n"] > 0
    assert stock_calls == []


@pytest.mark.parametrize("single_frame_pass", [False, True])
def test_multi_frame_and_cached_calls_keep_the_stock_forward(
    counted, stock_calls, single_frame_pass
):
    vae = _vae()
    z3 = torch.randn(1, 4, 3, 6, 6)
    with torch.no_grad():
        ref = vae.decode(z3).sample
    D.install_frame_pad_fix(types.SimpleNamespace(components = {"vae": vae}), MPS)
    if single_frame_pass:
        assert SF.install(vae)
    stock_calls.clear()
    with torch.no_grad():
        out = vae.decode(z3).sample
    assert torch.equal(ref, out)
    assert counted["n"] > 0  # the first chunk's one-frame, no-cache calls
    assert stock_calls
    assert all(frames != 1 or cached for frames, cached in stock_calls)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason = "needs MPS")
@pytest.mark.parametrize("frames", [1, 3])
def test_mps_decode_matches_cpu(frames):
    vae = _vae()
    z = torch.randn(1, 4, frames, 6, 6)
    with torch.no_grad():
        ref = vae.decode(z).sample
    vae = vae.to("mps")
    D.install_frame_pad_fix(types.SimpleNamespace(components = {"vae": vae}), MPS)
    with torch.no_grad():
        out = vae.decode(z.to("mps")).sample.cpu()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol = 1e-4, rtol = 1e-4)
