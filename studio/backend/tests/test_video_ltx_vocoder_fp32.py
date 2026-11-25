# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3/2.5 vocoder: float32 forward on bf16 weights (Lightricks' VocoderWithBWE), with a kill switch."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from core.inference import video_ltx2 as L  # noqa: E402


def _tiny_vocoder(dtype):
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    cfg = dict(L._VOCODER_CONFIG)
    cfg.update(
        in_channels = 16,
        hidden_channels = 32,
        upsample_kernel_sizes = [4, 4],
        upsample_factors = [2, 2],
        bwe_in_channels = 16,
        bwe_hidden_channels = 16,
        bwe_upsample_kernel_sizes = [12, 8],
        bwe_upsample_factors = [6, 4],
        filter_length = 32,
        hop_length = 8,
        window_length = 32,
        num_mel_channels = 8,
        input_sampling_rate = 8000,
        output_sampling_rate = 24000,
    )
    torch.manual_seed(0)
    voc = LTX2VocoderWithBWE.from_config(cfg).eval()
    with torch.no_grad():
        for p in voc.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return voc.to(dtype)


def _mel(dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 2, 12, 8, generator = g).to(dtype)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(L.VOCODER_FP32_ENV, raising = False)


@torch.no_grad()
def test_bf16_vocoder_matches_float32_copy_bit_for_bit():
    voc = _tiny_vocoder(torch.bfloat16)
    ref = copy.deepcopy(voc).float()
    L.install_vocoder_fp32_forward(voc)
    mel = _mel()
    out = voc(mel)
    assert out.dtype == torch.float32
    assert torch.equal(out, ref(mel.float()))
    # weights stay at their stored dtype after the call
    assert {p.dtype for p in voc.parameters()} == {torch.bfloat16}


@torch.no_grad()
def test_kill_switch_runs_the_stored_dtype(monkeypatch):
    voc = _tiny_vocoder(torch.bfloat16)
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    mel = _mel()
    monkeypatch.setenv(L.VOCODER_FP32_ENV, "0")
    out = voc(mel)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, base(mel))
    monkeypatch.delenv(L.VOCODER_FP32_ENV)
    assert voc(mel).dtype == torch.float32


@torch.no_grad()
def test_float32_vocoder_is_unchanged_and_install_is_idempotent():
    voc = _tiny_vocoder(torch.float32)
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    first = voc.forward
    L.install_vocoder_fp32_forward(voc)
    assert voc.forward is first
    mel = _mel(torch.float32)
    assert torch.equal(voc(mel), base(mel))
    # a bf16 mel into float32 weights is widened instead of raising a dtype mismatch
    assert torch.equal(voc(_mel()), base(_mel().float()))


def test_install_never_raises_on_a_non_module():
    L.install_vocoder_fp32_forward(object())


def test_loader_installs_the_float32_forward(monkeypatch):
    voc = _tiny_vocoder(torch.bfloat16)
    state = dict(voc.state_dict())

    class _Fake:
        @classmethod
        def from_config(cls, cfg):
            return copy.deepcopy(voc)

    import diffusers.pipelines.ltx2.vocoder as V

    monkeypatch.setattr(V, "LTX2VocoderWithBWE", _Fake)
    monkeypatch.setattr(L, "_build_from_config", lambda *a, **k: "audio_vae")
    monkeypatch.setattr(L, "_apply_rename", lambda s, r: s)
    audio_vae, out = L.load_ltx23_audio_vae_and_vocoder(
        {"x": 1}, state, variant = "dev", torch_dtype = torch.bfloat16, hf_token = None
    )
    assert audio_vae == "audio_vae" and out._unsloth_fp32_forward
