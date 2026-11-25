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
        # a bf16 checkpoint: weights and persistent buffers hold bf16 values; the non-persistent resampler
        # filter is computed at build time in float32
        for t in voc.state_dict().values():
            if t.is_floating_point():
                t.copy_(t.bfloat16().float())
    return voc.to(dtype)


def _reference_and_studio():
    """(float32 build of a bf16 checkpoint, the same model as Studio loads it: everything cast to bf16)."""
    ref = _tiny_vocoder(torch.float32)
    return ref, copy.deepcopy(ref).to(torch.bfloat16)


def _mel(dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 2, 12, 8, generator = g).to(dtype)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(L.VOCODER_FP32_ENV, raising = False)


@torch.no_grad()
def test_bf16_vocoder_matches_the_float32_build_bit_for_bit():
    ref, voc = _reference_and_studio()
    mel = _mel()
    # the bf16 cast rounded the resampler filter: a plain .float() copy is not the reference
    assert not torch.equal(copy.deepcopy(voc).float()(mel.float()), ref(mel.float()))
    L.install_vocoder_fp32_forward(voc)
    out = voc(mel)
    assert out.dtype == torch.float32
    assert torch.equal(out, ref(mel.float()))
    # weights and the stored filter keep their dtype: the float32 filter is swapped in for the call only
    assert {p.dtype for p in voc.parameters()} == {torch.bfloat16}
    assert voc.resampler.filter.dtype == torch.bfloat16
    assert torch.equal(voc(mel), out)


@torch.no_grad()
def test_filter_computed_in_bf16_is_rebuilt(monkeypatch):
    """diffusers' from_pretrained builds under torch_dtype as the default dtype: the filter is computed in bf16."""
    ref = _tiny_vocoder(torch.float32)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        bf16_filter = type(ref.resampler)(
            ratio = ref.resampler.ratio, window_type = "hann", persistent = False
        ).filter
    finally:
        torch.set_default_dtype(prev)
    voc = copy.deepcopy(ref).to(torch.bfloat16)
    voc.resampler._buffers["filter"] = bf16_filter
    assert not torch.equal(bf16_filter, ref.resampler.filter.bfloat16())
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    mel = _mel()
    assert torch.equal(voc(mel), ref(mel.float()))
    monkeypatch.setenv(L.VOCODER_FP32_ENV, "0")
    assert torch.equal(voc(mel), base(mel))


@torch.no_grad()
def test_kill_switch_runs_the_stored_dtype(monkeypatch):
    _, voc = _reference_and_studio()
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


def test_a_changed_resampler_filter_is_left_alone():
    _, voc = _reference_and_studio()
    with torch.no_grad():
        voc.resampler.filter.mul_(0.5)
    stored = voc.resampler.filter.clone()
    L.install_vocoder_fp32_forward(voc)
    assert torch.equal(voc.resampler.filter, stored)


def test_loader_installs_the_float32_forward(monkeypatch):
    _, voc = _reference_and_studio()
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


def _oom_on_float32(voc, exc):
    """The stage-1 generator raises ``exc`` whenever it is handed a float32 input (the float32 pass)."""
    real = voc.vocoder.forward
    calls = {"fp32": 0}

    def forward(x, *a, **k):
        if x.dtype == torch.float32:
            calls["fp32"] += 1
            raise exc
        return real(x, *a, **k)

    voc.vocoder.forward = forward
    return calls


@torch.no_grad()
def test_out_of_memory_runs_the_stored_dtype_and_logs_once(monkeypatch):
    _, voc = _reference_and_studio()
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    calls = _oom_on_float32(voc, oom("CUDA out of memory. Tried to allocate 96.00 MiB"))
    warned = []
    monkeypatch.setattr(L, "_fallback_logged", [False])
    monkeypatch.setattr(L.logger, "warning", lambda *a, **k: warned.append(a))
    mel = _mel()
    for _ in range(2):
        out = voc(mel)
        assert out.dtype == torch.bfloat16 and torch.equal(out, base(mel))
    assert calls["fp32"] == 2 and len(warned) == 1
    # the module is left as it was: stored filter, bf16 weights
    assert voc.resampler.filter.dtype == torch.bfloat16
    assert {p.dtype for p in voc.parameters()} == {torch.bfloat16}


@torch.no_grad()
def test_runtime_out_of_memory_message_also_falls_back(monkeypatch):
    _, voc = _reference_and_studio()
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    _oom_on_float32(voc, RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"))
    monkeypatch.setattr(L, "_fallback_logged", [False])
    assert torch.equal(voc(_mel()), base(_mel()))


@torch.no_grad()
def test_other_errors_are_not_swallowed():
    _, voc = _reference_and_studio()
    L.install_vocoder_fp32_forward(voc)
    _oom_on_float32(voc, ValueError("shape mismatch"))
    with pytest.raises(ValueError):
        voc(_mel())


def test_fp32_extra_scales_with_clip_length():
    small = L.vocoder_fp32_extra_bytes(torch.zeros(1, 2, 501, 64))
    big = L.vocoder_fp32_extra_bytes(torch.zeros(1, 2, 2004, 64))
    # the measured +104.5 MiB at 501 frames, with margin
    assert 105 << 20 < small < 300 << 20
    assert big > 3 * small
