# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3/2.5 vocoder: float32 forward on bf16 weights (Lightricks' VocoderWithBWE), with a kill switch.

Every behaviour test runs on a torch-only stand-in with the same layout (a ``vocoder`` generator, a hann
``resampler`` with a non-persistent filter, a persistent buffer), so it runs without diffusers; the same tests run
again on diffusers' real ``LTX2VocoderWithBWE`` when diffusers is installed."""

from __future__ import annotations

import copy
import math
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from core.inference import video_ltx2 as L  # noqa: E402


class _HannUpSample(torch.nn.Module):
    """diffusers' UpSample1d (hann branch): the filter is computed in the default dtype at build time."""

    def __init__(
        self,
        ratio = 2,
        kernel_size = None,
        window_type = "hann",
        padding_mode = "replicate",
        persistent = True,
    ):
        super().__init__()
        self.ratio = ratio
        self.padding_mode = padding_mode
        rolloff, width_l = 0.99, 6
        width = math.ceil(width_l / rolloff)
        self.kernel_size = 2 * width * ratio + 1
        self.pad = width
        self.pad_left = 2 * width * ratio
        self.pad_right = self.kernel_size - ratio
        t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
        window = torch.cos(t.clamp(-width_l, width_l) * math.pi / width_l / 2) ** 2
        self.register_buffer(
            "filter",
            (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1),
            persistent = persistent,
        )

    def forward(self, x):
        n = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode = self.padding_mode)
        f = self.filter.to(dtype = x.dtype, device = x.device).expand(n, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, f, stride = self.ratio, groups = n)
        return x[..., self.pad_left : -self.pad_right]


class _Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv1d(16, 32, 7, padding = 3)
        self.alpha = torch.nn.Parameter(torch.ones(1, 32, 1))
        self.convs = torch.nn.ModuleList(torch.nn.Conv1d(32, 32, 5, padding = 2) for _ in range(6))
        self.conv_out = torch.nn.Conv1d(32, 2, 7, padding = 3)

    def forward(self, mel):
        b, c, t, m = mel.shape
        x = self.conv_in(mel.transpose(2, 3).reshape(b, c * m, t))
        for conv in self.convs:
            x = x + conv(x + torch.sin(self.alpha * x) ** 2)
        return torch.tanh(self.conv_out(x))


class _FakeVocoderWithBWE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocoder = _Generator()
        self.register_buffer(
            "mel_basis", torch.linspace(0.5, 1.5, 2).view(1, 2, 1), persistent = True
        )
        self.resampler = _HannUpSample(ratio = 3, window_type = "hann", persistent = False)

    def forward(self, mel):
        x = self.vocoder(mel) * self.mel_basis
        return torch.clamp(self.resampler(x), -1, 1)


def _build_fake():
    return _FakeVocoderWithBWE()


def _build_diffusers():
    pytest.importorskip("diffusers")
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
    return LTX2VocoderWithBWE.from_config(cfg)


@pytest.fixture(params = ["fake", "diffusers"])
def build(request):
    raw = _build_fake if request.param == "fake" else _build_diffusers

    def make(dtype = torch.float32):
        torch.manual_seed(0)
        voc = raw().eval()
        with torch.no_grad():
            for p in voc.parameters():
                p.add_(torch.randn_like(p) * 0.05)
            # a bf16 checkpoint: weights and persistent buffers hold bf16 values; the non-persistent resampler
            # filter is computed at build time in float32
            for t in voc.state_dict().values():
                if t.is_floating_point():
                    t.copy_(t.bfloat16().float())
        return voc.to(dtype)

    make.kind = request.param
    return make


def _reference_and_studio(build):
    """(float32 build of a bf16 checkpoint, the same model as Studio loads it: everything cast to bf16)."""
    ref = build(torch.float32)
    return ref, copy.deepcopy(ref).to(torch.bfloat16)


def _mel(dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 2, 12, 8, generator = g).to(dtype)


def _dtypes(voc):
    return {
        t.dtype
        for t in list(voc.parameters()) + [b for n, b in voc.named_buffers() if "filter" not in n]
    }


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(L.VOCODER_FP32_ENV, raising = False)
    monkeypatch.setattr(L, "_fallback_logged", [False])


@torch.no_grad()
def test_bf16_vocoder_matches_the_float32_build_bit_for_bit(build):
    ref, voc = _reference_and_studio(build)
    mel = _mel()
    # the bf16 cast rounded the resampler filter: a plain .float() copy is not the reference
    assert not torch.equal(copy.deepcopy(voc).float()(mel.float()), ref(mel.float()))
    L.install_vocoder_fp32_forward(voc)
    out = voc(mel)
    assert out.dtype == torch.float32
    assert torch.equal(out, ref(mel.float()))
    # weights and persistent buffers keep their stored dtype
    assert _dtypes(voc) == {torch.bfloat16}
    assert torch.equal(voc(mel), out)
    # a later cast of the whole pipe rounds the float32 filter; the next call restores it
    voc.to(torch.bfloat16)
    assert torch.equal(voc(mel), out)


@torch.no_grad()
def test_filter_computed_in_bf16_is_rebuilt(build, monkeypatch):
    """diffusers' from_pretrained builds under torch_dtype as the default dtype: the filter is computed in bf16."""
    ref = build(torch.float32)
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
    # the switched-off path runs on the filter the vocoder was loaded with: bit-identical to main
    monkeypatch.setenv(L.VOCODER_FP32_ENV, "0")
    assert torch.equal(voc(mel), base(mel))


@torch.no_grad()
def test_kill_switch_runs_the_stored_dtype(build, monkeypatch):
    _, voc = _reference_and_studio(build)
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
def test_float32_vocoder_is_unchanged_and_install_is_idempotent(build):
    voc = build(torch.float32)
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


def test_a_changed_resampler_filter_is_left_alone(build):
    _, voc = _reference_and_studio(build)
    with torch.no_grad():
        voc.resampler.filter.mul_(0.5)
    stored = voc.resampler.filter.clone()
    L.install_vocoder_fp32_forward(voc)
    assert torch.equal(voc.resampler.filter, stored)


def test_loader_installs_the_float32_forward(monkeypatch):
    pytest.importorskip("diffusers")
    import diffusers.pipelines.ltx2.vocoder as V

    voc = _build_diffusers().to(torch.bfloat16)
    state = dict(voc.state_dict())

    class _Fake:
        @classmethod
        def from_config(cls, cfg):
            return copy.deepcopy(voc)

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


_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


@torch.no_grad()
def test_out_of_memory_runs_the_stored_dtype_and_logs_once(build, monkeypatch):
    _, voc = _reference_and_studio(build)
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    calls = _oom_on_float32(voc, _OOM("CUDA out of memory. Tried to allocate 96.00 MiB"))
    warned = []
    monkeypatch.setattr(L.logger, "warning", lambda *a, **k: warned.append(a))
    count = L.vocoder_fallback_count()
    mel = _mel()
    for _ in range(2):
        out = voc(mel)
        assert out.dtype == torch.bfloat16 and torch.equal(out, base(mel))
    assert calls["fp32"] == 2 and len(warned) == 1
    assert L.vocoder_fallback_count() == count + 2
    assert _dtypes(voc) == {torch.bfloat16}


@torch.no_grad()
def test_runtime_out_of_memory_message_also_falls_back(build):
    _, voc = _reference_and_studio(build)
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    _oom_on_float32(voc, RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"))
    assert torch.equal(voc(_mel()), base(_mel()))


@torch.no_grad()
def test_other_errors_are_not_swallowed(build):
    _, voc = _reference_and_studio(build)
    L.install_vocoder_fp32_forward(voc)
    _oom_on_float32(voc, ValueError("shape mismatch"))
    with pytest.raises(ValueError):
        voc(_mel())


@torch.no_grad()
def test_out_of_memory_while_converting_the_weights_leaves_the_module_intact(build, monkeypatch):
    """An allocation failure on the 2nd parameter's float32 copy: no weight changes dtype, this call and every
    later one (switched on or off) still renders, matching main."""
    ref, voc = _reference_and_studio(build)
    base = copy.deepcopy(voc)
    L.install_vocoder_fp32_forward(voc)
    ptrs = {p.data_ptr() for p in voc.parameters()}
    real_float = torch.Tensor.float
    seen = {"n": 0, "armed": True}

    def flaky_float(self, *a, **k):
        if seen["armed"] and self.data_ptr() in ptrs:
            seen["n"] += 1
            if seen["n"] == 2:
                raise _OOM("out of memory on the 2nd parameter")
        return real_float(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "float", flaky_float)
    mel = _mel()
    out = voc(mel)
    assert seen["n"] >= 2
    assert torch.equal(out, base(mel))
    assert _dtypes(voc) == {torch.bfloat16}
    seen["armed"] = False
    monkeypatch.setenv(L.VOCODER_FP32_ENV, "0")
    assert torch.equal(voc(mel), base(mel))
    monkeypatch.delenv(L.VOCODER_FP32_ENV)
    assert torch.equal(voc(mel), ref(mel.float()))


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_an_enclosing_autocast_does_not_lower_the_float32_pass(build, dtype):
    ref, voc = _reference_and_studio(build)
    voc32 = copy.deepcopy(ref)
    L.install_vocoder_fp32_forward(voc)
    L.install_vocoder_fp32_forward(voc32)
    mel = _mel()
    want = ref(mel.float())
    with torch.autocast("cpu", dtype = dtype):
        a, b = voc(mel), voc32(mel)
    assert a.dtype == b.dtype == torch.float32
    assert torch.equal(a, want) and torch.equal(b, want)


@pytest.mark.parametrize("stored", [torch.bfloat16, torch.float32])
def test_the_forward_compiles_without_graph_breaks(build, stored):
    import torch._dynamo as dynamo

    ref, voc = _reference_and_studio(build)
    voc = voc.to(stored)
    L.install_vocoder_fp32_forward(voc)
    mel = _mel()
    with torch.no_grad():
        dynamo.reset()
        explained = dynamo.explain(voc)(mel)
        assert explained.graph_break_count == 0, explained.break_reasons
        dynamo.reset()
        out = torch.compile(voc, fullgraph = True, backend = "eager")(mel)
    assert out.shape == ref(mel.float()).shape
    dynamo.reset()


def test_fp32_extra_scales_with_clip_length():
    small = L.vocoder_fp32_extra_bytes(torch.zeros(1, 2, 501, 64))
    big = L.vocoder_fp32_extra_bytes(torch.zeros(1, 2, 2004, 64))
    # the measured +104.5 MiB at 501 frames, with margin
    assert 105 << 20 < small < 300 << 20
    assert big > 3 * small


def test_vocoder_precision_keys_the_render_peaks(monkeypatch):
    from core.inference import video as V
    from core.inference import video_stream_residency as vr

    voc = _build_fake().to(torch.bfloat16)
    pipe = types.SimpleNamespace(vocoder = voc)
    assert (
        V._video_calibration_vocoder(pipe) == {}
    )  # not installed: every other vocoder's key is unchanged
    L.install_vocoder_fp32_forward(voc)
    on = V._video_calibration_vocoder(pipe)
    monkeypatch.setenv(L.VOCODER_FP32_ENV, "0")
    off = V._video_calibration_vocoder(pipe)
    assert on == {"vocoder_precision": "fp32"} and off == {"vocoder_precision": "stored"}
    assert vr.calibration_key("ltx-2.5", **on) != vr.calibration_key("ltx-2.5", **off)


def test_a_request_whose_vocoder_fell_back_records_no_keyed_peak(monkeypatch):
    import core.inference.diffusion_measured_peaks as mp
    import core.inference.diffusion_memory as dm
    from core.inference import video_stream_residency as vr

    monkeypatch.setattr(vr, "_PEAKS", {})
    monkeypatch.setattr(vr, "_live_graphs", lambda dev: 0)
    monkeypatch.setattr(vr, "_capture_generation", lambda dev: 0)
    monkeypatch.setattr(vr, "_store_key", lambda key: None)
    monkeypatch.setattr(mp, "open_window", lambda dev: object())
    monkeypatch.setattr(mp, "close_window", lambda dev, token: True)
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 3000 << 20)
    dev = torch.device("cuda", 0)

    def request(key, fall_back):
        module = types.SimpleNamespace(_unsloth_video_calibration_key = key)
        pipe = types.SimpleNamespace(transformer = module)
        vr._open_window(module, dev)
        module._unsloth_video_pending = (100, 1000 << 20, dev)
        module._unsloth_video_failed_captures = vr._failed_captures()
        module._unsloth_video_vocoder_fallbacks = vr._vocoder_fallbacks()
        if fall_back:
            L._note_fallback("out of memory in the float32 pass")
        vr.record_request_peak(pipe)

    fell = vr.calibration_key("ltx-2.5", vocoder_precision = "fp32", case = "fell")
    clean = vr.calibration_key("ltx-2.5", vocoder_precision = "fp32", case = "clean")
    request(fell, True)
    request(clean, False)
    assert vr.measured_extra_mib(fell, 100) is None
    assert vr.measured_extra_mib(clean, 100) == 2000
