# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple Silicon int8 weight-only; kernel math checked on CPU, dispatch tests need real MPS."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq
import core.inference.diffusion_transformer_quant as tq

_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
_HAS_OP = hasattr(torch, "_weight_int8pack_mm")


def _mps_target(dtype = None):
    return types.SimpleNamespace(device = "mps", dtype = torch.bfloat16 if dtype is None else dtype)


class _Toy(torch.nn.Module):
    def __init__(self, dim = 128):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.ff = torch.nn.Linear(dim, 2 * dim, bias = False)
        self.out = torch.nn.Linear(2 * dim, dim)

    def forward(self, x):
        return x + self.out(torch.nn.functional.gelu(self.ff(self.to_q(x))))


def _block_torchao(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setitem(sys.modules, "torchao.quantization", None)


@pytest.mark.parametrize("scheme", ["int8", "INT8", " int8 "])
def test_mps_bf16_takes_explicit_int8(scheme):
    assert tq.native_quant_host(_mps_target())
    assert tq.native_quant_scheme(_mps_target(), scheme) == "int8"
    assert tq.native_quant_scheme(_mps_target(), scheme, offload = True) == "int8"


@pytest.mark.parametrize("scheme", [None, "auto", "none", "off", "fp8", "mxfp8"])
def test_mps_refuses_everything_but_int8(scheme):
    # ``auto`` stays dense until measured on real Apple Silicon.
    assert tq.native_quant_scheme(_mps_target(), scheme) is None


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mps_without_bf16_is_not_native(dtype):
    target = _mps_target(getattr(torch, dtype))
    assert not tq.native_quant_host(target)
    assert tq.native_quant_scheme(target, "int8") is None


def test_mps_never_takes_int8_activations(monkeypatch):
    monkeypatch.setenv(nq.NATIVE_INT8_ACT_ENV, "1")
    assert tq.native_int8_act(_mps_target()) is False


def test_mps_family_deny_list_still_applies(monkeypatch):
    monkeypatch.setitem(tq._FAMILY_SCHEME_DENY, "z-image", frozenset({tq.TQ_INT8}))
    assert tq.native_quant_scheme(_mps_target(), "int8", family = "z-image") is None


def test_snap_scale_up_is_exact_in_the_dtype_and_never_clips():
    torch.manual_seed(0)
    scale = torch.rand(4096) * 0.05 + 1e-6
    for dtype in (torch.bfloat16, torch.float16):
        snapped = nq.snap_scale_up(scale, dtype)
        assert snapped.dtype == torch.float32
        assert torch.equal(snapped.to(dtype).float(), snapped)
        assert bool((snapped >= scale).all())
        assert float(((snapped - scale) / scale).max()) <= 2.0**-7


def test_quantize_transformer_on_mps_snaps_scales_and_skips_w8a8(monkeypatch):
    _block_torchao(monkeypatch)
    monkeypatch.setenv(nq.NATIVE_INT8_ACT_ENV, "1")
    torch.manual_seed(0)
    model = _Toy().to(torch.bfloat16)
    x = torch.randn(5, 128, dtype = torch.bfloat16)
    with torch.no_grad():
        ref = model(x).float()
    pipe = types.SimpleNamespace(transformer = model)
    # act_int8 = True is overridden on MPS.
    got = tq.quantize_transformer(pipe, _mps_target(), mode = "int8", min_features = 64, act_int8 = True)
    assert got == "int8"
    layers = [m for m in model.modules() if nq.is_native_linear(m)]
    assert len(layers) == 3
    for layer in layers:
        assert not layer.act_int8 and layer.rot_group == 0
        scale = layer.weight_scale.view(torch.float32)
        assert torch.equal(scale.to(torch.bfloat16).float(), scale)
        assert int(layer.weight_q.abs().max()) <= 127
    with torch.no_grad():
        out = model(x).float()
    assert ((out - ref).norm() / ref.norm()).item() < 0.05


def test_other_hosts_keep_unsnapped_scales():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64).to(torch.bfloat16)
    layer = nq.native_linear_class()(lin, "int8")
    scale = layer.weight_scale.view(torch.float32)
    expect = lin.weight.float().abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
    assert torch.equal(scale, expect)


def _layer(
    n_in = 128,
    n_out = 64,
    bias = True,
    dtype = torch.bfloat16,
):
    torch.manual_seed(0)
    lin = torch.nn.Linear(n_in, n_out, bias = bias).to(dtype)
    return lin, nq.native_linear_class()(lin, "int8", scale_dtype = dtype)


@pytest.mark.skipif(not _HAS_OP, reason = "torch has no _weight_int8pack_mm")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_int8pack_math_matches_the_dequantised_layer_on_cpu(dtype):
    _, layer = _layer(dtype = dtype)
    x = torch.randn(2, 7, 128, dtype = dtype)
    try:
        got = layer._forward_int8pack(x)
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"no CPU int8pack kernel for {dtype}: {exc}")
    assert got is not None and got.shape == (2, 7, 64) and got.dtype == dtype
    ref = torch.nn.functional.linear(
        x.float(), layer.dequantized_weight(torch.float32), layer.bias.float()
    )
    assert ((got.float() - ref).norm() / ref.norm()).item() < 1e-2


def test_inexact_scales_decline_the_kernel():
    # A silent cast of non-bf16-exact checkpoint scales would bias every row.
    lin = torch.nn.Linear(64, 32).to(torch.bfloat16)
    codes = torch.randint(-127, 128, (32, 64), dtype = torch.int8)
    scale = torch.full((32,), 1.0 + 2.0**-12)
    layer = nq.native_linear_class()(lin, "int8", codes = codes, scale = scale)
    assert layer._int8pack_scale(torch.bfloat16, "cpu") is None
    assert layer._forward_int8pack(torch.randn(3, 64, dtype = torch.bfloat16)) is None


def test_cpu_inputs_never_take_the_mps_kernel(monkeypatch):
    _, layer = _layer()
    monkeypatch.setattr(
        layer, "_forward_int8pack", lambda x: pytest.fail("int8pack on a CPU input")
    )
    x = torch.randn(4, 128, dtype = torch.bfloat16)
    assert torch.equal(
        layer(x), torch.nn.functional.linear(x, layer.dequantized_weight(x.dtype), layer.bias)
    )


@pytest.mark.skipif(not _MPS, reason = "needs an MPS device")
def test_mps_forward_runs_the_kernel_and_matches_dequantise(monkeypatch):
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "1")
    _, layer = _layer(n_in = 256, n_out = 96)
    layer = layer.to("mps")
    calls = []
    real = torch._weight_int8pack_mm
    monkeypatch.setattr(torch, "_weight_int8pack_mm", lambda *a: calls.append(1) or real(*a))
    x = torch.randn(3, 40, 256, dtype = torch.bfloat16, device = "mps")
    got = layer(x)
    assert calls, "the Metal kernel was not used"
    ref = torch.nn.functional.linear(
        x.float(), layer.dequantized_weight(torch.float32), layer.bias.float()
    )
    assert ((got.float() - ref).norm() / ref.norm()).item() < 1e-2
    xt = torch.randn(256, 40, dtype = torch.bfloat16, device = "mps").t()
    ref_t = torch.nn.functional.linear(
        xt.float(), layer.dequantized_weight(torch.float32), layer.bias.float()
    )
    assert ((layer(xt).float() - ref_t).norm() / ref_t.norm()).item() < 1e-2


@pytest.mark.skipif(not _MPS, reason = "needs an MPS device")
def test_mps_kill_switch_and_unaligned_shapes_keep_dequantise(monkeypatch):
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "1")
    monkeypatch.setattr(torch, "_weight_int8pack_mm", lambda *a: pytest.fail("kernel used"))
    _, odd = _layer(n_in = 128, n_out = 40)  # N % 32 != 0
    odd = odd.to("mps")
    odd(torch.randn(4, 128, dtype = torch.bfloat16, device = "mps"))
    _, layer = _layer()
    layer = layer.to("mps")
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "0")
    layer(torch.randn(4, 128, dtype = torch.bfloat16, device = "mps"))


def _image_gate(
    monkeypatch,
    family,
    pinned,
    model_kind = "pipeline",
):
    import core.inference.diffusion as d
    from core.inference.diffusion_families import _FAMILIES

    fam = next(f for f in _FAMILIES if f.name == family)
    monkeypatch.setattr(d, "effective_te_quant", lambda *a, **k: None)
    d.DiffusionBackend._assert_precision_for_target(
        None,
        fam,
        _mps_target(),
        model_kind = model_kind,
        pinned = pinned,
        te_mode = None,
        memory_mode = None,
        cpu_offload = False,
    )


@pytest.mark.parametrize("family", ["z-image", "qwen-image-2.1", "flux.2-klein"])
def test_image_gate_admits_int8_on_mps(monkeypatch, family):
    _image_gate(monkeypatch, family, "int8")


@pytest.mark.parametrize("pinned", ["fp8", "mxfp8"])
def test_image_gate_still_refuses_other_schemes_on_mps(monkeypatch, pinned):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, "z-image", pinned)


def test_image_gate_refuses_int8_on_a_gguf_load_on_mps(monkeypatch):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, "z-image", "int8", model_kind = "gguf")


def test_reloaded_scales_are_not_served_from_the_cache():
    _, layer = _layer()
    first = layer._int8pack_scale(torch.bfloat16, "cpu").clone()
    state = layer.state_dict()
    state["weight_scale"] = (state["weight_scale"].view(torch.float32) * 2).view(
        state["weight_scale"].dtype
    )
    layer.load_state_dict(state)
    assert torch.equal(layer._int8pack_scale(torch.bfloat16, "cpu").float(), first.float() * 2)


def _reported_layer(engaged):
    return types.SimpleNamespace(
        scheme = "int8",
        act_int8 = False,
        bias = None,
        weight_q = types.SimpleNamespace(device = torch.device("mps")),
        int8pack_engaged = lambda dtype: engaged,
    )


@pytest.mark.parametrize("engaged", [True, False])
def test_reason_names_the_metal_kernel_only_when_it_dispatches(monkeypatch, engaged):
    monkeypatch.setattr(nq, "_first_native_linear", lambda module: _reported_layer(engaged))
    assert ("Apple Silicon int8 kernel" in nq.native_quant_reason(object(), "int8")) is engaged


def test_kill_switch_and_alignment_close_the_kernel_for_the_status_line(monkeypatch):
    _, layer = _layer()
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "1")
    assert layer._int8pack_layer_ok(torch.bfloat16) is _HAS_OP
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "0")
    assert not layer._int8pack_layer_ok(torch.bfloat16)
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, "1")
    _, odd = _layer(n_out = 40)
    assert not odd._int8pack_layer_ok(torch.bfloat16)
    assert not layer.int8pack_engaged(torch.bfloat16)  # stored on CPU


@pytest.fixture(autouse = True)
def _no_probe_state(monkeypatch):
    monkeypatch.delenv(nq.NATIVE_MPS_INT8PACK_ENV, raising = False)
    monkeypatch.setattr(nq, "_INT8PACK_PROBED", {})


def _toy_native():
    toy = _Toy().to(torch.bfloat16)
    nq.apply_native_weight_quant(
        toy, "int8", filter_fn = lambda m, n: True, act_int8 = False, scale_dtype = torch.bfloat16
    )
    return toy


def test_unprobed_auto_mode_keeps_dequantise(_no_probe_state):
    # No verdict yet: the deterministic dequantise path, never the kernel by default.
    _, layer = _layer()
    assert not layer._int8pack_layer_ok(torch.bfloat16)
    assert layer._int8pack_layer_ok(torch.bfloat16, probe = True) is _HAS_OP


@pytest.mark.skipif(not _HAS_OP, reason = "torch has no _weight_int8pack_mm")
@pytest.mark.parametrize(
    "kernel_s, dense_s, expect",
    [
        (0.6, 0.8, True),  # M1-like: bf16 GEMM has no fast path
        (1.02, 1.0, False),  # M3-like: parity
        (4.1, 1.0, False),  # M5-like: 16-bit GEMM on the matrix units
        (0.95, 1.0, False),  # under the margin: not worth a nondeterministic kernel
    ],
)
def test_probe_takes_the_kernel_only_when_it_beats_bf16(
    _no_probe_state, monkeypatch, kernel_s, dense_s, expect
):
    seen = []

    def bench(layer, dtype, device):
        seen.append((layer.in_features, layer.out_features))
        return kernel_s, dense_s

    monkeypatch.setattr(nq, "_bench_int8pack_vs_dense", bench)
    toy = _toy_native()
    assert nq.probe_mps_int8pack(toy, torch.bfloat16, device = "cpu") is expect
    assert seen == [(128, 256)]  # the largest eligible layer, probed once
    assert nq.mps_int8pack_selected(torch.bfloat16) is expect
    assert nq.probe_mps_int8pack(toy, torch.bfloat16, device = "cpu") is expect
    assert len(seen) == 1


@pytest.mark.skipif(not _HAS_OP, reason = "torch has no _weight_int8pack_mm")
def test_probe_verdict_is_kept_per_model_shape(_no_probe_state, monkeypatch):
    # A second model in the same process with a different largest layer is probed again, and each model's layers
    # follow their own verdict; the same shape reuses the cached verdict.
    verdicts = {(128, 256): (0.6, 0.8), (256, 512): (4.1, 1.0)}
    seen = []

    def bench(layer, dtype, device):
        seen.append((layer.in_features, layer.out_features))
        return verdicts[(layer.in_features, layer.out_features)]

    monkeypatch.setattr(nq, "_bench_int8pack_vs_dense", bench)
    first = _toy_native()
    assert nq.probe_mps_int8pack(first, torch.bfloat16, device = "cpu") is True
    big = torch.nn.Sequential(torch.nn.Linear(256, 512, bias = False)).to(torch.bfloat16)
    nq.apply_native_weight_quant(
        big, "int8", filter_fn = lambda m, n: True, act_int8 = False, scale_dtype = torch.bfloat16
    )
    assert nq.probe_mps_int8pack(big, torch.bfloat16, device = "cpu") is False
    assert seen == [(128, 256), (256, 512)]
    assert first.ff._int8pack_layer_ok(torch.bfloat16) and not big[0]._int8pack_layer_ok(
        torch.bfloat16
    )
    again = _toy_native()
    assert nq.probe_mps_int8pack(again, torch.bfloat16, device = "cpu") is True
    assert len(seen) == 2 and again.ff._int8pack_layer_ok(torch.bfloat16)


@pytest.mark.parametrize("env, expect", [("0", False), ("1", True)])
def test_env_override_skips_the_probe(_no_probe_state, monkeypatch, env, expect):
    monkeypatch.setenv(nq.NATIVE_MPS_INT8PACK_ENV, env)
    monkeypatch.setattr(
        nq, "_bench_int8pack_vs_dense", lambda *a: pytest.fail("probed despite the override")
    )
    assert nq.probe_mps_int8pack(_toy_native(), torch.bfloat16, device = "cpu") is expect


@pytest.mark.skipif(not _HAS_OP, reason = "torch has no _weight_int8pack_mm")
def test_failed_probe_keeps_dequantise(_no_probe_state, monkeypatch):
    def boom(*a):
        raise RuntimeError("no Metal device")

    monkeypatch.setattr(nq, "_bench_int8pack_vs_dense", boom)
    assert nq.probe_mps_int8pack(_toy_native(), torch.bfloat16, device = "cpu") is False
    assert not nq.mps_int8pack_selected(torch.bfloat16)


@pytest.mark.skipif(not _HAS_OP, reason = "torch has no _weight_int8pack_mm")
def test_bench_times_both_arms_on_the_real_layer_shape(_no_probe_state):
    toy = _toy_native()
    try:
        kernel_s, dense_s = nq._bench_int8pack_vs_dense(toy.ff, torch.bfloat16, torch.device("cpu"))
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"no CPU int8pack kernel: {exc}")
    assert kernel_s > 0 and dense_s > 0


def test_quantize_transformer_on_mps_runs_the_probe(_no_probe_state, monkeypatch):
    _block_torchao(monkeypatch)
    calls = []
    monkeypatch.setattr(
        nq, "probe_mps_int8pack", lambda module, dtype, **k: calls.append(dtype) or False
    )
    pipe = types.SimpleNamespace(transformer = _Toy().to(torch.bfloat16))
    assert tq.quantize_transformer(pipe, _mps_target(), mode = "int8", min_features = 64) == "int8"
    assert calls == [torch.bfloat16]


@pytest.mark.parametrize("pinned", ["int8"])
def test_video_gate_admits_int8_on_mps(monkeypatch, pinned):
    # video.py shares native_quant_host, so Mac video int8 takes the same native path as images.
    import core.inference.video as v
    from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

    fam = next(f for f in VIDEO_FAMILIES if f.name == "wan2.2-ti2v-5b")
    monkeypatch.setattr(v, "effective_te_quant", lambda *a, **k: None)
    v._assert_video_precision_for_target(
        fam, _mps_target(), model_kind = "pipeline", transformer_quant = pinned
    )


def test_video_gate_still_refuses_fp8_on_mps(monkeypatch):
    import core.inference.video as v
    from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

    fam = next(f for f in VIDEO_FAMILIES if f.name == "wan2.2-ti2v-5b")
    monkeypatch.setattr(v, "effective_te_quant", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        v._assert_video_precision_for_target(
            fam, _mps_target(), model_kind = "pipeline", transformer_quant = "fp8"
        )


@pytest.mark.skipif(not _MPS, reason = "needs an MPS device")
def test_mps_probe_times_the_real_kernel_and_records_a_verdict():
    toy = _toy_native().to("mps")
    verdict = nq.probe_mps_int8pack(toy, torch.bfloat16)
    assert nq._INT8PACK_PROBED == {(torch.bfloat16, 128, 256): verdict}
    assert nq.mps_int8pack_selected(torch.bfloat16) is verdict
    assert toy.ff.int8pack_engaged(torch.bfloat16) is verdict
