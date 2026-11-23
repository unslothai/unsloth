# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from unsloth.kernels import fp8 as F8

L2 = 48 * 1024 * 1024  # L4
SMALL = (6144, 2048)  # Qwen3-1.7B MLP: 24 MiB in bf16, L2 resident
LARGE = (14336, 4096)  # Llama-3.1-8B MLP: 112 MiB in bf16


@pytest.fixture
def route(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "scaled_mm")
    monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", True)

    def run(
        rows,
        shape,
        l2 = L2,
        env = None,
    ):
        monkeypatch.setattr(F8, "_fp8_l2_cache_bytes", lambda device: l2)
        parsed = F8._parse_fp8_dequant_max_rows(env)
        monkeypatch.setattr(F8, "_FP8_DEQUANT_MAX_ROWS_ENV", parsed)
        monkeypatch.setattr(F8, "_FP8_DEQUANT_MAX_ROWS", 64 if parsed is None else parsed)
        N, K = shape
        with FakeTensorMode(), torch.no_grad():
            X = torch.empty(1, rows, K, device = "cuda", dtype = torch.bfloat16)
            w = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
            s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
            return F8._fp8_rowwise_direct_route(X, w, s)

    return run


@pytest.mark.parametrize("rows", [17, 32, 64])
def test_small_weight_stays_on_dequant_and_large_weight_goes_to_scaled_mm(route, rows):
    assert route(rows, SMALL) == "dequant"
    assert route(rows, LARGE) == "scaled_mm"


def test_row_bounds_are_unchanged(route):
    # 16 rows and below are the rows GEMV's; above 64 every weight already used scaled_mm.
    assert route(16, LARGE) == "dequant"
    assert route(65, SMALL) == "scaled_mm"


@pytest.mark.parametrize(
    "autocast_dtype, expected",
    [(None, "scaled_mm"), (torch.bfloat16, "scaled_mm"), (torch.float16, None)],
)
def test_autocast_in_another_dtype_keeps_the_compiled_path(
    route, monkeypatch, autocast_dtype, expected
):
    monkeypatch.setattr(
        torch, "is_autocast_enabled", lambda device_type = None: autocast_dtype is not None
    )
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device_type: autocast_dtype)
    assert route(32, LARGE) == expected


def test_unknown_l2_keeps_the_64_row_default(route):
    assert route(32, LARGE, l2 = 0) == "dequant"
    assert route(65, LARGE, l2 = 0) == "scaled_mm"


def test_env_override_sets_one_threshold_for_every_weight(route):
    assert route(32, LARGE, env = "64") == "dequant"
    assert route(17, SMALL, env = "16") == "scaled_mm"
    for bad in ("", "abc", "-1", "1.5", None):
        assert F8._parse_fp8_dequant_max_rows(bad) is None


def test_l2_probe_falls_back_to_zero(monkeypatch):
    def boom(device = None):
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    F8._probe_fp8_l2_cache_bytes.cache_clear()
    try:
        assert F8._probe_fp8_l2_cache_bytes(torch.device("cuda", 0)) == 0
    finally:
        F8._probe_fp8_l2_cache_bytes.cache_clear()


@pytest.mark.parametrize(
    "capability, l2, expected",
    [((8, 9), L2, L2), ((9, 0), 50 * 1024 * 1024, 0), ((12, 0), 128 * 1024 * 1024, 0)],
)
def test_l2_probe_only_reports_measured_arches(monkeypatch, capability, l2, expected):
    class Props:
        major, minor = capability
        L2_cache_size = l2

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device = None: Props())
    F8._probe_fp8_l2_cache_bytes.cache_clear()
    try:
        assert F8._probe_fp8_l2_cache_bytes(torch.device("cuda", 0)) == expected
    finally:
        F8._probe_fp8_l2_cache_bytes.cache_clear()


@pytest.mark.parametrize(
    "shape, scaled, dtype, out_broken",
    [
        (LARGE, True, torch.bfloat16, True),
        (SMALL, False, torch.bfloat16, True),
        (LARGE, False, torch.float32, True),
        (LARGE, True, torch.float32, False),
    ],
)
def test_autograd_function_uses_the_same_gate(monkeypatch, shape, scaled, dtype, out_broken):
    calls = []

    def fake_scaled_mm(a, b, **kwargs):
        calls.append("scaled_mm")
        return torch.empty(a.shape[0], b.shape[1], device = a.device, dtype = kwargs["out_dtype"])

    def fake_dequant(w, s, dtype):
        calls.append("dequant")
        return torch.empty(w.shape, device = w.device, dtype = dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    monkeypatch.setattr(F8, "weight_dequant", fake_dequant)
    monkeypatch.setattr(F8, "_fp8_l2_cache_bytes", lambda device: L2)
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", out_broken)
    monkeypatch.setattr(F8, "_scaled_mm_float32_out", lambda device: False)
    N, K = shape
    with FakeTensorMode(), torch.no_grad():
        X = torch.empty(1, 32, K, device = "cuda", dtype = dtype)
        w = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
        s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
        y = F8.FbgemmFp8Linear_matmul.apply(X, w, s, None, "scaled_mm")
    assert y.shape == (1, 32, N)
    assert calls == ["scaled_mm" if scaled else "dequant"]


def test_scaled_mm_asks_for_bf16_when_activations_are_fp16(monkeypatch):
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_DEQUANT_ONLY", True)
    assert F8._scaled_mm_out_dtype(torch.float16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.bfloat16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.float32) is torch.float32


def test_scaled_mm_keeps_fp16_output_where_torch_writes_fp16(monkeypatch):
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_DEQUANT_ONLY", False)
    assert F8._scaled_mm_out_dtype(torch.float16) is torch.float16
    assert F8._scaled_mm_out_dtype(torch.bfloat16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.float32) is torch.float32


def test_scaled_mm_fp16_workaround_follows_the_installed_torch():
    from torch.torch_version import TorchVersion

    broken = torch.__version__ < "2.14.0"
    detour = broken and not F8._SCALED_MM_FP16_BIAS_OUT
    assert F8._SCALED_MM_FP16_OUT_BROKEN is broken
    assert F8._SCALED_MM_FP16_DEQUANT_ONLY is detour
    assert F8._SCALED_MM_FP16_ZERO_BIAS is (broken and F8._SCALED_MM_FP16_BIAS_OUT)
    assert F8._scaled_mm_out_dtype(torch.float16) is (torch.bfloat16 if detour else torch.float16)
    assert TorchVersion("2.13.9+cu130") < "2.14.0"
    assert TorchVersion("2.14.0a0+git0") < "2.14.0"
    assert not TorchVersion("2.14.0+cu130") < "2.14.0"
    assert not TorchVersion("2.15.0") < "2.14.0"


def test_every_rowwise_scaled_mm_call_uses_the_out_dtype_helper():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(F8))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_scaled_mm"
    ]
    out = {}
    bias = {}
    for node in calls:
        for kw in node.keywords:
            if kw.arg == "out_dtype":
                out[node.lineno] = ast.unparse(kw.value)
            if kw.arg == "bias":
                bias[node.lineno] = ast.unparse(kw.value)
    data_calls = {
        line: v for line, v in out.items() if v not in ("torch.bfloat16", "torch.float32")
    }
    assert len(data_calls) >= 2, out
    assert all(v.startswith("_scaled_mm_out_dtype(") for v in data_calls.values()), out
    assert all(bias.get(line, "").startswith("_scaled_mm_fp16_bias(") for line in data_calls), bias


@pytest.mark.parametrize("float32_out, expected", [(True, "scaled_mm"), (False, "dequant")])
def test_float32_keeps_scaled_mm_where_rowwise_writes_float32(monkeypatch, float32_out, expected):
    calls = []

    def fake_scaled_mm(a, b, **kwargs):
        calls.append(("scaled_mm", kwargs["out_dtype"]))
        return torch.empty(a.shape[0], b.shape[1], device = a.device, dtype = kwargs["out_dtype"])

    def fake_dequant(w, s, dtype):
        calls.append(("dequant", dtype))
        return torch.empty(w.shape, device = w.device, dtype = dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    monkeypatch.setattr(F8, "weight_dequant", fake_dequant)
    monkeypatch.setattr(F8, "_fp8_l2_cache_bytes", lambda device: L2)
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", True)
    monkeypatch.setattr(F8, "_scaled_mm_float32_out", lambda device: float32_out)
    N, K = LARGE
    with FakeTensorMode(), torch.no_grad():
        X = torch.empty(1, 128, K, device = "cuda", dtype = torch.float32)
        w = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
        s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
        y = F8.FbgemmFp8Linear_matmul.apply(X, w, s, None, "scaled_mm")
    assert y.shape == (1, 128, N) and y.dtype == torch.float32
    assert calls == [(expected, torch.float32)]


@pytest.mark.parametrize("cutlass", [True, False])
def test_float32_probe_asks_scaled_mm_for_float32(monkeypatch, cutlass):
    seen = []

    def fake_scaled_mm(a, b, scale_a, scale_b, out_dtype, **kwargs):
        seen.append((tuple(scale_a.shape), tuple(scale_b.shape), out_dtype))
        if cutlass and out_dtype == torch.float32:
            raise ValueError(
                "Only bf16 and fp16 high precision output types are supported for row-wise scaling."
            )
        return torch.full((a.shape[0], b.shape[1]), 128.0, dtype = out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    F8._probe_scaled_mm_float32_out.cache_clear()
    try:
        assert F8._probe_scaled_mm_float32_out(torch.device("cpu")) is (not cutlass)
    finally:
        F8._probe_scaled_mm_float32_out.cache_clear()
    assert seen == [((128, 1), (1, 128), torch.float32)]


@pytest.mark.parametrize("rows, width", [(1, 32), (64, 32), (128, 33)])
def test_float32_dequant_calls_do_not_probe_scaled_mm(monkeypatch, rows, width):
    def unexpected_probe(device):
        raise AssertionError("dequant-only calls must not run the device probe")

    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", True)
    monkeypatch.setattr(F8, "_fp8_dequant_max_rows", lambda *args: 64)
    monkeypatch.setattr(F8, "_scaled_mm_float32_out", unexpected_probe)
    x = torch.randn(rows, 32)
    weight = torch.randn(width, 32).to(torch.float8_e4m3fn)
    scales = torch.ones(width, 1)
    result = F8.FbgemmFp8Linear_matmul.apply(x, weight, scales, None, "scaled_mm")
    torch.testing.assert_close(result, x @ weight.float().t())
