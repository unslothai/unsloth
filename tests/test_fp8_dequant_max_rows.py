# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""scaled_mm rowwise FP8 backend: 17-64 rows go to torch._scaled_mm only when the 16-bit weight spills L2 (CPU)."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from unsloth.kernels import fp8 as F8

L2 = 48 * 1024 * 1024  # L4
SMALL = (6144, 2048)  # Qwen3-1.7B MLP: 24 MiB in bf16, L2 resident
LARGE = (14336, 4096)  # Llama-3.1-8B MLP: 112 MiB in bf16


@pytest.fixture
def route(monkeypatch):
    # NVIDIA routing; on ROCm the direct path is off and every call keeps the compiled path.
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
    """Only sm89 was measured: H100 without FBGEMM (50 MiB L2) keeps the 64 row default."""

    class Props:
        major, minor = capability
        L2_cache_size = l2

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device = None: Props())
    F8._probe_fp8_l2_cache_bytes.cache_clear()
    try:
        assert F8._probe_fp8_l2_cache_bytes(torch.device("cuda", 0)) == expected
    finally:
        F8._probe_fp8_l2_cache_bytes.cache_clear()


@pytest.mark.parametrize("shape, scaled", [(LARGE, True), (SMALL, False)])
def test_autograd_function_uses_the_same_gate(monkeypatch, shape, scaled):
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
    N, K = shape
    with FakeTensorMode(), torch.no_grad():
        X = torch.empty(1, 32, K, device = "cuda", dtype = torch.bfloat16)
        w = torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn)
        s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
        y = F8.FbgemmFp8Linear_matmul.apply(X, w, s, None, "scaled_mm")
    assert y.shape == (1, 32, N)
    assert calls == ["scaled_mm" if scaled else "dequant"]


def test_scaled_mm_asks_for_bf16_when_activations_are_fp16(monkeypatch):
    """torch < 2.14's rowwise CUTLASS kernel writes bf16 bits into an fp16 output when there is no
    bias: fp16 calls request bf16 and cast back; bf16 / fp32 are unchanged."""
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", True)
    assert F8._scaled_mm_out_dtype(torch.float16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.bfloat16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.float32) is torch.float32


def test_scaled_mm_keeps_fp16_output_where_torch_writes_fp16(monkeypatch):
    """torch >= 2.14 honours out.dtype; a bf16 detour would lose precision and overflow 65408..65504."""
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_OUT_BROKEN", False)
    assert F8._scaled_mm_out_dtype(torch.float16) is torch.float16
    assert F8._scaled_mm_out_dtype(torch.bfloat16) is torch.bfloat16
    assert F8._scaled_mm_out_dtype(torch.float32) is torch.float32


def test_scaled_mm_fp16_workaround_follows_the_installed_torch():
    from torch.torch_version import TorchVersion

    broken = torch.__version__ < "2.14.0"
    assert F8._SCALED_MM_FP16_OUT_BROKEN is broken
    assert F8._scaled_mm_out_dtype(torch.float16) is (torch.bfloat16 if broken else torch.float16)
    # 2.14 pre-releases keep the workaround.
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
    for node in calls:
        for kw in node.keywords:
            if kw.arg == "out_dtype":
                out[node.lineno] = ast.unparse(kw.value)
    # The backend probe asks for bf16 directly; every activation-dtype call goes through the helper.
    data_calls = {line: v for line, v in out.items() if v != "torch.bfloat16"}
    assert len(data_calls) >= 2, out
    assert all(v.startswith("_scaled_mm_out_dtype(") for v in data_calls.values()), out
