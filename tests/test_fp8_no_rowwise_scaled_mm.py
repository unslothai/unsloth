# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Torch builds without the row-wise fp8 _scaled_mm (the Windows wheels raise "Rowwise scaling is not currently
supported on your device"; tensorwise still runs): fp8 LoRA, FP8 inference and load_in_fp8 fall back, on CPU
with a fake _scaled_mm."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from unsloth.kernels import fp8 as F8
from unsloth.kernels import fp8_lora as FL

WINDOWS_MESSAGE = "Rowwise scaling is not currently supported on your device"
# torch 2.10 Windows wheels (RTX 5090) get past that guard and fail inside cuBLASLt instead.
CUBLAS_MESSAGE = (
    "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic( ltHandle, "
    "computeDesc.descriptor(), Adesc.descriptor(), Bdesc.descriptor(), Cdesc.descriptor(), Ddesc.descriptor(), "
    "preference.descriptor(), 1, &heuristicResult, &returnedResult)"
)
E4M3 = torch.float8_e4m3fn


def _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype):
    out = (a.float() @ b.float()) * scale_a.float() * scale_b.float()
    return out.to(out_dtype or torch.bfloat16)


def _windows_scaled_mm(calls, message = WINDOWS_MESSAGE):
    """torch._scaled_mm as the Windows wheels run it: tensorwise only."""

    def scaled_mm(
        a,
        b,
        scale_a = None,
        scale_b = None,
        bias = None,
        out_dtype = None,
        use_fast_accum = False,
        **kwargs,
    ):
        rowwise = scale_a.numel() > 1 or scale_b.numel() > 1
        calls.append("row" if rowwise else "tensor")
        if rowwise:
            raise RuntimeError(message)
        out = _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype)
        return out if bias is None else out + bias

    return scaled_mm


@pytest.fixture(params = [WINDOWS_MESSAGE, CUBLAS_MESSAGE], ids = ["rowwise_guard", "cublaslt"])
def windows(monkeypatch, request):
    calls = []
    # Small test weights: lift the L4 size gate (tested on its own below).
    monkeypatch.setattr(F8, "_FP8_TENSORWISE_MIN_NUMEL", 0, raising = False)
    monkeypatch.setattr(F8, "_FP8_TENSORWISE_ANY_ROWS_NUMEL", 0, raising = False)
    monkeypatch.setattr(torch, "_scaled_mm", _windows_scaled_mm(calls, request.param))
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_BF16_ONLY", set())
    return calls


@pytest.mark.parametrize(
    "exc, oom",
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (torch.OutOfMemoryError("allocation failed"), True),
        (RuntimeError(WINDOWS_MESSAGE), False),
        (RuntimeError(CUBLAS_MESSAGE), False),
    ],
)
def test_oom_classifier(exc, oom):
    assert FL._is_oom(exc) is oom


def _rowwise_inputs(
    M = 32,
    N = 48,
    K = 64,
):
    torch.manual_seed(0)
    x = torch.randn(M, K)
    w = torch.randn(N, K) * 0.05
    w8, w_inv = FL.quantize_tensorwise(w, E4M3)
    return x, w8, w_inv, w_inv.reshape(1, 1).expand(1, N).contiguous()


def _row_reference(x, w8, w_inv, dtype):
    x8, x_inv = FL.quantize_rowwise(x, E4M3)
    return ((x8.float() @ w8.float().t()) * x_inv * w_inv).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_lora_forward_and_dgrad_keep_row_scaling(windows, dtype):
    x, w8, w_inv, w_inv_row = _rowwise_inputs()
    x = x.to(dtype)
    out = FL.fp8_lora_fwd(x, w8, w_inv_row)
    assert out.dtype == dtype and out.shape == (32, 48)
    # Per-row activation scales survive: the unit-scale tensorwise GEMM plus a float32 rescale.
    torch.testing.assert_close(out, _row_reference(x, w8, w_inv_row, dtype))
    assert FL._ROWWISE_UNSUPPORTED == {x.device}
    # Recorded once: the next call skips the row-wise attempt.
    calls_before = len(windows)
    FL.fp8_lora_fwd(x, w8, w_inv_row)
    assert windows[calls_before:] == ["tensor"]
    dy = torch.randn(32, 48).to(dtype)
    dx = FL.fp8_lora_dgrad(dy, w8, w_inv_row[:, :1].expand(1, 64).contiguous(), False)
    assert dx.shape == (32, 64) and torch.isfinite(dx).all()
    ref = dy.float() @ (w8.float() * w_inv)
    assert ((dx.float() - ref).norm() / ref.norm()) < 0.1


def test_fp8_lora_base_state_trains_on_the_row_recipe(windows, monkeypatch):
    monkeypatch.setattr(FL, "_SCALING", "row")
    W = torch.randn(48, 64).bfloat16()
    state = FL.Fp8LoRABaseState(W)
    assert state.scaling == "row" and state.w_inv_fwd.dim() == 2
    x = torch.randn(32, 64).bfloat16()
    y = state.matmul(x, W)
    ref = x.float() @ W.float().t()
    assert ((y.float() - ref).norm() / ref.norm()) < 0.08
    dx = state.dgrad(torch.randn(32, 48).bfloat16(), W)
    assert dx.shape == (32, 64) and torch.isfinite(dx).all()


def test_rowwise_scaled_mm_runs_at_unit_scale_instead_of_raising(windows):
    x8 = torch.ones(16, 32).to(E4M3)
    rows, cols = torch.full((16, 1), 0.5), torch.full((1, 16), 0.25)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        out = FL._rowwise_scaled_mm(x8, x8.t(), rows, cols, dtype, True)
        assert out.dtype == dtype and torch.equal(out.float(), torch.full((16, 16), 4.0))
    assert FL._rowwise_ok(torch.float16, x8.device)


def test_other_rowwise_errors_still_raise(monkeypatch):
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_BF16_ONLY", set())

    def broken(*args):
        raise RuntimeError("CUDA error: illegal memory access")

    monkeypatch.setattr(FL, "_scaled_mm", broken)
    x8 = torch.ones(16, 32).to(E4M3)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        FL._rowwise_scaled_mm(
            x8, x8.t(), torch.ones(16, 1), torch.ones(1, 16), torch.bfloat16, True
        )
    assert FL._ROWWISE_UNSUPPORTED == set()


@pytest.mark.parametrize("hip", [None, "7.2.0"], ids = ["cuda_build", "rocm_build"])
def test_rowwise_probe_is_cached_per_device(windows, monkeypatch, hip):
    # The answer comes from the real probe on every build, ROCm included (no torch.version.hip shortcut).
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    device = torch.device("cpu")
    assert FL.rowwise_supported(device) is False
    assert windows == ["tensor", "row"]
    assert FL.rowwise_supported(device) is False
    assert windows == ["tensor", "row"]  # cached per device
    assert FL._ROWWISE_UNSUPPORTED == {device}


def _probe_scaled_mm(row_error = None):
    calls = []

    def scaled_mm(a, b, scale_a, scale_b, out_dtype, fast_accum):
        calls.append("row" if scale_a.numel() > 1 else "tensor")
        if row_error is not None and scale_a.numel() > 1:
            raise row_error
        return _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype)

    return scaled_mm, calls


def test_rowwise_probe_keeps_row_when_it_runs_or_fails_otherwise(monkeypatch):
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    fake, calls = _probe_scaled_mm()
    monkeypatch.setattr(FL, "_scaled_mm", fake)
    assert FL.rowwise_supported(torch.device("cpu")) is True
    assert calls == ["tensor", "row"] and FL._ROWWISE_PROBED == {torch.device("cpu"): True}

    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    fake, calls = _probe_scaled_mm(torch.OutOfMemoryError("CUDA out of memory"))
    monkeypatch.setattr(FL, "_scaled_mm", fake)
    assert FL.rowwise_supported(torch.device("cpu")) is True
    assert FL._ROWWISE_PROBED == {}  # no verdict: probed again next time
    assert FL._ROWWISE_UNSUPPORTED == set()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(CUBLAS_MESSAGE),
        RuntimeError(WINDOWS_MESSAGE),
        RuntimeError("cutlass cannot initialize"),
    ],
)
def test_any_rowwise_probe_failure_means_missing(monkeypatch, error):
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    fake, calls = _probe_scaled_mm(error)
    monkeypatch.setattr(FL, "_scaled_mm", fake)
    assert FL.rowwise_supported(torch.device("cpu")) is False
    assert FL._ROWWISE_UNSUPPORTED == {torch.device("cpu")}


def test_wrong_rowwise_answer_means_missing(monkeypatch):
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)

    def zeros_for_rowwise(a, b, scale_a, scale_b, out_dtype, fast_accum):
        out = _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype)
        return out * 0 if scale_a.numel() > 1 else out

    monkeypatch.setattr(FL, "_scaled_mm", zeros_for_rowwise)
    assert FL.rowwise_supported(torch.device("cpu")) is False


def test_broken_context_after_failed_probe_is_no_verdict(monkeypatch):
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    fake, _ = _probe_scaled_mm(RuntimeError(CUBLAS_MESSAGE))
    monkeypatch.setattr(FL, "_scaled_mm", fake)
    checked = []
    monkeypatch.setattr(FL, "_cuda_context_ok", lambda device: checked.append(device) or False)
    assert FL.rowwise_supported(torch.device("cpu")) is True
    assert checked == [torch.device("cpu")]
    assert FL._ROWWISE_PROBED == {} and FL._ROWWISE_UNSUPPORTED == set()


def test_failed_probe_clears_cuda_state(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: events.append("sync"))
    real_ones = torch.ones

    def ones(
        *args,
        device = None,
        **kwargs,
    ):
        if device is not None and torch.device(device).type == "cuda":
            events.append("launch")
            return real_ones(*args, **kwargs)
        return real_ones(*args, device = device, **kwargs)

    monkeypatch.setattr(torch, "ones", ones)
    assert FL._cuda_context_ok(torch.device("cuda", 0)) is True
    assert events == ["sync", "launch"]
    events.clear()
    F8._clear_failed_probe(torch.device("cuda", 0))
    assert events == ["sync", "launch"]


def test_training_path_raises_when_the_probe_finds_rowwise(monkeypatch):
    # The probe GEMM runs, so a row-wise failure in training is a real error: never swallowed.
    monkeypatch.setattr(FL, "_ROWWISE_UNSUPPORTED", set(), raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_PROBED", {}, raising = False)
    monkeypatch.setattr(FL, "_ROWWISE_BF16_ONLY", set())

    def fails_on_big_rowwise(a, b, scale_a, scale_b, out_dtype, fast_accum):
        if scale_a.numel() > 1 and a.shape[0] != 64:
            raise RuntimeError(CUBLAS_MESSAGE)
        return _reference_scaled_mm(a, b, scale_a, scale_b, out_dtype)

    monkeypatch.setattr(FL, "_scaled_mm", fails_on_big_rowwise)
    x8 = torch.ones(16, 32).to(E4M3)
    with pytest.raises(RuntimeError, match = "CUBLAS_STATUS_NOT_SUPPORTED"):
        FL._rowwise_scaled_mm(
            x8, x8.t(), torch.ones(16, 1), torch.ones(1, 16), torch.bfloat16, True
        )
    assert FL._ROWWISE_UNSUPPORTED == set()


@pytest.mark.parametrize(
    "message", [WINDOWS_MESSAGE, CUBLAS_MESSAGE], ids = ["rowwise_guard", "cublaslt"]
)
def test_inference_backend_probe_picks_tensorwise(monkeypatch, message):
    calls = []
    monkeypatch.setattr(torch, "_scaled_mm", _windows_scaled_mm(calls, message))
    monkeypatch.setattr(F8, "_has_fbgemm_rowwise", lambda: False)
    monkeypatch.setattr(F8, "_FP8_ROWWISE_BACKEND_ENV", "")
    device = torch.device("cpu")
    assert F8._select_fp8_rowwise_backend(device) == "tensorwise"
    F8._probe_fp8_rowwise_backend.cache_clear()
    try:
        assert F8._probe_fp8_rowwise_backend(device) == "tensorwise"
    finally:
        F8._probe_fp8_rowwise_backend.cache_clear()

    def no_fp8(*args, **kwargs):
        raise RuntimeError(
            "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9"
        )

    monkeypatch.setattr(torch, "_scaled_mm", no_fp8)
    assert F8._select_fp8_rowwise_backend(device) == "dequant"


def test_inference_backend_env_override(monkeypatch):
    monkeypatch.setattr(F8, "_FP8_ROWWISE_BACKEND_ENV", "tensorwise")
    monkeypatch.setattr(F8, "_select_fp8_rowwise_backend", lambda device: pytest.fail("probed"))
    F8._probe_fp8_rowwise_backend.cache_clear()
    try:
        assert F8._probe_fp8_rowwise_backend(torch.device("cpu")) == "tensorwise"
    finally:
        F8._probe_fp8_rowwise_backend.cache_clear()


def _fp8_rowwise_weight(N, K):
    w = torch.randn(N, K) * 0.02
    scale = (w.abs().amax(1, keepdim = True) / 448).float()
    return (w / scale).to(E4M3), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tensorwise_backend_matches_rowwise_math(windows, dtype):
    torch.manual_seed(0)
    w, s = _fp8_rowwise_weight(256, 512)
    X = torch.randn(2, 96, 512).to(dtype)
    bias = torch.randn(256)
    out = F8.FbgemmFp8Linear_matmul.forward(F8._NO_CTX(), X, w, s, bias, "tensorwise")
    assert windows and set(windows) == {"tensor"}
    xq, xs = F8._quantize_fp8_per_row(X.reshape(-1, 512))
    expected = ((xq.float() @ w.float().t()) * xs * s.view(1, -1)).to(dtype) + bias
    assert out.shape == (2, 96, 256) and out.dtype == dtype
    torch.testing.assert_close(out, expected.to(dtype).reshape(2, 96, 256))
    dequant = F8.FbgemmFp8Linear_matmul.forward(F8._NO_CTX(), X, w, s, bias, "dequant")
    assert ((out.float() - dequant.float()).norm() / dequant.float().norm()) < 0.05


def test_tensorwise_backend_keeps_dequant_for_decode_rows(windows):
    w, s = _fp8_rowwise_weight(256, 512)
    X = torch.randn(1, 8, 512).bfloat16()
    F8.FbgemmFp8Linear_matmul.forward(F8._NO_CTX(), X, w, s, None, "tensorwise")
    assert windows == []


@pytest.mark.parametrize(
    "rows, dtype, route",
    [
        (64, torch.bfloat16, "dequant"),
        (65, torch.bfloat16, "tensorwise"),
        (4096, torch.bfloat16, "tensorwise"),
        (128, torch.float16, "tensorwise"),
    ],
)
def test_direct_route_takes_tensorwise(monkeypatch, rows, dtype, route):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "tensorwise")
    monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", True)
    monkeypatch.setattr(F8, "_FP8_TENSORWISE_ANY_ROWS_NUMEL", 0, raising = False)
    # Even where fp16 row-wise _scaled_mm must dequantize, the tensorwise backend writes float32 first.
    monkeypatch.setattr(F8, "_SCALED_MM_FP16_DEQUANT_ONLY", True)
    with FakeTensorMode(), torch.no_grad():
        X = torch.empty(1, rows, 512, device = "cuda", dtype = dtype)
        w = torch.empty(256, 512, device = "cuda", dtype = E4M3)
        s = torch.empty(256, 1, device = "cuda", dtype = torch.float32)
        assert F8._fp8_rowwise_direct_route(X, w, s) == route


def test_load_in_fp8_row_switches_to_block_without_rowwise(monkeypatch):
    from unsloth.models import loader_utils as LU

    monkeypatch.setattr(FL, "rowwise_supported", lambda device: False)
    monkeypatch.setattr(LU, "DEVICE_TYPE_TORCH", "cuda")
    monkeypatch.setattr(LU, "_allow_torchao_perblock_safetensors", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    pytest.importorskip("torchao")
    import torchao

    if LU.Version(torch.__version__) < LU.Version("2.9.0") or LU.Version(
        torchao.__version__
    ) < LU.Version("0.15.0"):
        pytest.skip("load_in_fp8 needs torch 2.9+ and torchao 0.15+")
    assert LU._get_fp8_mode_and_check_settings(True, False) == "block"
    assert LU._get_fp8_mode_and_check_settings("block", False) == "block"
    monkeypatch.setattr(FL, "rowwise_supported", lambda device: True)
    assert LU._get_fp8_mode_and_check_settings(True, False) == "row"


def test_tensorwise_size_gate():
    big, mid, small = 4096 * 12288, 4096 * 4096, 1024 * 3072
    assert F8._fp8_tensorwise_wins(65, big)
    assert not F8._fp8_tensorwise_wins(512, mid) and F8._fp8_tensorwise_wins(1024, mid)
    assert not F8._fp8_tensorwise_wins(8192, small)


@pytest.mark.parametrize(
    "rows, N, K, route",
    [
        (128, 4096, 4096, "dequant"),
        (1024, 4096, 4096, "tensorwise"),
        (128, 12288, 4096, "tensorwise"),
        (2048, 1024, 1024, "dequant"),
    ],
)
def test_direct_route_size_gate(monkeypatch, rows, N, K, route):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "tensorwise")
    monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", True)
    with FakeTensorMode(), torch.no_grad():
        X = torch.empty(1, rows, K, device = "cuda", dtype = torch.bfloat16)
        w = torch.empty(N, K, device = "cuda", dtype = E4M3)
        s = torch.empty(N, 1, device = "cuda", dtype = torch.float32)
        assert F8._fp8_rowwise_direct_route(X, w, s) == route
