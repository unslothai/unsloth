# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Rowwise FP8 direct-route routing (fake CUDA tensors) and parity (sm89+ GPU)."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from unsloth.kernels import fp8 as F8


@pytest.fixture
def spies(monkeypatch):
    calls = []
    # NVIDIA routing; ROCm runners report a HIP version and keep the compiled path (tested below).
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "scaled_mm")
    monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", True)
    monkeypatch.setattr(
        F8, "_fp8_rowwise_direct", lambda X, w, s, b, route: calls.append(route) or "direct"
    )
    monkeypatch.setattr(
        F8,
        "_fbgemm_fp8_linear_compiled",
        lambda X, w, s, b = None: calls.append("compiled") or "compiled",
    )
    return calls


def _args(
    rows,
    N = 256,
    K = 512,
    requires_grad = False,
):
    X = torch.empty(1, rows, K, device = "cuda", dtype = torch.bfloat16, requires_grad = requires_grad)
    w = torch.nn.Parameter(
        torch.empty(N, K, device = "cuda", dtype = torch.float8_e4m3fn), requires_grad = False
    )
    s = torch.nn.Parameter(
        torch.empty(N, 1, device = "cuda", dtype = torch.float32), requires_grad = False
    )
    return X, w, s


@pytest.mark.parametrize(
    "rows, route", [(17, "dequant"), (64, "dequant"), (65, "scaled_mm"), (4096, "scaled_mm")]
)
def test_inference_calls_take_the_direct_path(spies, rows, route):
    with FakeTensorMode(), torch.no_grad():
        assert F8.fbgemm_fp8_linear(*_args(rows)) == "direct"
    assert spies == [route]


def test_autograd_fbgemm_and_kill_switch_keep_the_compiled_path(spies, monkeypatch):
    with FakeTensorMode():
        assert F8.fbgemm_fp8_linear(*_args(128, requires_grad = True)) == "compiled"
        with torch.no_grad():
            X, w, s = _args(128)
            # Transposed weight (LoRA backward's W.t()) and odd scaled_mm dims stay compiled / dequant. Built with
            # empty_strided: view ops on fake CUDA tensors raise on CPU-only torch builds.
            Xt = torch.empty(1, 128, 256, device = "cuda", dtype = torch.bfloat16)
            wt = torch.empty_strided((512, 256), (1, 512), device = "cuda", dtype = torch.float8_e4m3fn)
            st = torch.empty(512, 1, device = "cuda", dtype = torch.float32)
            assert F8.fbgemm_fp8_linear(Xt, wt, st) == "compiled"
            X2, w2, s2 = _args(128, N = 200)
            assert F8.fbgemm_fp8_linear(X2, w2, s2) == "direct"
            assert F8.fbgemm_fp8_linear(*_args(F8._FP8_ROWWISE_DIRECT_MAX_ROWS + 1)) == "compiled"
            monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "fbgemm")
            assert F8.fbgemm_fp8_linear(X, w, s) == "compiled"
            monkeypatch.setattr(F8, "_fp8_rowwise_backend", lambda device: "scaled_mm")
            monkeypatch.setattr(F8, "_FP8_ROWWISE_DIRECT", False)
            assert F8.fbgemm_fp8_linear(X, w, s) == "compiled"
    assert spies == ["compiled", "compiled", "dequant", "compiled", "compiled", "compiled"]


def test_cpu_and_traced_calls_keep_the_compiled_path(spies, monkeypatch):
    with torch.no_grad():
        X = torch.zeros(1, 128, 512, dtype = torch.bfloat16)
        w = torch.zeros(256, 512, dtype = torch.float8_e4m3fn)
        s = torch.ones(256, 1)
        assert F8.fbgemm_fp8_linear(X, w, s) == "compiled"
        with FakeTensorMode():
            args = _args(128)
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            assert F8.fbgemm_fp8_linear(*args) == "compiled"
    assert spies == ["compiled", "compiled"]


_sm89 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@pytest.mark.skipif(not _sm89, reason = "rowwise FP8 GEMM parity needs an sm89+ GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_direct_path_is_bitwise_equal_to_the_compiled_path(dtype):
    if F8._fp8_rowwise_backend(torch.device("cuda", 0)) != "scaled_mm":
        pytest.skip("this GPU does not use the scaled_mm rowwise backend")
    torch.manual_seed(0)
    with torch.no_grad():
        for N, K in ((256, 512), (768, 2048), (512, 5120), (208, 4112), (256, 16400)):
            w = (torch.randn(N, K, device = "cuda") * 0.5).to(torch.float8_e4m3fn)
            for s in (
                torch.rand(N, 1, device = "cuda") * 0.01 + 1e-3,
                (torch.rand(N, 1, device = "cuda") + 0.5).to(torch.bfloat16),
            ):
                for rows in (17, 64, 65, 300):
                    X = (torch.randn(2, rows, K, device = "cuda") * 3).to(dtype)
                    X[0, 0] = 0
                    X[0, 1, ::7] = 1e4
                    X[1, 2, 5] = float("nan")
                    X[1, 3, 9] = float("inf")
                    bias = torch.randn(N, device = "cuda", dtype = torch.float32)
                    for b in (None, bias):
                        for ub in (None, torch.tensor([20.0], device = "cuda")):
                            if ub is not None:
                                w.input_scale_ub = ub
                            elif hasattr(w, "input_scale_ub"):
                                del w.input_scale_ub
                            route = F8._fp8_rowwise_direct_route(X, w, s)
                            assert route == (
                                "scaled_mm"
                                if 2 * rows > F8._fp8_dequant_max_rows(X.device, N * K)
                                and N % 16 == 0
                                and K % 16 == 0
                                else "dequant"
                            )
                            ref = F8._fbgemm_fp8_linear_compiled(X, w, s, b)
                            out = F8.fbgemm_fp8_linear(X, w, s, b)
                            assert out.shape == ref.shape and out.dtype == ref.dtype
                            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
                                N,
                                K,
                                rows,
                                s.dtype,
                                b is None,
                                ub,
                            )


def test_compile_disabled_keeps_the_eager_wrapper():
    # UNSLOTH_COMPILE_DISABLE runs the wrapper eagerly, whose dequant rounds the scale first: keep it.
    assert F8._fp8_rowwise_direct_enabled("1", compile_disabled = True) is False
    assert F8._fp8_rowwise_direct_enabled("1", compile_disabled = False) is True
    for off in ("0", "false", "No", " off "):
        assert F8._fp8_rowwise_direct_enabled(off, compile_disabled = False) is False


def test_rocm_keeps_the_compiled_path(spies, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0", raising = False)
    with FakeTensorMode(), torch.no_grad():
        assert F8.fbgemm_fp8_linear(*_args(32)) == "compiled"
        assert F8.fbgemm_fp8_linear(*_args(128)) == "compiled"
    assert spies == ["compiled", "compiled"]
