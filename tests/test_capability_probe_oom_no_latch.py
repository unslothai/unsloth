# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""A capability probe that runs out of memory (a nearly full card at the first forward) records no verdict: the
call takes the safe path and the next call probes again, instead of latching "unsupported" for the process."""

import sys
import types
from unittest import mock

import pytest
import torch

from unsloth.kernels import fp8 as F8
import unsloth.models._utils as U

CPU = torch.device("cpu")


def _oom():
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")


def _scaled_mm_failing_first(calls, failures = 1):
    def scaled_mm(
        a,
        b,
        scale_a = None,
        scale_b = None,
        out_dtype = None,
        **kwargs,
    ):
        calls.append(out_dtype)
        if len(calls) <= failures:
            raise _oom()
        return torch.full((a.shape[0], b.shape[1]), 128.0, dtype = out_dtype)

    return scaled_mm


@pytest.fixture
def fresh_fp8_caches(monkeypatch):
    monkeypatch.setattr(F8, "_has_fbgemm_rowwise", lambda: False)
    monkeypatch.setattr(F8, "_FP8_ROWWISE_BACKEND_ENV", "")
    F8._probe_fp8_rowwise_backend.cache_clear()
    F8._probe_scaled_mm_float32_out.cache_clear()
    yield
    F8._probe_fp8_rowwise_backend.cache_clear()
    F8._probe_scaled_mm_float32_out.cache_clear()


def test_rowwise_gemm_probe_gives_no_verdict_on_oom():
    def gemm(x):
        raise _oom()

    assert F8._rowwise_gemm_works(gemm, CPU) is None
    assert (
        F8._rowwise_gemm_works(lambda x: (_ for _ in ()).throw(RuntimeError("no kernel")), CPU)
        is False
    )


def test_rowwise_backend_reprobes_after_oom(monkeypatch, fresh_fp8_caches):
    calls = []
    monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_failing_first(calls))
    # First probe out of memory: dequant for this call only, not the slower tensorwise route and nothing cached.
    assert F8._probe_fp8_rowwise_backend(CPU) == "dequant"
    assert calls == [torch.bfloat16]
    assert F8._probe_fp8_rowwise_backend(CPU) == "scaled_mm"
    assert F8._probe_fp8_rowwise_backend(CPU) == "scaled_mm"
    assert calls == [torch.bfloat16, torch.bfloat16]


def test_fbgemm_probe_oom_does_not_fall_through(monkeypatch, fresh_fp8_caches):
    monkeypatch.setattr(F8, "_has_fbgemm_rowwise", lambda: True)
    import unsloth.import_fixes as IF

    monkeypatch.setattr(IF, "suppress_cuda_printf", lambda: mock.MagicMock())
    calls = []

    def fbgemm_rowwise(a, b, sa, sb):
        calls.append(1)
        if len(calls) == 1:
            raise _oom()
        return torch.full((a.shape[0], b.shape[0]), 128.0)

    fake_fbgemm = types.SimpleNamespace(f8f8bf16_rowwise = fbgemm_rowwise)
    monkeypatch.setattr(torch.ops, "fbgemm", fake_fbgemm, raising = False)
    monkeypatch.setattr(
        torch, "_scaled_mm", lambda *a, **k: pytest.fail("fell through to _scaled_mm")
    )
    assert F8._probe_fp8_rowwise_backend(CPU) == "dequant"
    assert F8._probe_fp8_rowwise_backend(CPU) == "fbgemm"


def test_scaled_mm_float32_out_probe_reprobes_after_oom(monkeypatch, fresh_fp8_caches):
    calls = []
    monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm_failing_first(calls))
    assert F8._probe_scaled_mm_float32_out(CPU) is False
    assert F8._probe_scaled_mm_float32_out(CPU) is True
    assert F8._probe_scaled_mm_float32_out(CPU) is True
    assert len(calls) == 2


def test_rocm_triton_probe_reprobes_after_oom():
    triton = sys.modules.get("triton") or types.ModuleType("triton")
    results = [_oom(), True]
    calls = []

    def probe(index):
        calls.append(index)
        result = results[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    with (
        mock.patch.dict(U._TRITON_RUNS_ON_DEVICE, clear = True),
        mock.patch.dict(sys.modules, {"triton": triton}),
        mock.patch.object(U.torch.version, "hip", "7.14", create = True),
        mock.patch.object(U, "_triton_probe", probe),
    ):
        assert U.triton_runs_on_device(0) is False
        assert 0 not in U._TRITON_RUNS_ON_DEVICE
        assert U.triton_runs_on_device(0) is True
        assert U.triton_runs_on_device(0) is True
    assert calls == [0, 0]
