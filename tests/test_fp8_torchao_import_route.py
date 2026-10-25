# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""torchao blockwise_fp8_gemm: found under both module paths, and routed away from sm_100 (no GPU needed)."""

import sys
import types

import pytest
import torch

from real_accelerator import has_real_cuda

from unsloth.kernels import fp8 as F8

OLD = "torchao.prototype.blockwise_fp8_inference.blockwise_quantization"
NEW = "torchao.kernel.blockwise_quantization"


def _stub(monkeypatch, name, fn):
    if fn is None:
        monkeypatch.setitem(sys.modules, name, None)  # the import raises ModuleNotFoundError
    else:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(blockwise_fp8_gemm = fn))


@pytest.mark.parametrize(
    "old, new, expect",
    [("old", "new", "old"), (None, "new", "new"), ("old", None, "old"), (None, None, None)],
)
def test_import_fallback(monkeypatch, old, new, expect):
    fns = {"old": lambda *a, **k: "old", "new": lambda *a, **k: "new"}
    _stub(monkeypatch, OLD, fns.get(old))
    _stub(monkeypatch, NEW, fns.get(new))
    got = F8._import_torchao_blockwise_gemm()
    assert (got is None) if expect is None else (got() == expect)


def test_import_skips_module_without_symbol(monkeypatch):
    monkeypatch.setitem(sys.modules, OLD, types.SimpleNamespace())
    _stub(monkeypatch, NEW, lambda: "new")
    assert F8._import_torchao_blockwise_gemm()() == "new"


def _devices(
    monkeypatch,
    tma_arch,
    capability = (9, 0),
):
    # tma_arch[i]: device i is sm_100; the others report `capability`.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(tma_arch))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(tma_arch))
    monkeypatch.setattr(F8, "_fp8_tma_gemm_arch_at", lambda index: tma_arch[index])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index = None: (10, 0) if tma_arch[index] else capability,
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(F8, "_FP8_TMA_GEMM", True)
    monkeypatch.setattr(F8, "_TensorDescriptor", object)


@pytest.mark.parametrize(
    "has_torchao, tma_arch, expect",
    [
        (False, [False], "triton"),
        (False, [True], "triton"),
        (True, [False], "torchao"),  # H100 / sm_120
        (True, [True], "torchao"),  # B200: torchao_block_matmul hands M > 64 to the TMA kernel
        (True, [True, True], "torchao"),
        (True, [True, False], "torchao"),
        (True, [], "torchao"),
    ],
)
def test_select(monkeypatch, has_torchao, tma_arch, expect):
    _devices(monkeypatch, tma_arch)
    monkeypatch.setattr(
        F8, "torchao_blockwise_gemm", (lambda *a, **k: None) if has_torchao else None
    )
    want = {"triton": F8.w8a8_block_fp8_matmul_triton, "torchao": F8.torchao_block_matmul}[expect]
    assert F8._select_fp8_block_matmul() is want


@pytest.mark.parametrize("switch", ["on", "kill_switch", "no_descriptor"])
def test_tma_prefers_triton_above_64_rows(monkeypatch, switch):
    _devices(monkeypatch, [True])
    if switch == "kill_switch":
        monkeypatch.setattr(F8, "_FP8_TMA_GEMM", False)
    elif switch == "no_descriptor":
        monkeypatch.setattr(F8, "_TensorDescriptor", None)
    dev = torch.device("cuda", 0)
    got = [F8._prefer_triton_block_matmul_at(dev, M) for M in (1, 16, 64, 65, 128, 8192)]
    assert got == ([False] * 3 + [True] * 3 if switch == "on" else [False] * 6)


def test_non_cuda_never_prefers_triton(monkeypatch):
    _devices(monkeypatch, [True], (8, 9))
    assert not F8._prefer_triton_block_matmul_at(torch.device("meta"), 8192)
    assert not F8._prefer_triton_block_matmul_at(torch.device("cpu"), 8192)


@pytest.mark.parametrize(
    "capability, expect",
    [((8, 9), "triton"), ((9, 0), "torchao"), ((12, 0), "torchao"), ((8, 0), "torchao")],
)
def test_select_by_capability(monkeypatch, capability, expect):
    _devices(monkeypatch, [False], capability)
    monkeypatch.setattr(F8, "torchao_blockwise_gemm", lambda *a, **k: None)
    want = {"triton": F8.w8a8_block_fp8_matmul_triton, "torchao": F8.torchao_block_matmul}[expect]
    assert F8._select_fp8_block_matmul() is want


def test_rocm_keeps_torchao(monkeypatch):
    _devices(monkeypatch, [False], (8, 9))
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(F8, "torchao_blockwise_gemm", lambda *a, **k: None)
    assert F8._select_fp8_block_matmul() is F8.torchao_block_matmul


def test_prefer_triton_per_device(monkeypatch):
    _devices(monkeypatch, [False, True])
    assert not F8._prefer_triton_block_matmul_at(torch.device("cuda", 0), 8192)
    assert F8._prefer_triton_block_matmul_at(torch.device("cuda", 1), 8192)
    assert not F8._prefer_triton_block_matmul_at(torch.device("cuda", 1), 64)
    _devices(monkeypatch, [False], (8, 9))
    assert F8._prefer_triton_block_matmul_at(torch.device("cuda", 0), 1)


def test_torchao_block_matmul_splits_by_rows(monkeypatch):
    # The opaque op computes M from the concrete tensor, so a compiled caller adds no guard on M.
    calls = []
    monkeypatch.setattr(F8, "_prefer_triton_block_matmul_at", lambda device, M: M > 64)
    monkeypatch.setattr(F8, "w8a8_block_fp8_matmul_triton", lambda *a, **k: calls.append("triton"))
    monkeypatch.setattr(
        F8, "torchao_blockwise_gemm", lambda *a, **k: calls.append("torchao") or a[0]
    )
    meta = lambda *shape: torch.empty(*shape, device = "meta", dtype = torch.float8_e4m3fn)
    w_s = torch.empty(2, 4, device = "meta")
    for M in (1, 64, 65, 4096):
        F8.torchao_block_matmul(
            meta(M, 512), meta(256, 512), torch.empty(M, 4, device = "meta"), w_s, [128, 128]
        )
    assert calls == ["torchao", "torchao", "triton", "triton"]


@pytest.mark.parametrize(
    "block_size, dtype, expect",
    [
        ([128, 128], torch.bfloat16, "torchao"),
        ([128, 128], torch.float16, "triton"),  # torchao writes bf16 only
        ([128, 128], torch.float32, "triton"),
        ([128, 64], torch.bfloat16, "triton"),  # torchao indexes the N scale with block_size[1]
    ],
)
def test_torchao_block_matmul_routes(monkeypatch, block_size, dtype, expect):
    calls = []
    monkeypatch.setattr(F8, "w8a8_block_fp8_matmul_triton", lambda *a, **k: calls.append("triton"))
    monkeypatch.setattr(
        F8,
        "torchao_blockwise_gemm",
        lambda *a, **k: calls.append(("torchao", k["block_size"])) or a[0],
    )
    meta = lambda *shape: torch.empty(*shape, device = "meta", dtype = torch.float8_e4m3fn)
    K, N = 512, 256
    act_s = torch.empty(64, K // block_size[1], device = "meta")
    w_s = torch.empty(N // block_size[0], K // block_size[1], device = "meta")
    F8.torchao_block_matmul(meta(64, K), meta(N, K), act_s, w_s, block_size, output_dtype = dtype)
    assert calls == (["triton"] if expect == "triton" else [("torchao", block_size[1])])


def test_torchao_block_matmul_routes_empty_inputs_to_triton(monkeypatch):
    # torchao buckets M with math.ceil(math.log2(M)), which raises for an expert that got no tokens.
    calls = []
    monkeypatch.setattr(F8, "_prefer_triton_block_matmul_at", lambda device, M: False)
    monkeypatch.setattr(F8, "w8a8_block_fp8_matmul_triton", lambda *a, **k: calls.append("triton"))
    monkeypatch.setattr(
        F8, "torchao_blockwise_gemm", lambda *a, **k: calls.append("torchao") or a[0]
    )
    meta = lambda *shape: torch.empty(*shape, device = "meta", dtype = torch.float8_e4m3fn)
    w_s = torch.empty(2, 4, device = "meta")
    for M in (0, 1):
        F8.torchao_block_matmul(
            meta(M, 512), meta(256, 512), torch.empty(M, 4, device = "meta"), w_s, [128, 128]
        )
    assert calls == ["triton", "torchao"]


@pytest.mark.skipif(
    not has_real_cuda() or F8.torchao_blockwise_gemm is None,
    reason = "needs CUDA and torchao's blockwise GEMM",
)
def test_fp8_block_matmul_empty_rows_on_gpu():
    # The real kernels: an empty activation gives an empty output instead of a math domain error.
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("block FP8 GEMM needs sm_89 or newer")
    K, N = 512, 256
    act_q = torch.empty(0, K, device = "cuda", dtype = torch.float8_e4m3fn)
    act_s = torch.empty(0, K // 128, device = "cuda")
    w_q = torch.randn(N, K, device = "cuda").to(torch.float8_e4m3fn)
    w_s = torch.ones(N // 128, K // 128, device = "cuda")
    out = F8.torchao_block_matmul(act_q, w_q, act_s, w_s, [128, 128])
    assert out.shape == (0, N) and out.dtype == torch.bfloat16
