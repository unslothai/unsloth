# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Below sm_80 a packed MXFP4 forward decodes the 16-bit weight for cuBLAS; it must not need more memory than the
tile kernel it replaced when the whole weight does not fit (T4 with little headroom)."""

import contextlib
import gc

import pytest
from llama_patch_isolation import restore_llama_patches  # noqa: F401
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py
import torch

import unsloth  # noqa: F401
import unsloth.models.mxfp4_compressed_linear as mcl

pytestmark = [
    pytest.mark.usefixtures("restore_llama_patches"),
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device"),
]


def _zoo():
    try:
        import unsloth_zoo.mxfp4_gemm as mg
    except Exception:
        pytest.skip("needs unsloth_zoo.mxfp4_gemm")
    if not hasattr(mg, "mxfp4_grouped_gemv_available"):
        pytest.skip("unsloth_zoo before the sm_80 tile GEMM floor: the tile kernel still runs here")
    return mg


@pytest.fixture
def sm75(monkeypatch):
    """The sm_75 policy on whatever GPU runs this: the verified tile GEMM is kept off, real kernels run."""
    mg = _zoo()
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
    for dtype in (torch.float16, torch.bfloat16):
        if not mg.mxfp4_grouped_gemv_available(device, dtype):
            pytest.skip("the fused MXFP4 kernel does not self-check on this GPU")
        monkeypatch.setitem(mg._AVAILABLE_DTYPE, (str(device), dtype, "tiles"), False)
        assert not mg.mxfp4_gemm_available(device, dtype)
    monkeypatch.setattr(mcl, "_DENSE_BUDGET", {}, raising = False)
    return device


@contextlib.contextmanager
def _headroom(mib, device):
    """Let the caching allocator hand out at most ``mib`` MiB more than is allocated now."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(device).total_memory
    cap = torch.cuda.memory_allocated(device) + mib * 2**20
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        torch.cuda.empty_cache()


def _weight(
    N,
    K,
    device,
    seed = 0,
):
    g = torch.Generator(device = device).manual_seed(seed)
    packed = torch.randint(0, 256, (N, K // 2), dtype = torch.uint8, device = device, generator = g)
    scale = torch.randint(118, 128, (N, K // 32), dtype = torch.uint8, device = device, generator = g)
    return packed, scale


def _single_shot(
    x,
    packed,
    scale,
    bias = None,
):
    # What #1899 runs below sm_80: the whole 16-bit weight, one cuBLAS call.
    weight = mcl.dequantize_mxfp4_packed(packed, scale, x.dtype)
    return torch.nn.functional.linear(x, weight, bias)


def test_t4_decode_fits_in_64mib_headroom(sm75):
    # 14336 x 4096 decodes to 112 MiB of fp16: the tile kernel needed 28 KiB, dequantize + cuBLAS must not OOM.
    device = sm75
    N, K = 14336, 4096
    packed, scale = _weight(N, K, device)
    x = torch.randn(1, K, dtype = torch.float16, device = device)
    want = _single_shot(x, packed, scale)
    mcl.mxfp4_packed_matmul(x, packed, scale)  # warm kernels and their self-checks
    with _headroom(64, device):
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        got = mcl.mxfp4_packed_matmul(x, packed, scale)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - start
    assert peak < 64 * 2**20
    torch.testing.assert_close(got, want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())
    # The next calls go straight to chunks: no second allocation failure.
    ooms = torch.cuda.memory_stats(device).get("num_ooms", 0)
    with _headroom(64, device):
        mcl.mxfp4_packed_matmul(x, packed, scale)
    assert torch.cuda.memory_stats(device).get("num_ooms", 0) == ooms


def test_weight_that_fits_keeps_the_single_shot_decode(sm75):
    device = sm75
    packed, scale = _weight(1024, 512, device, seed = 1)
    x = torch.randn(3, 512, dtype = torch.bfloat16, device = device)
    assert torch.equal(mcl.mxfp4_packed_matmul(x, packed, scale), _single_shot(x, packed, scale))
    assert not mcl._DENSE_BUDGET


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lead", [(), (2,)])
def test_row_chunks_match_the_single_shot_decode(sm75, dtype, rows, bias, lead):
    device = sm75
    N, K = 1000, 384  # 1000 rows: the last chunk is ragged
    packed, scale = _weight(N, K, device, seed = rows)
    x = torch.randn(*lead, rows, K, dtype = dtype, device = device)
    b = torch.randn(N, dtype = dtype, device = device) if bias else None
    want = _single_shot(x, packed, scale, b)
    row_bytes = K * x.element_size()
    mcl._DENSE_BUDGET[device.index] = [300 * row_bytes, 1024]  # 300-row chunks
    got = mcl.mxfp4_packed_matmul(x, packed, scale, bias = b)
    assert got.shape == want.shape and got.dtype == dtype
    # Same decoded rows, same math; cuBLAS may pick another reduction order for the narrower chunk (1 ulp).
    tol = 1.6e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(got, want, rtol = tol, atol = tol * want.abs().max().item())
    out = torch.empty_like(want)
    mcl._DENSE_BUDGET[device.index] = [300 * row_bytes, 1024]
    if not bias:
        assert mcl.mxfp4_packed_matmul(x, packed, scale, out = out).data_ptr() == out.data_ptr()
        torch.testing.assert_close(out, want, rtol = tol, atol = tol * want.abs().max().item())


def test_no_room_for_a_chunk_takes_the_tile_kernel(sm75, monkeypatch):
    device = sm75
    N, K = 4096, 4096
    packed, scale = _weight(N, K, device, seed = 3)
    x = torch.randn(2, K, dtype = torch.float16, device = device)
    want = _single_shot(x, packed, scale)
    calls = []
    tile = mcl._zoo_mxfp4_matmul
    monkeypatch.setattr(mcl, "_zoo_mxfp4_matmul", lambda *a, **k: calls.append(1) or tile(*a, **k))
    mcl._DENSE_BUDGET[device.index] = [64 * K * 2, 1024]  # 64 rows < the 128-row chunk floor
    got = mcl.mxfp4_packed_matmul(x, packed, scale)
    assert calls
    torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2 * want.abs().max().item())


def test_budget_is_retried_after_a_while(sm75):
    device = sm75
    packed, scale = _weight(512, 256, device, seed = 4)
    x = torch.randn(1, 256, dtype = torch.float16, device = device)
    mcl._DENSE_BUDGET[device.index] = [128 * 256 * 2, 2]
    mcl.mxfp4_packed_matmul(x, packed, scale)
    assert device.index in mcl._DENSE_BUDGET
    mcl.mxfp4_packed_matmul(x, packed, scale)
    assert device.index not in mcl._DENSE_BUDGET  # whole weight fitted again


def test_compiled_forward_is_bounded_without_graph_breaks(sm75):
    from torch._dynamo.utils import counters

    device = sm75
    N, K = 14336, 4096
    module = mcl.make_mxfp4_packed_linear(K, N, device = device)
    module.weight_packed.data, module.weight_scale.data = _weight(N, K, device, seed = 5)
    module.compute_dtype = torch.float16
    x = torch.randn(1, K, dtype = torch.float16, device = device)
    with torch.no_grad():
        want = module(x)
        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(module, fullgraph = True)
        torch.testing.assert_close(
            compiled(x), want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item()
        )
        assert sum(counters["graph_break"].values()) == 0
        with _headroom(64, device):
            got = compiled(x)
            torch.cuda.synchronize(device)
    torch.testing.assert_close(got, want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())
    torch._dynamo.reset()


def _fake_oom_dequant(monkeypatch, fail_rows):
    """``dequantize_mxfp4_packed`` raising an allocator OOM once for each entry of ``fail_rows`` (row counts, in
    order), as when the decoded weight allocated but its scratch, the output or the cuBLAS workspace did not."""
    real = mcl.dequantize_mxfp4_packed
    pending, seen = list(fail_rows), []

    def fake(
        packed,
        scale,
        dtype = torch.bfloat16,
        out = None,
    ):
        seen.append(packed.shape[0])
        if pending and packed.shape[0] == pending[0]:
            pending.pop(0)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        return real(packed, scale, dtype, out = out)

    monkeypatch.setattr(mcl, "dequantize_mxfp4_packed", fake)
    return seen, pending


def test_oom_after_the_weight_allocated_falls_back_to_chunks(sm75, monkeypatch):
    device = sm75
    N, K = 1024, 512
    packed, scale = _weight(N, K, device, seed = 6)
    x = torch.randn(2, K, dtype = torch.float16, device = device)
    want = _single_shot(x, packed, scale)
    # Whole weight attempt fails late, then the first (full-height) chunk fails too: half-height chunks finish.
    seen, pending = _fake_oom_dequant(monkeypatch, [N, N])
    got = mcl.mxfp4_packed_matmul(x, packed, scale)
    assert not pending and seen[:2] == [N, N] and seen[2] == N // 2
    assert device.index in mcl._DENSE_BUDGET
    torch.testing.assert_close(got, want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())


@pytest.mark.parametrize("compiling", [False, True])
def test_bias_never_writes_out_whatever_the_memory(sm75, monkeypatch, compiling):
    # The full-weight path with a bias returns a new tensor and leaves ``out`` alone: so must the fallbacks.
    device = sm75
    N, K = 1000, 384
    packed, scale = _weight(N, K, device, seed = 7)
    x = torch.randn(3, K, dtype = torch.float16, device = device)
    b = torch.randn(N, dtype = torch.float16, device = device)
    want = _single_shot(x, packed, scale, b)
    out = torch.full_like(want, 7.0)
    full = mcl.mxfp4_packed_matmul(x, packed, scale, out = out, bias = b)
    assert full.data_ptr() != out.data_ptr() and bool((out == 7.0).all())
    if compiling:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    mcl._DENSE_BUDGET[device.index] = [300 * K * x.element_size(), 1024]  # 300-row chunks
    got = mcl.mxfp4_packed_matmul(x, packed, scale, out = out, bias = b)
    assert got.data_ptr() != out.data_ptr() and bool((out == 7.0).all())
    torch.testing.assert_close(got, want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())


def _grad_oracle(x, packed, scale, bias, transpose):
    # fp64 oracle for d(sum(y)) / dx (and / dbias) of y = x @ W^T + b (or x @ W).
    w = mcl.dequantize_mxfp4_packed(packed, scale, torch.float32).double()
    xd = x.detach().double().requires_grad_()
    bd = None if bias is None else bias.detach().double().requires_grad_()
    y = xd @ w if transpose else torch.nn.functional.linear(xd, w, bd)
    y.sum().backward()
    return xd.grad, None if bd is None else bd.grad


def _assert_grad_close(got, want):
    err = (got.double() - want).norm() / want.norm()
    assert err.item() < 1e-2, err.item()


@pytest.mark.parametrize("path", ["chunks", "tile", "single_shot"])
@pytest.mark.parametrize("bias", [False, True])
def test_direct_helper_backward_is_exact_on_every_path(sm75, path, bias):
    # Each chunk's F.linear saved a view of the one reused decode buffer: backward read only the last chunk.
    device = sm75
    N, K = 512, 256
    packed, scale = _weight(N, K, device, seed = 8)
    x = torch.randn(3, K, dtype = torch.float16, device = device).requires_grad_()
    b = torch.randn(N, dtype = torch.float16, device = device).requires_grad_() if bias else None
    if path == "chunks":
        mcl._DENSE_BUDGET[device.index] = [128 * K * x.element_size(), 1024]  # four 128-row chunks
    elif path == "tile":
        mcl._DENSE_BUDGET[device.index] = [64 * K * x.element_size(), 1024]  # below the chunk floor
    y = mcl.mxfp4_packed_matmul(x, packed, scale, bias = b)
    assert y.requires_grad
    y.float().sum().backward()
    want_x, want_b = _grad_oracle(x, packed, scale, b, False)
    _assert_grad_close(x.grad, want_x)
    if bias:
        _assert_grad_close(b.grad, want_b)


def test_direct_helper_transpose_backward(sm75):
    device = sm75
    N, K = 512, 256
    packed, scale = _weight(N, K, device, seed = 9)
    g = torch.randn(3, N, dtype = torch.float16, device = device).requires_grad_()
    mcl._DENSE_BUDGET[device.index] = [
        128 * K * g.element_size(),
        1024,
    ]  # its backward is a chunked forward
    mcl.mxfp4_packed_matmul(g, packed, scale, transpose = True).float().sum().backward()
    want, _ = _grad_oracle(g, packed, scale, None, True)
    _assert_grad_close(g.grad, want)


def test_compiled_direct_helper_backward(sm75):
    # The bounded custom op has no autograd formula of its own: a grad-requiring call goes through the Function.
    device = sm75
    N, K = 512, 256
    packed, scale = _weight(N, K, device, seed = 10)
    x = torch.randn(3, K, dtype = torch.float16, device = device).requires_grad_()
    torch._dynamo.reset()
    f = torch.compile(lambda a: mcl.mxfp4_packed_matmul(a, packed, scale), fullgraph = True)
    f(x).float().sum().backward()
    want, _ = _grad_oracle(x, packed, scale, None, False)
    _assert_grad_close(x.grad, want)


@pytest.mark.parametrize("bias", [False, True])
def test_direct_helper_with_grad_keeps_the_out_contract(sm75, bias):
    # As without grad: a bias returns a new tensor and leaves ``out`` alone; no bias fills ``out``.
    device = sm75
    N, K = 512, 256
    packed, scale = _weight(N, K, device, seed = 11)
    x = torch.randn(3, K, dtype = torch.float16, device = device).requires_grad_()
    b = torch.randn(N, dtype = torch.float16, device = device) if bias else None
    mcl._DENSE_BUDGET[device.index] = [128 * K * x.element_size(), 1024]
    out = torch.full((3, N), 7.0, dtype = torch.float16, device = device)
    got = mcl.mxfp4_packed_matmul(x, packed, scale, out = out, bias = b)
    want = _single_shot(x.detach(), packed, scale, b)
    torch.testing.assert_close(got.detach(), want, rtol = 2e-3, atol = 2e-3 * want.abs().max().item())
    if bias:
        assert got.data_ptr() != out.data_ptr() and bool((out == 7.0).all())
    else:
        assert got.data_ptr() == out.data_ptr()
    got.float().sum().backward()
    want_x, _ = _grad_oracle(x, packed, scale, b, False)
    _assert_grad_close(x.grad, want_x)
