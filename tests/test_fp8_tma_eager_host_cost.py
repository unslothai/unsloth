# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""TMA block FP8 GEMM host cost: descriptors are built per call from the live tensors without
re-validation, and routing is unchanged in eager and under graph capture (no GPU needed)."""

import pytest
import torch

from unsloth.kernels import fp8 as F8

if F8._TensorDescriptor is None:
    pytest.skip("this Triton has no TensorDescriptor", allow_module_level = True)


class _Fake:
    def __init__(
        self,
        shape,
        dtype,
        ptr = 1 << 20,
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self.shape.numel()

    def element_size(self):
        return torch.empty((), dtype = self.dtype).element_size()


@pytest.fixture
def tma_device(monkeypatch):
    calls = []
    capturing = [False]
    monkeypatch.setattr(F8, "_FP8_TMA_GEMM", True)
    monkeypatch.setattr(F8, "_fp8_tma_gemm_arch_at", lambda index: True)
    monkeypatch.setattr(F8, "_FP8_TMA_SELF_CHECKED", {})
    monkeypatch.setattr(F8, "_fp8_tma_gemm_self_check", lambda index: calls.append(index) or True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    return calls, capturing


def _ok(
    M,
    N = 4096,
    K = 4096,
    dtype = torch.bfloat16,
    batch = None,
):
    lead = (M,) if batch is None else (batch, M // batch)
    A = _Fake(lead + (K,), torch.float8_e4m3fn)
    B = _Fake((N, K), torch.float8_e4m3fn)
    C = _Fake(lead + (N,), dtype)
    return F8._fp8_tma_gemm_ok(A, B, C, 128, 128, 128, False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [128, 512, 1023, 2048])
def test_eager_gemm_takes_tma_at_any_size(tma_device, M, dtype):
    # On B200 the pointer kernel is 1.4-1.8x slower on the GPU at M * N * K below 2**34, so eager
    # GEMMs keep TMA whenever it routes, as on main.
    assert _ok(M, dtype = dtype) is True
    assert _ok(M, dtype = dtype, batch = 1) is True


@pytest.mark.parametrize("M", [128, 512, 2048])
def test_graph_capture_takes_tma_at_any_size(tma_device, M):
    calls, capturing = tma_device
    _ok(M)  # an eager call first runs the self check
    capturing[0] = True
    assert _ok(M) is True


def test_capture_before_any_eager_call_keeps_pointer_kernel(tma_device):
    calls, capturing = tma_device
    capturing[0] = True
    assert _ok(4096) is False
    assert calls == []


def _reference(tensor, rows, cols, block):
    return F8._TensorDescriptor.from_tensor(tensor.view(rows, cols), list(block))


def _same(desc, ref, tensor):
    assert type(desc) is F8._TensorDescriptor
    assert desc.base.data_ptr() == ref.base.data_ptr() == tensor.data_ptr()
    assert list(desc.shape) == list(ref.shape)
    assert list(desc.strides) == list(ref.strides)
    assert list(desc.block_shape) == list(ref.block_shape)
    fields = {k: v for k, v in vars(ref).items() if k not in ("base", "shape", "strides")}
    assert {k: vars(desc)[k] for k in fields} == fields


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float16])
def test_descriptor_matches_from_tensor(dtype):
    t = torch.zeros(2, 96, 256, dtype = dtype)
    _same(F8._fp8_tma_descriptor(t, 192, 256, (128, 128)), _reference(t, 192, 256, (128, 128)), t)


def test_cached_fields_do_not_keep_the_first_tensor_alive():
    import gc
    import weakref

    F8._FP8_TMA_DESC_FIELDS.clear()
    t = torch.zeros(256, 256, dtype = torch.float16)
    ref = weakref.ref(t)
    F8._fp8_tma_descriptor(t, 256, 256, (128, 128))
    del t
    gc.collect()
    assert ref() is None
    assert all(
        k not in fields
        for fields in F8._FP8_TMA_DESC_FIELDS.values()
        for k in ("base", "shape", "strides")
    )


def test_sliced_rows_fall_back_to_from_tensor():
    # A weight sliced out of a wider buffer has a row pitch above its width: keep its real strides.
    wide = torch.zeros(256, 384, dtype = torch.float8_e4m3fn)
    t = wide[:, :256]
    desc = F8._fp8_tma_descriptor(t, 256, 256, (128, 128))
    assert list(desc.strides) == [384, 1]


def _launch(monkeypatch, A, B, C, M, N, K):
    seen = []

    class _Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append((grid, args, kwargs))

    monkeypatch.setattr(F8, "_w8a8_block_fp8_matmul_tma", _Kernel())
    As = torch.ones(M, K // 128)
    Bs = torch.ones(N // 128, K // 128)
    F8._launch_fp8_tma_gemm(A, B, C, As, Bs, M, N, K, 128, 128)
    ((grid, args, kwargs),) = seen
    return args[:3]


def test_launch_skips_per_call_validation(monkeypatch):
    M, N, K = 256, 384, 512
    A = torch.zeros(M, K, dtype = torch.float8_e4m3fn)
    B = torch.zeros(N, K, dtype = torch.float8_e4m3fn)
    C = torch.empty(M, N, dtype = torch.bfloat16)
    _launch(monkeypatch, A, B, C, M, N, K)
    built = []
    real = F8._TensorDescriptor.from_tensor
    monkeypatch.setattr(
        F8._TensorDescriptor,
        "from_tensor",
        staticmethod(lambda *a, **k: built.append(1) or real(*a, **k)),
    )
    for _ in range(5):
        a, b, c = _launch(monkeypatch, A, B, C, M, N, K)
    assert built == []
    _same(a, _reference(A, M, K, (128, 128)), A)
    _same(b, _reference(B, N, K, (128, 128)), B)
    _same(c, _reference(C, M, N, (128, 128)), C)


def test_replaced_or_edited_weight_is_never_stale(monkeypatch):
    M, N, K = 128, 256, 256
    A = torch.zeros(M, K, dtype = torch.float8_e4m3fn)
    C = torch.empty(M, N, dtype = torch.bfloat16)
    W = torch.nn.Parameter(torch.zeros(N, K, dtype = torch.float8_e4m3fn), requires_grad = False)
    _, b0, _ = _launch(monkeypatch, A, W, C, M, N, K)
    assert b0.base is W

    W.data.copy_(torch.ones(N, K).to(torch.float8_e4m3fn))  # edited in place
    _, b1, _ = _launch(monkeypatch, A, W, C, M, N, K)
    assert b1.base is W and b1.base.data_ptr() == W.data_ptr()
    assert torch.equal(b1.base.float(), torch.ones(N, K))

    W2 = torch.full((N, K), 2.0).to(torch.float8_e4m3fn)  # replaced by a new tensor
    _, b2, _ = _launch(monkeypatch, A, W2, C, M, N, K)
    assert b2.base is W2 and b2.base.data_ptr() == W2.data_ptr() != W.data_ptr()

    N2 = 384  # .data swapped for a different shape
    W.data = torch.zeros(N2, K, dtype = torch.float8_e4m3fn)
    C2 = torch.empty(M, N2, dtype = torch.bfloat16)
    _, b3, c3 = _launch(monkeypatch, A, W, C2, M, N2, K)
    _same(b3, _reference(W, N2, K, (128, 128)), W)
    _same(c3, _reference(C2, M, N2, (128, 128)), C2)


def test_empty_input_still_raises_like_from_tensor():
    t = torch.zeros(0, 256, dtype = torch.float8_e4m3fn)
    with pytest.raises(AssertionError):
        F8._fp8_tma_descriptor(t, 0, 256, (128, 128))
