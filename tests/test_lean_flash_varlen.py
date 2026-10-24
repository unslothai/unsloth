"""The lean eager varlen path must match torch's varlen_attn: same aten flash kernels, outputs and grads."""

import pytest
import torch

ad = pytest.importorskip("unsloth.utils.attention_dispatch")

needs_lean = pytest.mark.skipif(
    not torch.cuda.is_available() or not getattr(ad, "_LEAN_VARLEN", False),
    reason = "needs CUDA and torch varlen_attn",
)


def _run(fn, q, k, v, cu, mx, window, gqa, g):
    q, k, v = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    out = fn(q, k, v, cu, mx, window, gqa)
    out.backward(g)
    return out.detach(), q.grad, k.grad, v.grad


def _torch(q, k, v, cu, mx, window, gqa):
    return ad._TORCH_VARLEN_ATTN(q, k, v, cu, cu, mx, mx, scale = None, window_size = window, enable_gqa = gqa)


def _lean(q, k, v, cu, mx, window, gqa):
    return ad._LeanFlashVarlen.apply(q, k, v, cu, mx, window == (-1, 0), None, window[0], window[1])


@needs_lean
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hq,hkv", [(8, 8), (16, 8)])
@pytest.mark.parametrize("window", [(-1, 0), (31, 0)])
def test_lean_matches_torch_varlen(dtype, hq, hkv, window):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    torch.manual_seed(0)
    lens = [37, 128, 5, 200, 64]  # the last segment stands in for the collator's pad tail
    T, D = sum(lens), 128
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype = torch.int32, device = "cuda")
    q = torch.randn(T, hq, D, device = "cuda", dtype = dtype)
    k = torch.randn(T, hkv, D, device = "cuda", dtype = dtype)
    v = torch.randn(T, hkv, D, device = "cuda", dtype = dtype)
    g = torch.randn(T, hq, D, device = "cuda", dtype = dtype)
    ref = _run(_torch, q, k, v, cu, max(lens), window, hq != hkv, g)
    ref2 = _run(_torch, q, k, v, cu, max(lens), window, hq != hkv, g)
    got = _run(_lean, q, k, v, cu, max(lens), window, hq != hkv, g)
    assert torch.equal(got[0], ref[0])  # forward: identical kernel, identical bits
    for a, b, c in zip(got[1:], ref[1:], ref2[1:]):
        noise = (b.float() - c.float()).abs().max().item()  # flash backward accumulates dq atomically
        assert (a.float() - b.float()).abs().max().item() <= max(noise, 1e-6) * 2 + 1e-3


@needs_lean
def test_dispatch_uses_lean_in_eager_and_torch_when_compiling(monkeypatch):
    calls = []
    real = ad._LeanFlashVarlen.apply
    monkeypatch.setattr(ad._LeanFlashVarlen, "apply", lambda *a: calls.append(1) or real(*a))
    q = torch.randn(64, 4, 64, device = "cuda", dtype = torch.float16)
    cu = torch.tensor([0, 30, 64], dtype = torch.int32, device = "cuda")
    ad._varlen_attn_call(q, q, q, cu, 34, scale = None, window_size = (-1, 0), enable_gqa = False)
    expect = 0 if ad._varlen_can_use_cudnn(q, q, q, 34, [-1, 0], False) else 1
    assert len(calls) == expect
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    ad._varlen_attn_call(q, q, q, cu, 34, scale = None, window_size = (-1, 0), enable_gqa = False)
    assert len(calls) == expect
