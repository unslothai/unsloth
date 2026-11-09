# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU checks for diffusion_flux3_attn: cuDNN selection is a no-op, so this verifies the math, gates and trace, not cuDNN."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tf3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")

from core.inference import diffusion_flux3_attn as fa  # noqa: E402


@pytest.fixture(autouse = True)
def _cpu_cudnn(monkeypatch):
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.delenv(fa.STRIDED_ATTN_ENV, raising = False)
    fa.strided_processor_class.cache_clear()
    fa._PROBED.clear()
    yield
    fa.strided_processor_class.cache_clear()
    fa._PROBED.clear()


def _tiny():
    torch.manual_seed(0)
    model = tf3.Flux3Transformer2DModel(
        in_channels = 8,
        cond_in_channels = 8,
        num_layers = 1,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        timestep_channels = 16,
        axes_dims_rope = (4, 4, 4, 4),
    )
    return model.eval()


def _attn(dtype = torch.float32):
    torch.manual_seed(1)
    attn = tf3.Flux3Attention(query_dim = 64, heads = 4, dim_head = 16).to(dtype).eval()
    proc = attn.processor
    proc._attention_backend = "_native_cudnn"
    return attn, proc


def _rope(n, d = 16):
    g = torch.Generator().manual_seed(2)
    ang = torch.rand(n, d // 2, generator = g) * 6.28
    cos = torch.cos(ang).repeat_interleave(2, dim = -1)
    sin = torch.sin(ang).repeat_interleave(2, dim = -1)
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,seq", [(1, 40), (2, 33)])
def test_strided_processor_matches_the_cudnn_dispatch_bitwise(dtype, batch, seq):
    attn, stock = _attn(dtype)
    x = torch.randn(batch, seq, 64, generator = torch.Generator().manual_seed(3)).to(dtype)
    rope = _rope(seq)
    with torch.no_grad():
        want = stock(attn, x, None, rope)
        new = fa.strided_processor_class()()
        new._attention_backend = stock._attention_backend
        got = new(attn, x, None, rope)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.equal(got, want)


def test_masks_and_other_backends_take_the_stock_call(monkeypatch):
    attn, stock = _attn()
    new = fa.strided_processor_class()()
    calls = []
    real = tf3.Flux3AttnProcessor.__call__

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(tf3.Flux3AttnProcessor, "__call__", spy)
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        new._attention_backend = "_native_cudnn"
        new(attn, x, torch.ones(1, 12, dtype = torch.bool), _rope(12))
        new._attention_backend = None
        new(attn, x, None, _rope(12))
        assert len(calls) == 2
        new._attention_backend = "_native_cudnn"
        new(attn, x, None, _rope(12))
    assert len(calls) == 2  # the strided path did not call the stock processor


def _cuda_target(monkeypatch, probe = True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fa, "strided_matches", lambda *a, **k: probe)
    return SimpleNamespace(device = "cuda", torch_device = "cuda:0", dtype = torch.bfloat16)


def _pipe():
    model = _tiny()
    model.set_attention_backend("_native_cudnn")
    return SimpleNamespace(transformer = model)


def test_install_swaps_every_cudnn_processor_and_keeps_its_backend(monkeypatch):
    pipe = _pipe()
    target = _cuda_target(monkeypatch)
    n = len(fa._attention_modules(pipe.transformer))
    assert n > 0
    assert fa.install(pipe, "_native_cudnn", target) == n
    cls = fa.strided_processor_class()
    for attn in fa._attention_modules(pipe.transformer):
        assert type(attn.processor) is cls
        assert fa._backend_value(attn.processor._attention_backend) == "_native_cudnn"
    assert fa.install(pipe, "_native_cudnn", target) == 0  # idempotent


def test_install_declines(monkeypatch):
    target = _cuda_target(monkeypatch)
    assert fa.install(_pipe(), "_native_flash", target) == 0
    assert (
        fa.install(SimpleNamespace(transformer = torch.nn.Linear(2, 2)), "_native_cudnn", target) == 0
    )
    assert (
        fa.install(_pipe(), "_native_cudnn", SimpleNamespace(device = "cpu", torch_device = "cpu")) == 0
    )
    target = _cuda_target(monkeypatch, probe = False)
    assert fa.install(_pipe(), "_native_cudnn", target) == 0
    target = _cuda_target(monkeypatch)
    monkeypatch.setattr(fa, "_PROCESSOR_DIGESTS", frozenset({"0000000000000000"}))
    assert fa.install(_pipe(), "_native_cudnn", target) == 0
    monkeypatch.setattr(fa, "_PROCESSOR_DIGESTS", frozenset({"e3282b2e03be05d8"}))
    monkeypatch.setenv(fa.STRIDED_ATTN_ENV, "0")
    assert fa.install(_pipe(), "_native_cudnn", target) == 0


def test_processor_digest_is_the_pinned_diffusers_source():
    from core.inference.diffusion_autoopt.patches import source_digest
    assert source_digest(tf3.Flux3AttnProcessor.__call__) in fa._PROCESSOR_DIGESTS


def test_whole_transformer_matches_and_compiles_without_breaks(monkeypatch):
    from torch._dynamo.testing import CompileCounterWithBackend

    pipe = _pipe()
    g = torch.Generator().manual_seed(4)
    kw = {
        "hidden_states": torch.randn(1, 16, 8, generator = g),
        "encoder_hidden_states": torch.randn(1, 5, 24, generator = g),
        "timestep": torch.tensor([0.7]),
        "img_ids": torch.randint(0, 8, (16, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 8, (5, 4), generator = g).float(),
        "return_dict": False,
    }
    with torch.no_grad():
        want = pipe.transformer(**kw)[0]
        with monkeypatch.context() as m:  # the fake CUDA flag only for the install, not for dynamo
            assert fa.install(pipe, "_native_cudnn", _cuda_target(m)) > 0
        got = pipe.transformer(**kw)[0]
        assert torch.equal(got, want)
        torch._dynamo.reset()
        cc = CompileCounterWithBackend("inductor")
        blocks = list(pipe.transformer.single_transformer_blocks)
        eager = [b.forward for b in blocks]
        for b in blocks:
            b.forward = torch.compile(
                type(b).forward.__get__(b), backend = cc, fullgraph = True, dynamic = True
            )
        try:
            outs = [
                pipe.transformer(**dict(kw, timestep = torch.tensor([t])))[0] for t in (0.7, 0.5, 0.3)
            ]
        finally:
            for b, f in zip(blocks, eager):
                b.forward = f
            torch._dynamo.reset()
    assert cc.frame_count == len(blocks)  # fullgraph: no break; three steps, no recompile
    assert torch.allclose(outs[0], want, atol = 1e-5, rtol = 1e-5)
