# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference import diffusion_qwenimage21_rocm as chunks


@pytest.mark.parametrize(
    "architecture,setting,expected",
    [
        ("gfx1151", "auto", True),
        ("gfx1151:sramecc-", "auto", True),
        ("gfx1201", "auto", True),
        ("gfx1200", "auto", True),
        ("gfx1100", "auto", False),
        ("gfx1150", "auto", False),
        ("gfx942", "auto", False),
        ("gfx1201", "0", False),
        ("gfx1100", "1", True),
        ("gfx1151", "0", False),
    ],
)
def test_device_scope(monkeypatch, architecture, setting, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(chunks.QUERY_CHUNK_ENV, setting)
    monkeypatch.setattr(torch.version, "hip", "7.13")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(gcnArchName = architecture)
    )
    target = SimpleNamespace(backend = "rocm", device = "cuda", ordinal = 1)
    assert chunks._device_supported(target) is expected
    assert not chunks._device_supported(SimpleNamespace(backend = "cuda", device = "cuda"))
    assert not chunks._device_supported(SimpleNamespace(backend = "cpu", device = "cpu"))


@pytest.mark.parametrize(
    "reason",
    [
        "eligible",
        "cpu",
        "fp32",
        "small",
        "heads",
        "mask",
        "segments",
        "key_valid",
        "parallel",
        "grad",
        "compile",
        "backend",
    ],
)
def test_runtime_scope(monkeypatch, reason):
    torch = pytest.importorskip("torch")
    hidden = SimpleNamespace(
        device = SimpleNamespace(type = "cuda"), dtype = torch.bfloat16, shape = (1, 8192, 4096)
    )
    processor = SimpleNamespace(_parallel_config = None)
    attn = SimpleNamespace(heads = 32, inner_dim = 4096)
    mask = segments = key_valid = None
    if reason == "cpu":
        hidden.device.type = "cpu"
    if reason == "fp32":
        hidden.dtype = torch.float32
    if reason == "small":
        hidden.shape = (1, 4096, 4096)
    if reason == "heads":
        attn.heads = 16
    if reason == "mask":
        mask = object()
    if reason == "segments":
        segments = []
    if reason == "key_valid":
        key_valid = object()
    if reason == "parallel":
        processor._parallel_config = object()
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: reason == "grad")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: reason == "compile")
    monkeypatch.setattr(chunks, "_native_backend", lambda p: reason != "backend")
    assert chunks._can_chunk(processor, attn, hidden, mask, segments, key_valid) is (
        reason == "eligible"
    )


def test_chunks_keep_all_keys_and_values():
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    q = torch.randn(2, 1100, 2, 8)
    k = torch.randn(2, 1200, 2, 8)
    v = torch.randn_like(k)
    calls = []

    def dispatch(qpart, key, value, **kwargs):
        calls.append((qpart.shape[1], key, value, kwargs))
        return torch.nn.functional.scaled_dot_product_attention(
            qpart.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), **kwargs
        ).transpose(1, 2)

    expected = dispatch(q, k, v, scale = 0.2)
    calls.clear()
    actual = chunks._chunk_attention(q, k, v, dispatch, scale = 0.2)
    torch.testing.assert_close(actual, expected)
    assert [c[0] for c in calls] == [512, 512, 76]
    assert all(c[1] is k and c[2] is v and c[3] == {"scale": 0.2} for c in calls)


def test_processor_output_and_stock_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    torch.manual_seed(8)
    attn = module.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    stock = module.QwenImage21AttnProcessor()
    patched = chunks._processor_class()()
    hidden = torch.randn(1, 1100, 16)
    with torch.inference_mode():
        expected = stock(attn, hidden)
        monkeypatch.setattr(chunks, "_can_chunk", lambda *a: True)
        torch.testing.assert_close(patched(attn, hidden), expected)
        monkeypatch.setattr(chunks, "_can_chunk", lambda *a: False)
        mask = torch.ones(1, 1, 1100, 1100, dtype = torch.bool).tril()
        torch.testing.assert_close(
            patched(attn, hidden, attention_mask = mask), stock(attn, hidden, attention_mask = mask)
        )


def test_install_is_local_and_idempotent(monkeypatch):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    monkeypatch.setattr(chunks, "_device_supported", lambda target: True)
    attn = module.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    untouched = module.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    transformer = type("QwenImage21Transformer2DModel", (), {"modules": lambda self: [attn]})()
    pipe = SimpleNamespace(transformer = transformer)
    stock = attn.processor
    assert chunks.install(pipe, None)
    installed = attn.processor
    assert chunks.install(pipe, None)
    assert attn.processor is installed
    assert type(untouched.processor) is type(stock)
    assert type(installed) is chunks._processor_class()


def test_native_backend_honors_context_and_explicit_choice():
    dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
    processor = SimpleNamespace(_attention_backend = None)
    with dispatch.attention_backend("native"):
        assert chunks._native_backend(processor)
    with dispatch.attention_backend("_native_math"):
        assert not chunks._native_backend(processor)
        processor._attention_backend = "native"
        assert chunks._native_backend(processor)
    processor._attention_backend = "_native_math"
    assert not chunks._native_backend(processor)


def test_cached_decode_preserves_prefix_and_projections(monkeypatch):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    torch.manual_seed(9)
    attn = module.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    stock = module.QwenImage21AttnProcessor()
    patched = chunks._processor_class()()
    prefix = torch.randn(1, 7, 16)
    target = torch.randn(1, 1050, 16)
    cache = module.QwenImage21KVLayerCache()
    with torch.inference_mode():
        joined = torch.cat([prefix, target], dim = 1)
        before = stock(
            attn,
            joined,
            layer_cache = cache,
            kv_cache_mode = "extract",
            cache_write_slice = slice(0, 7),
            segments = [(0, 3, True), (3, 7, False)],
        )
        other_cache = module.QwenImage21KVLayerCache()
        monkeypatch.setattr(chunks, "_can_chunk", lambda *a: False)
        after = patched(
            attn,
            joined,
            layer_cache = other_cache,
            kv_cache_mode = "extract",
            cache_write_slice = slice(0, 7),
            segments = [(0, 3, True), (3, 7, False)],
        )
        torch.testing.assert_close(after, before, rtol = 0, atol = 0)
        torch.testing.assert_close(other_cache.k, cache.k, rtol = 0, atol = 0)
        expected = stock(attn, target, layer_cache = cache, kv_cache_mode = "cached")
        monkeypatch.setattr(chunks, "_can_chunk", lambda *a: True)
        actual = patched(attn, target, layer_cache = other_cache, kv_cache_mode = "cached")
        torch.testing.assert_close(actual, expected)
        assert other_cache.k.shape[1] == 7
