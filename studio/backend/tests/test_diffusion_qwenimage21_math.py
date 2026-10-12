# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference import diffusion_qwenimage21_math as bounded
from core.inference import diffusion_attention as attention

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


@pytest.fixture(autouse = True)
def math_backend():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from diffusers.models.attention_dispatch import attention_backend
    with sdpa_kernel([SDPBackend.MATH]), attention_backend("native"):
        yield


@pytest.mark.parametrize("mask_kind", ["none", "broadcast", "rows", "additive"])
def test_bounded_attention_matches_full_math_with_masks(mask_kind):
    torch.manual_seed(51)
    q = torch.randn(2, 769, 2, 8)
    k, v = torch.randn(2, 833, 2, 8), torch.randn(2, 833, 2, 8)
    mask = None
    if mask_kind == "broadcast":
        mask = torch.ones(2, 1, 1, 833, dtype = torch.bool)
        mask[1, ..., -7:] = False
    elif mask_kind in ("rows", "additive"):
        mask = (torch.arange(833)[None, :] <= torch.arange(769)[:, None] + 32)[None, None]
        if mask_kind == "additive":
            mask = torch.zeros_like(mask, dtype = torch.float32).masked_fill(~mask, float("-inf"))
    calls = []

    def dispatch(q, k, v, **kw):
        calls.append((q.shape[1], k.shape[1], kw["attn_mask"]))
        return qmod.dispatch_attention_fn(q, k, v, backend = "native", **kw)

    expected = dispatch(q, k, v, attn_mask = mask, dropout_p = 0.0)
    calls.clear()
    actual = bounded._bounded_attention(q, k, v, dispatch, mask = mask)
    torch.testing.assert_close(actual, expected)
    assert [c[:2] for c in calls] == [(512, 833), (257, 833)]


@pytest.mark.parametrize("padded", [False, True])
def test_segmented_prefill_and_cached_decode_match_stock(monkeypatch, padded):
    torch.manual_seed(7)
    attn = qmod.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    stock, patched = qmod.QwenImage21AttnProcessor(), bounded._processor_class()()
    states = torch.randn(2, 1800, 16)
    segments = [(0, 600, True), (600, 1130, False)]
    mask = torch.ones(2, 1800, dtype = torch.bool) if padded else None
    if padded:
        mask[1, 580:600] = False
    caches = [qmod.QwenImage21KVLayerCache(), qmod.QwenImage21KVLayerCache()]
    calls = []
    original = bounded._bounded_attention

    def record(q, k, v, dispatch, **kw):
        def checked(q, k, v, **args):
            calls.append(q.shape[1])
            return dispatch(q, k, v, **args)

        return original(q, k, v, checked, **kw)

    monkeypatch.setattr(bounded, "_bounded_attention", record)
    with torch.inference_mode():
        args = dict(
            segments = segments,
            key_valid = mask,
            kv_cache_mode = "extract",
            cache_write_slice = slice(0, 1130),
        )
        expected = stock(attn, states, layer_cache = caches[0], **args)
        actual = patched(attn, states, layer_cache = caches[1], **args)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(caches[0].k, caches[1].k, atol = 0, rtol = 0)
        torch.testing.assert_close(caches[0].v, caches[1].v, atol = 0, rtol = 0)
        assert max(calls) == 512 and len(calls) >= 6
        target = states[:, 1130:]
        key_mask = mask[:, None, None, :] if padded else None
        expected = stock(
            attn, target, layer_cache = caches[0], kv_cache_mode = "cached", attention_mask = key_mask
        )
        actual = patched(
            attn, target, layer_cache = caches[1], kv_cache_mode = "cached", attention_mask = key_mask
        )
        torch.testing.assert_close(actual, expected)
        assert caches[1].k.shape[1] == 1130


def make_pipe():
    modules = [qmod.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8) for _ in range(2)]
    transformer = type(
        "QwenImage21Transformer2DModel", (), {"modules": lambda self: iter(modules)}
    )()
    return SimpleNamespace(transformer = transformer), modules


def test_native_load_installs_fallback_and_guard_tracks_actual_processors(monkeypatch):
    from core.inference.diffusion import _quadratic_attention

    pipe, modules = make_pipe()
    target = SimpleNamespace(backend = "rocm", device = "cuda:0", dtype = torch.bfloat16)
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: True)
    monkeypatch.setattr("core.inference.diffusion.sdpa_math_only", lambda t: True)
    monkeypatch.setattr(attention, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(attention, "guard_rocm_fused_sdpa", lambda *a: ())
    assert _quadratic_attention(target, None, pipe)
    attention.apply_attention_backend(pipe, None, target = target)
    assert bounded.bounded_math_attention(pipe)
    assert not _quadratic_attention(target, None, pipe)
    installed = modules[0].processor
    assert bounded.install(pipe, target)
    assert modules[0].processor is installed
    modules[1].set_processor(qmod.QwenImage21AttnProcessor())
    assert _quadratic_attention(target, None, pipe)


@pytest.mark.parametrize("change", ["parallel", "custom", "non_native"])
def test_guard_never_credits_uncovered_processors(monkeypatch, change):
    pipe, modules = make_pipe()
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: True)
    assert bounded.install(pipe, SimpleNamespace(backend = "rocm"))
    if change == "parallel":
        modules[0].processor._parallel_config = object()
    elif change == "custom":
        modules[0].set_processor(qmod.QwenImage21AttnProcessor())
    else:
        modules[0].processor._attention_backend = "_native_math"
    assert not bounded.bounded_math_attention(pipe)


def test_healthy_fused_install_and_non_rocm_keep_stock(monkeypatch):
    pipe, modules = make_pipe()
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: False)
    assert not bounded.install(pipe, SimpleNamespace(backend = "rocm"))
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: True)
    assert not bounded.install(pipe, SimpleNamespace(backend = "cuda"))
    assert all(type(m.processor) is qmod.QwenImage21AttnProcessor for m in modules)


def test_causal_chunk_offsets_include_preceding_segments():
    torch.manual_seed(42)
    query = torch.randn(1, 769, 2, 8)
    key, value = torch.randn(1, 1000, 2, 8), torch.randn(1, 1000, 2, 8)
    allowed = (torch.arange(1000)[None, :] <= 231 + torch.arange(769)[:, None])[None, None]
    expected = qmod.dispatch_attention_fn(query, key, value, attn_mask = allowed, backend = "native")
    actual = bounded._bounded_attention(
        query, key, value, qmod.dispatch_attention_fn, causal_offset = 231, backend = "native"
    )
    torch.testing.assert_close(actual, expected)


def test_failed_bounded_processor_inspection_keeps_quadratic_guard(monkeypatch):
    from core.inference import diffusion

    monkeypatch.setattr(diffusion, "sdpa_math_only", lambda t: True)

    def failed(pipe):
        raise RuntimeError("processor unavailable")

    monkeypatch.setattr(bounded, "bounded_math_attention", failed)
    assert diffusion._quadratic_attention(object(), None, object())


def test_empty_prefix_matches_full_attention():
    torch.manual_seed(23)
    attn = qmod.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    states = torch.randn(1, 769, 16)
    with torch.inference_mode():
        expected = qmod.QwenImage21AttnProcessor()(attn, states, segments = [])
        actual = bounded._processor_class()()(attn, states, segments = [])
    torch.testing.assert_close(actual, expected)


def test_mps_installs_bounded_attention_without_trusting_the_probe(monkeypatch):
    from core.inference.diffusion import _quadratic_attention

    pipe, modules = make_pipe()
    target = SimpleNamespace(backend = "mps", device = "mps", dtype = torch.bfloat16)
    # MPS SDPA ignores sdpa_kernel, so the probe reports fused kernels it does not run.
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: False)
    warned = []
    monkeypatch.setattr(attention, "warn_if_sdpa_math_only", lambda *a, **k: warned.append(a))
    attention.apply_attention_backend(pipe, None, target = target)
    assert all(type(m.processor) is bounded._processor_class() for m in modules)
    assert bounded.bounded_math_attention(pipe)
    assert not _quadratic_attention(target, None, pipe)
    assert not warned


@pytest.mark.parametrize(
    "target, math_only, expected",
    [
        (SimpleNamespace(backend = "mps", device = "mps"), False, True),
        (SimpleNamespace(backend = "rocm", device = "cuda"), True, True),
        (SimpleNamespace(backend = "rocm", device = "cuda"), False, False),
        (SimpleNamespace(backend = "cuda", device = "cuda"), True, False),
        (SimpleNamespace(backend = "cpu", device = "cpu"), True, False),
    ],
)
def test_bounded_attention_targets(monkeypatch, target, math_only, expected):
    monkeypatch.setattr(attention, "sdpa_math_only", lambda t: math_only)
    assert bounded.needs_bounded_attention(target) is expected


def test_mps_budget_keeps_one_call_per_segment_when_the_scores_fit():
    torch.manual_seed(3)
    attn = qmod.QwenImage21Attention(dim = 16, heads = 2, dim_head = 8)
    stock, patched = qmod.QwenImage21AttnProcessor(), bounded._processor_class()()
    patched._unsloth_score_budget = 2**30
    states = torch.randn(1, 1800, 16)
    segments = [(0, 600, True), (600, 1130, False)]
    with torch.inference_mode():
        expected = stock(attn, states, segments = segments)
        actual = patched(attn, states, segments = segments)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "budget, rows",
    [(None, 512), (0, 512), (2 * 2 * 4096 * 4 * 700, 700), (2**40, 2**28 // (2 * 2 * 4096))],
)
def test_query_rows_follow_the_score_budget(budget, rows):
    q, k = torch.empty(2, 4096, 2, 8), torch.empty(2, 4096, 2, 8)
    assert bounded._query_rows(q, k, budget) == rows


def test_batched_2048_stays_under_the_mps_element_cap():
    q, k = (
        torch.empty(2, 16384, 32, 1, dtype = torch.bfloat16),
        torch.empty(2, 16640, 32, 1, dtype = torch.bfloat16),
    )
    rows = bounded._query_rows(q, k, 2**34)
    assert (
        1 <= rows < bounded.QUERY_CHUNK_SIZE and rows * 2 * 32 * 16640 <= bounded.MPS_SCORE_ELEMENTS
    )


def test_mps_score_budget_reads_the_override(monkeypatch):
    monkeypatch.setenv(bounded.SCORE_BUDGET_ENV, "256")
    assert bounded.mps_score_budget() == 256 * 2**20
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 16 * 2**30, raising = False)
    for unusable in ("", "inf", "1e309", "nan", "x"):
        monkeypatch.setenv(bounded.SCORE_BUDGET_ENV, unusable)
        assert bounded.mps_score_budget() == 2 * 2**30


def test_qwen_image_21_1024_splits_below_the_mps_element_cap():
    # One 32 x 4096 x 4352 call returned wrong values on macOS 15; 768x768 (2304 tokens) stays one call.
    k = torch.empty(1, 4352, 32, 1, dtype = torch.bfloat16)
    rows = bounded._query_rows(torch.empty(1, 4096, 32, 1, dtype = torch.bfloat16), k, 2**34)
    assert rows < 4096 and rows * 32 * 4352 <= bounded.MPS_SCORE_ELEMENTS
    k = torch.empty(1, 2560, 32, 1, dtype = torch.bfloat16)
    assert bounded._query_rows(torch.empty(1, 2304, 32, 1, dtype = torch.bfloat16), k, 2**34) >= 2304
