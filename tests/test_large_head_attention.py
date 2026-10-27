# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""head_dim 512 attention (unsloth/kernels/large_head_attention.py).

CPU: mask -> per-row key interval extraction, and the router leaving every unsupported call
to the original SDPA. CUDA: fwd + bwd against fp32 SDPA math over causal / packed / padded /
GQA shapes, and the router engaging on a Gemma-4 style call.
"""

import importlib.util
import os

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytest.importorskip("triton")

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "unsloth", "kernels", "large_head_attention.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_large_head_attention", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LH = _load()
_SM80 = has_real_cuda() and torch.cuda.get_device_capability()[0] >= 8


def _causal(S):
    i = torch.arange(S)
    return i[None, :] <= i[:, None]


def _packed(S, starts):
    doc = torch.zeros(S, dtype = torch.long)
    for s in starts:
        doc[s:] += 1
    return _causal(S) & (doc[None, :] == doc[:, None])


def test_intervals_causal_packed_padded_float():
    S = 37
    lo, hi = LH.intervals_from_mask(_causal(S)[None, None], 1, S)
    assert lo.tolist() == [[0] * S] and hi.tolist() == [list(range(S))]

    lo, hi = LH.intervals_from_mask(_packed(S, [5, 6, 20])[None, None], 1, S)
    assert lo[0, :5].tolist() == [0] * 5 and lo[0, 5].item() == 5 and lo[0, 6].item() == 6
    assert lo[0, 20:].tolist() == [20] * (S - 20) and hi.tolist() == [list(range(S))]

    i = torch.arange(S)
    right_pad = (_causal(S) & (i[None, :] < 30))[None, None]
    lo, hi = LH.intervals_from_mask(right_pad.expand(2, 1, S, S), 2, S)
    assert hi[1, 29:].tolist() == [29] * (S - 29)

    additive = torch.zeros(1, 1, S, S)
    additive[~_packed(S, [9])[None, None]] = torch.finfo(torch.float32).min
    lo, _ = LH.intervals_from_mask(additive, 3, S)
    assert lo.shape == (3, S) and lo[2, 9].item() == 9


def test_intervals_reject_unrepresentable_masks():
    S = 16
    i = torch.arange(S)
    left_pad = _causal(S) & (i[None, :] >= 3)  # rows 0..2 attend nothing
    assert LH.intervals_from_mask(left_pad[None, None], 1, S) is None
    holes = _causal(S).clone()
    holes[10, 4] = False
    assert LH.intervals_from_mask(holes[None, None], 1, S) is None
    per_head = _causal(S)[None, None].expand(1, 2, S, S)
    assert LH.intervals_from_mask(per_head, 1, S) is None
    soft = torch.zeros(1, 1, S, S)
    soft[0, 0, 5, 2] = -1.0  # a bias, not a mask
    assert LH.intervals_from_mask(soft, 1, S) is None
    assert LH.intervals_from_mask(_causal(S)[None, None, :, :8], 1, S) is None


def test_mask_verdict_is_cached_and_invalidated():
    S = 12
    mask = _causal(S)[None, None].clone()
    assert LH.intervals_from_mask(mask, 1, S) is not None
    mask[0, 0, 7, 2] = False
    assert LH.intervals_from_mask(mask, 1, S) is None


def test_inference_mask_is_rescanned_after_in_place_update():
    S = 12
    with torch.inference_mode():
        mask = _causal(S)[None, None].clone()
        assert LH.intervals_from_mask(mask, 1, S) is not None
        mask[0, 0, 7, 2] = False
        assert LH.intervals_from_mask(mask, 1, S) is None
        mask.copy_(_packed(S, [5, 7])[None, None])
        lo, hi = LH.intervals_from_mask(mask, 1, S)
        assert lo[0, 6].item() == 5 and hi[0, 4].item() == 4


def _sentinel():
    calls = []

    def original(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        calls.append(dict(dropout = dropout, scaling = scaling, is_causal = is_causal, **kwargs))
        return "original", None

    return calls, original


class _Module(torch.nn.Module):
    is_causal = True


def test_router_defers_everything_it_cannot_run():
    calls, original = _sentinel()
    router = LH._make_router(original)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    q = torch.zeros(1, 4, 8, 512)
    kv = torch.zeros(1, 2, 8, 512)
    # CPU tensors, a 256 head, dropout and a softcap never reach the kernel.
    assert router(_Module(), q, kv, kv, None, scaling = 1.0)[0] == "original"
    small = torch.zeros(1, 4, 8, 256)
    assert router(_Module(), small, small[:, :2], small[:, :2], None)[0] == "original"
    assert router(_Module(), q, kv, kv, None, dropout = 0.1)[0] == "original"
    assert router(_Module(), q, kv, kv, None, softcap = 30.0, sliding_window = None)[0] == "original"
    assert len(calls) == 4
    assert calls[0]["scaling"] == 1.0 and calls[3]["softcap"] == 30.0
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before


def test_kill_switch_skips_install(monkeypatch):
    monkeypatch.setenv("UNSLOTH_HD512_ATTENTION", "0")
    LH._enabled.cache_clear()
    try:
        assert LH.patch_large_head_attention() is False
    finally:
        monkeypatch.delenv("UNSLOTH_HD512_ATTENTION")
        LH._enabled.cache_clear()


def _reference(q, k, v, allowed):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, 1)
    v = v.repeat_interleave(groups, 1)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize("S", [17, 64, 131])
@pytest.mark.parametrize("heads", [(4, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("kind", ["causal", "packed", "right_pad"])
def test_kernel_matches_reference(S, heads, kind):
    H, HKV = heads
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.manual_seed(S + H)
    B, D = 2, 512
    allowed = (
        {
            "causal": _causal(S),
            "packed": _packed(S, [1, S // 3, S // 3 + 2]),
            "right_pad": _causal(S) & (torch.arange(S)[None, :] < S - 5),
        }[kind]
        .cuda()[None, None]
        .expand(B, 1, S, S)
    )
    q = torch.randn(B, S, H, D, device = "cuda", dtype = dtype).transpose(1, 2)
    k = torch.randn(B, S, HKV, D, device = "cuda", dtype = dtype).transpose(1, 2)
    v = torch.randn(B, S, HKV, D, device = "cuda", dtype = dtype).transpose(1, 2)
    grad = torch.randn(B, S, H, D, device = "cuda", dtype = dtype)
    lo, hi = LH.intervals_from_mask(allowed, B, S)

    ours_in = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    out = LH.large_head_attention(*ours_in, lo, hi)
    out.backward(grad)
    ref_in = [x.detach().float().requires_grad_() for x in (q, k, v)]
    ref = _reference(*ref_in, allowed)
    ref.backward(grad.float())

    for a, b in zip([out] + [x.grad for x in ours_in], [ref] + [x.grad for x in ref_in]):
        assert torch.isfinite(a).all()
        err = (a.float() - b).abs()
        assert err.max() <= 0.03 * b.abs().max() + 0.02
        assert err.mean() <= 0.01 * b.abs().mean() + 2e-3


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_engages_on_gemma4_global_call():
    calls, original = _sentinel()
    router = LH._make_router(original)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    q = torch.randn(1, 8, 40, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 1, 40, 512, device = "cuda", dtype = dtype)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    out, weights = router(_Module(), q, kv, kv, None, dropout = 0.0, scaling = 1.0, sliding_window = None)
    assert weights is None and out.shape == (1, 40, 8, 512) and out.is_contiguous()
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1 and not calls
    left_pad = _causal(40).cuda() & (torch.arange(40, device = "cuda")[None, :] >= 3)
    assert router(_Module(), q, kv, kv, left_pad[None, None])[0] == "original"


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_defers_trainable_additive_mask():
    calls, original = _sentinel()
    router = LH._make_router(original)
    dtype = _cuda_dtype()
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    additive = torch.zeros(1, 1, 24, 24, device = "cuda", dtype = dtype)
    additive[~_causal(24).cuda()[None, None]] = torch.finfo(dtype).min
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    assert router(_Module(), q, kv, kv, additive)[0] != "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1
    assert router(_Module(), q, kv, kv, additive.requires_grad_())[0] == "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1 and len(calls) == 1


def _cuda_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_self_check_leaves_global_rng_alone(monkeypatch):
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    LH._SELF_CHECK.pop((str(device), dtype), None)
    seeded = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: seeded.append(seed))
    state = torch.cuda.get_rng_state(device)
    assert LH._self_check(device, dtype) is True
    assert not seeded  # would reseed the model's other GPUs
    assert torch.equal(torch.cuda.get_rng_state(device), state)


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_self_check_under_inference_mode():
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    LH._SELF_CHECK.pop((str(device), dtype), None)
    with torch.inference_mode():
        assert LH._self_check(device, dtype) is True


def _self_check_in_checkpoint(device, dtype):
    # Cold first call from inside a non-reentrant checkpointed layer (DDP / FSDP2, Gemma-4
    # E-series): returns the cached verdict, the layer's run count and whether w got a grad.
    from torch.utils.checkpoint import checkpoint

    LH._SELF_CHECK.pop((str(device), dtype), None)
    calls = []
    w = (torch.randn(8, 8, device = device) * 0.1).requires_grad_()

    def layer(x):
        calls.append(1)
        h = (x @ w).sigmoid()
        LH._self_check(device, dtype)
        return (h @ w).sigmoid()

    checkpoint(layer, torch.randn(4, 8, device = device), use_reentrant = False).sum().backward()
    return LH._SELF_CHECK[(str(device), dtype)], len(calls), w.grad is not None


def test_self_check_stays_out_of_a_non_reentrant_checkpoint(monkeypatch):
    # The check's tensors were packed into the layer's checkpoint and its autograd.grad recomputed
    # the layer mid-forward: CheckpointError, cached as False, the kernel off for the process.
    def kernel(q, k, v, lo, hi, scale):
        groups = q.shape[1] // k.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1), is_causal = True
        )
        return out.transpose(1, 2)

    monkeypatch.setattr(LH, "large_head_attention", kernel)
    monkeypatch.setattr(LH, "_measured_faster_on", lambda device: True)
    monkeypatch.setattr(LH, "_dense_intervals", lambda *args: (None, None))
    key = (str(torch.device("cpu")), torch.float32)
    monkeypatch.setitem(LH._SELF_CHECK, key, None)
    assert _self_check_in_checkpoint(torch.device("cpu"), torch.float32) == (True, 2, True)


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_real_self_check_stays_out_of_a_non_reentrant_checkpoint():
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    assert _self_check_in_checkpoint(device, dtype) == (True, 2, True)


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_patch_self_checks_so_compiled_graphs_use_the_kernel(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    LH._SELF_CHECK.pop((str(device), dtype), None)
    model = torch.nn.Linear(4, 4).to(device = device, dtype = dtype)
    model.head_dim = 512
    saved = ALL_ATTENTION_FUNCTIONS["sdpa"]
    monkeypatch.setattr(LH, "_PATCHED", [False])
    try:
        assert LH.patch_large_head_attention(model) is True
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward

    calls, original = _sentinel()
    router = LH._make_router(original)
    q = torch.randn(1, 4, 24, 512, device = device, dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = device, dtype = dtype)
    torch._dynamo.reset()
    out, _ = torch.compile(router, backend = backend)(_Module(), q, kv, kv, None, scaling = 512**-0.5)
    assert not calls and out.shape == (1, 24, 4, 512)
    assert any("large_head_attention_fwd" in code for code in graphs)
    assert LH._SELF_CHECK.get((str(device), dtype)) is True


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_defers_autocast_dtype_change():
    calls, original = _sentinel()
    router = LH._make_router(original)
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = torch.float16)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = torch.float16)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    with torch.autocast("cuda", dtype = torch.float16):
        assert router(_Module(), q, kv, kv, None)[0] != "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1
    with torch.autocast("cuda", dtype = torch.bfloat16):
        assert router(_Module(), q, kv, kv, None)[0] == "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1 and len(calls) == 1


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_honours_math_sdpa_selection():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    calls, original = _sentinel()
    router = LH._make_router(original)
    dtype = _cuda_dtype()
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    with sdpa_kernel([SDPBackend.MATH]):
        assert router(_Module(), q, kv, kv, None)[0] == "original"
    if hasattr(torch._C, "_get_sdp_priority_order"):
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], set_priority = True):
            assert router(_Module(), q, kv, kv, None)[0] == "original"
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert router(_Module(), q, kv, kv, None)[0] != "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1


def test_router_keeps_wrapped_router_sentinels():
    # unsloth_zoo's gemma-4 sliding router re-patches on every model load unless the current sdpa
    # carries its sentinel; without it, it would wrap this router and the two would recurse.
    mapping, orig = {}, [None]

    def sdpa(module, query, key, value, attention_mask, **kwargs):
        return "sdpa", None

    def sliding(module, query, key, value, attention_mask, **kwargs):
        return orig[0](module, query, key, value, attention_mask, **kwargs)

    def patch_sliding():
        if getattr(mapping["sdpa"], "_unsloth_gemma4_flash", False):
            return
        orig[0] = mapping["sdpa"]
        sliding._unsloth_gemma4_flash = True
        mapping["sdpa"] = sliding

    mapping["sdpa"] = sdpa
    patch_sliding()
    router = LH._make_router(mapping["sdpa"])
    mapping["sdpa"] = router
    patch_sliding()  # second model load
    assert (
        mapping["sdpa"] is router
        and router._unsloth_large_head
        and router._unsloth_original is sliding
    )
    q = torch.randn(1, 2, 4, 64)
    assert mapping["sdpa"](None, q, q, q, None)[0] == "sdpa"


def test_install_sits_beneath_the_packed_router(monkeypatch):
    # A packed trainer built before the model load already installed the packed router; the route of a
    # packed call must not depend on which of the two was installed first.
    import sys
    import types

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def inner(module, query, key, value, attention_mask, **kwargs):
        return "inner", None

    packed = types.ModuleType("unsloth.utils.hf_packed_attention")
    packed._ORIG_SDPA = [inner]

    def packed_router(module, query, key, value, attention_mask, **kwargs):
        return packed._ORIG_SDPA[0](module, query, key, value, attention_mask, **kwargs)

    model = torch.nn.Linear(4, 4)
    model.head_dim = 512
    saved = ALL_ATTENTION_FUNCTIONS["sdpa"]
    monkeypatch.setitem(sys.modules, "unsloth.utils.hf_packed_attention", packed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(LH, "_PATCHED", [False])
    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = packed_router
        assert LH.patch_large_head_attention(model) is True
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is packed_router
        assert packed._ORIG_SDPA[0]._unsloth_large_head
        assert packed._ORIG_SDPA[0]._unsloth_original is inner
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved


def test_autotune_skips_the_flush_buffer_benchmarker():
    # Triton's default benchmarker allocates a 256 MB L2 flush buffer, enough to OOM a model that fits.
    for kernel in (LH._lh_fwd, LH._lh_bwd_dq, LH._lh_bwd_dkdv):
        assert kernel.do_bench is LH._bench_in_place


def test_self_check_oom_falls_back_to_sdpa(monkeypatch):
    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(LH, "large_head_attention", oom)
    monkeypatch.setattr(LH, "_measured_faster_on", lambda device: True)
    key = (str(torch.device("cpu")), torch.float32)
    monkeypatch.setitem(LH._SELF_CHECK, key, None)
    LH._SELF_CHECK.pop(key)
    assert LH._self_check(torch.device("cpu"), torch.float32) is False
    assert LH._SELF_CHECK[key] is False


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_autotune_of_a_new_shape_adds_no_flush_buffer():
    dtype = _cuda_dtype()
    # G = 3 is not used elsewhere in this file, so this call autotunes fwd and bwd from scratch.
    q = torch.randn(1, 6, 40, 512, device = "cuda", dtype = dtype, requires_grad = True)
    k = torch.randn(1, 2, 40, 512, device = "cuda", dtype = dtype, requires_grad = True)
    v = torch.randn(1, 2, 40, 512, device = "cuda", dtype = dtype, requires_grad = True)
    lo, hi = LH._dense_intervals(1, 40, True, q.device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    LH.large_head_attention(q, k, v, lo, hi).float().sum().backward()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base < 32 * 1024**2


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_falls_back_to_sdpa_on_kernel_oom(monkeypatch):
    calls, original = _sentinel()
    router = LH._make_router(original)
    dtype = _cuda_dtype()
    assert LH._self_check(torch.device("cuda", torch.cuda.current_device()), dtype) is True

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(LH, "_attention", oom)
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    before = LH.LARGE_HEAD_ENGAGED["fallback"]
    assert router(_Module(), q, kv, kv, None)[0] == "original"
    assert LH.LARGE_HEAD_ENGAGED["fallback"] == before + 1 and len(calls) == 1


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_skips_the_mask_scan_when_it_cannot_use_the_result(monkeypatch):
    # The scan needs a host sync, and a failed self check makes the result unusable.
    def scan(*args, **kwargs):
        raise AssertionError("mask scanned")

    calls, original = _sentinel()
    router = LH._make_router(original)
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    assert LH._self_check(device, dtype) is True
    monkeypatch.setattr(LH, "intervals_from_mask", scan)
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    monkeypatch.setitem(LH._SELF_CHECK, (str(device), dtype), False)
    assert router(_Module(), q, kv, kv, _causal(24).cuda()[None, None])[0] == "original"
    assert len(calls) == 1


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_leaves_an_explicit_is_causal_beside_a_mask_to_sdpa():
    calls, original = _sentinel()
    router = LH._make_router(original)
    dtype = _cuda_dtype()
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    allowed = torch.ones(1, 1, 24, 24, device = "cuda", dtype = torch.bool)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    assert router(_Module(), q, kv, kv, allowed, is_causal = True)[0] == "original"
    assert calls[-1]["is_causal"] is True and LH.LARGE_HEAD_ENGAGED["calls"] == before
    causal = _causal(24).cuda()[None, None]
    assert router(_Module(), q, kv, kv, causal)[0] != "original"
    assert router(_Module(), q, kv, kv, causal, is_causal = False)[0] != "original"
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 2 and len(calls) == 1


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_falls_back_to_sdpa_on_mask_scan_oom(monkeypatch):
    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

    calls, original = _sentinel()
    router = LH._make_router(original)
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    assert LH._self_check(device, dtype) is True
    monkeypatch.setattr(LH, "intervals_from_mask", oom)
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    before = LH.LARGE_HEAD_ENGAGED["fallback"]
    assert router(_Module(), q, kv, kv, _causal(24).cuda()[None, None])[0] == "original"
    assert LH.LARGE_HEAD_ENGAGED["fallback"] == before + 1 and len(calls) == 1


def test_pre_ampere_keeps_sdpa_without_compiling(monkeypatch):
    def kernel(*args, **kwargs):
        raise AssertionError("kernel compiled on sm75")

    monkeypatch.setattr(LH, "large_head_attention", kernel)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (7, 5))
    key = (str(torch.device("cpu")), torch.float16)
    monkeypatch.setitem(LH._SELF_CHECK, key, None)
    LH._SELF_CHECK.pop(key)
    assert LH._self_check(torch.device("cpu"), torch.float16) is False


def test_install_skipped_when_every_self_check_fails(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    class _Param:
        is_cuda = True
        device = torch.device("cuda", 0)
        dtype = torch.float16

    model = torch.nn.Linear(4, 4)
    model.head_dim = 512
    monkeypatch.setattr(model, "parameters", lambda: iter([_Param()]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(LH, "_PATCHED", [False])
    monkeypatch.setattr(LH, "_self_check", lambda device, dtype: False)
    saved = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        assert LH.patch_large_head_attention(model) is False
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is saved
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved


def test_enabled_reads_the_same_value_compiled_and_eager(monkeypatch):
    # An lru_cache wrapper is traced through by dynamo (with a warning), so a compiled caller
    # re-read the environment while eager callers kept the cached value.
    import sys

    # dynamo resolves globals through sys.modules; this test loads the file standalone.
    monkeypatch.setitem(sys.modules, LH.__name__, LH)
    monkeypatch.setenv("UNSLOTH_HD512_ATTENTION", "1")
    LH._enabled.cache_clear()
    try:
        assert LH._enabled() is True
        monkeypatch.setenv("UNSLOTH_HD512_ATTENTION", "0")

        def f(x):
            return x + 1 if LH._enabled() else x - 1

        x = torch.zeros(3)
        torch._dynamo.reset()
        compiled = torch.compile(f, backend = "eager", fullgraph = True)(x)
        assert torch.equal(compiled, f(x))
        assert torch.equal(compiled, x + 1)
    finally:
        monkeypatch.delenv("UNSLOTH_HD512_ATTENTION")
        LH._enabled.cache_clear()
        torch._dynamo.reset()


def _install_with_params(monkeypatch, params, **attrs):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = torch.nn.Linear(4, 4)
    model.head_dim = 512
    for name, value in attrs.items():
        setattr(model, name, value)
    monkeypatch.setattr(model, "parameters", lambda: iter(params))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(LH, "_PATCHED", [False])
    checked = []
    monkeypatch.setattr(
        LH, "_self_check", lambda device, dtype: checked.append((device, dtype)) or True
    )
    saved = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        installed = LH.patch_large_head_attention(model)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = saved
    return installed, checked


class _FakeParam:
    is_cuda = True
    device = torch.device("cuda", 0)

    def __init__(
        self,
        dtype,
        quant_dtype = None,
    ):
        self.dtype = dtype
        if quant_dtype is not None:
            self.quant_state = type("QuantState", (), {"dtype": quant_dtype})()


def test_patch_self_checks_the_compute_dtype_of_4bit_only_shards(monkeypatch):
    # A GPU holding only bnb 4-bit weights (uint8 storage), e.g. embeddings offloaded to CPU, must
    # still get its load-time check, else compiled blocks never use the kernel there.
    cpu = type(
        "CpuParam", (), {"is_cuda": False, "device": torch.device("cpu"), "dtype": torch.bfloat16}
    )()
    installed, checked = _install_with_params(
        monkeypatch, [cpu, _FakeParam(torch.uint8, torch.bfloat16)]
    )
    assert installed is True
    assert checked == [(torch.device("cuda", 0), torch.bfloat16)]


def test_patch_self_checks_the_compute_dtype_of_float_packed_4bit(monkeypatch):
    # bnb_4bit_quant_storage = float16 with bfloat16 compute: the storage dtype is a half type, but
    # attention runs in bfloat16, so that is the verdict compiled blocks look up.
    installed, checked = _install_with_params(
        monkeypatch, [_FakeParam(torch.float16, torch.bfloat16)]
    )
    assert installed is True
    assert checked == [(torch.device("cuda", 0), torch.bfloat16)]


def test_patch_skips_the_fp16_self_check_under_forced_float32(monkeypatch):
    # zoo's forced float32 Gemma-4 attention runs float32 SDPA directly, so the router never sees it.
    installed, checked = _install_with_params(
        monkeypatch, [_FakeParam(torch.float16)], _unsloth_forced_float32 = True
    )
    assert installed is False and checked == []
    installed, checked = _install_with_params(monkeypatch, [_FakeParam(torch.float16)])
    assert installed is True and checked == [(torch.device("cuda", 0), torch.float16)]


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_uses_the_kernel_for_masks_made_under_inference_mode():
    # generate() / eval build their masks under inference_mode. Sending those to SDPA repeated the
    # KV heads and kept the dense mask (OOM at 8192 tokens where the kernel fits); they are rescanned
    # per call instead, since an inference tensor has no version counter to cache the verdict on.
    calls, original = _sentinel()
    router = LH._make_router(original)
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    assert LH._self_check(device, dtype) is True
    q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
    kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
    with torch.inference_mode():
        mask = _causal(24).cuda()[None, None]
        before = LH.LARGE_HEAD_ENGAGED["calls"]
        out, _ = router(_Module(), q, kv, kv, mask)
        reference, _ = router(_Module(), q, kv, kv, None)
    assert not isinstance(out, str) and len(calls) == 0
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 2
    torch.testing.assert_close(out, reference)


class _FakeCudaOperand:
    """A meta tensor that reports CUDA, so `_shapes_qualify` can be checked without a GPU."""

    def __init__(
        self,
        shape,
        stride,
        dtype = torch.float16,
    ):
        self._t = torch.empty_strided(shape, stride, device = "meta", dtype = dtype)
        self.shape, self.dtype, self.is_cuda, self.device = self._t.shape, dtype, True, "cuda:0"

    def dim(self):
        return self._t.dim()

    def numel(self):
        return self._t.numel()

    def stride(self, dim = None):
        return self._t.stride() if dim is None else self._t.stride(dim)


def test_shapes_qualify_bounds_the_reachable_row_offset():
    # 2,560 elements, but row 4 starts 2**31 elements in: int32 row offsets wrap in the kernel.
    dense = _FakeCudaOperand((1, 1, 5, 512), (2560, 2560, 512, 1))
    wide = _FakeCudaOperand((1, 1, 5, 512), (0, 0, 1 << 29, 1))
    assert LH._shapes_qualify(dense, dense, dense)
    assert not LH._shapes_qualify(wide, dense, dense)
    assert not LH._shapes_qualify(dense, wide, dense)
    assert not LH._shapes_qualify(dense, dense, wide)
    # Batch and head offsets are int64 in the kernels, so a wide batch stride still qualifies.
    batch = _FakeCudaOperand((2, 1, 5, 512), (1 << 31, 2560, 512, 1))
    assert LH._shapes_qualify(batch, batch, batch)
    # The last row's last element at exactly 2**31 - 1 is the largest offset that still fits.
    edge = _FakeCudaOperand((1, 1, 2, 512), (0, 0, LH._INT32_MAX - 511, 1))
    dense_k = _FakeCudaOperand((1, 1, 2, 512), (1024, 1024, 512, 1))
    assert LH._shapes_qualify(edge, dense_k, dense_k)
    over = _FakeCudaOperand((1, 1, 2, 512), (0, 0, LH._INT32_MAX - 510, 1))
    assert not LH._shapes_qualify(over, dense_k, dense_k)


def _wide_row_stride(shape, dtype, seed):
    # The storage runs to the last row's offset: 2**29 elements per row, over 4 GiB for five rows.
    need = ((shape[-2] - 1) * (1 << 29) + shape[-1]) * torch.empty((), dtype = dtype).element_size()
    if torch.cuda.mem_get_info()[0] < need + (1 << 30):
        pytest.skip(reason = f"needs {need / 2**30:.1f} GiB free for a 2**29 element row stride")
    gen = torch.Generator(device = "cuda").manual_seed(seed)
    x = torch.empty_strided(shape, (0, 0, 1 << 29, 1), device = "cuda", dtype = dtype)
    x.copy_(torch.randn(shape, device = "cuda", dtype = dtype, generator = gen))
    return x


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_sends_a_wide_row_stride_query_to_sdpa():
    dtype = _cuda_dtype()
    S = 5
    q = _wide_row_stride((1, 1, S, 512), dtype, 0)
    k = torch.randn(1, 1, S, 512, device = "cuda", dtype = dtype)
    v = torch.randn_like(k)

    def original(module, query, key, value, attention_mask, **kwargs):
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal = True)
        return out.transpose(1, 2).contiguous(), None

    router = LH._make_router(original)
    before = LH.LARGE_HEAD_ENGAGED["calls"]
    out, _ = router(_Module(), q, k, v, None)
    torch.cuda.synchronize()
    assert LH.LARGE_HEAD_ENGAGED["calls"] == before
    ref = _reference(q.float(), k.float(), v.float(), _causal(S).cuda())
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max() <= 0.03 * ref.abs().max() + 0.02
    del q
    torch.cuda.empty_cache()


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_backward_copies_a_wide_row_stride_grad():
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    S = 5
    torch.manual_seed(1)
    q, k, v = [torch.randn(1, 1, S, 512, device = "cuda", dtype = dtype) for _ in range(3)]
    lo, hi = LH._dense_intervals(1, S, True, q.device)
    ours_in = [x.clone().requires_grad_() for x in (q, k, v)]
    out = LH.large_head_attention(*ours_in, lo, hi)
    grad = _wide_row_stride((1, 1, S, 512), dtype, 2)  # [B, H, S, D], as the kernel reads dO
    out.backward(grad.transpose(1, 2))
    torch.cuda.synchronize()
    ref_in = [x.detach().float().requires_grad_() for x in (q, k, v)]
    _reference(*ref_in, _causal(S).cuda()).backward(grad.float().transpose(1, 2))
    for a, b in zip([x.grad for x in ours_in], [x.grad for x in ref_in]):
        assert torch.isfinite(a).all()
        assert (a.float() - b).abs().max() <= 0.03 * b.abs().max() + 0.02
    del grad
    torch.cuda.empty_cache()


@pytest.mark.skipif(not _SM80, reason = "the router engages on sm80+ only")
def test_router_captures_into_a_cuda_graph_with_an_inference_mask():
    # The mask scan syncs the host ("operation not permitted when stream is capturing"), and a graph
    # would replay one scan's verdict for whatever the mask buffer holds later, so capture uses SDPA.
    def original(module, query, key, value, attention_mask, **kwargs):
        key, value = key.repeat_interleave(2, 1), value.repeat_interleave(2, 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask = attention_mask
        )
        return out.transpose(1, 2).contiguous(), None

    router = LH._make_router(original)
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    assert LH._self_check(device, dtype) is True
    with torch.inference_mode():
        q = torch.randn(1, 4, 24, 512, device = "cuda", dtype = dtype)
        kv = torch.randn(1, 2, 24, 512, device = "cuda", dtype = dtype)
        mask = _causal(24).cuda()[None, None]
        eager_masked = router(_Module(), q, kv, kv, mask)[0]
        eager_dense = router(_Module(), q, kv, kv, None)[0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                router(_Module(), q, kv, kv, mask)
                router(_Module(), q, kv, kv, None)
        stream.synchronize()
        before = LH.LARGE_HEAD_ENGAGED["calls"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream = stream):
            masked = router(_Module(), q, kv, kv, mask)[0]
            dense = router(_Module(), q, kv, kv, None)[0]
        # The unmasked call still runs the kernel inside the graph; the masked one runs SDPA.
        assert LH.LARGE_HEAD_ENGAGED["calls"] == before + 1
        graph.replay()
        torch.cuda.synchronize()
    assert LH._SELF_CHECK[(str(device), dtype)] is True
    torch.testing.assert_close(masked, eager_masked, atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(dense, eager_dense)
