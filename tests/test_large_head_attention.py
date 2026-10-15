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

    def original(module, query, key, value, attention_mask, dropout = 0.0,
                 scaling = None, is_causal = None, **kwargs):
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
    allowed = {
        "causal": _causal(S),
        "packed": _packed(S, [1, S // 3, S // 3 + 2]),
        "right_pad": _causal(S) & (torch.arange(S)[None, :] < S - 5),
    }[kind].cuda()[None, None].expand(B, 1, S, S)
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_self_check_leaves_global_rng_alone(monkeypatch):
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    LH._SELF_CHECK.pop((str(device), dtype), None)
    seeded = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: seeded.append(seed))
    state = torch.cuda.get_rng_state(device)
    assert LH._self_check(device, dtype) is True
    assert not seeded  # would reseed the model's other GPUs
    assert torch.equal(torch.cuda.get_rng_state(device), state)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_self_check_under_inference_mode():
    device, dtype = torch.device("cuda", torch.cuda.current_device()), _cuda_dtype()
    LH._SELF_CHECK.pop((str(device), dtype), None)
    with torch.inference_mode():
        assert LH._self_check(device, dtype) is True


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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
    out, _ = torch.compile(router, backend = backend)(_Module(), q, kv, kv, None, scaling = 512 ** -0.5)
    assert not calls and out.shape == (1, 24, 4, 512)
    assert any("large_head_attention_fwd" in code for code in graphs)
    assert LH._SELF_CHECK.get((str(device), dtype)) is True


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
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
