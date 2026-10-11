# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed rows on the transformers modeling path run varlen attention, and nothing else changes."""

import inspect
import types

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

from unsloth.utils import hf_packed_attention as hpa


def _block_causal(lengths, device = "cpu"):
    segment = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).to(device)
    return ((segment[:, None] == segment[None, :]).tril())[None, None]


def _hf_window_mask(
    lengths,
    window,
    device = "cpu",
):
    """The mask transformers builds for a packed row on a sliding layer: its own mask functions."""
    from transformers.masking_utils import (
        and_masks,
        find_packed_sequence_indices,
        packed_sequence_mask_function,
        sliding_window_causal_mask_function,
    )

    position_ids = torch.cat([torch.arange(n) for n in lengths])[None].to(device)
    packed = find_packed_sequence_indices(position_ids)
    fn = and_masks(
        sliding_window_causal_mask_function(window), packed_sequence_mask_function(packed)
    )
    idx = torch.arange(sum(lengths), device = device)
    zero = torch.zeros((), dtype = torch.long, device = device)
    return fn(zero, zero, idx[:, None], idx[None, :])[None, None]


def _cu(lengths, device = "cpu"):
    return torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype = torch.int32, device = device
    )


@pytest.fixture(autouse = True)
def _fresh_state():
    saved = hpa._ORIG_SDPA[0]
    hpa._MASK_CHECK.clear()
    yield
    hpa._MASK_CHECK.clear()
    hpa._ORIG_SDPA[0] = saved


def test_block_causal_mask_is_recognised():
    lengths = [3, 5, 2]
    assert hpa._mask_is_block_causal(_block_causal(lengths), _cu(lengths), 10)


@pytest.mark.parametrize(
    "edit",
    ["cross_document", "bidirectional", "window", "plain_causal", "pad_tail", "float"],
)
def test_any_other_mask_keeps_the_masked_path(edit):
    lengths = [3, 5, 2]
    mask, cu, total = _block_causal(lengths), _cu(lengths), 10
    if edit == "cross_document":
        mask = mask.clone()
        mask[0, 0, 4, 1] = True
    elif edit == "bidirectional":
        mask = mask.clone()
        mask[0, 0, 3, 4] = True
    elif edit == "window":
        mask = mask.clone()
        mask[0, 0, 7, 3] = False
    elif edit == "plain_causal":
        mask = torch.ones(10, 10, dtype = torch.bool).tril()[None, None]
    elif edit == "pad_tail":
        cu = _cu([3, 5])
    elif edit == "float":
        mask = torch.where(mask, 0.0, float("-inf"))
    assert not hpa._mask_is_block_causal(mask, cu, total)


@pytest.mark.parametrize("window", [1, 3, 4, 128])
def test_windowed_mask_is_recognised_only_with_its_window(window):
    lengths = [2, 7, 130, 4]
    mask, cu, total = _hf_window_mask(lengths, window), _cu(lengths), sum(lengths)
    assert hpa._mask_is_block_causal(mask, cu, total, window)
    # One off either way, or no window at all, is a different band for the 130-token document.
    for wrong in (window - 1 or None, window + 1, None):
        hpa._MASK_CHECK.clear()
        assert not hpa._mask_is_block_causal(mask, cu, total, wrong)


def test_documents_inside_the_window_are_plain_block_causal():
    lengths = [3, 4, 2]
    mask = _hf_window_mask(lengths, 4)
    assert torch.equal(mask, _block_causal(lengths))
    assert hpa._mask_is_block_causal(mask, _cu(lengths), 9, 4)


def test_sliding_and_full_layers_keep_their_own_verdicts():
    lengths = [6, 9]
    cu = _cu(lengths)
    full, sliding = _block_causal(lengths), _hf_window_mask(lengths, 3)
    for _ in range(3):
        assert hpa._mask_is_block_causal(full, cu, 15)
        assert hpa._mask_is_block_causal(sliding, cu, 15, 3)
    assert hpa.HF_PACKED_ATTENTION_STATS["mask_checks"] >= 2
    checks = hpa.HF_PACKED_ATTENTION_STATS["mask_checks"]
    assert hpa._mask_is_block_causal(full, cu, 15)
    assert hpa._mask_is_block_causal(sliding, cu, 15, 3)
    assert hpa.HF_PACKED_ATTENTION_STATS["mask_checks"] == checks
    # The same mask object asked about another window is checked, not answered from the cache.
    assert not hpa._mask_is_block_causal(full, cu, 15, 3)


def test_mask_verdict_is_cached_per_mask_object():
    lengths = [4, 4]
    mask, cu = _block_causal(lengths), _cu(lengths)
    assert hpa._mask_is_block_causal(mask, cu, 8)
    # Same object: the cached verdict stands. A new object is checked again.
    assert hpa._mask_is_block_causal(mask, cu, 8)
    other = torch.ones(8, 8, dtype = torch.bool).tril()[None, None]
    assert not hpa._mask_is_block_causal(other, cu, 8)


def _calls():
    seen = []

    def orig(
        module,
        q,
        k,
        v,
        mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        seen.append(kwargs)
        return "orig", None

    return seen, orig


def test_unpacked_calls_go_straight_through(monkeypatch):
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    q = torch.zeros(2, 4, 6, 8)
    out = hpa._sdpa_packed_varlen(types.SimpleNamespace(), q, q, q, None, scaling = 0.3, foo = 1)
    assert out == ("orig", None) and seen == [{"foo": 1}]


_CUDA = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device")


@_CUDA
@pytest.mark.parametrize(
    "why",
    [
        "none",
        "window_inside_documents",
        "sliding_wrong_window",
        "zero_window",
        "float_window",
        "softcap",
        "sinks",
        "fp32",
        "dropout",
        "batched",
        "cpu",
        "head_dim_512",
        "mla_value_width",
        "autocast_dtype",
        "paged_cache",
        "no_mask",
        "module_softcap",
        "not_causal",
        "positional",
        "kv_batched",
    ],
)
def test_packed_calls_that_varlen_cannot_express_fall_back(monkeypatch, why):
    """Each reason alone sends the call to the wrapped sdpa; "none" is the control that takes varlen."""
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if not (HAS_FLASH_ATTENTION or HAS_XFORMERS or hpa._torch_varlen(torch.device("cuda", 0))):
        pytest.skip("needs a varlen kernel")
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths, D = [3, 3], 64
    module = types.SimpleNamespace(is_causal = True)
    device, dtype, B = "cuda", torch.bfloat16, 1
    kwargs = {"packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32)}
    dropout, mask, v_width = 0.0, _block_causal(lengths, "cuda"), D
    if why == "window_inside_documents":
        # Both documents fit the window, so the band is the plain block-causal mask.
        module.sliding_window = 4
    elif why == "sliding_wrong_window":
        # The layer says 2, the mask has no band.
        module.sliding_window = 2
    elif why == "zero_window":
        kwargs["sliding_window"] = 0
    elif why == "float_window":
        module.sliding_window = 4.0
    elif why == "softcap":
        module.config = types.SimpleNamespace(attn_logit_softcapping = 50.0)
    elif why == "module_softcap":
        module.attn_logit_softcapping = 30.0
    elif why == "sinks":
        module.sinks = torch.zeros(4)
    elif why == "fp32":
        dtype = torch.float32
    elif why == "dropout":
        dropout = 0.1
    elif why == "batched":
        B = 2
    elif why == "cpu":
        device, mask = "cpu", _block_causal(lengths)
    elif why == "head_dim_512":
        D = v_width = 512
    elif why == "mla_value_width":
        D, v_width = 192, 128
    elif why == "paged_cache":
        kwargs["cache"] = object()
    elif why == "no_mask":
        mask = None
    q = torch.randn(B, 4, 6, D, device = device, dtype = dtype)
    v = torch.randn(B, 4, 6, v_width, device = device, dtype = dtype)
    k = q
    if why == "kv_batched":
        k, v = q.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1)
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    if why == "autocast_dtype":
        with torch.autocast("cuda", dtype = torch.float16):
            out = hpa._sdpa_packed_varlen(module, q, k, v, mask, dropout = dropout, **kwargs)
    elif why == "not_causal":
        out = hpa._sdpa_packed_varlen(module, q, k, v, mask, is_causal = False, **kwargs)
    elif why == "positional":
        out = hpa._sdpa_packed_varlen(module, q, k, v, mask, 0.0, **kwargs)
    else:
        out = hpa._sdpa_packed_varlen(module, q, k, v, mask, dropout = dropout, **kwargs)
    if why in ("none", "window_inside_documents"):
        assert out[0].shape == (1, 6, 4, D) and seen == []
        assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before["fast"] + 1
    else:
        assert out == ("orig", None)
        assert hpa.HF_PACKED_ATTENTION_STATS["fallback"] == before["fallback"] + 1


def test_an_edited_mask_is_checked_again():
    lengths = [3, 3]
    mask, cu = _block_causal(lengths).clone(), _cu(lengths)
    assert hpa._mask_is_block_causal(mask, cu, 6)
    mask[0, 0, 3, 0] = True
    assert not hpa._mask_is_block_causal(mask, cu, 6)
    with torch.inference_mode():
        frozen, frozen_cu = _block_causal(lengths), _cu(lengths)
    assert hpa._mask_is_block_causal(frozen, frozen_cu, 6)


def test_a_missing_mask_is_not_block_causal():
    assert not hpa._mask_is_block_causal(None, _cu([6]), 6)


def _plain_sdpa(
    module,
    q,
    k,
    v,
    mask,
    dropout = 0.0,
    scaling = None,
    is_causal = None,
    **kwargs,
):
    # Not transformers' sdpa_attention_forward: compiling that after importing unsloth trips a Dynamo
    # guard bug on torch <= 2.13, with or without this wrapper.
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = mask, scale = scaling)
    return out.transpose(1, 2), None


@_CUDA
def test_fullgraph_compile_keeps_the_wrapped_sdpa(monkeypatch):
    sdpa_attention_forward = _plain_sdpa
    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths = [3, 3]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 1, training = False)
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    mask, psl = _block_causal(lengths, "cuda"), torch.tensor(lengths, dtype = torch.int32)
    fn = torch.compile(
        lambda q: hpa._sdpa_packed_varlen(module, q, q, q, mask, packed_seq_lengths = psl)[0],
        backend = "eager",
        fullgraph = True,
    )
    torch.testing.assert_close(fn(q), sdpa_attention_forward(module, q, q, q, mask)[0])


@_CUDA
@pytest.mark.parametrize("route", ["torch", "dispatch"])
def test_reentrant_checkpoint_of_grouped_heads(monkeypatch, route):
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if route == "torch" and hpa._torch_varlen(torch.device("cuda", 0)) is None:
        pytest.skip("this torch has no window_size / enable_gqa varlen_attn")
    if route == "dispatch":
        if not (HAS_FLASH_ATTENTION or HAS_XFORMERS):
            pytest.skip("needs xformers or flash-attn")
        monkeypatch.setattr(hpa, "_torch_varlen", lambda device: None)
    from torch.utils.checkpoint import checkpoint
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths = [40, 24]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 4, training = True)
    q = torch.randn(1, 8, 64, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(1, 2, 64, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    v = torch.randn(1, 2, 64, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    mask, psl = _block_causal(lengths, "cuda"), torch.tensor(lengths, dtype = torch.int32)
    fn = lambda q, k, v: hpa._sdpa_packed_varlen(module, q, k, v, mask, packed_seq_lengths = psl)[0]
    out = checkpoint(fn, q, k, v, use_reentrant = True)
    out.float().sum().backward()
    ref = sdpa_attention_forward(module, q, k, v, mask)[0]
    torch.testing.assert_close(out.float(), ref.float(), atol = 2e-2, rtol = 2e-2)


def test_kill_switch(monkeypatch):
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.setenv("UNSLOTH_HF_PACKED_VARLEN", "0")
    q = torch.zeros(1, 4, 6, 8, dtype = torch.bfloat16)
    lengths = torch.tensor([3, 3], dtype = torch.int32)
    out = hpa._sdpa_packed_varlen(
        types.SimpleNamespace(), q, q, q, _block_causal([3, 3]), packed_seq_lengths = lengths
    )
    assert out == ("orig", None)
    assert not hpa.enable_hf_packed_attention()


def test_install_forwards_router_sentinels_and_runs_once(monkeypatch):
    mapping_mod = pytest.importorskip("transformers.modeling_utils")
    mapping = mapping_mod.ALL_ATTENTION_FUNCTIONS

    def router(*args, **kwargs):
        return None

    router._unsloth_gemma4_flash = True
    monkeypatch.setitem(mapping, "sdpa", router)
    hpa._ORIG_SDPA[0] = None
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    assert hpa.enable_hf_packed_attention()
    installed = mapping["sdpa"]
    assert installed is hpa._sdpa_packed_varlen and hpa._ORIG_SDPA[0] is router
    # A zoo router re-running on the next load sees its own sentinel and leaves the chain alone.
    assert getattr(installed, "_unsloth_gemma4_flash", False)
    # A router stacked above later never gets wrapped a second time.
    monkeypatch.setitem(mapping, "sdpa", router)
    assert hpa.enable_hf_packed_attention()
    assert mapping["sdpa"] is router and hpa._ORIG_SDPA[0] is router


@_CUDA
@pytest.mark.parametrize("window", [None, 3, 4, 128])
@pytest.mark.parametrize("route", ["torch", "torch_2_11", "dispatch"])
@pytest.mark.parametrize("n_kv", [8, 2])
def test_varlen_matches_sdpa_over_the_dense_mask(monkeypatch, n_kv, route, window):
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if route != "dispatch" and hpa._torch_varlen(torch.device("cuda", 0)) is None:
        pytest.skip("this torch has no window_size varlen_attn")
    if route == "torch_2_11":
        # torch 2.11's varlen_attn has scale / window_size but no enable_gqa.
        native = hpa._TORCH_VARLEN[0]

        def without_gqa(*args, **kwargs):
            assert "enable_gqa" not in kwargs
            return native(*args, **kwargs)

        monkeypatch.setattr(hpa, "_TORCH_VARLEN", [without_gqa])
        monkeypatch.setattr(hpa, "_TORCH_VARLEN_GQA", [False])
    if route == "dispatch":
        if not (HAS_FLASH_ATTENTION or HAS_XFORMERS):
            pytest.skip("needs xformers or flash-attn")
        monkeypatch.setattr(hpa, "_torch_varlen", lambda device: None)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    torch.manual_seed(0)
    lengths = [37, 100, 2, 5, 230]
    T, H, D = sum(lengths), 8, 64
    module = types.SimpleNamespace(
        is_causal = True, num_key_value_groups = H // n_kv, training = True, sliding_window = window
    )
    q = torch.randn(1, H, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(1, n_kv, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    v = torch.randn(1, n_kv, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    mask = (
        _block_causal(lengths, "cuda")
        if window is None
        else _hf_window_mask(lengths, window, "cuda")
    )
    psl = torch.tensor(lengths, dtype = torch.int32)
    before = hpa.HF_PACKED_ATTENTION_STATS["fast"]
    fast, _ = hpa._sdpa_packed_varlen(module, q, k, v, mask, scaling = 0.11, packed_seq_lengths = psl)
    assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before + 1
    ref, _ = sdpa_attention_forward(module, q, k, v, mask, scaling = 0.11)
    assert fast.shape == ref.shape
    torch.testing.assert_close(fast.float(), ref.float(), atol = 2e-2, rtol = 2e-2)
    g = torch.randn_like(ref)
    grads_fast = torch.autograd.grad(fast, (q, k, v), g)
    grads_ref = torch.autograd.grad(ref, (q, k, v), g)
    for a, b in zip(grads_fast, grads_ref):
        torch.testing.assert_close(a.float(), b.float(), atol = 5e-2, rtol = 5e-2)


def test_new_lengths_tensor_with_a_cached_mask_is_checked_again():
    lengths = [3, 3]
    mask = _block_causal(lengths)
    assert hpa._mask_is_block_causal(mask, _cu(lengths), 6)
    # A different cu_seqlens object must not be compared elementwise against the cached one.
    assert hpa._mask_is_block_causal(mask, _cu(lengths), 6)


def test_long_masks_are_checked_in_row_blocks(monkeypatch):
    monkeypatch.setattr(hpa, "_MASK_CHECK_ROWS", 4)
    lengths = [5, 7, 6]
    mask = _block_causal(lengths).clone()
    assert hpa._mask_is_block_causal(mask, _cu(lengths), 18)
    mask[0, 0, 17, 0] = True
    assert not hpa._mask_is_block_causal(mask, _cu(lengths), 18)


@_CUDA
def test_compiling_leaves_the_counters_alone(monkeypatch):
    hpa._ORIG_SDPA[0] = _plain_sdpa
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths = [3, 3]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 1, training = False)
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    mask, psl = _block_causal(lengths, "cuda"), torch.tensor(lengths, dtype = torch.int32)
    fn = torch.compile(
        lambda q: hpa._sdpa_packed_varlen(module, q, q, q, mask, packed_seq_lengths = psl)[0],
        backend = "eager",
        fullgraph = True,
    )
    positional = torch.compile(
        lambda q: hpa._sdpa_packed_varlen(module, q, q, q, mask, 0.0, packed_seq_lengths = psl)[0],
        backend = "eager",
        fullgraph = True,
    )
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    fn(q)
    positional(q)
    assert hpa.HF_PACKED_ATTENTION_STATS == before


@_CUDA
@pytest.mark.parametrize("mode", ["training_no_grad", "eval_grad", "eval_no_grad"])
def test_int32_guard_follows_training_or_grad_mode(monkeypatch, mode):
    """A reentrant checkpoint's no_grad forward (training) and its recompute both fall back; a pure
    no_grad eval forward allocates no backward buffer and keeps varlen."""
    import unsloth.utils.attention_dispatch as ad

    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(ad, "_varlen_backward_overflows_int32", lambda *a: True)
    lengths = [3, 3]
    module = types.SimpleNamespace(is_causal = True, training = mode == "training_no_grad")
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    psl = torch.tensor(lengths, dtype = torch.int32)
    with torch.set_grad_enabled(mode == "eval_grad"):
        out = hpa._sdpa_packed_varlen(
            module, q, q, q, _block_causal(lengths, "cuda"), packed_seq_lengths = psl
        )
    if mode == "eval_no_grad":
        if hpa._torch_varlen(q.device) is None and not (ad.HAS_FLASH_ATTENTION or ad.HAS_XFORMERS):
            pytest.skip("needs a varlen kernel")
        assert out[0].shape == (1, 6, 4, 64)
    else:
        assert out == ("orig", None)


@_CUDA
def test_a_refusing_torch_varlen_hands_the_call_back(monkeypatch):
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)

    def refuse(*args, **kwargs):
        raise RuntimeError("GQA is not supported with the cuDNN backend.")

    monkeypatch.setattr(hpa, "_TORCH_VARLEN", [refuse])
    monkeypatch.setattr(hpa, "_TORCH_VARLEN_DEVICES", {})
    lengths = [3, 3]
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    psl = torch.tensor(lengths, dtype = torch.int32)
    out = hpa._sdpa_packed_varlen(
        types.SimpleNamespace(is_causal = True),
        q,
        q,
        q,
        _block_causal(lengths, "cuda"),
        packed_seq_lengths = psl,
    )
    assert out == ("orig", None)
    assert hpa._TORCH_VARLEN == [None]


def test_torch_varlen_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_VARLEN", "0")
    monkeypatch.setattr(hpa, "_TORCH_VARLEN", [])
    monkeypatch.setattr(hpa, "_TORCH_VARLEN_GQA", [])
    assert hpa._torch_varlen(torch.device("cuda", 0)) is None
    assert hpa._TORCH_VARLEN == [None]


@_CUDA
@pytest.mark.parametrize("policy", ["math_only", "flash_disabled", "math_first", "deterministic"])
def test_an_sdpa_policy_keeps_every_varlen_kernel_off(monkeypatch, policy):
    """The wrapped sdpa runs, not torch varlen nor the xformers / flash-attn dispatch; outside, varlen runs."""
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from unsloth.utils import attention_dispatch as ad

    if policy == "math_first" and "set_priority" not in inspect.signature(sdpa_kernel).parameters:
        pytest.skip("this torch has no sdpa_kernel(set_priority = True)")
    calls = []

    def spy(q, *args, **kwargs):
        calls.append(1)
        return torch.zeros_like(q)

    monkeypatch.setattr(hpa, "_TORCH_VARLEN", [spy])
    monkeypatch.setattr(hpa, "_TORCH_VARLEN_DEVICES", {})
    monkeypatch.setattr(ad, "run_attention", lambda **kwargs: calls.append(2) or kwargs["Q"])
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths = [3, 3]
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    k = torch.randn(1, 2, 6, 64, device = "cuda", dtype = torch.bfloat16)
    module = types.SimpleNamespace(is_causal = True)

    def run():
        return hpa._sdpa_packed_varlen(
            module,
            q,
            k,
            k,
            _block_causal(lengths, "cuda"),
            packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
        )

    if policy == "math_only":
        context = sdpa_kernel([SDPBackend.MATH])
    elif policy == "flash_disabled":
        context = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    elif policy == "math_first":
        context = sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION], set_priority = True)
    else:
        context = contextlib.nullcontext()
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
    try:
        with context:
            assert run() == ("orig", None)
    finally:
        if policy == "deterministic":
            torch.use_deterministic_algorithms(deterministic)
    assert calls == [] and len(seen) == 1
    run()
    assert calls == [1]


@_CUDA
@pytest.mark.parametrize("window", [None, 4])
def test_tiny_mistral_packed_row_matches_the_masked_path(monkeypatch, window):
    """A real sliding-window decoder: packed logits and grads with the wrapper equal the wrapped sdpa."""
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if not (HAS_FLASH_ATTENTION or HAS_XFORMERS or hpa._torch_varlen(torch.device("cuda", 0))):
        pytest.skip("no varlen kernel on this build")
    hpa._ORIG_SDPA[0] = None
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", original)
    assert hpa.enable_hf_packed_attention()
    torch.manual_seed(0)
    config = transformers.MistralConfig(
        vocab_size = 128,
        hidden_size = 256,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 64,
        sliding_window = window,
        attn_implementation = "sdpa",
    )
    model = transformers.MistralForCausalLM(config).cuda().to(torch.bfloat16)
    lengths = [3, 11, 7]
    ids = torch.randint(0, 128, (1, sum(lengths)), device = "cuda")
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None].cuda()
    psl = torch.tensor(lengths, dtype = torch.int32)

    def run():
        model.zero_grad()
        out = model(
            input_ids = ids, position_ids = position_ids, packed_seq_lengths = psl, use_cache = False
        )
        out.logits.float().square().mean().backward()
        return out.logits.detach().float(), [p.grad.float() for p in model.parameters()]

    before = hpa.HF_PACKED_ATTENTION_STATS["fast"]
    fast_logits, fast_grads = run()
    assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before + 2
    monkeypatch.setenv("UNSLOTH_HF_PACKED_VARLEN", "0")
    ref_logits, ref_grads = run()
    torch.testing.assert_close(fast_logits, ref_logits, atol = 3e-2, rtol = 3e-2)
    for a, b in zip(fast_grads, ref_grads):
        torch.testing.assert_close(a, b, atol = 3e-2, rtol = 5e-2)
