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
# torch varlen is never used on ROCm (_torch_varlen returns None there).
_TORCH_VARLEN_HOST = pytest.mark.skipif(
    torch.version.hip is not None, reason = "torch varlen is off on ROCm"
)


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
        "head_dim_640",
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
    elif why == "head_dim_640":
        # Past both flash (256) and the efficient route's tested width (512).
        D = v_width = 640
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
@pytest.mark.parametrize("window", [None, 4])
def test_rocm_keeps_windowed_rows_off_flash_attn(monkeypatch, window):
    """flash-attn's ROCm backend lists sliding windows as work in progress."""
    from unsloth.utils import attention_dispatch as ad

    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    dispatched = []
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(ad, "select_attention_backend", lambda use_varlen = False: ad.FLASH_VARLEN)
    monkeypatch.setattr(ad, "run_attention", lambda **kwargs: dispatched.append(1) or kwargs["Q"])
    lengths = [3, 5]
    q = torch.randn(1, 4, 8, 64, device = "cuda", dtype = torch.bfloat16)
    mask = (
        _block_causal(lengths, "cuda")
        if window is None
        else _hf_window_mask(lengths, window, "cuda")
    )
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 1, sliding_window = window)
    hpa._sdpa_packed_varlen(
        module, q, q, q, mask, packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32)
    )
    if window is None:
        assert dispatched and not seen
    else:
        assert seen and not dispatched


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
@_TORCH_VARLEN_HOST
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
@_TORCH_VARLEN_HOST
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
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    # transformers' own entry: an earlier test can leave the installed wrapper in the mapping.
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", sdpa_attention_forward)
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


# Gemma-4 global layers: head_dim 512, past every flash kernel, take SDPA's efficient kernel over cu_seqlens.
_EFFICIENT_HOST = pytest.mark.skipif(
    torch.version.hip is not None, reason = "the efficient varlen route is off on ROCm"
)


@pytest.fixture
def _efficient_fresh(monkeypatch):
    monkeypatch.setattr(hpa, "_EFFICIENT_VARLEN", [])
    monkeypatch.setattr(hpa, "_EFFICIENT_DEVICES", {})
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.delenv(hpa._EFFICIENT_ENV, raising = False)


def _wide_inputs(lengths, H, n_kv, D, scale = 0.1):
    T = sum(lengths)

    def make(heads):
        x = torch.randn(1, heads, T, D, device = "cuda", dtype = torch.bfloat16) * scale
        return x.requires_grad_()

    return make(H), make(n_kv), make(n_kv)


@_CUDA
@_EFFICIENT_HOST
@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("n_kv", [8, 2, 1])
def test_wide_heads_match_sdpa_over_the_dense_mask(_efficient_fresh, n_kv, window):
    """Output and dQ / dK / dV equal transformers' sdpa over the real mask, Gemma-4 scaling 1.0."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    torch.manual_seed(0)
    lengths = [37, 100, 2, 5, 230]
    H, D = 8, 512
    module = types.SimpleNamespace(
        is_causal = True, num_key_value_groups = H // n_kv, training = True, sliding_window = window
    )
    q, k, v = _wide_inputs(lengths, H, n_kv, D)
    mask = (
        _block_causal(lengths, "cuda")
        if window is None
        else _hf_window_mask(lengths, window, "cuda")
    )
    psl = torch.tensor(lengths, dtype = torch.int32)
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    fast, _ = hpa._sdpa_packed_varlen(module, q, k, v, mask, scaling = 1.0, packed_seq_lengths = psl)
    assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == before["efficient"] + 1
    assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before["fast"] + 1
    ref, _ = sdpa_attention_forward(module, q, k, v, mask, scaling = 1.0)
    assert fast.shape == ref.shape == (1, sum(lengths), H, D)
    torch.testing.assert_close(fast.float(), ref.float(), atol = 1e-2, rtol = 1e-2)
    g = torch.randn_like(ref)
    for a, b in zip(torch.autograd.grad(fast, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a.float(), b.float(), atol = 2e-2, rtol = 2e-2)


@_CUDA
@_EFFICIENT_HOST
def test_wide_heads_keep_documents_apart(_efficient_fresh):
    """Each document of the packed row equals that document attended alone."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    torch.manual_seed(1)
    lengths = [60, 3, 130]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 4, training = True)
    q, k, v = _wide_inputs(lengths, 8, 2, 512)
    fast, _ = hpa._sdpa_packed_varlen(
        module,
        q,
        k,
        v,
        _block_causal(lengths, "cuda"),
        scaling = 1.0,
        packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
    )
    start = 0
    for n in lengths:
        sl = slice(start, start + n)
        alone, _ = sdpa_attention_forward(
            module, q[:, :, sl], k[:, :, sl], v[:, :, sl], None, scaling = 1.0, is_causal = True
        )
        torch.testing.assert_close(fast[:, sl].float(), alone.float(), atol = 1e-2, rtol = 1e-2)
        start += n


@_CUDA
@_EFFICIENT_HOST
@pytest.mark.parametrize(
    "why",
    [
        "none",
        "cross_document",
        "wrong_window",
        "softcap",
        "sinks",
        "dropout",
        "fp32",
        "head_dim_640",
        "kill_switch",
        "deterministic",
        "math_only",
        "efficient_disabled",
        "no_op",
    ],
)
def test_wide_heads_fall_back_to_the_wrapped_sdpa(_efficient_fresh, monkeypatch, why):
    """Each reason alone sends a head_dim 512 call to the wrapped sdpa; "none" is the control."""
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel

    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    lengths, D, dtype = [5, 7], 512, torch.bfloat16
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 2)
    mask, dropout, context = _block_causal(lengths, "cuda"), 0.0, contextlib.nullcontext()
    if why == "cross_document":
        mask = mask.clone()
        mask[0, 0, 6, 1] = True
    elif why == "wrong_window":
        module.sliding_window = 3  # the mask has no band
    elif why == "softcap":
        module.config = types.SimpleNamespace(attn_logit_softcapping = 50.0)
    elif why == "sinks":
        module.sinks = torch.zeros(4)
    elif why == "dropout":
        dropout = 0.1
    elif why == "fp32":
        dtype = torch.float32
    elif why == "head_dim_640":
        D = 640
    elif why == "kill_switch":
        monkeypatch.setenv(hpa._EFFICIENT_ENV, "0")
    elif why == "math_only":
        context = sdpa_kernel([SDPBackend.MATH])
    elif why == "efficient_disabled":
        context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
    elif why == "no_op":
        monkeypatch.setattr(hpa, "_EFFICIENT_VARLEN", [False])
    q = torch.randn(1, 4, 12, D, device = "cuda", dtype = dtype)
    k = torch.randn(1, 2, 12, D, device = "cuda", dtype = dtype)
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    deterministic = torch.are_deterministic_algorithms_enabled()
    if why == "deterministic":
        torch.use_deterministic_algorithms(True)
    try:
        with context:
            out = hpa._sdpa_packed_varlen(
                module,
                q,
                k,
                k,
                mask,
                dropout = dropout,
                scaling = 1.0,
                packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
            )
    finally:
        torch.use_deterministic_algorithms(deterministic)
    if why == "none":
        assert out[0].shape == (1, 12, 4, D) and seen == []
        assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == before["efficient"] + 1
    else:
        assert out == ("orig", None)
        assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == before["efficient"]
        assert hpa.HF_PACKED_ATTENTION_STATS["fallback"] == before["fallback"] + 1


@_CUDA
@_EFFICIENT_HOST
def test_a_refusing_efficient_kernel_hands_the_call_back(_efficient_fresh, monkeypatch):
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig

    def refuse(*args, **kwargs):
        raise RuntimeError("No available kernel.")

    monkeypatch.setattr(hpa._EfficientVarlenAttention, "apply", refuse)
    lengths = [3, 3]
    q = torch.randn(1, 4, 6, 512, device = "cuda", dtype = torch.bfloat16)
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    out = hpa._sdpa_packed_varlen(
        types.SimpleNamespace(is_causal = True),
        q,
        q,
        q,
        _block_causal(lengths, "cuda"),
        packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
    )
    assert out == ("orig", None) and len(seen) == 1
    assert hpa._EFFICIENT_VARLEN == [False]
    assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before["fast"]
    assert hpa.HF_PACKED_ATTENTION_STATS["fallback"] == before["fallback"] + 1


@_CUDA
@_EFFICIENT_HOST
def test_the_efficient_kill_switch_is_read_on_every_call(_efficient_fresh, monkeypatch):
    q = torch.randn(1, 4, 12, 512, device = "cuda", dtype = torch.bfloat16)
    assert hpa._efficient_varlen_takes(q, None)
    monkeypatch.setenv(hpa._EFFICIENT_ENV, "0")
    assert not hpa._efficient_varlen_takes(q, None)
    monkeypatch.delenv(hpa._EFFICIENT_ENV)
    assert hpa._efficient_varlen_takes(q, None)


@_CUDA
@_EFFICIENT_HOST
def test_a_non_contiguous_head_dim_is_copied_not_refused(_efficient_fresh):
    cu = torch.tensor([0, 5, 12], device = "cuda", dtype = torch.int32)
    base = torch.randn(1, 4, 12, 1024, device = "cuda", dtype = torch.bfloat16)
    q = base[..., ::2]  # head dim 512 with stride 2
    assert q.stride(-1) == 2
    out = hpa._efficient_varlen(q, q, q, cu, 7, 512**-0.5, None)
    ref = hpa._efficient_varlen(q.contiguous(), q.contiguous(), q.contiguous(), cu, 7, 512**-0.5, None)
    torch.testing.assert_close(out, ref)


@_CUDA
@_EFFICIENT_HOST
def test_a_cuda_fault_in_the_efficient_kernel_is_not_a_refusal(_efficient_fresh, monkeypatch):
    if hpa._ACCELERATOR_ERROR is None:
        pytest.skip("this torch has no AcceleratorError")
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig

    def fault(*args, **kwargs):
        raise hpa._ACCELERATOR_ERROR("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(hpa._EfficientVarlenAttention, "apply", fault)
    lengths = [3, 3]
    q = torch.randn(1, 4, 6, 512, device = "cuda", dtype = torch.bfloat16)
    with pytest.raises(hpa._ACCELERATOR_ERROR):
        hpa._sdpa_packed_varlen(
            types.SimpleNamespace(is_causal = True),
            q,
            q,
            q,
            _block_causal(lengths, "cuda"),
            packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
        )
    assert seen == [] and hpa._EFFICIENT_VARLEN != [False]


@_CUDA
@_EFFICIENT_HOST
@pytest.mark.parametrize("backend", ["sdpa", "flash_varlen"])
def test_a_shape_the_flash_impl_refuses_takes_the_efficient_kernel(_efficient_fresh, monkeypatch, backend):
    """FA4 on sm100 lacks head_dim 256 + window: without flash-attn / xformers that row ran SDPA over
    the dense mask; now the efficient kernel. With flash-attn installed, run_attention keeps it."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from unsloth.utils import attention_dispatch as ad
    from unsloth.utils import flash_impl

    if hpa._torch_varlen(torch.device("cuda", 0)) is None:
        pytest.skip("no torch varlen_attn to refuse")
    monkeypatch.setattr(flash_impl, "flash_impl_allows", lambda head_dim, window, device: False)
    monkeypatch.setattr(ad, "select_attention_backend", lambda use_varlen = False: backend)
    dispatched = []
    monkeypatch.setattr(ad, "run_attention", lambda **kwargs: dispatched.append(1) or kwargs["Q"].transpose(1, 2))
    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    torch.manual_seed(2)
    lengths, window = [9, 20, 4], 6
    module = types.SimpleNamespace(
        is_causal = True, num_key_value_groups = 4, training = True, sliding_window = window
    )
    q, k, v = _wide_inputs(lengths, 8, 2, 256)
    mask = _hf_window_mask(lengths, window, "cuda")
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    fast, _ = hpa._sdpa_packed_varlen(
        module, q, k, v, mask, scaling = 1.0, packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32)
    )
    if backend == "flash_varlen":
        assert dispatched == [1]
        assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == before["efficient"]
        return
    assert dispatched == []
    assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == before["efficient"] + 1
    ref, _ = sdpa_attention_forward(module, q, k, v, mask, scaling = 1.0)
    torch.testing.assert_close(fast.float(), ref.float(), atol = 1e-2, rtol = 1e-2)
    g = torch.randn_like(ref)
    for a, b in zip(torch.autograd.grad(fast, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a.float(), b.float(), atol = 2e-2, rtol = 2e-2)


@_CUDA
@_EFFICIENT_HOST
def test_wide_heads_under_reentrant_checkpoint(_efficient_fresh):
    from torch.utils.checkpoint import checkpoint
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    lengths = [40, 24]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 4, training = True)
    q, k, v = _wide_inputs(lengths, 8, 2, 512)
    mask, psl = _block_causal(lengths, "cuda"), torch.tensor(lengths, dtype = torch.int32)
    fn = lambda q, k, v: hpa._sdpa_packed_varlen(module, q, k, v, mask, scaling = 1.0, packed_seq_lengths = psl)[0]
    out = checkpoint(fn, q, k, v, use_reentrant = True)
    g = torch.randn_like(out)
    out.backward(g)
    grads = [x.grad for x in (q, k, v)]
    ref = sdpa_attention_forward(module, q, k, v, mask, scaling = 1.0)[0]
    torch.testing.assert_close(out.float(), ref.float(), atol = 1e-2, rtol = 1e-2)
    for a, b in zip(grads, torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a.float(), b.float(), atol = 2e-2, rtol = 2e-2)


@_CUDA
@_EFFICIENT_HOST
def test_wide_heads_leave_a_compiled_caller_alone(_efficient_fresh):
    """Under fullgraph compile the wrapper returns the wrapped sdpa: no graph break, no counters."""
    hpa._ORIG_SDPA[0] = _plain_sdpa
    lengths = [3, 5]
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 1, training = False)
    q = torch.randn(1, 4, 8, 512, device = "cuda", dtype = torch.bfloat16)
    mask, psl = _block_causal(lengths, "cuda"), torch.tensor(lengths, dtype = torch.int32)
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    fn = torch.compile(
        lambda q: hpa._sdpa_packed_varlen(module, q, q, q, mask, scaling = 1.0, packed_seq_lengths = psl)[0],
        backend = "eager",
        fullgraph = True,
    )
    torch.testing.assert_close(fn(q), _plain_sdpa(module, q, q, q, mask, scaling = 1.0)[0])
    assert hpa.HF_PACKED_ATTENTION_STATS == before


@_CUDA
@_EFFICIENT_HOST
@pytest.mark.parametrize("k_eq_v", [False, True])
def test_tiny_gemma4_packed_row_matches_each_document(_efficient_fresh, monkeypatch, k_eq_v):
    """Real Gemma4ForCausalLM with global_head_dim 512: the global layer takes the efficient kernel,
    and packed logits / grads equal the wrapped sdpa and each document alone."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("this transformers has no Gemma 4")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    hpa._ORIG_SDPA[0] = None
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    # transformers' own entry: an earlier test can leave the installed wrapper in the mapping.
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", sdpa_attention_forward)
    assert hpa.enable_hf_packed_attention()
    torch.manual_seed(0)
    config = transformers.Gemma4TextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        hidden_size_per_layer_input = 8,
        intermediate_size = 128,
        num_hidden_layers = 2,
        layer_types = ["sliding_attention", "full_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_global_key_value_heads = 1 if k_eq_v else None,
        attention_k_eq_v = k_eq_v,
        head_dim = 256,
        global_head_dim = 512,
        sliding_window = 4,
        attn_implementation = "sdpa",
    )
    model = transformers.Gemma4ForCausalLM(config).cuda().to(torch.bfloat16)
    global_attn = model.model.layers[1].self_attn
    assert global_attn.head_dim == 512 and model.model.layers[0].self_attn.head_dim == 256
    lengths = [3, 11, 7]
    ids = torch.randint(0, 128, (1, sum(lengths)), device = "cuda")
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None].cuda()
    psl = torch.tensor(lengths, dtype = torch.int32)

    def run():
        model.zero_grad()
        out = model(input_ids = ids, position_ids = position_ids, packed_seq_lengths = psl, use_cache = False)
        out.logits.float().square().mean().backward()
        return out.logits.detach().float(), [p.grad.float() for p in model.parameters() if p.grad is not None]

    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    fast_logits, fast_grads = run()
    stats = hpa.HF_PACKED_ATTENTION_STATS
    assert stats["efficient"] == before["efficient"] + 1
    # Without torch varlen (torch 2.10 and older) the windowed sliding layer may have no varlen route.
    assert stats["fast"] + stats["fallback"] == before["fast"] + before["fallback"] + 2
    if hpa._torch_varlen(ids.device) is not None:
        assert stats["fast"] == before["fast"] + 2
    monkeypatch.setenv("UNSLOTH_HF_PACKED_VARLEN", "0")
    ref_logits, ref_grads = run()
    torch.testing.assert_close(fast_logits, ref_logits, atol = 3e-2, rtol = 3e-2)
    for a, b in zip(fast_grads, ref_grads):
        torch.testing.assert_close(a, b, atol = 3e-2, rtol = 5e-2)
    with torch.no_grad():
        start = 0
        for n in lengths:
            alone = model(input_ids = ids[:, start : start + n], use_cache = False).logits.float()
            torch.testing.assert_close(fast_logits[:, start : start + n], alone, atol = 3e-2, rtol = 3e-2)
            start += n
