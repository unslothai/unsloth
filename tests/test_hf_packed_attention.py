"""Packed rows on the transformers modeling path run varlen attention, and nothing else changes."""

import types

import pytest
import torch

from unsloth.utils import hf_packed_attention as hpa


def _block_causal(lengths, device = "cpu"):
    segment = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).to(device)
    return ((segment[:, None] == segment[None, :]).tril())[None, None]


def _cu(lengths, device = "cpu"):
    return torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype = torch.int32, device = device
    )


@pytest.fixture(autouse = True)
def _fresh_state():
    saved = hpa._ORIG_SDPA[0]
    hpa._MASK_CHECK[:] = [None, None, None]
    yield
    hpa._MASK_CHECK[:] = [None, None, None]
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


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


@_CUDA
@pytest.mark.parametrize(
    "why",
    [
        "none",
        "sliding",
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
    ],
)
def test_packed_calls_that_varlen_cannot_express_fall_back(monkeypatch, why):
    """Each reason alone sends the call to the wrapped sdpa; "none" is the control that takes varlen."""
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if not (HAS_FLASH_ATTENTION or HAS_XFORMERS):
        pytest.skip("needs xformers or flash-attn")
    seen, orig = _calls()
    hpa._ORIG_SDPA[0] = orig
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths, D = [3, 3], 64
    module = types.SimpleNamespace(is_causal = True)
    device, dtype, B = "cuda", torch.bfloat16, 1
    kwargs = {"packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32)}
    dropout, mask, v_width = 0.0, _block_causal(lengths, "cuda"), D
    if why == "sliding":
        module.sliding_window = 4
    elif why == "softcap":
        module.config = types.SimpleNamespace(attn_logit_softcapping = 50.0)
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
    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
    if why == "autocast_dtype":
        with torch.autocast("cuda", dtype = torch.float16):
            out = hpa._sdpa_packed_varlen(module, q, q, v, mask, dropout = dropout, **kwargs)
    else:
        out = hpa._sdpa_packed_varlen(module, q, q, v, mask, dropout = dropout, **kwargs)
    if why == "none":
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


@_CUDA
def test_fullgraph_compile_keeps_the_wrapped_sdpa(monkeypatch):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

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
def test_reentrant_checkpoint_of_grouped_heads(monkeypatch):
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if not (HAS_FLASH_ATTENTION or HAS_XFORMERS):
        pytest.skip("needs xformers or flash-attn")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("n_kv", [8, 2])
def test_varlen_matches_sdpa_over_the_dense_mask(monkeypatch, n_kv):
    from unsloth.utils.attention_dispatch import HAS_FLASH_ATTENTION, HAS_XFORMERS

    if not (HAS_FLASH_ATTENTION or HAS_XFORMERS):
        pytest.skip("needs xformers or flash-attn")
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    torch.manual_seed(0)
    lengths = [37, 100, 5, 114]
    T, H, D = sum(lengths), 8, 64
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = H // n_kv, training = True)
    q = torch.randn(1, H, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(1, n_kv, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    v = torch.randn(1, n_kv, T, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    mask = _block_causal(lengths, "cuda")
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
