# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed / padding-free training moves a decoder routed to flex only for its head dim onto "sdpa",
where the packed varlen wrapper runs; every other routing is left alone."""

import types

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u
from unsloth.utils import hf_packed_attention as hpa

transformers = pytest.importorskip("transformers")


def _gemma(
    head_dim = 256,
    impl = "flex_attention",
    model_type = None,
):
    config = transformers.GemmaConfig(
        vocab_size = 128,
        hidden_size = 256,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = head_dim,
    )
    config._attn_implementation = impl
    if model_type is not None:
        config.model_type = model_type
    return config


@pytest.fixture
def routed(monkeypatch):
    """A box where head_dim 256 routes to flex and torch's varlen kernel runs."""
    for name in ("UNSLOTH_PACKED_VARLEN_OVER_FLEX", "UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(u, "_flex_kernels_fit_large_head_dim", lambda: True)
    monkeypatch.setattr(u, "_sdpa_reaches_cudnn_at_head_dim_256", lambda: False)
    monkeypatch.setattr(hpa, "_ORIG_SDPA", [lambda *args, **kwargs: None])
    monkeypatch.setattr(hpa, "_torch_varlen", lambda device: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


def test_head_dim_flex_moves_to_sdpa(routed):
    config = _gemma()
    assert u.use_sdpa_for_packed_head_dim_flex(types.SimpleNamespace(config = config))
    assert config._attn_implementation == "sdpa"


@pytest.mark.parametrize(
    "why",
    [
        "kill_switch",
        "flex_forced",
        "flex_refused",
        "gemma3_family",
        "head_dim_512",
        "head_dim_128",
        "already_sdpa",
        "no_torch_varlen",
        "one_device_without_varlen",
        "wrapper_not_installed",
        "cudnn_takes_head_dim_256",
    ],
)
def test_every_other_routing_is_left_alone(routed, monkeypatch, why):
    config = _gemma()
    if why == "kill_switch":
        monkeypatch.setenv("UNSLOTH_PACKED_VARLEN_OVER_FLEX", "0")
    elif why == "flex_forced":
        monkeypatch.setenv("UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM", "1")
    elif why == "flex_refused":
        monkeypatch.setenv("UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM", "0")
    elif why == "gemma3_family":
        config = _gemma(model_type = "gemma3_text")
    elif why == "head_dim_512":
        config = _gemma(head_dim = 512)
    elif why == "head_dim_128":
        config = _gemma(head_dim = 128)
    elif why == "already_sdpa":
        config = _gemma(impl = "sdpa")
    elif why == "no_torch_varlen":
        monkeypatch.setattr(hpa, "_torch_varlen", lambda device: None)
    elif why == "one_device_without_varlen":
        monkeypatch.setattr(hpa, "_torch_varlen", lambda device: None if device.index else object())
    elif why == "wrapper_not_installed":
        monkeypatch.setattr(hpa, "_ORIG_SDPA", [None])
    elif why == "cudnn_takes_head_dim_256":
        monkeypatch.setattr(u, "_sdpa_reaches_cudnn_at_head_dim_256", lambda: True)
    before = config._attn_implementation
    assert not u.use_sdpa_for_packed_head_dim_flex(types.SimpleNamespace(config = config))
    assert config._attn_implementation == before


def test_padding_free_metadata_makes_the_switch(routed, monkeypatch):
    from unsloth.utils import packing

    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    config = _gemma()
    collator = types.SimpleNamespace(padding_free = True, torch_call = lambda examples: {})
    trainer = types.SimpleNamespace(data_collator = collator, args = types.SimpleNamespace())
    packing.enable_padding_free_metadata(types.SimpleNamespace(config = config), trainer)
    assert config._attn_implementation == "sdpa"


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device")
def test_switched_model_runs_packed_rows_on_varlen_like_flex(monkeypatch):
    if hpa._torch_varlen(torch.device("cuda", 0)) is None:
        pytest.skip("this torch has no window_size varlen_attn")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", ALL_ATTENTION_FUNCTIONS["sdpa"])
    monkeypatch.setattr(hpa, "_ORIG_SDPA", [None])
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    assert hpa.enable_hf_packed_attention()
    monkeypatch.setattr(u, "_flex_kernels_fit_large_head_dim", lambda: True)
    monkeypatch.setattr(u, "_sdpa_reaches_cudnn_at_head_dim_256", lambda: False)
    for name in ("UNSLOTH_PACKED_VARLEN_OVER_FLEX", "UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM"):
        monkeypatch.delenv(name, raising = False)
    torch.manual_seed(0)
    model = transformers.GemmaForCausalLM(_gemma()).cuda().to(torch.bfloat16)
    model.config._attn_implementation = "flex_attention"
    lengths = [5, 70, 53]
    ids = torch.randint(0, 128, (1, sum(lengths)), device = "cuda")
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None].cuda()
    psl = torch.tensor(lengths, dtype = torch.int32)

    def logits():
        with torch.no_grad():
            return model(
                input_ids = ids, position_ids = position_ids, packed_seq_lengths = psl, use_cache = False
            ).logits.float()

    flex = logits()
    assert u.use_sdpa_for_packed_head_dim_flex(model)
    before = hpa.HF_PACKED_ATTENTION_STATS["fast"]
    varlen = logits()
    assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before + 2
    torch.testing.assert_close(varlen, flex, atol = 3e-2, rtol = 3e-2)
