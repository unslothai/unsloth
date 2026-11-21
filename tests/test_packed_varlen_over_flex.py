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
    dtype = torch.bfloat16,
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
    config.dtype = dtype
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
    # SDPA backend state for these shapes: flash enabled, flash picked (not math).
    monkeypatch.setattr(hpa, "_SDPA_MATH", [-1, lambda *args: (True, 0)])


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


@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only = True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize(
    "why",
    [
        "float32",
        "float32_embeddings",
        "dtype_unknown",
        "deterministic",
        "flash_sdpa_disabled",
        "math_first_priority",
        "softcap_config",
        "sinks_module",
        "no_sdpa_model_type",
    ],
)
def test_rows_the_varlen_wrapper_declines_stay_on_flex(routed, monkeypatch, request, why):
    """The wrapper would send these to dense-mask SDPA (slower than flex, ~2x memory at head_dim 256)."""
    config = _gemma()
    model = types.SimpleNamespace(config = config)
    if why == "float32":
        config = _gemma(dtype = torch.float32)
        model = types.SimpleNamespace(config = config)
    elif why == "float32_embeddings":
        embedding = torch.nn.Embedding(4, 4)
        model.get_input_embeddings = lambda: embedding
    elif why == "dtype_unknown":
        config = _gemma(dtype = None)
        model = types.SimpleNamespace(config = config)
    elif why == "deterministic":
        request.getfixturevalue("deterministic")
    elif why == "flash_sdpa_disabled":
        monkeypatch.setattr(hpa, "_SDPA_MATH", [-1, lambda *args: (False, 0)])
    elif why == "math_first_priority":
        monkeypatch.setattr(hpa, "_SDPA_MATH", [-1, lambda *args: (True, -1)])
    elif why == "softcap_config":
        config.attn_logit_softcapping = 50.0
    elif why == "sinks_module":
        attention = types.SimpleNamespace(sinks = torch.zeros(2))
        model.modules = lambda: iter([attention])
    elif why == "no_sdpa_model_type":
        monkeypatch.setattr(u, "_NO_SDPA_MODEL_TYPES", {"gemma"})
    assert not u.use_sdpa_for_packed_head_dim_flex(model)
    assert config._attn_implementation == "flex_attention"


def test_explicit_flex_request_stays_flex(routed):
    assert u.requests_flex_attention("flex_attention")
    assert u.requests_flex_attention({"text_config": "flex_attention"})
    assert not u.requests_flex_attention(None) and not u.requests_flex_attention("sdpa")
    config = _gemma()
    model = types.SimpleNamespace(config = config)
    setattr(model, u._EXPLICIT_FLEX_ATTR, True)
    assert not u.use_sdpa_for_packed_head_dim_flex(model)
    assert config._attn_implementation == "flex_attention"


def test_explicit_flex_is_per_model_not_per_type(routed):
    # Two models of one type: the one loaded with an explicit flex request keeps flex, the other switches.
    explicit, routed_model = (
        types.SimpleNamespace(config = _gemma()),
        types.SimpleNamespace(config = _gemma()),
    )
    setattr(explicit, u._EXPLICIT_FLEX_ATTR, True)
    assert u.use_sdpa_for_packed_head_dim_flex(routed_model)
    assert not u.use_sdpa_for_packed_head_dim_flex(explicit)
    assert explicit.config._attn_implementation == "flex_attention"
    assert routed_model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("compiled", ["torch_compile_arg", "compiled_model"])
def test_whole_model_compile_keeps_flex(routed, monkeypatch, compiled):
    # The outer compile inlines the blocks, where the varlen wrapper falls back to dense-mask SDPA.
    from unsloth.utils import packing

    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    config = _gemma()
    model = types.SimpleNamespace(config = config)
    args = types.SimpleNamespace()
    if compiled == "torch_compile_arg":
        args.torch_compile = True
    else:
        model._orig_mod = types.SimpleNamespace(config = config)
    collator = types.SimpleNamespace(padding_free = True, torch_call = lambda examples: {})
    trainer = types.SimpleNamespace(data_collator = collator, args = args)
    packing.enable_padding_free_metadata(model, trainer)
    assert config._attn_implementation == "flex_attention"


@pytest.mark.parametrize(
    "batch,max_length,switched", [(2, 4096, True), (8, 4096, False), (1, 32768, False)]
)
def test_long_flattened_rows_keep_flex(routed, monkeypatch, batch, max_length, switched):
    # SDPA builds a dense [T, T] mask for the whole flattened row before the varlen wrapper runs.
    from unsloth.utils import packing

    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    config = _gemma()
    args = types.SimpleNamespace(max_length = max_length, per_device_train_batch_size = batch)
    collator = types.SimpleNamespace(padding_free = True, torch_call = lambda examples: {})
    trainer = types.SimpleNamespace(data_collator = collator, args = args)
    packing.enable_padding_free_metadata(types.SimpleNamespace(config = config), trainer)
    assert config._attn_implementation == ("sdpa" if switched else "flex_attention")


@pytest.mark.parametrize(
    "eval_dataset,eval_batch,pad_multiple,switched",
    [(None, 8, None, True), ([1], 8, None, False), ([1], 1, None, True), (None, 8, 8, False)],
)
def test_eval_rows_and_aligned_tails_keep_flex(
    routed, monkeypatch, eval_dataset, eval_batch, pad_multiple, switched
):
    # Eval batches flatten into one row too; a padded-to-multiple tail is declined by the varlen wrapper.
    from unsloth.utils import packing

    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    config = _gemma()
    args = types.SimpleNamespace(
        max_length = 8192, per_device_train_batch_size = 1, per_device_eval_batch_size = eval_batch
    )
    collator = types.SimpleNamespace(
        padding_free = True, torch_call = lambda examples: {}, pad_to_multiple_of = pad_multiple
    )
    trainer = types.SimpleNamespace(data_collator = collator, args = args, eval_dataset = eval_dataset)
    packing.enable_padding_free_metadata(types.SimpleNamespace(config = config), trainer)
    assert config._attn_implementation == ("sdpa" if switched else "flex_attention")


def test_padding_free_metadata_makes_the_switch(routed, monkeypatch):
    from unsloth.utils import packing

    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    config = _gemma()
    collator = types.SimpleNamespace(padding_free = True, torch_call = lambda examples: {})
    trainer = types.SimpleNamespace(data_collator = collator, args = types.SimpleNamespace())
    packing.enable_padding_free_metadata(types.SimpleNamespace(config = config), trainer)
    assert config._attn_implementation == "sdpa"


def test_a_document_too_long_to_split_under_int32_stays_on_flex(routed):
    """The wrapper splits many short documents under flash's int32 varlen backward limit, but one
    document past it alone would go to dense-mask SDPA: (L + 128) * 2 heads * 256 >= 2**31."""
    over = 2**31 // (2 * 256) - 128
    config = _gemma()
    assert not u.use_sdpa_for_packed_head_dim_flex(types.SimpleNamespace(config = config), over)
    assert config._attn_implementation == "flex_attention"
    assert u.use_sdpa_for_packed_head_dim_flex(types.SimpleNamespace(config = config), over - 1)
    assert config._attn_implementation == "sdpa"


def test_packing_passes_the_trainer_max_length(routed, monkeypatch):
    from unsloth.utils import packing
    monkeypatch.setattr(hpa, "enable_hf_packed_attention", lambda: True)
    for max_length, impl in ((2**22, "flex_attention"), (8192, "sdpa")):
        config = _gemma()
        collator = types.SimpleNamespace(padding_free = True, torch_call = lambda examples: {})
        trainer = types.SimpleNamespace(
            data_collator = collator, args = types.SimpleNamespace(max_length = max_length)
        )
        packing.enable_padding_free_metadata(types.SimpleNamespace(config = config), trainer)
        assert config._attn_implementation == impl


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
