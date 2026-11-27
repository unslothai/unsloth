# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Compiled flex mask: no graph breaks, same mask as eager."""

import pytest

torch = pytest.importorskip("torch")
masking_utils = pytest.importorskip("transformers.masking_utils")
qwen3 = pytest.importorskip("transformers.models.qwen3.configuration_qwen3")

from unsloth.import_fixes import (  # noqa: E402
    _FLEX_MASK_PATCH_FLAG,
    _flex_mask_reads_padding_values,
    fix_transformers_flex_mask_graph_breaks,
)

import inspect  # noqa: E402

# transformers before 5.0 builds flex masks from cache_position and never reads the padding values.
NEW_SIGNATURE = "q_length" in inspect.signature(masking_utils.flex_attention_mask).parameters
needs_new_signature = pytest.mark.skipif(not NEW_SIGNATURE, reason = "old flex mask signature")

create_causal_mask = getattr(
    masking_utils, "_unsloth_original_create_causal_mask", masking_utils.create_causal_mask
)


def _config():
    config = qwen3.Qwen3Config(
        hidden_size = 16,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
        num_hidden_layers = 2,
    )
    config._attn_implementation = "flex_attention"
    return config


def _build(
    config,
    embeds,
    attention_mask,
    position_ids = None,
):
    return create_causal_mask(
        config = config,
        inputs_embeds = embeds,
        attention_mask = attention_mask,
        past_key_values = None,
        position_ids = position_ids,
    )


def _elements(block_mask, length):
    q, kv = torch.arange(length)[:, None], torch.arange(length)[None]
    zero = torch.zeros((), dtype = torch.long)
    return torch.stack(
        [block_mask.mask_mod(torch.tensor(b), zero, q, kv) for b in range(block_mask.shape[0])]
    )


@needs_new_signature
def test_compiled_flex_mask_has_no_graph_breaks_and_matches_eager():
    fix_transformers_flex_mask_graph_breaks()
    fix_transformers_flex_mask_graph_breaks()
    assert getattr(masking_utils.create_block_mask, _FLEX_MASK_PATCH_FLAG, False)
    config = _config()
    compiled = torch.compile(_build, backend = "eager", fullgraph = True, dynamic = True)
    cases = [
        (torch.ones(2, 200, dtype = torch.long), None),
        (torch.tensor([[1] * 300, [1] * 250 + [0] * 50]), None),
        (None, torch.cat([torch.arange(100), torch.arange(60), torch.arange(40)])[None]),
    ]
    for attention_mask, position_ids in cases:
        length = (attention_mask if attention_mask is not None else position_ids).shape[-1]
        embeds = torch.zeros(1 if attention_mask is None else 2, length, 16)
        eager = _build(config, embeds, attention_mask, position_ids)
        traced = compiled(config, embeds, attention_mask, position_ids)
        assert torch.equal(eager.to_dense(), traced.to_dense())
        assert torch.equal(_elements(eager, length), _elements(traced, length))
    assert not _elements(traced, 200)[0, 100:160, :100].any()


@needs_new_signature
def test_a_wrapper_forwarding_by_position_keeps_the_compiled_path():
    """unsloth_zoo wraps the registered flex mask builder above this one and forwards `bound.args`,
    i.e. by position; the patch must still drop the `fast_all` branch, or fullgraph tracing breaks."""
    fix_transformers_flex_mask_graph_breaks()
    interface = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
    registered = interface["flex_attention"]
    ours = registered
    while ours is not None and not getattr(ours, _FLEX_MASK_PATCH_FLAG, False):
        ours = getattr(ours, "__wrapped__", None)
    assert ours is not None, "the flex mask patch is not in the registered chain"
    signature = inspect.signature(ours)

    def positional(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return ours(*bound.args, **bound.kwargs)

    interface.register("flex_attention", positional)
    try:
        config = _config()
        compiled = torch.compile(_build, backend = "eager", fullgraph = True, dynamic = True)
        attention_mask = torch.tensor([[1] * 300, [1] * 250 + [0] * 50])
        embeds = torch.zeros(2, 300, 16)
        eager = _build(config, embeds, attention_mask)
        traced = compiled(config, embeds, attention_mask)
        assert torch.equal(eager.to_dense(), traced.to_dense())
        assert torch.equal(_elements(eager, 300), _elements(traced, 300))
    finally:
        interface.register("flex_attention", registered)
        torch._dynamo.reset()


def test_probe_reads_only_value_branching_builders():
    def branches(
        batch_size,
        q_length,
        kv_length,
        attention_mask = None,
        device = "cpu",
        **kwargs,
    ):
        if attention_mask is not None and not attention_mask.all():
            pass
        return masking_utils.create_block_mask(None)

    def applies(
        batch_size,
        q_length,
        kv_length,
        attention_mask = None,
        device = "cpu",
        **kwargs,
    ):
        return masking_utils.create_block_mask(None)

    builder = masking_utils.create_block_mask
    assert _flex_mask_reads_padding_values(masking_utils, branches)
    assert not _flex_mask_reads_padding_values(masking_utils, applies)
    assert masking_utils.create_block_mask is builder


@needs_new_signature
def test_eager_calls_reach_the_original():
    fix_transformers_flex_mask_graph_breaks()
    patched = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"]
    original = getattr(patched, "__wrapped__", patched)
    calls = []
    builder = masking_utils.create_block_mask
    masking_utils.create_block_mask = lambda *args, **kwargs: calls.append(kwargs) or "mask"
    try:
        mask = torch.ones(1, 4, dtype = torch.bool)
        kwargs = dict(batch_size = 1, q_length = 4, kv_length = 4, attention_mask = mask)
        assert patched(**kwargs) == original(**kwargs) == "mask"
        assert calls[0].keys() == calls[1].keys()
    finally:
        masking_utils.create_block_mask = builder


@pytest.mark.skipif(NEW_SIGNATURE, reason = "new flex mask signature")
def test_old_flex_mask_builder_is_left_alone():
    original = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"]
    fix_transformers_flex_mask_graph_breaks()
    assert masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"] is original
