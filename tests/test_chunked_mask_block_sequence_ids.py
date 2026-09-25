# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Llama-4 static-cache generate on transformers 5.17 (`block_sequence_ids` to chunked mask)."""

import inspect

import pytest

torch = pytest.importorskip("torch")
masking_utils = pytest.importorskip("transformers.masking_utils")

from unsloth.import_fixes import (  # noqa: E402
    _CHUNKED_MASK_PATCH_FLAG,
    _chunked_mask_rejects_block_sequence_ids,
    _swap_function_references,
    fix_transformers_chunked_mask_block_sequence_ids,
)

if not hasattr(masking_utils, "create_chunked_causal_mask"):
    pytest.skip("this transformers has no create_chunked_causal_mask", allow_module_level = True)


def _tiny_llama4():
    try:
        from transformers import Llama4ForCausalLM, Llama4TextConfig
    except Exception:
        pytest.skip("this transformers has no Llama-4")
    torch.manual_seed(0)
    config = Llama4TextConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 64,
        intermediate_size_mlp = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_local_experts = 2,
        num_experts_per_tok = 1,
        attention_chunk_size = 8,
        max_position_embeddings = 64,
        interleave_moe_layer_step = 1,
    )
    return Llama4ForCausalLM(config).eval()


def _generate(model, cache_implementation):
    kwargs = {"max_new_tokens": 3, "do_sample": False}
    if cache_implementation is not None:
        kwargs["cache_implementation"] = cache_implementation
    with torch.no_grad():
        return model.generate(torch.tensor([[1, 2, 3, 4]]), **kwargs)


@pytest.fixture
def unpatched():
    # conftest's `import unsloth` may already have installed the fix, so unwrap it first.
    live = masking_utils.create_chunked_causal_mask
    if getattr(live, _CHUNKED_MASK_PATCH_FLAG, False):
        original = live.__wrapped__
        _swap_function_references(masking_utils, live, original)
    else:
        original = live
    try:
        yield original
    finally:
        current = masking_utils.create_chunked_causal_mask
        if current is not live:
            _swap_function_references(masking_utils, current, live)


def test_static_cache_generate_matches_dynamic(unpatched):
    if not _chunked_mask_rejects_block_sequence_ids(masking_utils):
        pytest.skip("this transformers does not pass block_sequence_ids to chunked masks")
    model = _tiny_llama4()
    with pytest.raises(TypeError, match = "block_sequence_ids"):
        _generate(model, "static")

    fix_transformers_chunked_mask_block_sequence_ids()
    patched = masking_utils.create_chunked_causal_mask
    assert getattr(patched, _CHUNKED_MASK_PATCH_FLAG, False)
    assert patched.__wrapped__ is unpatched
    assert masking_utils.LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING["chunked_attention"] is patched
    from transformers.models.llama4 import modeling_llama4

    assert modeling_llama4.create_chunked_causal_mask is patched

    static = _generate(model, "static")
    dynamic = _generate(model, None)
    assert static.shape == (1, 7)
    assert torch.equal(static, dynamic)

    fix_transformers_chunked_mask_block_sequence_ids()
    assert masking_utils.create_chunked_causal_mask is patched


def test_block_sequence_ids_are_honoured_not_dropped(unpatched):
    if not _chunked_mask_rejects_block_sequence_ids(masking_utils):
        pytest.skip("this transformers does not pass block_sequence_ids to chunked masks")
    fix_transformers_chunked_mask_block_sequence_ids()
    patched = masking_utils.create_chunked_causal_mask
    config = _tiny_llama4().config
    config._attn_implementation = "sdpa"
    length = 12
    kwargs = {
        "config": config,
        "inputs_embeds": torch.zeros(1, length, config.hidden_size),
        "attention_mask": torch.ones(1, length, dtype = torch.long),
        "past_key_values": None,
        "position_ids": torch.arange(length)[None],
        "allow_is_causal_skip": False,
    }
    plain = patched(**kwargs)
    assert torch.equal(plain, unpatched(**kwargs))
    all_text = patched(**kwargs, block_sequence_ids = torch.full((1, length), -1))
    assert torch.equal(all_text, plain)
    # Media block spanning the chunk boundary at 8: 6..9 must see each other both ways.
    ids = torch.full((1, length), -1)
    ids[0, 6:10] = 0
    blocked = patched(**kwargs, block_sequence_ids = ids)[0, 0]
    assert not plain[0, 0, 6, 9] and not plain[0, 0, 9, 6]
    assert all(blocked[q, k] for q in range(6, 10) for k in range(6, 10))
    outside = torch.ones(length, length, dtype = torch.bool)
    outside[6:10, 6:10] = False
    assert torch.equal(blocked[outside], plain[0, 0][outside])


def test_function_that_accepts_the_kwarg_is_left_untouched(unpatched, monkeypatch):
    def create_chunked_causal_mask(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        block_sequence_ids = None,
    ):
        return None

    assert "block_sequence_ids" in inspect.signature(create_chunked_causal_mask).parameters
    monkeypatch.setattr(masking_utils, "create_chunked_causal_mask", create_chunked_causal_mask)
    assert not _chunked_mask_rejects_block_sequence_ids(masking_utils)
    fix_transformers_chunked_mask_block_sequence_ids()
    assert masking_utils.create_chunked_causal_mask is create_chunked_causal_mask


def test_noop_where_generate_never_passes_the_kwarg(unpatched):
    if _chunked_mask_rejects_block_sequence_ids(masking_utils):
        pytest.skip("this transformers needs the fix")
    fix_transformers_chunked_mask_block_sequence_ids()
    assert masking_utils.create_chunked_causal_mask is unpatched
