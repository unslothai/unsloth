# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained on a model loaded from a pre-quantized bitsandbytes checkpoint.

transformers 5.x attaches a `WeightConverter` running `Bnb4bitDeserialize` to
`model._weight_conversions` on such a load, and `save_pretrained` reverses it through
`revert_weight_conversion`; the op has no `reverse_op`, so the save raises
`NotImplementedError` (unslothai/unsloth#638 thread: merged_4bit_forced and plain
save_pretrained of a bnb-4bit model). Real transformers objects throughout.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from unsloth.import_fixes import (  # noqa: E402
    _BNB_PREQUANTIZED_SAVE_FLAG,
    _bnb_deserialize_ops_without_reverse,
    fix_transformers_bnb_prequantized_save,
)


def _modules():
    try:
        from transformers import core_model_loading, modeling_utils
        from transformers.integrations import bitsandbytes as bnb_integration
    except Exception:
        pytest.skip("this transformers has no core_model_loading")
    if not os.path.isfile(getattr(core_model_loading, "__file__", None) or ""):
        pytest.skip("core_model_loading is unsloth's transformers 4.x stand-in")
    if not hasattr(bnb_integration, "Bnb4bitDeserialize"):
        pytest.skip("this transformers has no Bnb4bitDeserialize")
    return core_model_loading, modeling_utils, bnb_integration


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    return LlamaForCausalLM(config)


def _bnb_converter(bnb_integration, core_model_loading):
    return core_model_loading.WeightConverter(
        source_patterns = [
            "weight.nested_absmax",
            "weight.nested_quant_map",
            "weight.quant_map",
            "weight.absmax",
            "weight.quant_state.bitsandbytes__nf4",
            "weight.quant_state.bitsandbytes__fp4",
            "weight",
        ],
        target_patterns = "weight",
        operations = [bnb_integration.Bnb4bitDeserialize(None)],
    )


def _unwrapped(fn):
    while getattr(fn, _BNB_PREQUANTIZED_SAVE_FLAG, False):
        fn = fn.__wrapped__
    return fn


def test_upstream_revert_raises_on_bnb_converter():
    core_model_loading, _, bnb_integration = _modules()
    if not _bnb_deserialize_ops_without_reverse():
        pytest.skip("this transformers implements Bnb4bitDeserialize.reverse_op")
    model = _tiny_llama()
    model._weight_conversions = [_bnb_converter(bnb_integration, core_model_loading)]
    original = _unwrapped(core_model_loading.revert_weight_conversion)
    with pytest.raises(NotImplementedError):
        original(model, model.state_dict())


def test_patched_revert_skips_bnb_converter_and_keeps_others():
    core_model_loading, modeling_utils, bnb_integration = _modules()
    if not _bnb_deserialize_ops_without_reverse():
        pytest.skip("this transformers implements Bnb4bitDeserialize.reverse_op")
    fix_transformers_bnb_prequantized_save()
    fix_transformers_bnb_prequantized_save()
    patched = core_model_loading.revert_weight_conversion
    assert getattr(patched, _BNB_PREQUANTIZED_SAVE_FLAG, False)
    assert not getattr(patched.__wrapped__, _BNB_PREQUANTIZED_SAVE_FLAG, False)
    assert modeling_utils.revert_weight_conversion is patched

    model = _tiny_llama()
    state_dict = model.state_dict()
    renaming = core_model_loading.WeightRenaming("^old_lm_head", "^lm_head")
    conversions = [renaming, _bnb_converter(bnb_integration, core_model_loading)]
    model._weight_conversions = conversions

    saved = patched(model, dict(state_dict))
    assert model._weight_conversions is conversions

    model._weight_conversions = [renaming]
    expected = patched.__wrapped__(model, dict(state_dict))
    assert set(saved) == set(expected)
    assert "old_lm_head.weight" in saved
    for key in expected:
        assert torch.equal(saved[key], expected[key])

    model._weight_conversions = [_bnb_converter(bnb_integration, core_model_loading)]
    saved = patched(model, dict(state_dict))
    assert set(saved) == set(state_dict)
    for key in state_dict:
        assert torch.equal(saved[key], state_dict[key])
