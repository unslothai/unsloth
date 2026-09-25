# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""A text-only load of a VLM checkpoint (MiniMax-M3) must still apply the VLM's checkpoint conversions, else MoE / indexer weights load randomly initialised."""

import pytest


@pytest.fixture(scope = "module")
def tiny_minimax(tmp_path_factory):
    try:
        import unsloth  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"unsloth does not import here: {e}")
    import torch
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "minimax_m3_vl" not in CONFIG_MAPPING:
        pytest.skip("this transformers has no MiniMax-M3")
    from transformers import AutoModelForImageTextToText

    config = AutoConfig.for_model("minimax_m3_vl")
    text = config.text_config
    layers = 3
    for key, value in dict(
        hidden_size = 64,
        intermediate_size = 32,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        rotary_dim = 8,
        dense_intermediate_size = 128,
        shared_intermediate_size = 32,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        vocab_size = 256,
        index_n_heads = 2,
        index_head_dim = 16,
        index_block_size = 16,
        index_topk_blocks = 2,
    ).items():
        setattr(text, key, value)
    text.mlp_layer_types = ["dense"] + ["sparse"] * (layers - 1)
    text.layer_types = ["full_attention"] + ["minimax_m3_sparse"] * (layers - 1)
    vision = config.vision_config
    for key, value in dict(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 2
    ).items():
        setattr(vision, key, value)
    config.projector_hidden_size = 64
    config.image_token_index = config.video_token_index = 200
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny_minimax_m3")
    model.save_pretrained(path)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return path, config, reference


def test_saved_checkpoint_has_the_original_layout(tiny_minimax):
    path, _, _ = tiny_minimax
    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    keys = set()
    for file in files:
        with safe_open(str(file), "pt") as handle:
            keys |= set(handle.keys())
    assert any(".block_sparse_moe.experts.0.w1.weight" in k for k in keys), sorted(keys)[:5]


def test_text_only_load_keeps_every_decoder_weight(tiny_minimax):
    import torch
    from transformers import AutoModelForCausalLM
    from unsloth.models._utils import _apply_text_only_key_mapping, _get_text_only_config

    path, config, reference = tiny_minimax
    text_config = _get_text_only_config(config, str(path))
    kwargs = {}
    _apply_text_only_key_mapping(kwargs, config, text_config)
    assert kwargs.get("key_mapping")
    decoder = AutoModelForCausalLM.from_pretrained(
        path, config = text_config, dtype = torch.bfloat16, **kwargs
    )
    assert type(decoder).__name__ == "MiniMaxM3VLForCausalLM"
    loaded = decoder.state_dict()
    compared = 0
    for key, value in loaded.items():
        ref_key = key.replace("model.", "model.language_model.", 1)
        if ref_key not in reference:
            ref_key = key
        assert ref_key in reference, key
        assert torch.equal(value, reference[ref_key]), key
        compared += 1
    assert any("experts.gate_up_proj" in k for k in loaded)
    assert any("indexer" in k for k in loaded)
    assert compared == len(loaded)


def test_only_prefix_moves_are_left_to_the_key_mapping():
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightConverter
    except Exception:
        pytest.skip("transformers without checkpoint conversion mappings")
    from unsloth.models._utils import _parent_conversions_for_text_only

    if get_checkpoint_conversion_mapping("minimax_m3_vl") is None:
        pytest.skip("this transformers has no MiniMax-M3")
    carried = _parent_conversions_for_text_only("minimax_m3_vl")
    sources = [
        str(s)
        for t in carried
        for s in (
            t.source_patterns
            if isinstance(t.source_patterns, (list, tuple))
            else [t.source_patterns]
        )
    ]
    assert not any(s.startswith("^") for s in sources)
    assert any(isinstance(t, WeightConverter) for t in carried)
    assert _parent_conversions_for_text_only("a_model_type_with_no_mapping") == []


def test_other_loads_are_untouched(tiny_minimax):
    import torch
    from transformers import AutoModelForImageTextToText

    path, _, reference = tiny_minimax
    model = AutoModelForImageTextToText.from_pretrained(path, dtype = torch.bfloat16)
    state = model.state_dict()
    assert all(torch.equal(state[k], reference[k]) for k in reference)


def test_the_parent_conversions_are_per_thread(tiny_minimax):
    """Carried conversions are thread-local to the requesting load; the lookup is never swapped."""
    import threading

    import transformers.conversion_mapping as conversion_mapping
    from unsloth.models import _utils

    path, config, _ = tiny_minimax
    text_config = _utils._get_text_only_config(config, str(path))
    _utils._apply_text_only_key_mapping({}, config, text_config)
    text_type = text_config.model_type
    lookup = conversion_mapping.get_checkpoint_conversion_mapping
    assert getattr(lookup, "_unsloth_text_only_carry", False)
    assert lookup(text_type) is None

    seen = {}
    inside, done = threading.Event(), threading.Event()

    def other_load():
        inside.wait(5)
        seen["other"] = conversion_mapping.get_checkpoint_conversion_mapping(text_type)
        done.set()

    thread = threading.Thread(target = other_load)
    thread.start()
    _utils._TEXT_ONLY_LOOKUP_OVERRIDES.value = {text_type: ["carried"]}
    try:
        seen["this"] = conversion_mapping.get_checkpoint_conversion_mapping(text_type)
        inside.set()
        done.wait(5)
    finally:
        _utils._TEXT_ONLY_LOOKUP_OVERRIDES.value = None
    thread.join(5)
    assert seen == {"this": ["carried"], "other": None}
    assert conversion_mapping.get_checkpoint_conversion_mapping is lookup
