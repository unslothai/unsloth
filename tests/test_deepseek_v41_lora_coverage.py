# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DeepSeek-V4.1 (community port) LoRA / QLoRA load rules: grouped o_a_proj stays 16-bit, padding-free is refused."""

import pytest

torch = pytest.importorskip("torch")


def test_4bit_keeps_the_grouped_output_projection_in_16bit():
    from unsloth.models.vision import _architecture_skip_modules

    # bitsandbytes swaps any nn.Linear subclass for a plain Linear4bit, dropping the per-group forward.
    for model_types in (
        ["deepseek_v41"],
        ["deepseek_v41", "deepseek_v41_text"],
        ["deepseek_v41_text"],
    ):
        assert "o_a_proj" in _architecture_skip_modules(model_types)
    assert "o_a_proj" not in _architecture_skip_modules(["deepseek_v4"])
    assert _architecture_skip_modules(["llama"]) == []


def test_padding_free_is_refused_for_deepseek_v41():
    from unsloth.utils.packing import packing_boundary_blocker

    # Engram n-grams and the CSA2 compressor run over the whole row, ignoring packed boundaries.
    for model_type in ("deepseek_v41", "deepseek_v41_text"):
        assert packing_boundary_blocker(None, (model_type,)) == model_type

    class DeepseekV41NgramHashState(torch.nn.Module):
        pass

    holder = torch.nn.Module()
    holder.hash_state = DeepseekV41NgramHashState()
    assert packing_boundary_blocker(holder) == "DeepseekV41NgramHashState"
    assert packing_boundary_blocker(None, ("llama",)) is None


def test_vision_config_without_image_text_class_resolves_to_causal_lm():
    import importlib.util

    if importlib.util.find_spec("transformers.models.deepseek_v41") is None:
        pytest.skip("needs the DeepSeek-V4.1 transformers port")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config

    from unsloth.models._utils import resolve_model_class

    # The composite config carries a vision_config (so the loader calls it a VLM), but the port maps only a
    # causal LM; loader.py now falls back to it instead of raising "Unrecognized configuration class".
    config = DeepseekV41Config(text_config = {"num_hidden_layers": 4, "num_nextn_predict_layers": 0})
    assert resolve_model_class(AutoModelForImageTextToText, config) is None
    assert resolve_model_class(AutoModelForCausalLM, config).__name__ == "DeepseekV41ForCausalLM"
    import re

    import unsloth.models.loader as loader_module

    with open(loader_module.__file__) as handle:
        source = handle.read()
    assert re.search(r"auto_model = AutoModelForCausalLM\s+is_vlm = False", source)


def test_deepseek_v41_blocks_are_not_region_compiled():
    from unsloth.models import _block_compile
    assert "DeepseekV41DecoderLayer" in _block_compile._SKIP


def test_no_placement_keeps_only_the_engram_table_off_the_device_map():
    import importlib.util

    if importlib.util.find_spec("transformers.models.deepseek_v41") is None:
        pytest.skip("needs the DeepSeek-V4.1 transformers port")
    from transformers.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
    from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM

    from unsloth.models.loader_utils import exclude_no_placement_params, no_placement_tensor_names

    config = DeepseekV41TextConfig(
        vocab_size = 256,
        hidden_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        head_dim = 64,
        qk_rope_head_dim = 16,
        q_lora_rank = 32,
        o_groups = 2,
        o_lora_rank = 16,
        moe_intermediate_size = 32,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        compress_ratios = [0, 0, 2, 1],
        kv_source_layer_ids = [2, 3],
        index_source_layer_ids = [2, 3],
        candidate_source_layer_id = 3,
        index_n_heads = 2,
        index_head_dim = 32,
        hc_mult = 2,
        engram_layer_ids = [1],
        engram_num_embeddings = [64],
        engram_head_dim = 32,
        engram_vocab_size = 3,
        engram_n_heads = 2,
        engram_compressed_vocab_size = 8,
        num_nextn_predict_layers = 0,
    )
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
    # `_no_placement_params = ["weight", "weight_scale_inv"]` is relative to the no-split engram table; a suffix
    # match kept every weight of the model on CPU.
    assert no_placement_tensor_names(model) == {
        "model.engram_tables.1.weight",
        "model.engram_tables.1.weight_scale_inv",
    }
    device_map = exclude_no_placement_params({"": 0}, DeepseekV41ForCausalLM, config)
    assert "model.engram_tables" not in device_map and device_map.get("lm_head") == 0
    assert device_map.get("model.layers") == 0 and "model.engram_tables.1" not in device_map


def test_deepseek_v41_loads_with_model_compile_disabled():
    import re

    import unsloth.models.loader as loader_module

    with open(loader_module.__file__) as handle:
        source = handle.read()
    branch = re.search(
        r'elif "deepseek_v41," in model_types_all or "deepseek_v41_text," in model_types_all:(.*?)elif "csm"',
        source,
        re.S,
    )
    assert branch and 'os.environ["UNSLOTH_COMPILE_DISABLE"] = "partial"' in branch.group(1)
    # The trailing comma keeps deepseek_v4 (and any other prefix) out of the branch.
    assert "deepseek_v41," not in "deepseek_v4,"
