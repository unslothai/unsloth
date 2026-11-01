# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth.precision selectors over every model family transformers ships (tiny configs on the meta device, CPU
only): decoder_layers is the language decoder's block list, find_output_heads / find_embeddings / find_routers
return the expected modules (routers only for MoE families, never an expert / shared expert / dense MLP), and
hp_filter(first_n = 1, last_n = 1) protects exactly the first and last language blocks.

unsloth.precision is loaded standalone (it imports only torch), so this runs in any venv with transformers."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
import transformers

_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "precision")


def _load_precision():
    name = "_unsloth_precision_standalone"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_PKG, "__init__.py"), submodule_search_locations = [_PKG]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load_precision()

TXT = dict(
    vocab_size = 128,
    hidden_size = 64,
    intermediate_size = 128,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 16,
    pad_token_id = 0,
    bos_token_id = 1,
    eos_token_id = 2,
)
MAMBA = dict(mamba_d_state = 8, mamba_d_conv = 4, mamba_expand = 2)


def _cls(name):
    c = getattr(transformers, name, None)
    if c is None:
        pytest.skip(f"{name} not in transformers {transformers.__version__}")
    return c


def _build(model_cls, config_cls, meta = True, **kwargs):
    config = _cls(config_cls)(**kwargs)
    if meta:
        with torch.device("meta"):
            return _cls(model_cls)(config)
    torch.manual_seed(0)
    return _cls(model_cls)(config)


def _causal(arch, n = 3, **kw):
    return lambda: _build(f"{arch}ForCausalLM", f"{arch}Config", **{**TXT, "num_hidden_layers": n, **kw})


def _siglip(n = 5):
    return dict(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = n,
        num_attention_heads = 4,
        image_size = 32,
        patch_size = 8,
    )


def _text(n = 3, **kw):
    return {**TXT, "num_hidden_layers": n, **kw}


# name -> (builder, decoder list paths (first one present wins), decoder length, router fqn suffixes, heads)
# heads: None = {the module get_output_embeddings returns}; otherwise the exact expected set.
FAMILIES = {
    "llama": (_causal("Llama"), ["model.layers"], 3, (), None),
    "mistral": (_causal("Mistral"), ["model.layers"], 3, (), None),
    "qwen2": (_causal("Qwen2"), ["model.layers"], 3, (), None),
    "qwen3_1layer": (_causal("Qwen3", 1), ["model.layers"], 1, (), None),
    "qwen3_4layer": (_causal("Qwen3", 4), ["model.layers"], 4, (), None),
    "gemma2": (_causal("Gemma2"), ["model.layers"], 3, (), None),
    "gemma3_text": (
        lambda: _build("Gemma3ForCausalLM", "Gemma3TextConfig", **_text()),
        ["model.layers"],
        3,
        (),
        None,
    ),
    "gemma3_mm": (
        lambda: _build(
            "Gemma3ForConditionalGeneration",
            "Gemma3Config",
            text_config = _text(),
            vision_config = _siglip(),
            mm_tokens_per_image = 4,
        ),
        ["model.language_model.layers", "language_model.model.layers"],
        3,
        (),
        None,
    ),
    "gemma3n_text": (
        lambda: _build(
            "Gemma3nForCausalLM",
            "Gemma3nTextConfig",
            **_text(
                4,
                laurel_rank = 8,
                hidden_size_per_layer_input = 8,
                altup_num_inputs = 4,
                num_kv_shared_layers = 0,
                layer_types = ["sliding_attention", "full_attention"] * 2,
                activation_sparsity_pattern = [0.0] * 4,
            ),
        ),
        ["model.layers"],
        4,
        (),
        None,
    ),
    "phi3": (_causal("Phi3"), ["model.layers"], 3, (), None),
    "granite": (_causal("Granite"), ["model.layers"], 3, (), None),
    "granitemoe": (
        _causal("GraniteMoe", num_local_experts = 4, num_experts_per_tok = 2),
        ["model.layers"],
        3,
        (".block_sparse_moe.router",),
        None,
    ),
    "granitemoehybrid": (
        _causal(
            "GraniteMoeHybrid",
            4,
            num_local_experts = 4,
            num_experts_per_tok = 2,
            shared_intermediate_size = 64,
            layer_types = ["mamba", "attention", "mamba", "attention"],
            mamba_n_heads = 8,
            mamba_d_head = 16,
            mamba_n_groups = 1,
            mamba_d_state = 8,
            mamba_d_conv = 4,
            mamba_expand = 2,
            mamba_chunk_size = 16,
        ),
        ["model.layers"],
        4,
        (".block_sparse_moe.router",),
        None,
    ),
    "qwen2_moe": (
        _causal(
            "Qwen2Moe",
            num_experts = 8,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            shared_expert_intermediate_size = 64,
        ),
        ["model.layers"],
        3,
        (".mlp.gate",),
        None,
    ),
    "qwen3_moe": (
        _causal("Qwen3Moe", num_experts = 8, num_experts_per_tok = 2, moe_intermediate_size = 32),
        ["model.layers"],
        3,
        (".mlp.gate",),
        None,
    ),
    "qwen3_next": (
        _causal(
            "Qwen3Next",
            4,
            num_experts = 8,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            shared_expert_intermediate_size = 64,
            layer_types = ["linear_attention", "full_attention"] * 2,
            linear_num_value_heads = 4,
            linear_num_key_heads = 2,
            linear_key_head_dim = 16,
            linear_value_head_dim = 16,
            linear_conv_kernel_dim = 4,
        ),
        ["model.layers"],
        4,
        (".mlp.gate",),
        None,
    ),
    "mixtral": (
        _causal("Mixtral", num_local_experts = 4, num_experts_per_tok = 2),
        ["model.layers"],
        3,
        (".block_sparse_moe.gate", ".mlp.gate"),
        None,
    ),
    "olmoe": (
        _causal("Olmoe", num_experts = 8, num_experts_per_tok = 2),
        ["model.layers"],
        3,
        (".mlp.gate",),
        None,
    ),
    "gpt_oss": (
        _causal(
            "GptOss",
            num_local_experts = 4,
            num_experts_per_tok = 2,
            layer_types = ["sliding_attention", "full_attention", "sliding_attention"],
        ),
        ["model.layers"],
        3,
        (".mlp.router",),
        None,
    ),
    "deepseek_v3": (
        _causal(
            "DeepseekV3",
            n_routed_experts = 8,
            num_experts_per_tok = 2,
            n_shared_experts = 1,
            first_k_dense_replace = 1,
            moe_intermediate_size = 32,
            n_group = 1,
            topk_group = 1,
            kv_lora_rank = 16,
            q_lora_rank = 16,
            qk_rope_head_dim = 8,
            qk_nope_head_dim = 8,
            v_head_dim = 16,
        ),
        ["model.layers"],
        3,
        (".mlp.gate",),
        None,
    ),
    "glm4_moe": (
        _causal(
            "Glm4Moe",
            n_routed_experts = 8,
            num_experts_per_tok = 2,
            n_shared_experts = 1,
            first_k_dense_replace = 1,
            moe_intermediate_size = 32,
            n_group = 1,
            topk_group = 1,
        ),
        ["model.layers"],
        3,
        (".mlp.gate",),
        None,
    ),
    "llama4_text": (
        lambda: _build(
            "Llama4ForCausalLM",
            "Llama4TextConfig",
            **_text(
                num_local_experts = 4,
                num_experts_per_tok = 1,
                interleave_moe_layer_step = 1,
                intermediate_size_mlp = 128,
            ),
        ),
        ["model.layers"],
        3,
        (".feed_forward.router",),
        None,
    ),
    "llama4_mm": (
        lambda: _build(
            "Llama4ForConditionalGeneration",
            "Llama4Config",
            text_config = _text(
                num_local_experts = 4,
                num_experts_per_tok = 1,
                interleave_moe_layer_step = 1,
                intermediate_size_mlp = 128,
            ),
            vision_config = dict(
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 5,
                num_attention_heads = 4,
                image_size = 32,
                patch_size = 8,
                vision_output_dim = 64,
                projector_input_dim = 64,
                projector_output_dim = 64,
                pixel_shuffle_ratio = 0.5,
            ),
        ),
        ["language_model.model.layers", "model.language_model.layers"],
        3,
        (".feed_forward.router",),
        None,
    ),
    "jamba": (
        _causal(
            "Jamba",
            4,
            num_experts = 4,
            num_experts_per_tok = 2,
            expert_layer_period = 2,
            expert_layer_offset = 1,
            attn_layer_period = 2,
            attn_layer_offset = 1,
            use_mamba_kernels = False,
            **MAMBA,
        ),
        ["model.layers"],
        4,
        (".feed_forward.router",),
        None,
    ),
    "zamba": (
        _causal(
            "Zamba",
            6,
            attn_layer_period = 2,
            attn_layer_offset = 1,
            mamba_dt_rank = 8,
            n_mamba_heads = 2,
            use_mamba_kernels = False,
            **MAMBA,
        ),
        ["model.layers"],
        6,
        (),
        None,
    ),
    "zamba2": (
        _causal(
            "Zamba2",
            4,
            layers_block_type = ["mamba", "hybrid", "mamba", "hybrid"],
            mamba_headdim = 16,
            mamba_ngroups = 1,
            n_mamba_heads = 8,
            num_mem_blocks = 1,
            use_mamba_kernels = False,
            **MAMBA,
        ),
        ["model.layers"],
        4,
        (),
        None,
    ),
    "falcon_h1": (
        _causal(
            "FalconH1",
            mamba_d_ssm = 128,
            mamba_n_heads = 8,
            mamba_d_head = 16,
            mamba_n_groups = 1,
            mamba_chunk_size = 16,
            **MAMBA,
        ),
        ["model.layers"],
        3,
        (),
        None,
    ),
    "nemotron_h": (
        _causal(
            "NemotronH",
            hybrid_override_pattern = "M*-",
            mamba_num_heads = 8,
            mamba_head_dim = 16,
            n_groups = 1,
            ssm_state_size = 8,
            conv_kernel = 4,
            expand = 2,
            chunk_size = 16,
        ),
        ["model.layers", "backbone.layers"],
        3,
        (),
        None,
    ),
    "nemotron_h_moe": (
        _causal(
            "NemotronH",
            4,
            layers_block_type = ["linear_attention", "moe", "full_attention", "moe"],
            n_routed_experts = 8,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            moe_shared_expert_intermediate_size = 64,
            mamba_num_heads = 8,
            mamba_head_dim = 16,
            n_groups = 1,
            ssm_state_size = 8,
            conv_kernel = 4,
            expand = 2,
            chunk_size = 16,
        ),
        ["model.layers"],
        4,
        (".mixer.gate",),
        None,
    ),
    "mamba": (
        lambda: _build(
            "MambaForCausalLM",
            "MambaConfig",
            vocab_size = 128,
            hidden_size = 64,
            state_size = 8,
            num_hidden_layers = 3,
            use_mambapy = False,
        ),
        ["backbone.layers"],
        3,
        (),
        None,
    ),
    "llava": (
        lambda: _build(
            "LlavaForConditionalGeneration",
            "LlavaConfig",
            text_config = dict(model_type = "llama", **_text()),
            vision_config = dict(model_type = "clip_vision_model", **_siglip()),
        ),
        ["model.language_model.layers", "language_model.model.layers"],
        3,
        (),
        None,
    ),
    "qwen2_vl": (
        lambda: _build(
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLConfig",
            text_config = _text(rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]}),
            vision_config = dict(depth = 5, embed_dim = 32, hidden_size = 64, num_heads = 4, mlp_ratio = 2),
            **_text(rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]}),
        ),
        ["model.language_model.layers", "model.layers"],
        3,
        (),
        None,
    ),
    "qwen2_5_vl": (
        lambda: _build(
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5_VLConfig",
            text_config = _text(rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]}),
            vision_config = dict(
                depth = 5,
                hidden_size = 32,
                intermediate_size = 64,
                out_hidden_size = 64,
                num_heads = 4,
                fullatt_block_indexes = [4],
            ),
            **_text(rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]}),
        ),
        ["model.language_model.layers", "model.layers"],
        3,
        (),
        None,
    ),
    "mllama": (
        lambda: _build(
            "MllamaForConditionalGeneration",
            "MllamaConfig",
            text_config = dict(
                vocab_size = 128,
                hidden_size = 64,
                intermediate_size = 128,
                num_hidden_layers = 4,
                num_attention_heads = 4,
                num_key_value_heads = 2,
                cross_attention_layers = [1],
                pad_token_id = 0,
                rope_scaling = {"rope_type": "default"},
            ),
            vision_config = dict(
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 5,
                num_global_layers = 2,
                attention_heads = 4,
                image_size = 32,
                patch_size = 8,
                vision_output_dim = 64,
                intermediate_layers_indices = [1],
            ),
        ),
        ["model.language_model.layers", "language_model.model.layers"],
        4,
        (),
        None,
    ),
    "idefics3": (
        lambda: _build(
            "Idefics3ForConditionalGeneration",
            "Idefics3Config",
            text_config = dict(model_type = "llama", **_text()),
            vision_config = dict(
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 5,
                num_attention_heads = 4,
                image_size = 32,
                patch_size = 8,
            ),
            scale_factor = 2,
        ),
        ["model.text_model.layers"],
        3,
        (),
        None,
    ),
    "smolvlm": (
        lambda: _build(
            "SmolVLMForConditionalGeneration",
            "SmolVLMConfig",
            text_config = dict(model_type = "llama", **_text()),
            vision_config = dict(
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 5,
                num_attention_heads = 4,
                image_size = 32,
                patch_size = 8,
            ),
            scale_factor = 2,
        ),
        ["model.text_model.layers"],
        3,
        (),
        None,
    ),
    "paligemma": (
        lambda: _build(
            "PaliGemmaForConditionalGeneration",
            "PaliGemmaConfig",
            text_config = dict(model_type = "gemma", **_text()),
            vision_config = dict(model_type = "siglip_vision_model", **_siglip()),
            projection_dim = 64,
            hidden_size = 64,
        ),
        ["model.language_model.layers", "language_model.model.layers"],
        3,
        (),
        None,
    ),
    "whisper": (
        lambda: _build(
            "WhisperForConditionalGeneration",
            "WhisperConfig",
            vocab_size = 128,
            d_model = 32,
            encoder_layers = 5,
            decoder_layers = 3,
            encoder_attention_heads = 4,
            decoder_attention_heads = 4,
            encoder_ffn_dim = 64,
            decoder_ffn_dim = 64,
            pad_token_id = 0,
            bos_token_id = 1,
            eos_token_id = 2,
            decoder_start_token_id = 1,
        ),
        ["model.decoder.layers"],
        3,
        (),
        None,
    ),
    "t5": (
        lambda: _build(
            "T5ForConditionalGeneration",
            "T5Config",
            vocab_size = 128,
            d_model = 32,
            d_kv = 8,
            d_ff = 64,
            num_layers = 5,
            num_decoder_layers = 3,
            num_heads = 4,
            decoder_start_token_id = 0,
        ),
        ["decoder.block"],
        3,
        (),
        None,
    ),
    "bert_encoder": (
        lambda: _build(
            "BertModel",
            "BertConfig",
            vocab_size = 128,
            hidden_size = 32,
            num_hidden_layers = 3,
            num_attention_heads = 4,
            intermediate_size = 64,
        ),
        ["encoder.layer"],
        3,
        (),
        set(),
    ),
    "bert_seq_cls": (
        lambda: _build(
            "BertForSequenceClassification",
            "BertConfig",
            vocab_size = 128,
            hidden_size = 32,
            num_hidden_layers = 3,
            num_attention_heads = 4,
            intermediate_size = 64,
            num_labels = 3,
        ),
        ["bert.encoder.layer"],
        3,
        (),
        {"classifier"},
    ),
    "llama_seq_cls": (
        lambda: _build("LlamaForSequenceClassification", "LlamaConfig", **_text(), num_labels = 3),
        ["model.layers"],
        3,
        (),
        {"score"},
    ),
    "qwen3_tied_real_weights": (
        lambda: _build(
            "Qwen3ForCausalLM", "Qwen3Config", meta = False, **_text(tie_word_embeddings = True)
        ),
        ["model.layers"],
        3,
        (),
        {"lm_head"},
    ),
}


def _lists(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_selectors_on_family(family):
    build, paths, n, router_suffixes, heads = FAMILIES[family]
    try:
        model = build()
    except pytest.skip.Exception:
        raise
    except Exception as e:  # a config this transformers cannot build is not a selector result
        pytest.skip(f"{family} does not build in transformers {transformers.__version__}: {e!r}")
    names = dict(model.named_modules())
    lists = _lists(model)
    path = next((p for p in paths if p in lists), None)
    assert path is not None, (family, paths, sorted(lists)[:20])
    assert len(lists[path]) == n

    # decoder_layers: the language decoder's block list.
    assert P.decoder_layers(model) == (path, n), family

    # hp_filter(first_n = 1, last_n = 1): exactly the first and last language blocks.
    is_hp = P.hp_filter(model, P.HPLayerPolicy(output_head = False, embeddings = False, routers = False, first_n = 1, last_n = 1))
    assert set(is_hp.keep) == {f"{path}.0", f"{path}.{n - 1}"}, (family, is_hp.keep)
    for i in range(n):
        for sub, _ in lists[path][i].named_modules():
            fqn = f"{path}.{i}.{sub}" if sub else f"{path}.{i}"
            assert is_hp(None, fqn) == (i in (0, n - 1)), (family, fqn)
    for fqn in names:
        if fqn and not (fqn == path or fqn.startswith(path + ".")):
            assert not is_hp(None, fqn), (family, fqn)

    # Output heads.
    out = model.get_output_embeddings()
    expected_heads = heads if heads is not None else {k for k, v in names.items() if v is out}
    assert set(P.find_output_heads(model)) == expected_heads, family

    # Embeddings: the input embedding is found, and nothing found is a Linear.
    emb = P.find_embeddings(model)
    inp = model.get_input_embeddings()
    assert any(names[k] is inp for k in emb), family
    assert not any(isinstance(names[k], torch.nn.Linear) and names[k] is not inp for k in emb), family

    # Routers: exactly the router modules, never an expert, shared expert or dense MLP.
    expected_routers = {
        k for k in names if router_suffixes and k.endswith(router_suffixes) and "expert" not in k
    }
    assert set(P.find_routers(model)) == expected_routers, family
