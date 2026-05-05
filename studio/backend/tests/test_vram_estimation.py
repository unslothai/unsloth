# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from utils.hardware.vram_estimation import (
    ModelArchConfig,
    TrainingVramConfig,
    extract_arch_config,
    compute_model_weights_bytes,
    compute_total_params,
    compute_lora_params,
    compute_lora_adapter_bytes,
    compute_optimizer_bytes,
    compute_gradient_bytes,
    compute_activation_bytes,
    estimate_training_vram,
    DEFAULT_TARGET_MODULES,
)


def _gb(b: int) -> float:
    return b / (1024**3)


LLAMA_8B = ModelArchConfig(
    hidden_size = 4096,
    num_hidden_layers = 32,
    num_attention_heads = 32,
    num_key_value_heads = 8,
    intermediate_size = 14336,
    vocab_size = 128256,
    tie_word_embeddings = False,
)

QWEN_05B = ModelArchConfig(
    hidden_size = 896,
    num_hidden_layers = 24,
    num_attention_heads = 14,
    num_key_value_heads = 2,
    intermediate_size = 4864,
    vocab_size = 151936,
    tie_word_embeddings = True,
)

MOE_CONFIG = ModelArchConfig(
    hidden_size = 4096,
    num_hidden_layers = 32,
    num_attention_heads = 32,
    num_key_value_heads = 8,
    intermediate_size = 14336,
    vocab_size = 32000,
    tie_word_embeddings = False,
    num_experts = 8,
)

DEEPSEEK_V3 = ModelArchConfig(
    hidden_size = 7168,
    num_hidden_layers = 61,
    num_attention_heads = 128,
    num_key_value_heads = 128,
    intermediate_size = 18432,
    vocab_size = 129280,
    tie_word_embeddings = False,
    num_experts = 256,
    moe_intermediate_size = 2048,
    n_shared_experts = 1,
    num_dense_layers = 3,
    q_lora_rank = 1536,
    kv_lora_rank = 512,
    qk_nope_head_dim = 128,
    qk_rope_head_dim = 64,
    v_head_dim = 128,
)

QWEN3_MOE_30B = ModelArchConfig(
    hidden_size = 2048,
    num_hidden_layers = 48,
    num_attention_heads = 32,
    num_key_value_heads = 4,
    intermediate_size = 8192,
    vocab_size = 151936,
    tie_word_embeddings = True,
    num_experts = 128,
    moe_intermediate_size = 768,
    n_shared_experts = 0,
    num_dense_layers = 0,
)

GLM4_MOE = ModelArchConfig(
    hidden_size = 4096,
    num_hidden_layers = 46,
    num_attention_heads = 96,
    num_key_value_heads = 8,
    intermediate_size = 10944,
    vocab_size = 151552,
    tie_word_embeddings = False,
    num_experts = 128,
    moe_intermediate_size = 1408,
    n_shared_experts = 1,
    num_dense_layers = 1,
)

GPT_OSS = ModelArchConfig(
    hidden_size = 6144,
    num_hidden_layers = 64,
    num_attention_heads = 64,
    num_key_value_heads = 8,
    intermediate_size = 2880,
    vocab_size = 200064,
    tie_word_embeddings = False,
    num_experts = 128,
    moe_intermediate_size = None,
    n_shared_experts = 0,
    num_dense_layers = 0,
)

STRUCTURED_MIXED = ModelArchConfig(
    hidden_size = 256,
    num_hidden_layers = 6,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    intermediate_size = 512,
    vocab_size = 1024,
    tie_word_embeddings = True,
    head_dim = 80,
    global_head_dim = 96,
    num_global_key_value_heads = 1,
    attention_k_eq_v = True,
    layer_types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
)

STRUCTURED_SHARED = ModelArchConfig(
    hidden_size = 192,
    num_hidden_layers = 4,
    num_attention_heads = 6,
    num_key_value_heads = 2,
    intermediate_size = 384,
    vocab_size = 512,
    tie_word_embeddings = True,
    head_dim = 32,
    num_kv_shared_layers = 2,
    use_double_wide_mlp = True,
    vocab_size_per_layer_input = 128,
    hidden_size_per_layer_input = 48,
    quant_4bit_factor = 3.6,
)

QUANT_SKIP_STRUCTURED = replace(
    STRUCTURED_SHARED,
    quantization_skip_modules = [
        "model.layers.0.self_attn.q_proj",
        "language_model.model.layers.1.mlp",
        "layers.2",
        "vision_tower",
        "embed_tokens",
    ],
)


class TestExtractArchConfig(unittest.TestCase):
    def test_basic_config(self):
        hf_config = SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 128256,
            tie_word_embeddings = False,
        )
        arch = extract_arch_config(hf_config)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.hidden_size, 4096)
        self.assertEqual(arch.num_hidden_layers, 32)
        self.assertEqual(arch.num_key_value_heads, 8)
        self.assertIsNone(arch.num_experts)

    def test_vlm_text_config(self):
        text_cfg = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 24,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 8192,
            vocab_size = 32000,
            tie_word_embeddings = True,
        )
        hf_config = SimpleNamespace(text_config = text_cfg)
        arch = extract_arch_config(hf_config)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.hidden_size, 2048)

    def test_moe_detection(self):
        hf_config = SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_local_experts = 8,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_experts, 8)

    def test_missing_fields_returns_none(self):
        hf_config = SimpleNamespace(hidden_size = 4096)
        arch = extract_arch_config(hf_config)
        self.assertIsNone(arch)

    def test_intermediate_size_list(self):
        hf_config = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 24,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = [8192, 8192],
            vocab_size = 32000,
            tie_word_embeddings = True,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.intermediate_size, 8192)

    def test_structural_and_quantization_fields_are_config_derived(self):
        hf_config = SimpleNamespace(
            hidden_size = 256,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            intermediate_size = 512,
            vocab_size = 1024,
            tie_word_embeddings = True,
            head_dim = 80,
            global_head_dim = 96,
            num_global_key_value_heads = 1,
            attention_k_eq_v = True,
            layer_types = ["sliding_attention", "full_attention"],
            num_kv_shared_layers = 1,
            use_double_wide_mlp = True,
            vocab_size_per_layer_input = 128,
            hidden_size_per_layer_input = 48,
            quantization_config = {
                "bnb_4bit_use_double_quant": True,
                "llm_int8_skip_modules": ["model.layers.0.self_attn"],
            },
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.head_dim, 80)
        self.assertEqual(arch.global_head_dim, 96)
        self.assertEqual(arch.num_global_key_value_heads, 1)
        self.assertTrue(arch.attention_k_eq_v)
        self.assertEqual(arch.layer_types, ["sliding_attention", "full_attention"])
        self.assertEqual(arch.num_kv_shared_layers, 1)
        self.assertTrue(arch.use_double_wide_mlp)
        self.assertEqual(arch.vocab_size_per_layer_input, 128)
        self.assertEqual(arch.hidden_size_per_layer_input, 48)
        self.assertEqual(arch.quantization_skip_modules, ["model.layers.0.self_attn"])
        self.assertEqual(arch.quant_4bit_factor, 3.6)


class TestModelWeightsBytes(unittest.TestCase):
    def test_llama_8b_fp16(self):
        weight_bytes = compute_model_weights_bytes(LLAMA_8B, "full", False)
        weight_gb = _gb(weight_bytes)
        self.assertGreater(weight_gb, 14.0)
        self.assertLess(weight_gb, 18.0)

    def test_llama_8b_qlora_4bit(self):
        weight_bytes = compute_model_weights_bytes(LLAMA_8B, "qlora", True)
        weight_gb = _gb(weight_bytes)
        self.assertGreater(weight_gb, 4.0)
        self.assertLess(weight_gb, 7.0)

    def test_4bit_smaller_than_fp16(self):
        fp16 = compute_model_weights_bytes(LLAMA_8B, "full", False)
        q4 = compute_model_weights_bytes(LLAMA_8B, "qlora", True)
        self.assertLess(q4, fp16)
        ratio = fp16 / q4
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 4.0)

    def test_moe_larger_than_dense(self):
        dense = compute_model_weights_bytes(LLAMA_8B, "full", False)
        moe = compute_model_weights_bytes(MOE_CONFIG, "full", False)
        self.assertGreater(moe, dense * 3)


class TestLoraParams(unittest.TestCase):
    def test_llama_8b_default_modules_rank16(self):
        lora_p = compute_lora_params(LLAMA_8B, 16, DEFAULT_TARGET_MODULES)
        total_p = compute_total_params(LLAMA_8B)
        ratio = lora_p / total_p
        self.assertGreater(ratio, 0.005)
        self.assertLess(ratio, 0.05)

    def test_higher_rank_more_params(self):
        r16 = compute_lora_params(LLAMA_8B, 16, DEFAULT_TARGET_MODULES)
        r64 = compute_lora_params(LLAMA_8B, 64, DEFAULT_TARGET_MODULES)
        self.assertAlmostEqual(r64 / r16, 4.0, places = 1)

    def test_fewer_modules_fewer_params(self):
        all_mods = compute_lora_params(LLAMA_8B, 16, DEFAULT_TARGET_MODULES)
        qv_only = compute_lora_params(LLAMA_8B, 16, ["q_proj", "v_proj"])
        self.assertLess(qv_only, all_mods)

    def test_moe_mlp_modules_scale_with_experts(self):
        dense_lora = compute_lora_params(
            LLAMA_8B, 16, ["gate_proj", "up_proj", "down_proj"]
        )
        moe_lora = compute_lora_params(
            MOE_CONFIG, 16, ["gate_proj", "up_proj", "down_proj"]
        )
        ratio = moe_lora / dense_lora
        self.assertAlmostEqual(ratio, 8.0, delta = 0.5)

    def test_structured_moe_mlp_modules_scale_with_experts(self):
        structured_moe = replace(QWEN3_MOE_30B, head_dim = 128)
        dense_like = replace(
            structured_moe,
            num_experts = None,
            moe_intermediate_size = None,
        )
        target_modules = ["gate_proj", "up_proj", "down_proj"]
        dense_lora = compute_lora_params(dense_like, 16, target_modules)
        moe_lora = compute_lora_params(structured_moe, 16, target_modules)
        self.assertGreater(moe_lora, dense_lora * 20)

    def test_attention_modules_same_for_moe(self):
        dense_attn = compute_lora_params(
            LLAMA_8B, 16, ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        moe_attn = compute_lora_params(
            MOE_CONFIG, 16, ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.assertEqual(dense_attn, moe_attn)

    def test_all_linear_uses_default_text_modules(self):
        text_only = compute_lora_params(STRUCTURED_MIXED, 16, DEFAULT_TARGET_MODULES)
        all_linear = compute_lora_params(STRUCTURED_MIXED, 16, ["all-linear"])
        self.assertEqual(all_linear, text_only)

    def test_structural_layer_shapes_are_config_driven(self):
        unstructured_arch = replace(
            STRUCTURED_MIXED,
            head_dim = None,
            global_head_dim = None,
            num_global_key_value_heads = None,
            attention_k_eq_v = False,
            layer_types = None,
        )
        self.assertNotEqual(
            compute_lora_params(unstructured_arch, 16, ["all-linear"]),
            compute_lora_params(STRUCTURED_MIXED, 16, ["all-linear"]),
        )
        self.assertNotEqual(
            compute_model_weights_bytes(unstructured_arch, "qlora", True),
            compute_model_weights_bytes(STRUCTURED_MIXED, "qlora", True),
        )

    def test_shared_kv_and_per_layer_inputs_change_weight_count(self):
        unstructured_arch = replace(
            STRUCTURED_SHARED,
            head_dim = None,
            num_kv_shared_layers = 0,
            use_double_wide_mlp = False,
        )
        self.assertNotEqual(
            compute_model_weights_bytes(unstructured_arch, "qlora", True),
            compute_model_weights_bytes(STRUCTURED_SHARED, "qlora", True),
        )


class TestOptimizerBytes(unittest.TestCase):
    def test_adamw_8bit(self):
        self.assertEqual(compute_optimizer_bytes(1_000_000, "adamw_8bit"), 4_000_000)

    def test_adamw_torch(self):
        self.assertEqual(compute_optimizer_bytes(1_000_000, "adamw_torch"), 6_000_000)

    def test_sgd(self):
        self.assertEqual(compute_optimizer_bytes(1_000_000, "sgd"), 4_000_000)

    def test_unknown_defaults_to_4(self):
        self.assertEqual(compute_optimizer_bytes(1_000_000, "some_new_opt"), 4_000_000)


class TestGradientBytes(unittest.TestCase):
    def test_fp16_gradients(self):
        self.assertEqual(compute_gradient_bytes(1_000_000), 2_000_000)


class TestActivationBytes(unittest.TestCase):
    def test_no_gc_scales_with_layers(self):
        act_none = compute_activation_bytes(LLAMA_8B, 2, 2048, "none")
        act_gc = compute_activation_bytes(LLAMA_8B, 2, 2048, "true")
        self.assertGreater(act_none, act_gc * 10)

    def test_unsloth_gc_smaller_than_standard(self):
        act_true = compute_activation_bytes(LLAMA_8B, 2, 2048, "true")
        act_unsloth = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth")
        self.assertLess(act_unsloth, act_true)

    def test_lora_activations_smaller_than_full_ft(self):
        full_ft = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth", is_lora = False)
        lora = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth", is_lora = True)
        self.assertLess(lora, full_ft)

    def test_scales_with_batch_size(self):
        act_bsz2 = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth")
        act_bsz4 = compute_activation_bytes(LLAMA_8B, 4, 2048, "unsloth")
        self.assertAlmostEqual(act_bsz4 / act_bsz2, 2.0, delta = 0.1)

    def test_scales_with_seq_len(self):
        act_2k = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth")
        act_4k = compute_activation_bytes(LLAMA_8B, 2, 4096, "unsloth")
        self.assertAlmostEqual(act_4k / act_2k, 2.0, delta = 0.1)

    def test_flash_attention_uses_linear_path(self):
        flash = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
            attention_implementation = "flash_attention_2",
        )
        default = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
        )
        self.assertEqual(flash, default)

    def test_sdpa_attention_uses_linear_path(self):
        flash = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
            attention_implementation = "flash_attention_2",
        )
        sdpa = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
            attention_implementation = "sdpa",
        )
        self.assertEqual(sdpa, flash)

    def test_non_flash_attention_uses_quadratic_path(self):
        seq_len = 4096
        expected_quadratic = (
            1 * STRUCTURED_MIXED.num_attention_heads * seq_len * seq_len * 2 * 12.0
        )
        for attention_implementation in ("eager", "unknown_impl", None):
            with self.subTest(attention_implementation = attention_implementation):
                non_flash = compute_activation_bytes(
                    STRUCTURED_MIXED,
                    1,
                    seq_len,
                    "unsloth",
                    is_lora = True,
                    attention_implementation = attention_implementation,
                )
                self.assertEqual(non_flash, int(expected_quadratic))

    def test_non_flash_attention_without_gc_scales_quadratic_path_by_layers(self):
        seq_len = 4096
        one_layer = (
            1 * STRUCTURED_MIXED.num_attention_heads * seq_len * seq_len * 2 * 12.0
        )
        non_flash = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            seq_len,
            "none",
            is_lora = True,
            attention_implementation = "eager",
        )
        self.assertEqual(non_flash, int(one_layer * STRUCTURED_MIXED.num_hidden_layers))
        self.assertGreater(non_flash, int(one_layer))


class TestQuantizationSkips(unittest.TestCase):
    def test_skipped_language_layers_stay_fp16(self):
        no_skips = replace(QUANT_SKIP_STRUCTURED, quantization_skip_modules = [])
        skipped = compute_model_weights_bytes(QUANT_SKIP_STRUCTURED, "qlora", True)
        quantized = compute_model_weights_bytes(no_skips, "qlora", True)
        self.assertGreater(skipped, quantized)

    def test_non_language_skips_do_not_double_count_text_weights(self):
        arch = replace(
            QUANT_SKIP_STRUCTURED,
            quantization_skip_modules = ["vision_tower", "embed_tokens"],
        )
        no_skips = replace(QUANT_SKIP_STRUCTURED, quantization_skip_modules = [])
        self.assertEqual(
            compute_model_weights_bytes(arch, "qlora", True),
            compute_model_weights_bytes(no_skips, "qlora", True),
        )

    def test_double_quant_factor_reduces_quantized_weight_storage(self):
        default_quant = replace(STRUCTURED_MIXED, quant_4bit_factor = 16 / 5)
        double_quant = replace(STRUCTURED_MIXED, quant_4bit_factor = 3.6)
        self.assertLess(
            compute_model_weights_bytes(double_quant, "qlora", True),
            compute_model_weights_bytes(default_quant, "qlora", True),
        )

    def test_prefixed_parent_and_child_skips_do_not_double_count(self):
        parent_only = replace(
            QUANT_SKIP_STRUCTURED,
            quantization_skip_modules = ["language_model.model.layers.1.mlp"],
        )
        parent_and_child = replace(
            QUANT_SKIP_STRUCTURED,
            quantization_skip_modules = [
                "language_model.model.layers.1.mlp",
                "language_model.model.layers.1.mlp.gate_proj",
                "model.layers.1.mlp.up_proj",
            ],
        )
        self.assertEqual(
            compute_model_weights_bytes(parent_and_child, "qlora", True),
            compute_model_weights_bytes(parent_only, "qlora", True),
        )

    def test_vlm_prefix_skip_module_does_not_match_text_alias(self):
        # vision_tower-prefixed skips must not shadow text aliases sharing the
        # same suffix.
        baseline = replace(QUANT_SKIP_STRUCTURED, quantization_skip_modules = [])
        vlm_skip = replace(
            QUANT_SKIP_STRUCTURED,
            quantization_skip_modules = [
                "vision_tower.model.layers.0.self_attn.q_proj",
                "vision_tower.model.layers.1.mlp",
            ],
        )
        self.assertEqual(
            compute_model_weights_bytes(vlm_skip, "qlora", True),
            compute_model_weights_bytes(baseline, "qlora", True),
        )

    def test_mla_skip_module_uses_authoritative_attn_total(self):
        from utils.hardware.vram_estimation import (
            _build_text_module_elements,
            _compute_attn_elements,
        )

        mla = ModelArchConfig(
            hidden_size = 2048,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 16,
            intermediate_size = 8192,
            vocab_size = 32000,
            tie_word_embeddings = False,
            q_lora_rank = 512,
            kv_lora_rank = 128,
            qk_nope_head_dim = 64,
            qk_rope_head_dim = 32,
            v_head_dim = 64,
        )
        elements, _ = _build_text_module_elements(mla)
        self.assertEqual(
            elements["text.layers.0.self_attn"],
            _compute_attn_elements(mla),
        )


class TestEstimateTrainingVram(unittest.TestCase):
    def test_llama_8b_qlora_reasonable_total(self):
        config = TrainingVramConfig(
            training_method = "qlora",
            batch_size = 2,
            max_seq_length = 2048,
            lora_rank = 16,
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = True,
        )
        breakdown = estimate_training_vram(LLAMA_8B, config)
        total_gb = _gb(breakdown.total)
        self.assertGreater(total_gb, 5.0)
        self.assertLess(total_gb, 12.0)

    def test_llama_8b_full_ft_reasonable_total(self):
        config = TrainingVramConfig(
            training_method = "full",
            batch_size = 2,
            max_seq_length = 2048,
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = False,
        )
        breakdown = estimate_training_vram(LLAMA_8B, config)
        total_gb = _gb(breakdown.total)
        self.assertGreater(total_gb, 50.0)
        self.assertLess(total_gb, 75.0)

    def test_qlora_much_less_than_full_ft(self):
        qlora_config = TrainingVramConfig(
            training_method = "qlora",
            load_in_4bit = True,
            batch_size = 2,
            max_seq_length = 2048,
        )
        full_config = TrainingVramConfig(
            training_method = "full",
            load_in_4bit = False,
            batch_size = 2,
            max_seq_length = 2048,
        )
        qlora = estimate_training_vram(LLAMA_8B, qlora_config)
        full = estimate_training_vram(LLAMA_8B, full_config)
        self.assertLess(qlora.total, full.total / 3)

    def test_qwen_05b_qlora_fits_in_4gb(self):
        config = TrainingVramConfig(
            training_method = "qlora",
            batch_size = 2,
            max_seq_length = 2048,
            lora_rank = 16,
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = True,
        )
        breakdown = estimate_training_vram(QWEN_05B, config)
        total_gb = _gb(breakdown.total)
        self.assertLess(total_gb, 5.0)

    def test_breakdown_components_positive(self):
        config = TrainingVramConfig(training_method = "qlora", load_in_4bit = True)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        self.assertGreater(breakdown.model_weights, 0)
        self.assertGreater(breakdown.lora_adapters, 0)
        self.assertGreater(breakdown.optimizer_states, 0)
        self.assertGreater(breakdown.gradients, 0)
        self.assertGreater(breakdown.activations, 0)
        self.assertGreater(breakdown.cuda_overhead, 0)

    def test_full_ft_no_lora_adapters(self):
        config = TrainingVramConfig(training_method = "full", load_in_4bit = False)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        self.assertEqual(breakdown.lora_adapters, 0)

    def test_to_gb_dict_keys(self):
        config = TrainingVramConfig(training_method = "qlora", load_in_4bit = True)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        gb_dict = breakdown.to_gb_dict()
        expected_keys = {
            "model_weights_gb",
            "lora_adapters_gb",
            "optimizer_states_gb",
            "gradients_gb",
            "activations_gb",
            "cuda_overhead_gb",
            "total_gb",
        }
        self.assertEqual(set(gb_dict.keys()), expected_keys)

    def test_total_equals_sum_of_parts(self):
        config = TrainingVramConfig(training_method = "qlora", load_in_4bit = True)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        parts_sum = (
            breakdown.model_weights
            + breakdown.lora_adapters
            + breakdown.optimizer_states
            + breakdown.gradients
            + breakdown.activations
            + breakdown.cuda_overhead
        )
        self.assertEqual(breakdown.total, parts_sum)

    def test_larger_batch_increases_total(self):
        small = TrainingVramConfig(
            training_method = "qlora",
            load_in_4bit = True,
            batch_size = 1,
        )
        large = TrainingVramConfig(
            training_method = "qlora",
            load_in_4bit = True,
            batch_size = 8,
        )
        small_v = estimate_training_vram(LLAMA_8B, small)
        large_v = estimate_training_vram(LLAMA_8B, large)
        self.assertGreater(large_v.total, small_v.total)

    def test_adamw_fp32_uses_more_optimizer_memory(self):
        opt8 = TrainingVramConfig(
            training_method = "full",
            load_in_4bit = False,
            optimizer = "adamw_8bit",
        )
        opt32 = TrainingVramConfig(
            training_method = "full",
            load_in_4bit = False,
            optimizer = "adamw_torch",
        )
        v8 = estimate_training_vram(LLAMA_8B, opt8)
        v32 = estimate_training_vram(LLAMA_8B, opt32)
        self.assertAlmostEqual(
            v32.optimizer_states / v8.optimizer_states, 1.5, delta = 0.1
        )

    def test_min_gpu_vram_treats_activations_as_per_gpu_fixed(self):
        config = TrainingVramConfig(training_method = "qlora", load_in_4bit = True)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        shardable = (
            breakdown.model_weights
            + breakdown.lora_adapters
            + breakdown.optimizer_states
            + breakdown.gradients
        )
        per_gpu_fixed = breakdown.activations + breakdown.cuda_overhead
        for n_gpus in (1, 2, 4):
            self.assertEqual(
                breakdown.min_gpu_vram(n_gpus),
                shardable // n_gpus + per_gpu_fixed,
            )

    def test_qlora_gradient_floor_is_capped_by_trainable_scale(self):
        config = TrainingVramConfig(
            training_method = "qlora",
            batch_size = 1,
            max_seq_length = 512,
            lora_rank = 16,
            target_modules = ["all-linear"],
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = True,
        )
        breakdown = estimate_training_vram(LLAMA_8B, config)
        lora_params = compute_lora_params(LLAMA_8B, 16, DEFAULT_TARGET_MODULES)
        optimizer_bytes = compute_optimizer_bytes(lora_params, "adamw_8bit")
        weight_floor = int(breakdown.model_weights * 0.15)

        self.assertEqual(
            breakdown.gradients,
            max(breakdown.activations_computed, optimizer_bytes),
        )
        self.assertLess(breakdown.gradients, weight_floor)
        self.assertEqual(breakdown.activations, breakdown.activations_computed)

    def test_full_finetuning_gradient_floor_remains_uncapped(self):
        config = TrainingVramConfig(
            training_method = "full",
            batch_size = 1,
            max_seq_length = 512,
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = False,
        )
        expected_floor = int(
            compute_model_weights_bytes(LLAMA_8B, "full", False) * 0.15
        )
        with patch(
            "utils.hardware.vram_estimation.compute_gradient_bytes",
            return_value = 1,
        ):
            breakdown = estimate_training_vram(LLAMA_8B, config)
        self.assertEqual(breakdown.gradients, expected_floor)

    def test_non_flash_attention_flows_into_training_estimate(self):
        config = TrainingVramConfig(
            training_method = "qlora",
            batch_size = 1,
            max_seq_length = 4096,
            lora_rank = 16,
            target_modules = ["all-linear"],
            gradient_checkpointing = "unsloth",
            optimizer = "adamw_8bit",
            load_in_4bit = True,
            attention_implementation = "eager",
        )
        breakdown = estimate_training_vram(STRUCTURED_MIXED, config)
        self.assertEqual(breakdown.activations, breakdown.activations_computed)
        self.assertGreater(
            breakdown.activations,
            compute_activation_bytes(
                STRUCTURED_MIXED,
                1,
                4096,
                "unsloth",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
        )


class TestExtractArchConfigMoE(unittest.TestCase):
    def test_deepseek_v3_shared_experts(self):
        hf_config = SimpleNamespace(
            hidden_size = 7168,
            num_hidden_layers = 61,
            num_attention_heads = 128,
            num_key_value_heads = 128,
            intermediate_size = 18432,
            vocab_size = 129280,
            tie_word_embeddings = False,
            n_routed_experts = 256,
            moe_intermediate_size = 2048,
            n_shared_experts = 1,
            first_k_dense_replace = 3,
            q_lora_rank = 1536,
            kv_lora_rank = 512,
            qk_nope_head_dim = 128,
            qk_rope_head_dim = 64,
            v_head_dim = 128,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_experts, 256)
        self.assertEqual(arch.n_shared_experts, 1)
        self.assertEqual(arch.num_dense_layers, 3)
        self.assertEqual(arch.q_lora_rank, 1536)
        self.assertEqual(arch.kv_lora_rank, 512)

    def test_qwen3_moe_decoder_sparse_step(self):
        hf_config = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 48,
            num_attention_heads = 32,
            num_key_value_heads = 4,
            intermediate_size = 8192,
            vocab_size = 151936,
            tie_word_embeddings = True,
            num_local_experts = 128,
            moe_intermediate_size = 768,
            decoder_sparse_step = 1,
            mlp_only_layers = [],
            head_dim = 128,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_experts, 128)
        self.assertEqual(arch.num_dense_layers, 0)
        self.assertEqual(arch.head_dim, 128)
        self.assertIsNone(arch.q_lora_rank)
        total_b = compute_total_params(arch) / 1e9
        self.assertGreater(total_b, 20)
        self.assertLess(total_b, 50)

    def test_qwen3_moe_with_mlp_only_layers(self):
        hf_config = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 24,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 8192,
            vocab_size = 151936,
            tie_word_embeddings = True,
            num_local_experts = 60,
            moe_intermediate_size = 1408,
            decoder_sparse_step = 1,
            mlp_only_layers = [0, 1, 2, 3],
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_dense_layers, 4)

    def test_glm4_moe_first_k_dense(self):
        hf_config = SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 46,
            num_attention_heads = 96,
            num_key_value_heads = 8,
            intermediate_size = 10944,
            vocab_size = 151552,
            tie_word_embeddings = False,
            n_routed_experts = 128,
            moe_intermediate_size = 1408,
            n_shared_experts = 1,
            first_k_dense_replace = 1,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_dense_layers, 1)
        self.assertEqual(arch.n_shared_experts, 1)

    def test_gpt_oss_no_moe_intermediate(self):
        hf_config = SimpleNamespace(
            hidden_size = 6144,
            num_hidden_layers = 64,
            num_attention_heads = 64,
            num_key_value_heads = 8,
            intermediate_size = 2880,
            vocab_size = 200064,
            tie_word_embeddings = False,
            num_local_experts = 128,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_experts, 128)
        self.assertIsNone(arch.moe_intermediate_size)
        self.assertEqual(arch.num_dense_layers, 0)

    def test_backward_compat_no_new_fields(self):
        hf_config = SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 128256,
            tie_word_embeddings = False,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.n_shared_experts, 0)
        self.assertEqual(arch.num_dense_layers, 0)
        self.assertIsNone(arch.q_lora_rank)
        self.assertFalse(arch.moe_has_dense_mlp)

    def test_enable_moe_block_extracted_as_moe_has_dense_mlp(self):
        hf_config = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 8,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 4096,
            vocab_size = 32000,
            tie_word_embeddings = True,
            num_experts = 8,
            moe_intermediate_size = 1024,
            head_dim = 128,
            layer_types = ["full_attention"] * 8,
            enable_moe_block = True,
        )
        arch = extract_arch_config(hf_config)
        self.assertTrue(arch.moe_has_dense_mlp)


class TestParallelDenseMoE(unittest.TestCase):
    def _arch(self, **overrides):
        base = ModelArchConfig(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 1024,
            vocab_size = 1024,
            tie_word_embeddings = True,
            num_experts = 8,
            moe_intermediate_size = 512,
            num_dense_layers = 0,
            head_dim = 64,
            layer_types = ["full_attention"] * 4,
        )
        return replace(base, **overrides)

    def test_total_params_includes_parallel_dense_when_enable_moe_block(self):
        without_parallel = self._arch(moe_has_dense_mlp = False)
        with_parallel = self._arch(moe_has_dense_mlp = True)
        self.assertGreater(
            compute_total_params(with_parallel),
            compute_total_params(without_parallel),
        )

    def test_lora_params_includes_parallel_dense_when_enable_moe_block(self):
        without_parallel = self._arch(moe_has_dense_mlp = False)
        with_parallel = self._arch(moe_has_dense_mlp = True)
        target = ["gate_proj", "up_proj", "down_proj"]
        self.assertGreater(
            compute_lora_params(with_parallel, 16, target),
            compute_lora_params(without_parallel, 16, target),
        )

    def test_activation_bytes_includes_parallel_dense_when_enable_moe_block(self):
        without_parallel = self._arch(moe_has_dense_mlp = False)
        with_parallel = self._arch(moe_has_dense_mlp = True)
        self.assertGreater(
            compute_activation_bytes(
                with_parallel,
                1,
                2048,
                "unsloth",
                is_lora = True,
            ),
            compute_activation_bytes(
                without_parallel,
                1,
                2048,
                "unsloth",
                is_lora = True,
            ),
        )

    def test_layer_aggregates_split_dense_mlp_from_experts(self):
        from utils.hardware.vram_estimation import _build_text_module_elements

        with_parallel = self._arch(moe_has_dense_mlp = True)
        elements, _ = _build_text_module_elements(with_parallel)
        moe_only = (
            with_parallel.hidden_size
            * with_parallel.moe_intermediate_size
            * 3
            * with_parallel.num_experts
            + with_parallel.num_experts * with_parallel.hidden_size
        )
        dense_only = with_parallel.hidden_size * with_parallel.intermediate_size * 3
        # why: under gemma4 enable_moe_block, the layer's `self.experts` is a
        # sibling of `self.mlp`; the `text.layers.<i>.mlp` aggregate must
        # cover the dense path only, with experts in their own aggregate.
        self.assertEqual(elements["text.layers.0.mlp"], dense_only)
        self.assertEqual(elements["text.layers.0.experts"], moe_only)


class TestDenseLayerIndices(unittest.TestCase):
    def test_non_prefix_mlp_only_layers_preserve_position(self):
        hf_config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 8,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = True,
            num_local_experts = 4,
            moe_intermediate_size = 512,
            decoder_sparse_step = 1,
            mlp_only_layers = [3, 5],
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(arch.num_dense_layers, 2)
        self.assertIn(3, arch.dense_layer_indices)
        self.assertIn(5, arch.dense_layer_indices)
        self.assertNotIn(0, arch.dense_layer_indices)

    def test_first_k_dense_replace_indices_are_prefix(self):
        hf_config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 6,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            n_routed_experts = 8,
            moe_intermediate_size = 512,
            first_k_dense_replace = 2,
        )
        arch = extract_arch_config(hf_config)
        self.assertEqual(tuple(arch.dense_layer_indices), (0, 1))


class TestKvSharedLayer(unittest.TestCase):
    def test_fully_shared_kv_returns_false_matching_upstream(self):
        from utils.hardware.vram_estimation import _is_kv_shared_layer

        arch = ModelArchConfig(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 1024,
            vocab_size = 1024,
            num_kv_shared_layers = 4,
        )
        for i in range(arch.num_hidden_layers):
            self.assertFalse(_is_kv_shared_layer(arch, i))

    def test_partial_share_returns_true_for_tail_layers(self):
        from utils.hardware.vram_estimation import _is_kv_shared_layer

        arch = ModelArchConfig(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 1024,
            vocab_size = 1024,
            num_kv_shared_layers = 2,
        )
        self.assertFalse(_is_kv_shared_layer(arch, 0))
        self.assertFalse(_is_kv_shared_layer(arch, 1))
        self.assertTrue(_is_kv_shared_layer(arch, 2))
        self.assertTrue(_is_kv_shared_layer(arch, 3))


class TestFlexAttentionLinear(unittest.TestCase):
    def test_flex_attention_treated_as_linear(self):
        flash = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
            attention_implementation = "flash_attention_2",
        )
        flex = compute_activation_bytes(
            STRUCTURED_MIXED,
            1,
            4096,
            "unsloth",
            is_lora = True,
            attention_implementation = "flex_attention",
        )
        self.assertEqual(flex, flash)


class TestNonStructuredParallelDense(unittest.TestCase):
    def _arch(self, **overrides):
        base = ModelArchConfig(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 4096,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_experts = 8,
            moe_intermediate_size = 768,
            num_dense_layers = 0,
            moe_has_dense_mlp = True,
        )
        return replace(base, **overrides)

    def test_skip_module_uses_intermediate_size_for_parallel_dense(self):
        from utils.hardware.vram_estimation import _build_text_module_elements

        arch = self._arch()
        elements, _ = _build_text_module_elements(arch)
        gate_proj = elements["text.layers.0.mlp.gate_proj"]
        self.assertEqual(gate_proj, arch.hidden_size * arch.intermediate_size)


class TestPerLayerInputAccounting(unittest.TestCase):
    def _arch(self, **overrides):
        base = ModelArchConfig(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            head_dim = 64,
            layer_types = ["full_attention"] * 4,
            vocab_size_per_layer_input = 256,
            hidden_size_per_layer_input = 96,
        )
        return replace(base, **overrides)

    def test_per_layer_input_increases_total_params(self):
        with_ple = self._arch()
        without_ple = replace(with_ple, hidden_size_per_layer_input = 0)
        self.assertGreater(
            compute_total_params(with_ple),
            compute_total_params(without_ple),
        )

    def test_per_layer_input_modules_count_quantizable_block(self):
        with_ple = self._arch()
        without_ple = replace(with_ple, hidden_size_per_layer_input = 0)
        # The PLE block adds: model_projection (hd*nl*pli), per_layer_input_gate
        # (hd*pli per layer) + per_layer_projection (pli*hd per layer) as
        # quantizable text linears.
        n_layers = with_ple.num_hidden_layers
        hd = with_ple.hidden_size
        pli = with_ple.hidden_size_per_layer_input
        expected_quantizable_extra = (
            hd * (n_layers * pli) + (hd * pli) * n_layers + (pli * hd) * n_layers
        )
        delta = compute_total_params(with_ple) - compute_total_params(without_ple)
        self.assertGreaterEqual(delta, expected_quantizable_extra)

    def test_all_linear_lora_excludes_per_layer_input_modules(self):
        # why: Unsloth's get_peft_regex requires module names to contain a
        # component tag (mlp/attn/...); PLE module names (per_layer_input_gate,
        # per_layer_projection, per_layer_model_projection) lack any tag, so
        # all-linear training does NOT attach LoRA to them.
        arch = self._arch()
        without_ple = replace(arch, hidden_size_per_layer_input = 0)
        self.assertEqual(
            compute_lora_params(arch, 16, ["all-linear"]),
            compute_lora_params(without_ple, 16, ["all-linear"]),
        )

    def test_explicit_target_modules_does_not_add_per_layer_input(self):
        arch = self._arch()
        without_ple = replace(arch, hidden_size_per_layer_input = 0)
        self.assertEqual(
            compute_lora_params(arch, 16, ["q_proj", "v_proj"]),
            compute_lora_params(without_ple, 16, ["q_proj", "v_proj"]),
        )


class TestDenseMlpLayerFallback(unittest.TestCase):
    def test_falls_back_to_count_when_indices_empty(self):
        from utils.hardware.vram_estimation import _is_dense_mlp_layer

        arch = ModelArchConfig(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 1024,
            vocab_size = 1024,
            num_experts = 4,
            moe_intermediate_size = 256,
            num_dense_layers = 2,
        )
        self.assertTrue(_is_dense_mlp_layer(arch, 0))
        self.assertTrue(_is_dense_mlp_layer(arch, 1))
        self.assertFalse(_is_dense_mlp_layer(arch, 2))
        self.assertFalse(_is_dense_mlp_layer(arch, 3))


class TestExpertsSkipGranularity(unittest.TestCase):
    def _arch(self):
        return ModelArchConfig(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 1024,
            vocab_size = 1024,
            tie_word_embeddings = True,
            num_experts = 8,
            moe_intermediate_size = 512,
            num_dense_layers = 0,
            head_dim = 64,
            layer_types = ["full_attention"] * 4,
            moe_has_dense_mlp = True,
        )

    def test_experts_skip_excludes_parallel_dense_projections(self):
        no_skip = self._arch()
        skip_experts = replace(
            no_skip,
            quantization_skip_modules = ["model.layers.0.mlp.experts"],
        )
        skip_full_mlp = replace(
            no_skip,
            quantization_skip_modules = ["model.layers.0.mlp"],
        )
        bytes_no_skip = compute_model_weights_bytes(no_skip, "qlora", True)
        bytes_skip_experts = compute_model_weights_bytes(skip_experts, "qlora", True)
        bytes_skip_mlp = compute_model_weights_bytes(skip_full_mlp, "qlora", True)
        # why: under gemma4 enable_moe_block, `self.experts` is a sibling of
        # `self.mlp`; skipping `model.layers.0.mlp` should cover only the
        # dense MLP, while `model.layers.0.mlp.experts` covers the routed
        # experts. Routed experts have far more params than the dense MLP,
        # so skipping experts must add more bytes than skipping the dense
        # path.
        self.assertGreater(bytes_skip_experts, bytes_no_skip)
        self.assertGreater(bytes_skip_mlp, bytes_no_skip)
        self.assertGreater(bytes_skip_experts, bytes_skip_mlp)


class TestSharedExperts(unittest.TestCase):
    def test_shared_experts_increase_weight_bytes(self):
        no_shared = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_experts = 64,
            moe_intermediate_size = 1407,
            n_shared_experts = 0,
        )
        with_shared = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_experts = 64,
            moe_intermediate_size = 1407,
            n_shared_experts = 2,
        )
        w_no = compute_model_weights_bytes(no_shared, "full", False)
        w_yes = compute_model_weights_bytes(with_shared, "full", False)
        self.assertGreater(w_yes, w_no)
        delta_per_layer = 4096 * 1407 * 3 * 2
        expected_delta = delta_per_layer * 32 * 2
        actual_delta = w_yes - w_no
        self.assertAlmostEqual(
            actual_delta, expected_delta, delta = expected_delta * 0.01
        )

    def test_deepseek_v3_params_in_range(self):
        total = compute_total_params(DEEPSEEK_V3)
        total_b = total / 1e9
        self.assertGreater(total_b, 600)
        self.assertLess(total_b, 750)


class TestMLA(unittest.TestCase):
    def test_mla_different_from_standard(self):
        from utils.hardware.vram_estimation import _compute_attn_elements

        mla_arch = DEEPSEEK_V3
        std_arch = ModelArchConfig(
            hidden_size = 7168,
            num_hidden_layers = 61,
            num_attention_heads = 128,
            num_key_value_heads = 128,
            intermediate_size = 18432,
            vocab_size = 129280,
        )
        mla_attn = _compute_attn_elements(mla_arch)
        std_attn = _compute_attn_elements(std_arch)
        self.assertNotEqual(mla_attn, std_attn)

    def test_mla_lora_produces_values(self):
        lora_p = compute_lora_params(DEEPSEEK_V3, 16, ["q_proj", "v_proj", "o_proj"])
        self.assertGreater(lora_p, 0)

    def test_mla_with_head_dim_does_not_route_through_structured(self):
        from utils.hardware.vram_estimation import _uses_structured_layer_shapes

        mla_with_head_dim = replace(DEEPSEEK_V3, head_dim = 128)
        self.assertFalse(_uses_structured_layer_shapes(mla_with_head_dim))
        self.assertEqual(
            compute_lora_params(DEEPSEEK_V3, 16, ["q_proj", "v_proj", "o_proj"]),
            compute_lora_params(mla_with_head_dim, 16, ["q_proj", "v_proj", "o_proj"]),
        )


class TestDenseMoEMix(unittest.TestCase):
    def test_dense_layers_change_total(self):
        all_moe = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 46,
            num_attention_heads = 96,
            num_key_value_heads = 8,
            intermediate_size = 10944,
            vocab_size = 151552,
            tie_word_embeddings = False,
            num_experts = 128,
            moe_intermediate_size = 1408,
            n_shared_experts = 1,
            num_dense_layers = 0,
        )
        mixed = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 46,
            num_attention_heads = 96,
            num_key_value_heads = 8,
            intermediate_size = 10944,
            vocab_size = 151552,
            tie_word_embeddings = False,
            num_experts = 128,
            moe_intermediate_size = 1408,
            n_shared_experts = 1,
            num_dense_layers = 1,
        )
        w_all = compute_model_weights_bytes(all_moe, "full", False)
        w_mixed = compute_model_weights_bytes(mixed, "full", False)
        self.assertNotEqual(w_all, w_mixed)

    def test_glm4_moe_params_reasonable(self):
        total = compute_total_params(GLM4_MOE)
        total_b = total / 1e9
        self.assertGreater(total_b, 80)
        self.assertLess(total_b, 120)

    def test_qwen3_moe_30b_params_reasonable(self):
        total = compute_total_params(QWEN3_MOE_30B)
        total_b = total / 1e9
        self.assertGreater(total_b, 20)
        self.assertLess(total_b, 50)

    def test_gpt_oss_uses_intermediate_size(self):
        total = compute_total_params(GPT_OSS)
        total_b = total / 1e9
        self.assertGreater(total_b, 350)
        self.assertLess(total_b, 500)

    def test_lora_dense_vs_moe_layers_differ(self):
        all_moe = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 10,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_experts = 8,
            moe_intermediate_size = 1024,
            num_dense_layers = 0,
        )
        mixed = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 10,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_experts = 8,
            moe_intermediate_size = 1024,
            num_dense_layers = 5,
        )
        lora_all = compute_lora_params(
            all_moe, 16, ["gate_proj", "up_proj", "down_proj"]
        )
        lora_mix = compute_lora_params(mixed, 16, ["gate_proj", "up_proj", "down_proj"])
        self.assertNotEqual(lora_all, lora_mix)


class TestMlpLayerTypesDispatch(unittest.TestCase):
    def _hf(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 64,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 128,
            vocab_size = 1000,
            tie_word_embeddings = True,
            num_local_experts = 4,
            moe_intermediate_size = 32,
            **fields,
        )
        return SimpleNamespace(text_config = text_config, quantization_config = {})

    def test_mlp_layer_types_drives_dense_indices(self):
        hf = self._hf(mlp_layer_types = ["sparse", "dense", "sparse", "dense"])
        arch = extract_arch_config(hf)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.dense_layer_indices, (1, 3))
        self.assertEqual(arch.num_dense_layers, 2)

    def test_mlp_layer_types_takes_priority_over_first_k_dense_replace(self):
        hf = self._hf(
            mlp_layer_types = ["dense", "sparse", "dense", "sparse"],
            first_k_dense_replace = 3,
        )
        arch = extract_arch_config(hf)
        self.assertEqual(arch.dense_layer_indices, (0, 2))

    def test_mlp_layer_types_ignores_unknown_entries(self):
        hf = self._hf(mlp_layer_types = ["dense", "moe", "dense", "linear"])
        arch = extract_arch_config(hf)
        self.assertEqual(arch.dense_layer_indices, (0, 2))

    def test_mlp_layer_types_shorter_than_layers_only_marks_present(self):
        hf = self._hf(mlp_layer_types = ["dense", "sparse"])
        arch = extract_arch_config(hf)
        self.assertEqual(arch.dense_layer_indices, (0,))

    def test_empty_mlp_layer_types_falls_through_to_first_k(self):
        hf = self._hf(mlp_layer_types = [], first_k_dense_replace = 2)
        arch = extract_arch_config(hf)
        self.assertEqual(arch.dense_layer_indices, (0, 1))


class TestPerLayerInputSkipAlias(unittest.TestCase):
    def _hf(self, skip):
        text_config = SimpleNamespace(
            hidden_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 128,
            vocab_size = 1000,
            tie_word_embeddings = True,
            hidden_size_per_layer_input = 8,
            vocab_size_per_layer_input = 256,
        )
        return SimpleNamespace(
            text_config = text_config,
            quantization_config = {"llm_int8_skip_modules": list(skip)},
        )

    def test_per_layer_input_gate_skip_pulls_nonzero_delta(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(self._hf(["model.layers.0.per_layer_input_gate"]))
        delta = _compute_skipped_quantizable_elements(arch)
        self.assertEqual(delta, arch.hidden_size * arch.hidden_size_per_layer_input)

    def test_per_layer_model_projection_skip_pulls_global_delta(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(self._hf(["model.per_layer_model_projection"]))
        delta = _compute_skipped_quantizable_elements(arch)
        self.assertEqual(
            delta,
            arch.hidden_size
            * arch.num_hidden_layers
            * arch.hidden_size_per_layer_input,
        )

    def test_layer_aggregate_skip_includes_per_layer_input_modules(self):
        from utils.hardware.vram_estimation import (
            _compute_skipped_quantizable_elements,
        )

        arch_with = extract_arch_config(self._hf(["model.layers.0"]))
        # The text.layers.0 aggregate must include the PLE per-layer modules,
        # so the same skip on a config without PLE produces a smaller value.
        arch_without = extract_arch_config(
            SimpleNamespace(
                text_config = SimpleNamespace(
                    hidden_size = 64,
                    num_hidden_layers = 2,
                    num_attention_heads = 4,
                    num_key_value_heads = 4,
                    intermediate_size = 128,
                    vocab_size = 1000,
                    tie_word_embeddings = True,
                    hidden_size_per_layer_input = 0,
                    vocab_size_per_layer_input = 0,
                ),
                quantization_config = {"llm_int8_skip_modules": ["model.layers.0"]},
            )
        )
        self.assertGreater(
            _compute_skipped_quantizable_elements(arch_with),
            _compute_skipped_quantizable_elements(arch_without),
        )


class TestAllLinearStringHandling(unittest.TestCase):
    def test_compute_lora_params_accepts_bare_all_linear_string(self):
        list_form = compute_lora_params(LLAMA_8B, 16, ["all-linear"])
        str_form = compute_lora_params(LLAMA_8B, 16, "all-linear")
        self.assertEqual(list_form, str_form)
        self.assertGreater(list_form, 0)

    def test_compute_lora_params_string_with_underscores_normalized(self):
        list_form = compute_lora_params(LLAMA_8B, 16, ["all_linear"])
        str_form = compute_lora_params(LLAMA_8B, 16, "all_linear")
        self.assertEqual(list_form, str_form)
        self.assertGreater(str_form, 0)


class TestSharedExpertVariants(unittest.TestCase):
    def _hf(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 256,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            num_local_experts = 8,
            moe_intermediate_size = 128,
            **fields,
        )
        return SimpleNamespace(text_config = text_config, quantization_config = {})

    def test_shared_expert_intermediate_size_extracted_and_infers_count(self):
        arch = extract_arch_config(self._hf(shared_expert_intermediate_size = 64))
        self.assertEqual(arch.shared_expert_intermediate_size, 64)
        self.assertEqual(arch.n_shared_experts, 1)

    def test_num_shared_experts_alias_extracted(self):
        arch = extract_arch_config(self._hf(num_shared_experts = 2))
        self.assertEqual(arch.n_shared_experts, 2)

    def test_n_shared_experts_takes_priority_over_alias(self):
        arch = extract_arch_config(self._hf(n_shared_experts = 3, num_shared_experts = 99))
        self.assertEqual(arch.n_shared_experts, 3)

    def test_shared_expert_size_separate_from_routed_changes_weight_count(self):
        from utils.hardware.vram_estimation import _compute_moe_mlp_elements

        arch_separate = extract_arch_config(
            self._hf(shared_expert_intermediate_size = 64)
        )
        arch_implicit = extract_arch_config(self._hf(n_shared_experts = 1))
        # Different shared sizes (64 vs default moe_intermediate_size=128) must
        # produce different MoE element counts.
        self.assertNotEqual(
            _compute_moe_mlp_elements(arch_separate),
            _compute_moe_mlp_elements(arch_implicit),
        )

    def test_shared_expert_gate_counted_only_for_qwen_style(self):
        from utils.hardware.vram_estimation import _compute_moe_mlp_elements

        # Qwen-style: shared_expert_intermediate_size set -> shared_expert_gate counted.
        qwen_arch = extract_arch_config(self._hf(shared_expert_intermediate_size = 64))
        hd = qwen_arch.hidden_size
        ms = qwen_arch.moe_intermediate_size
        ne = qwen_arch.num_experts
        ss = qwen_arch.shared_expert_intermediate_size
        expected = hd * ms * 3 * ne + ne * hd + hd * ss * 3 * 1 + 1 * hd
        self.assertEqual(_compute_moe_mlp_elements(qwen_arch), expected)

        # Non-Qwen shared experts (e.g. Exaone-MoE) -> no shared_expert_gate.
        plain_arch = extract_arch_config(self._hf(n_shared_experts = 1))
        hd = plain_arch.hidden_size
        ms = plain_arch.moe_intermediate_size
        ne = plain_arch.num_experts
        expected_plain = hd * ms * 3 * ne + ne * hd + hd * ms * 3 * 1
        self.assertEqual(_compute_moe_mlp_elements(plain_arch), expected_plain)


class TestSharedExpertActivation(unittest.TestCase):
    def _make(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            num_local_experts = 4,
            moe_intermediate_size = 64,
            **fields,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_shared_expert_increases_activation_bytes(self):
        with_shared = self._make(shared_expert_intermediate_size = 64)
        without = self._make()
        self.assertGreater(
            compute_activation_bytes(
                with_shared,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
            compute_activation_bytes(
                without,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
        )

    def test_shared_expert_plus_dense_block_compose(self):
        # gemma4 enable_moe_block with hypothetical shared expert: dense + routed
        # + shared all live per layer; mlp_size should sum all three terms.
        from utils.hardware.vram_estimation import _layer_qkv_mlp_sizes

        arch = self._make(
            enable_moe_block = True,
            shared_expert_intermediate_size = 32,
            head_dim = 64,
            layer_types = ["full_attention"] * 4,
        )
        _, mlp_size = _layer_qkv_mlp_sizes(arch, 0)
        # routed (64) + shared (32) + parallel dense intermediate (1024)
        self.assertEqual(mlp_size, 64 + 32 + 1024)


class TestPerLayerInputActivation(unittest.TestCase):
    def _make(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            **fields,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_ple_increases_activation_bytes(self):
        with_ple = self._make(
            hidden_size_per_layer_input = 64,
            vocab_size_per_layer_input = 256,
        )
        without = self._make()
        self.assertGreater(
            compute_activation_bytes(
                with_ple,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
            compute_activation_bytes(
                without,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
        )

    def test_ple_zero_does_not_inflate_activations(self):
        without = self._make(hidden_size_per_layer_input = 0)
        baseline = self._make()
        self.assertEqual(
            compute_activation_bytes(
                without,
                2,
                512,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
            compute_activation_bytes(
                baseline,
                2,
                512,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
        )


class TestKvSharedActivation(unittest.TestCase):
    def _make(self, kv_shared):
        text_config = SimpleNamespace(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            head_dim = 64,
            num_kv_shared_layers = kv_shared,
            layer_types = ["full_attention"] * 4,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_kv_shared_layers_keep_activation_bytes(self):
        shared = self._make(kv_shared = 2)
        full = self._make(kv_shared = 0)
        self.assertEqual(
            compute_activation_bytes(
                shared,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
            compute_activation_bytes(
                full,
                2,
                1024,
                "none",
                is_lora = True,
                attention_implementation = "flash_attention_2",
            ),
        )


class TestSparseMoeSkipAliases(unittest.TestCase):
    def _hf(self, skip, **fields):
        text_config = SimpleNamespace(
            hidden_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 256,
            vocab_size = 1000,
            tie_word_embeddings = False,
            num_local_experts = 4,
            moe_intermediate_size = 64,
            **fields,
        )
        return SimpleNamespace(
            text_config = text_config,
            quantization_config = {"llm_int8_skip_modules": list(skip)},
        )

    def test_gemma4_layers_experts_alias_pulls_routed(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(
            self._hf(["model.layers.0.experts"], enable_moe_block = True)
        )
        self.assertGreater(_compute_skipped_quantizable_elements(arch), 0)

    def test_qwen_shared_expert_skip_pulls_only_shared(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(
            self._hf(
                ["model.layers.0.mlp.shared_expert"],
                shared_expert_intermediate_size = 32,
            )
        )
        # shared_expert delta only -- routed mlp.experts is NOT skipped.
        delta = _compute_skipped_quantizable_elements(arch)
        self.assertGreater(delta, 0)
        full_layer = extract_arch_config(
            self._hf(
                ["model.layers.0.mlp"],
                shared_expert_intermediate_size = 32,
            )
        )
        self.assertGreater(
            _compute_skipped_quantizable_elements(full_layer),
            delta,
        )

    def test_exaone_shared_experts_plural_alias(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(
            self._hf(
                ["model.layers.0.mlp.shared_experts"],
                num_shared_experts = 1,
            )
        )
        self.assertGreater(_compute_skipped_quantizable_elements(arch), 0)


class TestAllLinearMoELoraExclusion(unittest.TestCase):
    def _arch(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 256,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 512,
            vocab_size = 1000,
            tie_word_embeddings = False,
            num_local_experts = 8,
            moe_intermediate_size = 64,
            **fields,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_all_linear_drops_routed_moe_expert_lora(self):
        arch = self._arch()
        all_linear = compute_lora_params(arch, 8, "all-linear")
        explicit = compute_lora_params(arch, 8, ["gate_proj", "up_proj", "down_proj"])
        self.assertLess(all_linear, explicit)

    def test_all_linear_drops_shared_expert_lora(self):
        arch = self._arch(shared_expert_intermediate_size = 32)
        all_linear = compute_lora_params(arch, 8, "all-linear")
        explicit = compute_lora_params(arch, 8, ["gate_proj", "up_proj", "down_proj"])
        # explicit includes routed + shared MoE; all-linear includes neither.
        self.assertLess(all_linear, explicit)

    def test_all_linear_includes_attention_lora(self):
        arch = self._arch()
        all_linear = compute_lora_params(arch, 8, "all-linear")
        attn_only = compute_lora_params(
            arch, 8, ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        # all-linear still attaches to attention nn.Linear modules.
        self.assertGreaterEqual(all_linear, attn_only)


class TestExplicitPerLayerInputLora(unittest.TestCase):
    def _arch(self):
        text_config = SimpleNamespace(
            hidden_size = 256,
            num_hidden_layers = 3,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 512,
            vocab_size = 1000,
            tie_word_embeddings = False,
            hidden_size_per_layer_input = 32,
            vocab_size_per_layer_input = 128,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_explicit_per_layer_input_gate_returns_nonzero(self):
        arch = self._arch()
        result = compute_lora_params(arch, 16, ["per_layer_input_gate"])
        self.assertGreater(result, 0)

    def test_explicit_per_layer_projection_returns_nonzero(self):
        arch = self._arch()
        result = compute_lora_params(arch, 16, ["per_layer_projection"])
        self.assertGreater(result, 0)

    def test_explicit_per_layer_model_projection_returns_nonzero(self):
        arch = self._arch()
        result = compute_lora_params(arch, 16, ["per_layer_model_projection"])
        self.assertGreater(result, 0)

    def test_explicit_ple_string_target_handled(self):
        # Bare-string target with a PLE name should not be iterated char-by-char.
        arch = self._arch()
        list_form = compute_lora_params(arch, 16, ["per_layer_input_gate"])
        str_form = compute_lora_params(arch, 16, "per_layer_input_gate")
        self.assertEqual(list_form, str_form)


class TestTopKExpertActivation(unittest.TestCase):
    def _make(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 512,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            num_local_experts = 8,
            moe_intermediate_size = 64,
            **fields,
        )
        return extract_arch_config(
            SimpleNamespace(text_config = text_config, quantization_config = {})
        )

    def test_num_experts_per_tok_extracted(self):
        arch = self._make(num_experts_per_tok = 4)
        self.assertEqual(arch.num_experts_per_tok, 4)

    def test_top_k_experts_alias_extracted(self):
        arch = self._make(top_k_experts = 8)
        self.assertEqual(arch.num_experts_per_tok, 8)

    def test_default_top_k_one_unchanged(self):
        arch = self._make()
        self.assertEqual(arch.num_experts_per_tok, 1)

    def test_top_k_scales_moe_activation(self):
        single = self._make()
        multi = self._make(num_experts_per_tok = 8)
        single_act = compute_activation_bytes(
            single,
            2,
            512,
            "none",
            is_lora = True,
            attention_implementation = "flash_attention_2",
        )
        multi_act = compute_activation_bytes(
            multi,
            2,
            512,
            "none",
            is_lora = True,
            attention_implementation = "flash_attention_2",
        )
        self.assertGreater(multi_act, single_act)


class TestErnieMoEListConfig(unittest.TestCase):
    def _hf(self, **fields):
        text_config = SimpleNamespace(
            hidden_size = 256,
            num_hidden_layers = 4,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 1024,
            vocab_size = 1000,
            tie_word_embeddings = False,
            **fields,
        )
        return SimpleNamespace(text_config = text_config, quantization_config = {})

    def test_list_moe_intermediate_size_scalarized(self):
        arch = extract_arch_config(
            self._hf(
                moe_num_experts = 32,
                moe_intermediate_size = [1536, 512],
            )
        )
        # why: ERNIE 4.5 VL MoE encodes [text_routed, vision_routed]; the
        # second element is the vision-routed expert width, not the shared
        # expert width. Shared experts are sized from the text-routed width
        # (= moe_intermediate_size[0]) when moe_num_shared_experts is set.
        self.assertEqual(arch.moe_intermediate_size, 1536)
        self.assertIsNone(arch.shared_expert_intermediate_size)
        self.assertEqual(arch.n_shared_experts, 0)

    def test_moe_num_experts_alias_extracted(self):
        arch = extract_arch_config(
            self._hf(
                moe_num_experts = 64,
                moe_intermediate_size = 1024,
            )
        )
        self.assertEqual(arch.num_experts, 64)

    def test_moe_num_shared_experts_alias_extracted(self):
        arch = extract_arch_config(
            self._hf(
                moe_num_experts = 16,
                moe_num_shared_experts = 2,
                moe_intermediate_size = 1024,
            )
        )
        self.assertEqual(arch.n_shared_experts, 2)

    def test_explicit_shared_size_overrides_list_second_element(self):
        arch = extract_arch_config(
            self._hf(
                moe_num_experts = 8,
                moe_intermediate_size = [1536, 512],
                shared_expert_intermediate_size = 256,
            )
        )
        # Explicit shared size wins over moe_intermediate_size[1].
        self.assertEqual(arch.shared_expert_intermediate_size, 256)


class TestSuffixSkipModuleMatch(unittest.TestCase):
    def _hf(self, skip):
        text_config = SimpleNamespace(
            hidden_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            intermediate_size = 256,
            vocab_size = 1000,
            tie_word_embeddings = False,
        )
        return SimpleNamespace(
            text_config = text_config,
            quantization_config = {"llm_int8_skip_modules": list(skip)},
        )

    def test_q_proj_suffix_skip_matches_all_layers(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(self._hf(["q_proj"]))
        delta = _compute_skipped_quantizable_elements(arch)
        # 2 layers * hd * hd of q_proj weight elements.
        self.assertEqual(delta, 2 * arch.hidden_size * arch.hidden_size)

    def test_self_attn_aggregate_skip_matches_aggregate(self):
        from utils.hardware.vram_estimation import _compute_skipped_quantizable_elements

        arch = extract_arch_config(self._hf(["self_attn"]))
        # The aggregate text.layers.<i>.self_attn matches; total covers both layers.
        delta = _compute_skipped_quantizable_elements(arch)
        self.assertGreater(delta, 0)

    def test_vision_prefix_skip_does_not_match_text_alias(self):
        from utils.hardware.vram_estimation import _module_path_matches

        # vision_tower-prefixed full path must NOT match text-tower aliases.
        self.assertFalse(
            _module_path_matches(
                "vision_tower.model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.q_proj",
            )
        )


class TestMultimodalFullModelBytes(unittest.TestCase):
    def test_extra_bytes_added_when_safetensors_exceeds_text_arch(self):
        from utils.hardware import hardware as hardware_module

        config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
        )
        # Force safetensors size >>> arch text-only bytes.
        big_safetensors = 20 * 1024**3
        with (
            patch.object(
                hardware_module,
                "_load_config_for_gpu_estimate",
                return_value = config,
            ),
            patch.object(
                hardware_module,
                "estimate_fp16_model_size_bytes",
                return_value = (big_safetensors, "safetensors"),
            ),
            patch.object(
                hardware_module,
                "_determine_attention_impl_for_gpu_estimate",
                return_value = "flash_attention_2",
            ),
            patch.object(
                hardware_module,
                "get_visible_gpu_count",
                return_value = 1,
            ),
        ):
            _, metadata = hardware_module.estimate_required_model_memory_gb(
                "fake/model",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )
        self.assertEqual(metadata.get("estimation_mode"), "detailed")
        # model_weights_gb must reflect the extra non-text bytes (>5 GB
        # since text-only arch_fp16 is small for these dims).
        self.assertGreater(metadata["vram_breakdown"]["model_weights_gb"], 5.0)

    def test_no_extra_when_safetensors_smaller_than_text_arch(self):
        from utils.hardware import hardware as hardware_module

        config = SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 11008,
            vocab_size = 32000,
            tie_word_embeddings = False,
        )
        tiny_safetensors = 100  # bytes, deliberately absurdly small
        with (
            patch.object(
                hardware_module,
                "_load_config_for_gpu_estimate",
                return_value = config,
            ),
            patch.object(
                hardware_module,
                "estimate_fp16_model_size_bytes",
                return_value = (tiny_safetensors, "safetensors"),
            ),
            patch.object(
                hardware_module,
                "_determine_attention_impl_for_gpu_estimate",
                return_value = "flash_attention_2",
            ),
            patch.object(
                hardware_module,
                "get_visible_gpu_count",
                return_value = 1,
            ),
        ):
            required, metadata = hardware_module.estimate_required_model_memory_gb(
                "fake/model",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )
        # No negative extra; required_gb stays a positive finite number.
        self.assertGreater(required, 0)


class TestLlama4ArchExtraction(unittest.TestCase):
    def _llama4_text_config(self, **fields):
        base = dict(
            hidden_size = 2048,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 8192,
            intermediate_size_mlp = 16384,
            vocab_size = 32000,
            tie_word_embeddings = True,
            num_local_experts = 4,
            num_experts_per_tok = 2,
        )
        base.update(fields)
        return SimpleNamespace(**base)

    def test_llama4_moe_layers_dispatch_uses_explicit_indices(self):
        from utils.hardware.vram_estimation import _compute_dense_layer_indices

        cfg = SimpleNamespace(num_hidden_layers = 4, moe_layers = [1, 3])
        self.assertEqual(_compute_dense_layer_indices(cfg, 4), (0, 2))

    def test_llama4_moe_layers_takes_priority_over_first_k_dense_replace(self):
        from utils.hardware.vram_estimation import _compute_dense_layer_indices

        cfg = SimpleNamespace(
            num_hidden_layers = 6,
            moe_layers = [2, 4],
            first_k_dense_replace = 4,
        )
        self.assertEqual(_compute_dense_layer_indices(cfg, 6), (0, 1, 3, 5))

    def test_dense_intermediate_size_picks_up_intermediate_size_mlp(self):
        from utils.hardware.vram_estimation import _dense_mlp_size

        arch = extract_arch_config(self._llama4_text_config(moe_layers = [1, 3]))
        self.assertIsNotNone(arch)
        self.assertEqual(arch.intermediate_size, 8192)
        self.assertEqual(arch.dense_intermediate_size, 16384)
        self.assertEqual(_dense_mlp_size(arch), 16384)

    def test_auto_attaches_one_shared_expert_at_routed_width(self):
        from utils.hardware.vram_estimation import _shared_expert_size

        arch = extract_arch_config(self._llama4_text_config(moe_layers = [1, 3]))
        self.assertIsNotNone(arch)
        self.assertEqual(arch.n_shared_experts, 1)
        self.assertIsNone(arch.shared_expert_intermediate_size)
        self.assertEqual(_shared_expert_size(arch), arch.intermediate_size)

    def test_non_llama4_config_leaves_dense_intermediate_size_none(self):
        from utils.hardware.vram_estimation import _dense_mlp_size

        cfg = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            intermediate_size = 4096,
            vocab_size = 32000,
            tie_word_embeddings = True,
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertIsNone(arch.dense_intermediate_size)
        self.assertEqual(_dense_mlp_size(arch), 4096)

    def test_intermediate_size_mlp_without_moe_does_not_force_shared_expert(self):
        cfg = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 8192,
            intermediate_size_mlp = 16384,
            vocab_size = 32000,
            tie_word_embeddings = True,
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.dense_intermediate_size, 16384)
        self.assertEqual(arch.n_shared_experts, 0)


class TestDbrxFfnConfigExtraction(unittest.TestCase):
    def test_extracts_moe_fields_from_ffn_subconfig(self):
        ffn = SimpleNamespace(moe_num_experts = 4, moe_top_k = 2, ffn_hidden_size = 1024)
        cfg = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            ffn_config = ffn,
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.num_experts, 4)
        self.assertEqual(arch.num_experts_per_tok, 2)
        self.assertEqual(arch.moe_intermediate_size, 1024)

    def test_top_level_attrs_take_precedence_over_ffn_config(self):
        ffn = SimpleNamespace(moe_num_experts = 4, moe_top_k = 2, ffn_hidden_size = 1024)
        cfg = SimpleNamespace(
            hidden_size = 2048,
            num_hidden_layers = 4,
            num_attention_heads = 16,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            ffn_config = ffn,
            num_local_experts = 16,
            num_experts_per_tok = 8,
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.num_experts, 16)
        self.assertEqual(arch.num_experts_per_tok, 8)


class TestErniePhaseModuloDispatch(unittest.TestCase):
    def test_phase_modulo_with_interval_two_matches_decoder(self):
        from utils.hardware.vram_estimation import _compute_dense_layer_indices

        cfg = SimpleNamespace(
            num_hidden_layers = 10,
            moe_layer_start_index = 2,
            moe_layer_end_index = 8,
            moe_layer_interval = 2,
        )
        # Decoder gates by ((i + 1) % 2 == 0) AND 2 <= i <= 8 -> MoE = {3, 5, 7}.
        self.assertEqual(_compute_dense_layer_indices(cfg, 10), (0, 1, 2, 4, 6, 8, 9))

    def test_phase_modulo_with_interval_three(self):
        from utils.hardware.vram_estimation import _compute_dense_layer_indices

        cfg = SimpleNamespace(
            num_hidden_layers = 9,
            moe_layer_start_index = 0,
            moe_layer_end_index = -1,
            moe_layer_interval = 3,
        )
        self.assertEqual(_compute_dense_layer_indices(cfg, 9), (0, 1, 3, 4, 6, 7))


class TestErnieVlSharedExpertWidth(unittest.TestCase):
    def test_shared_expert_width_uses_text_routed_not_vision(self):
        from utils.hardware.vram_estimation import (
            _compute_shared_moe_elements,
            _shared_expert_size,
        )

        cfg = SimpleNamespace(
            text_config = SimpleNamespace(
                hidden_size = 1024,
                num_hidden_layers = 4,
                num_attention_heads = 8,
                num_key_value_heads = 4,
                intermediate_size = 2048,
                vocab_size = 32000,
                tie_word_embeddings = False,
                moe_num_experts = 8,
                moe_num_shared_experts = 2,
                moe_intermediate_size = [1536, 512],
            ),
            quantization_config = {},
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertIsNone(arch.shared_expert_intermediate_size)
        self.assertEqual(arch.moe_intermediate_size, 1536)
        self.assertEqual(arch.n_shared_experts, 2)
        self.assertEqual(_shared_expert_size(arch), 1536)
        self.assertEqual(_compute_shared_moe_elements(arch), 1024 * 1536 * 3 * 2)

    def test_qwen_style_explicit_shared_expert_size_still_adds_gate(self):
        from utils.hardware.vram_estimation import _compute_shared_moe_elements

        cfg = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            num_local_experts = 8,
            moe_intermediate_size = 256,
            shared_expert_intermediate_size = 768,
        )
        arch = extract_arch_config(cfg)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.shared_expert_intermediate_size, 768)
        self.assertEqual(arch.n_shared_experts, 1)
        self.assertEqual(
            _compute_shared_moe_elements(arch),
            1024 * 768 * 3 + 1 * 1024,
        )


if __name__ == "__main__":
    unittest.main()
