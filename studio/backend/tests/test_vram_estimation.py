# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import unittest
from types import SimpleNamespace

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
        dense_lora = compute_lora_params(LLAMA_8B, 16, ["gate_proj", "up_proj", "down_proj"])
        moe_lora = compute_lora_params(MOE_CONFIG, 16, ["gate_proj", "up_proj", "down_proj"])
        ratio = moe_lora / dense_lora
        self.assertAlmostEqual(ratio, 8.0, delta = 0.5)

    def test_attention_modules_same_for_moe(self):
        dense_attn = compute_lora_params(LLAMA_8B, 16, ["q_proj", "k_proj", "v_proj", "o_proj"])
        moe_attn = compute_lora_params(MOE_CONFIG, 16, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(dense_attn, moe_attn)


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

    def test_lora_activations_much_smaller(self):
        full_ft = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth", is_lora = False)
        lora = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth", is_lora = True)
        self.assertLess(lora, full_ft)
        self.assertLess(lora, full_ft / 3)

    def test_scales_with_batch_size(self):
        act_bsz2 = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth")
        act_bsz4 = compute_activation_bytes(LLAMA_8B, 4, 2048, "unsloth")
        self.assertAlmostEqual(act_bsz4 / act_bsz2, 2.0, delta = 0.1)

    def test_scales_with_seq_len(self):
        act_2k = compute_activation_bytes(LLAMA_8B, 2, 2048, "unsloth")
        act_4k = compute_activation_bytes(LLAMA_8B, 2, 4096, "unsloth")
        self.assertAlmostEqual(act_4k / act_2k, 2.0, delta = 0.1)


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
            training_method = "qlora", load_in_4bit = True,
            batch_size = 2, max_seq_length = 2048,
        )
        full_config = TrainingVramConfig(
            training_method = "full", load_in_4bit = False,
            batch_size = 2, max_seq_length = 2048,
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
            "model_weights_gb", "lora_adapters_gb", "optimizer_states_gb",
            "gradients_gb", "activations_gb", "cuda_overhead_gb", "total_gb",
        }
        self.assertEqual(set(gb_dict.keys()), expected_keys)

    def test_total_equals_sum_of_parts(self):
        config = TrainingVramConfig(training_method = "qlora", load_in_4bit = True)
        breakdown = estimate_training_vram(LLAMA_8B, config)
        parts_sum = (
            breakdown.model_weights + breakdown.lora_adapters
            + breakdown.optimizer_states + breakdown.gradients
            + breakdown.activations + breakdown.cuda_overhead
        )
        self.assertEqual(breakdown.total, parts_sum)

    def test_larger_batch_increases_total(self):
        small = TrainingVramConfig(
            training_method = "qlora", load_in_4bit = True, batch_size = 1,
        )
        large = TrainingVramConfig(
            training_method = "qlora", load_in_4bit = True, batch_size = 8,
        )
        small_v = estimate_training_vram(LLAMA_8B, small)
        large_v = estimate_training_vram(LLAMA_8B, large)
        self.assertGreater(large_v.total, small_v.total)

    def test_adamw_fp32_uses_more_optimizer_memory(self):
        opt8 = TrainingVramConfig(
            training_method = "full", load_in_4bit = False, optimizer = "adamw_8bit",
        )
        opt32 = TrainingVramConfig(
            training_method = "full", load_in_4bit = False, optimizer = "adamw_torch",
        )
        v8 = estimate_training_vram(LLAMA_8B, opt8)
        v32 = estimate_training_vram(LLAMA_8B, opt32)
        self.assertAlmostEqual(v32.optimizer_states / v8.optimizer_states, 1.5, delta = 0.1)


if __name__ == "__main__":
    unittest.main()
