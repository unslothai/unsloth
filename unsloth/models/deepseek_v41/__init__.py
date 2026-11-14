# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Native DeepSeek-V4.1 (``deepseek_v41`` / ``deepseek_v41_text``) for transformers releases that do not ship it.

``register_deepseek_v41()`` registers the configs with ``AutoConfig`` and the models with ``AutoModel`` /
``AutoModelForCausalLM`` (idempotent), plus, on transformers 5 (``core_model_loading``), the renaming table that
loads the released checkpoint (DeepSeek-native key names, per-expert weights) into this layout. When transformers
itself provides ``transformers.models.deepseek_v41`` nothing is registered and that implementation is used.
"""

import importlib.util

__all__ = [
    "DEEPSEEK_V41_MODEL_TYPES",
    "register_deepseek_v41",
    "transformers_has_deepseek_v41",
]

DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
_REGISTERED = False


def transformers_has_deepseek_v41():
    try:
        return importlib.util.find_spec("transformers.models.deepseek_v41") is not None
    except Exception:
        return False


def _conversion_rules():
    """The released checkpoint's names -> this module tree (relative to the text model). Same table as
    transformers' deepseek_v4 entry plus the V4.1 parts (compressor / indexer / engram). No ``$`` anchors: the
    table also runs in reverse on save, and a reversed ``$`` would be written into the key."""
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
    )

    R = WeightRenaming
    return [
        R(source_patterns = r"^layers\.(\d+)\.engram\.embed\.", target_patterns = r"engram_tables.\1."),
        R(source_patterns = r"\.attn\.", target_patterns = r".self_attn."),
        R(source_patterns = r"\.ffn\.", target_patterns = r".mlp."),
        R(source_patterns = r"^embed\.weight", target_patterns = "embed_tokens.weight"),
        R(source_patterns = r"^head\.weight", target_patterns = "lm_head.weight"),
        R(source_patterns = r"\.attn_norm\.", target_patterns = r".input_layernorm."),
        R(source_patterns = r"\.ffn_norm\.", target_patterns = r".post_attention_layernorm."),
        R(source_patterns = r"\.hc_attn_fn", target_patterns = r".attn_hc.fn"),
        R(source_patterns = r"\.hc_attn_base", target_patterns = r".attn_hc.base"),
        R(source_patterns = r"\.hc_attn_scale", target_patterns = r".attn_hc.scale"),
        R(source_patterns = r"\.hc_ffn_fn", target_patterns = r".ffn_hc.fn"),
        R(source_patterns = r"\.hc_ffn_base", target_patterns = r".ffn_hc.base"),
        R(source_patterns = r"\.hc_ffn_scale", target_patterns = r".ffn_hc.scale"),
        R(source_patterns = r"\.attn_sink", target_patterns = r".sinks"),
        R(source_patterns = r"\.wq_a\.", target_patterns = r".q_a_proj."),
        R(source_patterns = r"\.q_norm\.", target_patterns = r".q_a_norm."),
        R(source_patterns = r"\.wq_b\.", target_patterns = r".q_b_proj."),
        R(source_patterns = r"\.self_attn\.wkv\.", target_patterns = r".self_attn.kv_proj."),
        R(source_patterns = r"\.compressor\.wkv\.", target_patterns = r".compressor.kv_proj."),
        R(source_patterns = r"\.compressor\.wgate\.", target_patterns = r".compressor.gate_proj."),
        R(source_patterns = r"\.compressor\.norm\.", target_patterns = r".compressor.kv_norm."),
        R(source_patterns = r"\.indexer\.wk\.", target_patterns = r".indexer.k_proj."),
        R(source_patterns = r"\.wo_a\.", target_patterns = r".o_a_proj."),
        R(source_patterns = r"\.wo_b\.", target_patterns = r".o_b_proj."),
        R(source_patterns = r"\.gate\.bias_vl", target_patterns = r".gate.e_score_correction_bias_vl"),
        R(source_patterns = r"\.gate\.bias", target_patterns = r".gate.e_score_correction_bias"),
        R(source_patterns = r"\.shared_experts\.w1\.", target_patterns = r".shared_experts.gate_proj."),
        R(source_patterns = r"\.shared_experts\.w2\.", target_patterns = r".shared_experts.down_proj."),
        R(source_patterns = r"\.shared_experts\.w3\.", target_patterns = r".shared_experts.up_proj."),
        WeightConverter(
            source_patterns = [r"\.experts.*.w1.weight", r"\.experts.*.w3.weight"],
            target_patterns = r"\.experts.gate_up_proj",
            operations = [MergeModulelist(dim = 0), Concatenate(dim = 1)],
        ),
        WeightConverter(
            source_patterns = r"\.experts.*.w2.weight",
            target_patterns = r"\.experts.down_proj",
            operations = [MergeModulelist(dim = 0)],
        ),
    ]


def _register_conversions():
    try:
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    except Exception:
        return False
    try:
        rules = _conversion_rules()
    except Exception:
        return False
    for key in (
        "deepseek_v41",
        "deepseek_v41_text",
        "DeepseekV41ForCausalLM",
        "DeepseekV41TextModel",
    ):
        register_checkpoint_conversion_mapping(key, list(rules), overwrite = True)
    return True


def register_deepseek_v41():
    """Register the native classes with the Auto factories; returns (config_class, model_class) or None when
    transformers ships its own deepseek_v41."""
    global _REGISTERED
    if transformers_has_deepseek_v41():
        return None
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from .configuration import DeepseekV41Config, DeepseekV41TextConfig
    from .modeling import DeepseekV41ForCausalLM, DeepseekV41TextModel

    if not _REGISTERED:
        for model_type, config_class in (
            ("deepseek_v41", DeepseekV41Config),
            ("deepseek_v41_text", DeepseekV41TextConfig),
        ):
            try:
                AutoConfig.register(model_type, config_class)
            except ValueError:
                pass
        for auto, config_class, model_class in (
            (AutoModelForCausalLM, DeepseekV41Config, DeepseekV41ForCausalLM),
            (AutoModel, DeepseekV41TextConfig, DeepseekV41TextModel),
        ):
            try:
                auto.register(config_class, model_class)
            except ValueError:
                pass
        _register_conversions()
        _REGISTERED = True
    return DeepseekV41Config, DeepseekV41ForCausalLM
