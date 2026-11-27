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

"""GGUF -> transformers weight mappings for architectures transformers' GGUF quantizer does not map itself
(it maps qwen35, qwen35moe and lfm2), in its own WeightRenaming / WeightConverter terms. Each mirrors what
transformers' legacy loader does to the same file (`modeling_gguf_pytorch_utils` TENSOR_PROCESSORS), with
row permutations kept on the packed blocks. Imported only on transformers >= 5.18."""

import torch
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.integrations.gguf.gguf_conversion_mapping import (
    DENSE_DECODER_RENAMINGS,
    ConcatenateRows,
    PermuteRows,
    SubtractOne,
    Unsqueeze,
)

__all__ = ["EXTRA_GGUF_ARCHS"]

_NORMS = [
    WeightRenaming(r"^output_norm\.", "model.norm."),
    WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
]
_QK_NORMS = [
    WeightRenaming(r"\.attn_q_norm\.", ".self_attn.q_norm."),
    WeightRenaming(r"\.attn_k_norm\.", ".self_attn.k_norm."),
]


def _text(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _rope_unpermute(rows, n_head):
    """Row order that undoes llama.cpp's q / k interleave (legacy `_reverse_permute_weights`)."""
    dim = rows // n_head // 2
    return torch.arange(rows, device = "cpu").reshape(n_head, dim, 2).transpose(1, 2).reshape(-1)


def _llama(config):
    text = _text(config)
    heads, kv_heads = text.num_attention_heads, text.num_key_value_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // heads
    renamings = (
        DENSE_DECODER_RENAMINGS
        + _NORMS
        + [WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm.")]
    )
    permute = [
        WeightConverter(
            source_patterns = r"self_attn\.q_proj\.weight",
            target_patterns = "self_attn.q_proj.weight",
            operations = [PermuteRows(_rope_unpermute(heads * head_dim, heads))],
        ),
        WeightConverter(
            source_patterns = r"self_attn\.k_proj\.weight",
            target_patterns = "self_attn.k_proj.weight",
            operations = [PermuteRows(_rope_unpermute(kv_heads * head_dim, kv_heads))],
        ),
    ]
    return renamings + permute


def _plain_decoder(config):
    """qwen2 / mistral: llama.cpp only renames (no q / k interleave)."""
    return (
        DENSE_DECODER_RENAMINGS
        + _NORMS
        + [WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm.")]
    )


def _qwen3(config):
    return _plain_decoder(config) + _QK_NORMS


def _routed_experts(shared = False):
    renamings = [
        WeightRenaming(r"\.ffn_gate_inp\.", ".mlp.gate."),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
    ]
    converters = [
        # gate first, up second, as transformers chunks gate_up_proj.
        WeightConverter(
            source_patterns = [r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns = ".mlp.experts.gate_up_proj",
            operations = [ConcatenateRows(dim = 1)],
        )
    ]
    if shared:
        renamings += [
            WeightRenaming(r"\.ffn_gate_inp_shexp\.", ".mlp.shared_expert_gate."),
            WeightRenaming(r"\.ffn_gate_shexp\.", ".mlp.shared_expert.gate_proj."),
            WeightRenaming(r"\.ffn_up_shexp\.", ".mlp.shared_expert.up_proj."),
            WeightRenaming(r"\.ffn_down_shexp\.", ".mlp.shared_expert.down_proj."),
        ]
        converters.append(
            WeightConverter(
                source_patterns = "mlp.shared_expert_gate.weight",
                target_patterns = "mlp.shared_expert_gate.weight",
                operations = [Unsqueeze(0, expected_ndim = 2)],
            )
        )
    return renamings, converters


def _qwen3moe(config):
    renamings, converters = _routed_experts()
    return _qwen3(config) + renamings + converters


def _qwen2moe(config):
    renamings, converters = _routed_experts(shared = True)
    return _plain_decoder(config) + renamings + converters


def _gemma(config):
    """gemma2 / gemma3 text: llama.cpp stores every norm as `w + 1`."""
    renamings = DENSE_DECODER_RENAMINGS + [
        WeightRenaming(r"\.post_attention_norm\.", ".post_attention_layernorm."),
        WeightRenaming(r"\.ffn_norm\.", ".pre_feedforward_layernorm."),
        WeightRenaming(r"\.post_ffw_norm\.", ".post_feedforward_layernorm."),
    ]
    offset = [
        WeightConverter(
            source_patterns = r"^output_norm\.weight",
            target_patterns = "model.norm.weight",
            operations = [SubtractOne()],
        )
    ]
    for gguf_leaf, hf_leaf in (
        ("attn_norm", "input_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
    ):
        offset.append(
            WeightConverter(
                source_patterns = rf"\.{gguf_leaf}\.weight",
                target_patterns = f".{hf_leaf}.weight",
                operations = [SubtractOne()],
            )
        )
    for hf_leaf in (
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        offset.append(
            WeightConverter(
                source_patterns = rf"{hf_leaf}\.weight",
                target_patterns = f"{hf_leaf}.weight",
                operations = [SubtractOne()],
            )
        )
    return renamings + offset


# gguf `general.architecture` -> builder taking the model config
EXTRA_GGUF_ARCHS = {
    "llama": _llama,
    "mistral": _plain_decoder,
    "qwen2": _plain_decoder,
    "qwen3": _qwen3,
    "qwen2moe": _qwen2moe,
    "qwen3moe": _qwen3moe,
    "gemma2": _gemma,
    "gemma3": _gemma,
}
