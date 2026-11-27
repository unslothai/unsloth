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
from transformers.core_model_loading import ConversionOps, WeightConverter, WeightRenaming
from transformers.integrations.gguf.gguf_conversion_mapping import (
    DENSE_DECODER_RENAMINGS,
    ConcatenateRows,
    PermuteRows,
    SubtractOne,
    Unsqueeze,
)

__all__ = ["EXTRA_GGUF_ARCHS", "EXTRA_GGUF_CONFIG_ARCHS"]

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


# ----------------------------------------------------------------------------------------------------------
# Inkling 2 (`inkling2` = Small, `inkling2_loop` = Mini): llama.cpp's conversion/inkling2.py writes the text
# backbone under ggml leaf names. Beyond the renames it changes four things, each undone here on the values it
# stores (every one of them F32 in the file, or rows only, so the quantized linears and expert stacks stay packed):
#   1. the dynamic short convolutions of the full-attention layers store `static_weight` (K, C) as its flipped
#      transpose; the static ones store `conv1d.weight` (C, 1, K) squeezed to (C, K);
#   2. `relative_position_bias.proj` (d_rel, extent) is stored transposed;
#   3. the unembedding keeps the padded vocabulary rows, which transformers' lm_head drops;
#   4. the loop model's message queries, one (L, 4, H) parameter, are stored per loop layer as (4, H).
# ----------------------------------------------------------------------------------------------------------


def _one(input_dict):
    tensor = next(iter(input_dict.values()))
    return tensor[0] if isinstance(tensor, list) else tensor


class _TransposeFlip(ConversionOps):
    """`t.T.flip(0)`: undoes the `flip(0).T` llama.cpp applies to a dynamic short convolution's static kernel."""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return {target_patterns[0]: _one(input_dict).t().flip(0).contiguous()}


class _Transpose(ConversionOps):
    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return {target_patterns[0]: _one(input_dict).t().contiguous()}


class _KeepRows(ConversionOps):
    """The first `rows` rows; safe on packed blocks (a block never spans two rows)."""

    supports_packed = True

    def __init__(self, rows):
        self.rows = int(rows)

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        return {target_patterns[0]: _one(input_dict)[: self.rows].contiguous()}


class _StackSources(ConversionOps):
    """One tensor per source pattern, stacked on a new leading axis in pattern order."""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensors = []
        for pattern in source_patterns:
            value = input_dict[pattern]
            tensors.append(value[0] if isinstance(value, list) else value)
        return {target_patterns[0]: torch.stack(tensors, dim = 0).contiguous()}


def _inkling2(config):
    text = _text(config)
    renamings = [
        WeightRenaming(r"^token_embd_norm\.", "model.embed_tokens.embed_norm."),
        WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
        WeightRenaming(r"^output_norm\.", "model.norm."),
        WeightRenaming(r"^output_res_query\.weight", "model.final_input.query"),
        WeightRenaming(r"^blk\.", "model.layers."),
        WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj.linear."),
        WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj.linear."),
        WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj.linear."),
        WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj.linear."),
        WeightRenaming(r"\.attn_gate\.", ".self_attn.o_proj.gate_proj."),
        WeightRenaming(r"\.attn_sink_proj\.", ".self_attn.sink_proj."),
        WeightRenaming(r"\.attn_rel\.", ".self_attn.relative_position_bias.linear."),
        WeightRenaming(r"\.attn_q_clip\.weight", ".self_attn.q_proj.norm.scale"),
        WeightRenaming(r"\.attn_rel_clip\.weight", ".self_attn.relative_position_bias.norm.scale"),
        WeightRenaming(r"\.attn_k_conv_proj\.", ".self_attn.k_proj.sconv.weight_proj."),
        WeightRenaming(r"\.attn_v_conv_proj\.", ".self_attn.v_proj.sconv.weight_proj."),
        WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
        WeightRenaming(r"\.attn_q_inp_norm\.", ".q_input_layernorm."),
        WeightRenaming(r"\.attn_k_inp_norm\.", ".k_input_layernorm."),
        WeightRenaming(r"\.attn_v_inp_norm\.", ".v_input_layernorm."),
        WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
        WeightRenaming(r"\.attn_res_query\.weight", ".attention_input.query"),
        WeightRenaming(r"\.ffn_res_query\.weight", ".mlp_input.query"),
        WeightRenaming(r"\.ffn_gate_inp\.", ".mlp.gate."),
        WeightRenaming(r"\.exp_probs_b\.bias", ".mlp.gate.e_score_correction_bias"),
        WeightRenaming(r"\.ffn_gate_up_exps\.weight", ".mlp.experts.gate_up_proj"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
    ]
    converters = [
        WeightConverter(
            source_patterns = r"^output\.weight",
            target_patterns = "lm_head.weight",
            operations = [_KeepRows(text.unpadded_vocab_size)],
        ),
        WeightConverter(
            source_patterns = r"\.attn_rel_bias\.weight",
            target_patterns = ".self_attn.relative_position_bias.proj",
            operations = [_Transpose()],
        ),
        WeightConverter(
            source_patterns = r"\.attn_post_conv\.weight",
            target_patterns = ".post_attention_convolution.conv1d.weight",
            operations = [Unsqueeze(1)],
        ),
        WeightConverter(
            source_patterns = r"\.ffn_post_conv\.weight",
            target_patterns = ".post_mlp_convolution.conv1d.weight",
            operations = [Unsqueeze(1)],
        ),
    ]
    # Sliding-window layers hold a static short convolution, the full-attention ones a dynamic one.
    for layer, kind in enumerate(text.layer_types):
        for proj in ("k", "v"):
            source = rf"\.layers\.{layer}\.attn_{proj}_conv\.weight"
            if kind == "hybrid_sliding":
                target, operations = (
                    f".layers.{layer}.self_attn.{proj}_proj.sconv.conv1d.weight",
                    [Unsqueeze(1)],
                )
            else:
                target, operations = (
                    f".layers.{layer}.self_attn.{proj}_proj.sconv.static_weight",
                    [_TransposeFlip()],
                )
            converters.append(
                WeightConverter(
                    source_patterns = source, target_patterns = target, operations = operations
                )
            )
    return renamings + converters


def _inkling2_loop(config):
    text = _text(config)
    renamings = []
    for gguf_leaf, hf_leaf in (("q", "q"), ("k", "k"), ("v", "v"), ("ffn", "mlp")):
        renamings += [
            WeightRenaming(
                rf"\.loop_{gguf_leaf}_ctrl_down\.", f".{hf_leaf}_message_gate.down_proj."
            ),
            WeightRenaming(
                rf"\.loop_{gguf_leaf}_ctrl_forget\.", f".{hf_leaf}_message_gate.forget_gate_proj."
            ),
            WeightRenaming(
                rf"\.loop_{gguf_leaf}_ctrl_gain\.", f".{hf_leaf}_message_gate.message_gate_proj."
            ),
        ]
    loop_layers = range(text.loop_layer_index_start, text.loop_layer_index_end)
    queries = WeightConverter(
        source_patterns = [
            rf"^model\.layers\.{layer}\.loop_msg_query\.weight" for layer in loop_layers
        ],
        target_patterns = "model.message_input.query",
        operations = [_StackSources()],
    )
    return _inkling2(config) + renamings + [queries]


def _gguf_values(
    metadata,
    prefix,
    name,
    default = None,
):
    value = metadata.get(f"{prefix}.{name}", default)
    if value is None:
        raise KeyError(
            f"Unsloth: the GGUF has no `{prefix}.{name}`, which the Inkling 2 config needs."
        )
    return value


def _per_layer(value, layers):
    return list(value) if isinstance(value, (list, tuple)) else [value] * layers


def _inkling2_config(
    metadata,
    tensor_names,
    prefix = "inkling2",
):
    """transformers' Inkling 2 text config from the GGUF metadata llama.cpp writes (conversion/inkling2.py)."""
    get = lambda name, default = None: _gguf_values(metadata, prefix, name, default)  # noqa: E731
    layers = get("block_count")
    sliding = [bool(v) for v in _per_layer(get("attention.sliding_window_pattern"), layers)]
    heads = _per_layer(get("attention.head_count"), layers)
    kv_heads = _per_layer(get("attention.head_count_kv"), layers)

    def pick(values, want_sliding):
        chosen = {v for v, s in zip(values, sliding) if s == want_sliding}
        if len(chosen) > 1:
            raise NotImplementedError(
                f"Unsloth: Inkling 2 GGUF with mixed per-layer head counts {values}."
            )
        return chosen.pop() if chosen else values[0]

    if int(metadata.get(f"{prefix}.expert_gating_func", 2)) != 2:
        raise NotImplementedError(
            "Unsloth: Inkling 2 GGUFs route with sigmoid scores; this file does not."
        )
    head_dim = get("attention.key_length")
    config = {
        "model_type": f"{prefix}_text",
        "architectures": [
            "Inkling2LoopForCausalLM" if prefix == "inkling2_loop" else "Inkling2ForCausalLM"
        ],
        "vocab_size": get("vocab_size"),
        "unpadded_vocab_size": get("vocab_size_unpadded"),
        "hidden_size": get("embedding_length"),
        "num_hidden_layers": layers,
        "num_attention_heads": pick(heads, False),
        "num_key_value_heads": pick(kv_heads, False),
        "head_dim": head_dim,
        "swa_num_attention_heads": pick(heads, True),
        "swa_num_key_value_heads": pick(kv_heads, True),
        "swa_head_dim": get("attention.key_length_swa", head_dim),
        "sliding_window_size": get("attention.sliding_window"),
        "layer_types": ["hybrid_sliding" if s else "hybrid" for s in sliding],
        "d_rel": get("attention.relative_length"),
        "rel_extent": get("attention.relative_extent"),
        "log_scaling_n_floor": get("attention.log_scale_floor"),
        "log_scaling_alpha": get("attention.log_scale_alpha"),
        "qk_l2_clip_threshold": get("attention.qk_clip"),
        "rel_l2_clip_threshold": get("attention.relative_clip"),
        "max_position_embeddings": get("context_length"),
        "rms_norm_eps": get("attention.layer_norm_rms_epsilon"),
        "conv_kernel_size": get("shortconv.l_cache"),
        "n_layers_per_attn_res_block": get("attn_res.block_size"),
        "n_routed_experts": get("expert_count"),
        "num_experts_per_tok": get("expert_used_count"),
        "moe_intermediate_size": get("expert_feed_forward_length"),
        "route_scale": get("expert_weights_scale"),
        "mlp_gate_softcap": get("activation.situ_beta"),
        "mlp_up_softcap": get("activation.situ_linear_beta"),
        "final_logit_softcapping": get("final_logit_softcapping"),
        # lm_head drops the padded vocabulary rows, so it is never the embedding; the tensor names may not be at
        # hand either (the first shard of a split file holds metadata only).
        "tie_word_embeddings": False,
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }
    if prefix == "inkling2_loop":
        if metadata.get("inkling2_loop.loop.varying", False):
            raise NotImplementedError(
                "Unsloth: variable-pass Inkling 2 loop GGUFs are not supported."
            )
        config |= {
            "loop_layer_index_start": get("loop.start"),
            "loop_layer_index_end": get("loop.end"),
            "loop_num_passes": get("loop.count"),
            "loop_gate_rank": get("loop.gate_rank"),
        }
    return config


def _inkling2_loop_config(metadata, tensor_names):
    return _inkling2_config(metadata, tensor_names, prefix = "inkling2_loop")


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
    "inkling2": _inkling2,
    "inkling2_loop": _inkling2_loop,
}

# gguf `general.architecture` -> (metadata, tensor names) -> config dict, for files transformers' GGUF config reader
# does not know.
EXTRA_GGUF_CONFIG_ARCHS = {
    "inkling2": _inkling2_config,
    "inkling2_loop": _inkling2_loop_config,
}
