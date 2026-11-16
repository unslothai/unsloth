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

"""DeepSeek-V4.1 configs. Field names are the released config.json's (deepseek-ai/DeepSeek-V4.1-Flash-1017);
defaults are the released shapes. Plain ``__init__`` classes so they load on transformers 4.57 and 5.x alike.

RoPE is kept in our own fields (``rope_theta``, ``compress_rope_theta``, ``rope_yarn_*``) rather than
``rope_scaling`` / ``rope_parameters``, whose handling differs between transformers versions. The released
``rope_scaling`` (YaRN) is read once in ``__init__``.
"""

from transformers.configuration_utils import PretrainedConfig

__all__ = ["DeepseekV41TextConfig", "DeepseekV41Config"]


def _read_yarn(rope_scaling, rope_parameters):
    """(factor, original_max_position_embeddings, beta_fast, beta_slow) from either spelling."""
    src = rope_scaling if isinstance(rope_scaling, dict) else None
    if src is None and isinstance(rope_parameters, dict):
        src = rope_parameters
    if not src:
        return None
    return (
        float(src.get("factor", 1.0)),
        int(src.get("original_max_position_embeddings", 0) or 0),
        float(src.get("beta_fast", 32)),
        float(src.get("beta_slow", 1)),
    )


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    # TRL / MoE helpers read num_experts; transformers' MoE code reads num_local_experts.
    attribute_map = {"num_experts": "n_routed_experts", "num_local_experts": "n_routed_experts"}
    # TP: query heads sharded, the single shared KV head replicated, o_a_proj sharded by whole groups so o_b_proj
    # reduces rowwise. Needs num_attention_heads % tp == 0 and o_groups % tp == 0.
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.self_attn.o_a_proj": "colwise",
        "layers.*.self_attn.o_b_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    # PP: KV / index sources share group state with later layers, so stages start only at
    # modeling.pipeline_safe_cut_points(config).
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "pre_mix"], ["hidden_states", "pre_mix"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size = 129280,
        hidden_size = 5120,
        moe_intermediate_size = 2304,
        num_hidden_layers = 40,
        num_attention_heads = 64,
        num_key_value_heads = 1,
        head_dim = 512,
        qk_rope_head_dim = 64,
        q_lora_rank = 1280,
        o_lora_rank = 1024,
        o_groups = 8,
        hidden_act = "silu",
        swiglu_limit = 10.0,
        rms_norm_eps = 1e-20,
        attention_bias = False,
        attention_dropout = 0.0,
        initializer_range = 0.02,
        use_cache = True,
        tie_word_embeddings = False,
        max_position_embeddings = 1048576,
        rope_theta = 10000.0,
        compress_rope_theta = 160000.0,
        rope_scaling = None,
        rope_parameters = None,
        rope_yarn_factor = None,
        rope_yarn_original_max_position_embeddings = None,
        rope_yarn_beta_fast = None,
        rope_yarn_beta_slow = None,
        n_routed_experts = 384,
        n_shared_experts = 1,
        num_experts_per_tok = 6,
        scoring_func = "sqrtsoftplus",
        topk_method = "noaux_tc",
        norm_topk_prob = True,
        routed_scaling_factor = 1.5,
        gate_temperature = 1.0,
        sliding_window = 128,
        compress_ratios = None,
        kv_source_layer_ids = (2, 8, 14, 20),
        index_source_layer_ids = (2, 8, 14, 20, 24, 28, 32, 36),
        index_n_heads = 32,
        index_head_dim = 128,
        index_topk = 512,
        candidate_source_layer_id = 20,
        candidate_topk_blocks = 2048,
        candidate_block_size = 8,
        hc_mult = 4,
        hc_sinkhorn_iters = 20,
        hc_eps = 1e-6,
        engram_layer_ids = (1, 14),
        engram_num_embeddings = (384006168, 384016682),
        engram_max_ngram_size = 4,
        engram_vocab_size = 16000000,
        engram_n_heads = 8,
        engram_head_dim = 256,
        engram_pad_token_id = 2,
        engram_compressed_vocab_size = 99092,
        engram_block_size = 32,
        num_nextn_predict_layers = 3,
        dspark_block_size = 5,
        dspark_noise_token_id = 128799,
        dspark_target_layer_ids = (37, 38, 39),
        dspark_markov_rank = 256,
        dspark_n_routed_experts = 128,
        dspark_num_experts_per_tok = 3,
        image_token_id = 129264,
        pad_token_id = 2,
        bos_token_id = 0,
        eos_token_id = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = float(rope_theta)
        self.compress_rope_theta = float(compress_rope_theta)
        yarn = _read_yarn(rope_scaling, rope_parameters)
        if yarn is None:
            yarn = (16.0, 65536, 32.0, 1.0)
        self.rope_yarn_factor = float(rope_yarn_factor if rope_yarn_factor is not None else yarn[0])
        self.rope_yarn_original_max_position_embeddings = int(
            rope_yarn_original_max_position_embeddings
            if rope_yarn_original_max_position_embeddings is not None
            else yarn[1]
        )
        self.rope_yarn_beta_fast = float(
            rope_yarn_beta_fast if rope_yarn_beta_fast is not None else yarn[2]
        )
        self.rope_yarn_beta_slow = float(
            rope_yarn_beta_slow if rope_yarn_beta_slow is not None else yarn[3]
        )
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.gate_temperature = gate_temperature
        self.sliding_window = sliding_window
        if compress_ratios is None:
            compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * num_nextn_predict_layers
        self.compress_ratios = list(compress_ratios)
        self.kv_source_layer_ids = list(kv_source_layer_ids)
        self.index_source_layer_ids = list(index_source_layer_ids)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = (
            -1 if candidate_source_layer_id is None else candidate_source_layer_id
        )
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = list(engram_layer_ids or ())
        self.engram_num_embeddings = list(engram_num_embeddings or ())
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.engram_block_size = engram_block_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = list(dspark_target_layer_ids or ())
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        self.image_token_id = image_token_id
        if len(self.compress_ratios) < num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries for {num_hidden_layers} layers."
            )
        _drop_unloadable_fp8_config(kwargs)
        super().__init__(
            pad_token_id = pad_token_id,
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            tie_word_embeddings = tie_word_embeddings,
            **kwargs,
        )


def _drop_unloadable_fp8_config(kwargs):
    """The release's FP8 quantization_config (ue8m0 32x32 blocks, fp4 experts) on a transformers whose FP8 quantizer
    cannot read it: those versions load the release through convert.py, dequantized, so the config must not ask
    for FP8 (or make Unsloth's loader treat the model as pre-quantized and skip a requested 4-bit load)."""
    qc = kwargs.get("quantization_config")
    if qc is None:
        return
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if str(getattr(method, "value", method)).lower() != "fp8":
        return
    from .convert import released_needs_conversion

    if released_needs_conversion():
        kwargs.pop("quantization_config")


class DeepseekV41Config(PretrainedConfig):
    """The released composite config: ``text_config`` drives the language model; ``vision_config`` is kept as a
    plain dict (the vision tower is not built by the text CausalLM)."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config = None,
        vision_config = None,
        image_token_id = 129264,
        tie_word_embeddings = False,
        **kwargs,
    ):
        if text_config is None:
            # A flat text config (saved from a model built on DeepseekV41TextConfig): its fields are the text config.
            import inspect

            fields = set(inspect.signature(DeepseekV41TextConfig.__init__).parameters) - {
                "self",
                "kwargs",
            }
            flat = {
                k: kwargs.pop(k) for k in list(kwargs) if k in fields and k != "tie_word_embeddings"
            }
            text_config = DeepseekV41TextConfig(**flat)
        elif isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(
                **{k: v for k, v in text_config.items() if k != "model_type"}
            )
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        _drop_unloadable_fp8_config(kwargs)
        super().__init__(tie_word_embeddings = tie_word_embeddings, **kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config

    def __getattr__(self, name):
        # Model code and Unsloth helpers read text fields (hidden_size, num_hidden_layers, ...) off the top config.
        if name.startswith("_") or name in ("text_config", "vision_config"):
            raise AttributeError(name)
        text = self.__dict__.get("text_config")
        if text is not None and hasattr(text, name):
            return getattr(text, name)
        raise AttributeError(name)
