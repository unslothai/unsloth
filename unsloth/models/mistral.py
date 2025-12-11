# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .llama import *
import os
from ._utils import __version__
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.hf_utils import dtype_from_config
from ..utils.packing import (
    get_packed_info_from_kwargs,
    mask_packed_sequence_boundaries,
)
from ..utils.attention_dispatch import (
    AttentionConfig,
    AttentionContext,
    run_attention,
    select_attention_backend,
)
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
)

# For Pytorch 2.1.1
try:
    from transformers.models.mistral.modeling_mistral import (
        MistralSdpaAttention,
        MistralFlashAttention2,
    )
except:
    MistralSdpaAttention = MistralAttention
    MistralFlashAttention2 = MistralAttention
from unsloth_zoo.utils import Version, _get_dtype


def MistralAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention

    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    seq_info = get_packed_info_from_kwargs(kwargs, Q.device)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Extend RoPE dynamically to fit in VRAM
    self.rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)
    cos, sin = self.rotary_emb.get_cached(kv_seq_len, Q.device.index)

    rope_position_ids = (
        position_ids if position_ids is not None else kwargs.get("position_ids")
    )
    # Useful for LongRoPE
    Q, K = fast_rope_embedding(Q, K, cos, sin, rope_position_ids)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    past_key_value = (K, V) if use_cache else None

    # Attention module
    sw_cfg = getattr(self.config, "sliding_window", None)
    sw = kv_seq_len if (sw_cfg is None or sw_cfg == "null") else sw_cfg
    window_size = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)

    use_varlen = (
        seq_info is not None and past_key_value is None and window_size == (-1, -1)
    )
    backend = select_attention_backend(use_varlen)
    attention_config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_groups,
        flash_dense_kwargs = {"causal": True, "window_size": window_size},
        flash_varlen_kwargs = {
            "dropout_p": 0.0,
            "causal": True,
            "softmax_scale": getattr(self, "softmax_scale", None),
        },
    )
    context = AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = kv_seq_len,
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = hidden_states.requires_grad,
        seq_info = seq_info,
        attention_mask = attention_mask,
        causal_mask = causal_mask,
    )

    A = run_attention(config = attention_config, context = context, Q = Q, K = K, V = V)
    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


def MistralForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    num_logits_to_keep: Optional[int] = 0,
    logits_to_keep: Optional[int] = 0,
    *args,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if causal_mask is None and past_key_values is None:
        bsz, q_len = input_ids.shape
        sliding_window = getattr(self.config, "sliding_window", None)

        if HAS_XFORMERS:
            # Always create causal mask for xformers
            if (
                sliding_window is None
                or sliding_window == "null"
                or sliding_window <= 0
            ):
                causal_mask = xformers.attn_bias.LowerTriangularMask()
            elif q_len <= sliding_window:
                causal_mask = xformers.attn_bias.LowerTriangularMask()
            else:
                causal_mask = xformers.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                    [q_len] * bsz
                ).make_local_attention(window_size = sliding_window)

            # If attention_mask exists, it will be handled in the attention forward

        else:
            # Not using xformers - need to create attention masks
            if (
                sliding_window is None
                or sliding_window == "null"
                or sliding_window <= 0
                or q_len <= sliding_window
            ):
                # Fully causal mask
                causal_mask_values = torch.triu(
                    torch.full((q_len, q_len), -torch.inf, device = input_ids.device),
                    diagonal = 1,
                )
            else:
                # Sliding window attention
                q_indices = torch.arange(q_len, device = input_ids.device).view(-1, 1)
                k_indices = torch.arange(q_len, device = input_ids.device).view(1, -1)

                causal_bool_mask = k_indices <= q_indices
                window_bool_mask = (q_indices - k_indices) < sliding_window

                causal_mask_values = torch.where(
                    causal_bool_mask & window_bool_mask, 0.0, -torch.inf
                )

            # Combine with existing attention_mask if present
            if attention_mask is None:
                attention_mask = causal_mask_values[None, None, :, :].expand(
                    bsz, 1, q_len, q_len
                )
            else:
                # attention_mask should be [bsz, 1, q_len, q_len] or broadcastable
                # Add causal mask to existing attention mask
                if attention_mask.dim() == 2:
                    # [bsz, seq_len] -> [bsz, 1, 1, seq_len]
                    attention_mask = attention_mask[:, None, None, :]
                    attention_mask = attention_mask.expand(bsz, 1, q_len, q_len)
                attention_mask = attention_mask + causal_mask_values[None, None, :, :]

            attention_mask = attention_mask.to(
                dtype = _get_dtype(dtype_from_config(self.config))
            )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    self.model._has_no_labels = labels is None

    if past_key_values is not None:
        outputs = LlamaModel_fast_forward_inference(
            self,
            input_ids,
            past_key_values,
            position_ids = position_ids,
            attention_mask = attention_mask,
        )
    else:
        outputs = self.model(
            input_ids = input_ids,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            **kwargs,
        )

    hidden_states = outputs[0]

    bsz, q_len, hd = hidden_states.shape
    lm_head = self.lm_head.weight
    lm_head_device = lm_head.device

    # Move items to same device as lm_head
    hidden_states = hidden_states.to(lm_head_device)
    if labels is not None:
        labels = labels.to(lm_head_device)

    # If we are in GRPO mode, return raw hidden states
    if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
        num_logits_to_keep = max(num_logits_to_keep, logits_to_keep)
        if num_logits_to_keep != 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        return CausalLMOutputWithPast(
            loss = None,
            logits = hidden_states,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )

    if bsz == 1 and q_len == 1:
        logits = torch.mv(lm_head, hidden_states.ravel().to(lm_head.dtype))
        logits = logits.unsqueeze(0).unsqueeze(0)
    elif num_logits_to_keep != 0:
        logits = self.lm_head(
            hidden_states[:, -num_logits_to_keep:, :].to(lm_head.dtype)
        )
    else:
        RETURN_LOGITS = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
        # < 1024 Normal Unsloth uses less VRAM!
        if bsz * q_len <= 1024 and not RETURN_LOGITS:
            # Use unsloth_fused_ce_loss which actually calculates the best chunk size to reduce VRAM usage
            RETURN_LOGITS = False

        if not RETURN_LOGITS and labels is not None:
            n_items = kwargs.get("num_items_in_batch", None) or kwargs.get(
                "n_items", None
            )
            logit_softcapping = getattr(self.config, "final_logit_softcapping", 0)

            # loss = fused_linear_cross_entropy(
            #     hidden_states = hidden_states,
            #     lm_weight = lm_head,
            #     labels = labels,
            #     num_items_in_batch = n_items,
            #     logit_softcapping = logit_softcapping,
            # )
            loss = unsloth_fused_ce_loss(
                trainer = None,
                hidden_states = hidden_states,
                lm_head_weight = lm_head,
                lm_head_bias = None,
                labels = labels,
                mask = None,
                n_items = n_items,
                scaling = getattr(self, "accelerator_scaler", None),
                target_gb = None,
                torch_compile = True,
                logit_softcapping = logit_softcapping,
            )
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            output = CausalLMOutputWithPast(
                loss = loss,
                logits = EMPTY_LOGITS,
                past_key_values = outputs.past_key_values,
                hidden_states = outputs.hidden_states,
                attentions = outputs.attentions,
            )
            return output
        pass
        logits = self.lm_head(hidden_states.to(lm_head.dtype))
    logits = logits.to(_get_dtype(dtype_from_config(self.config)))

    loss = None
    if labels is not None:
        shift_logits = logits
        # if not hasattr(self, "extra_ignored_labels"):
        #     # Fixes https://github.com/unslothai/unsloth/issues/10
        #     self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda:0")
        # pass
        # shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = -100
        mask_packed_sequence_boundaries(
            shift_labels,
            kwargs.get("packed_seq_lengths"),
        )
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
            n_items = kwargs.get("num_items_in_batch", None)
            or kwargs.get("n_items", None),
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss = loss,
        logits = logits,
        past_key_values = outputs.past_key_values,
        hidden_states = outputs.hidden_states,
        attentions = outputs.attentions,
    )


# Transformers had to update for Mistral Nemo 12b since Attention is (5120, 4096) now.
def patch_mistral_nemo_attention(function):
    function = function.replace(
        "(self.head_dim * self.config.num_attention_heads) != self.config.hidden_size",
        "False",
    )
    function = function.replace(
        "self.head_dim = self.config.hidden_size // self.config.num_attention_heads",
        "self.head_dim = config.head_dim",
    )
    function = function.replace(
        "self.o_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)",
        "self.o_proj = nn.Linear(self.config.num_attention_heads * self.head_dim, self.config.hidden_size, bias=False)",
    )
    return function


class FastMistralModel(FastLlamaModel):
    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name = "mistral",
            rope_module = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module = MistralAttention,
        )
        # Just for Mistral Nemo models!
        if function is not None and init_name is not None:
            function = patch_mistral_nemo_attention(function)
            # if True:#init_name is not None:
            exec(function, globals())
            MistralAttention.__init__ = eval(init_name)
        MistralAttention.forward = MistralAttention_fast_forward
        MistralSdpaAttention.forward = MistralAttention_fast_forward
        MistralFlashAttention2.forward = MistralAttention_fast_forward
        MistralDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        MistralModel.forward = LlamaModel_fast_forward
        MistralForCausalLM.forward = MistralForCausalLM_fast_forward
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(MistralForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inference can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.mistral.modeling_mistral

        transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        return

    @staticmethod
    def from_pretrained(
        model_name = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,  # Mistral does not support RoPE scaling
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = token,
            device_map = device_map,
            rope_scaling = rope_scaling,
            fix_tokenizer = fix_tokenizer,
            model_patcher = FastMistralModel,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
