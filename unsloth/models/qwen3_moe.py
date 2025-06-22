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
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from .qwen3 import (
    Qwen3Attention_fast_forward,
    FastQwen3Model,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
)
# For Pytorch 2.1.1
# TODO: Transformers moved to `attention_interface`. So we might not need these anymore
# try:
#     from transformers.models.qwen3_moe.modeling_qwen3_moe import (
#         Qwen3SdpaAttention,
#         Qwen3FlashAttention2,
#     )
# except:
#     Qwen3SdpaAttention   = Qwen3Attention
#     Qwen3FlashAttention2 = Qwen3Attention
# pass
from unsloth_zoo.utils import Version, _get_dtype


torch_nn_functional_softmax = torch.nn.functional.softmax
def Qwen3MoeSparseMoeBlock_fast_forward(self, X, temp_gate = None, temp_up = None):
    # adapted from https://github.com/huggingface/transformers/pull/36878/files#diff-0855b77fc27ad9449158a1c74953f909b011c00de7125f7c8e68d0ff209c092aR356-R370
    
    bsz, seq_len, hd = X.shape
    X = X.view(-1, hd)

    router_logits = fast_linear_forward(self.gate_proj, X, out = temp_gate) #pretty much the only change from transformers implementation.

    routing_weights = torch_nn_functional_softmax(router_logits, dim = -1)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(X.dtype)
    final_X = torch.zeros(
        (bsz * seq_len, hd), dtype=X.dtype, device=X.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = X[None, top_x].reshape(-1, hd)
        current_X = expert_layer(current_state) * routing_weights[top_x, idx, None] # Qwen3MoeMLP.forward = fast_swiglu_inference takes care of making this faster. Analogous to Dense models' MLP

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_X.index_add_(0, top_x, current_X.to(X.dtype))
    final_X = final_X.reshape(bsz, seq_len, hd)
    return final_X, router_logits
pass


def Qwen3MoeDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    output_router_logits:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    position_embeddings:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
):
    residual = hidden_states

    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings = position_embeddings,
            _flag_for_generation=self._flag_for_generation,
        )
        hidden_states = residual + hidden_states

        # MoE Router MLP
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(self.post_attention_layernorm, hidden_states)
        hidden_states, router_logits = Qwen3MoeSparseMoeBlock_fast_forward(self.mlp, hidden_states)
        hidden_states = residual + hidden_states
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MoE Router MLP
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if output_router_logits: outputs += (router_logits,)
    if use_cache: outputs += (present_key_value,)
    return outputs



class FastQwen3MoeModel(FastQwen3Model):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "Qwen3Moe",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = Qwen3MoeAttention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen3MoeAttention.__init__  = eval(init_name)
        pass
        Qwen3MoeAttention      .forward = Qwen3Attention_fast_forward
        # Qwen3SdpaAttention   .forward = Qwen3Attention_fast_forward
        # Qwen3FlashAttention2 .forward = Qwen3Attention_fast_forward
        Qwen3MoeSparseMoeBlock .forward = Qwen3MoeSparseMoeBlock_fast_forward
        Qwen3MoeMLP            .forward = fast_swiglu_inference # This is analogous to Dense models' MLP
        Qwen3MoeDecoderLayer   .forward = Qwen3MoeDecoderLayer_fast_forward
        Qwen3MoeModel          .forward = LlamaModel_fast_forward
        Qwen3MoeForCausalLM    .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(Qwen3MoeForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py\
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.Qwen3Moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding = LlamaRotaryEmbedding
        return
    pass


    @staticmethod
    def from_pretrained(  #TODO: Change after release
        model_name        = "Qwen/Qwen3-7B",
        max_seq_length    = 4096,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None,
        fix_tokenizer     = True,
        model_patcher     = None,
        tokenizer_name    = None,
        trust_remote_code = False,
        **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastQwen3Model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
    pass
pass