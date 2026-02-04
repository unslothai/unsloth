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
from ._utils import __version__
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralSparseMoeBlock,
    MixtralDecoderLayer,
    MixtralModel,
    MixtralForCausalLM,
)
from ..kernels.moe.grouped_gemm.interface import grouped_gemm
from ..kernels.moe.grouped_gemm.reference.moe_ops import (
    calculate_topk,
    get_routing_indices,
)

def MixtralSparseMoeBlock_fast_forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Router
    router_logits = self.gate(hidden_states)
    
    # Calculate topk and routing weights
    routing_weights, selected_experts = calculate_topk(
        router_logits,
        top_k = self.top_k,
        use_sigmoid = False,
        renormalize = True,
    )
    routing_weights = routing_weights.to(hidden_states.dtype)

    # Get routing indices for grouped gemm
    token_counts_by_expert, gather_indices = get_routing_indices(
        selected_experts, self.num_experts
    )

    # If weights aren't cached, stack them. 
    # In practice, Unsloth should do this during patching.
    if not hasattr(self, "gate_up_proj"):
        # This is a fallback if pre-patching didn't happen correctly
        expert_w1 = torch.stack([expert.w1.weight for expert in self.experts]) # [E, N, K]
        expert_w3 = torch.stack([expert.w3.weight for expert in self.experts]) # [E, N, K]
        expert_w2 = torch.stack([expert.w2.weight for expert in self.experts]) # [E, K, N]
        
        # Fuse w1 and w3 for efficiency in the first grouped gemm
        self.gate_up_proj = torch.cat([expert_w1, expert_w3], dim = 1) # [E, 2N, K]
        self.down_proj = expert_w2 # [E, K, N]

    # First GEMM: Gate & Up Projection
    # M_total = batch * seq * topk
    inter_states = grouped_gemm(
        X = hidden_states,
        W = self.gate_up_proj,
        m_sizes = token_counts_by_expert,
        gather_indices = gather_indices,
        topk = self.top_k,
        permute_x = True,
        permute_y = False,
    )

    # Activation (SiLU) and Gating
    # Split into gate and up
    gate, up = inter_states.chunk(2, dim = -1)
    inter_states = F.silu(gate) * up

    # Second GEMM: Down Projection
    final_states = grouped_gemm(
        X = inter_states,
        W = self.down_proj,
        m_sizes = token_counts_by_expert,
        gather_indices = gather_indices,
        topk = self.top_k,
        permute_x = False,
        permute_y = True,
    )

    # Merge topk weights
    final_states = (
        final_states.view(batch_size * sequence_length, self.top_k, hidden_dim)
        * routing_weights[..., None]
    )
    final_states = final_states.sum(dim = 1)

    return final_states.view(batch_size, sequence_length, hidden_dim), router_logits

class FastMixtralModel(FastLlamaModel):
    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name = "mixtral",
            rope_module = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module = MixtralAttention,
        )
        if init_name is not None:
            exec(function, globals())
            MixtralAttention.__init__ = eval(init_name)
        
        MixtralAttention.forward = LlamaAttention_fast_forward
        # Mixtral specific patches
        MixtralSparseMoeBlock.forward = MixtralSparseMoeBlock_fast_forward
        MixtralDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        MixtralModel.forward = LlamaModel_fast_forward
        MixtralForCausalLM.forward = CausalLM_fast_forward(
            LlamaModel_fast_forward_inference
        )
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(MixtralForCausalLM)

        # Patch rotary embedding
        import transformers.models.mixtral.modeling_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        return

    @staticmethod
    def from_pretrained(
        model_name = "mistralai/Mixtral-8x7B-v0.1",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
        token = None,
        device_map = "sequential",
        rope_scaling = None,
        fix_tokenizer = True,
        model_patcher = None,
        tokenizer_name = None,
        trust_remote_code = False,
        cfg_model_name = None,
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
            model_patcher = FastMixtralModel,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            cfg_model_name = cfg_model_name,
            **kwargs,
        )
