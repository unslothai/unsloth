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
    fast_swiglu_inference,
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

    router_logits = fast_linear_forward(
        self.gate_proj, X, out = temp_gate
    )  # pretty much the only change from transformers implementation.

    routing_weights = torch_nn_functional_softmax(
        router_logits, dim = -1, dtype = torch.float32
    )
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim = -1)
    routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(X.dtype)
    final_X = torch.zeros((bsz * seq_len, hd), dtype = torch.float32, device = X.device)

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes = self.num_experts
    ).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = X[None, top_x].reshape(-1, hd)
        current_X = (
            expert_layer(current_state) * routing_weights[top_x, idx, None]
        )  # Qwen3MoeMLP.forward = fast_swiglu_inference takes care of making this faster. Analogous to Dense models' MLP

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_X.index_add_(0, top_x, current_X.to(X.dtype))
    final_X = final_X.reshape(bsz, seq_len, hd)
    return final_X, router_logits


def Qwen3MoeDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
):
    residual = hidden_states

    if use_cache and hasattr(
        self, "_flag_for_generation"
    ):  # past_key_value is not None:
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(
            self.input_layernorm, hidden_states
        )
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            padding_mask = padding_mask,
            position_embeddings = position_embeddings,
            _flag_for_generation = self._flag_for_generation,
        )
        hidden_states += residual

        # MoE Router MLP
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference(
            self.post_attention_layernorm, hidden_states
        )
        hidden_states, router_logits = Qwen3MoeSparseMoeBlock_fast_forward(
            self.mlp, hidden_states
        )
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            causal_mask = causal_mask,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            padding_mask = padding_mask,
            position_embeddings = position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MoE Router MLP
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if output_router_logits:
        outputs += (router_logits,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


class FastQwen3MoeModel(FastQwen3Model):
    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name = "Qwen3Moe",
            rope_module = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module = Qwen3MoeAttention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen3MoeAttention.__init__ = eval(init_name)
        Qwen3MoeAttention.forward = Qwen3Attention_fast_forward
        # Qwen3SdpaAttention   .forward = Qwen3Attention_fast_forward
        # Qwen3FlashAttention2 .forward = Qwen3Attention_fast_forward
        Qwen3MoeSparseMoeBlock.forward = Qwen3MoeSparseMoeBlock_fast_forward
        Qwen3MoeMLP.forward = (
            fast_swiglu_inference  # This is analogous to Dense models' MLP
        )
        Qwen3MoeDecoderLayer.forward = Qwen3MoeDecoderLayer_fast_forward
        Qwen3MoeModel.forward = LlamaModel_fast_forward
        Qwen3MoeForCausalLM.forward = CausalLM_fast_forward(
            LlamaModel_fast_forward_inference
        )
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(Qwen3MoeForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inference can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py\
        import transformers.models.qwen3_moe.modeling_qwen3_moe

        transformers.models.Qwen3Moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        return

    @staticmethod
    def from_pretrained(  # TODO: Change after release
        model_name = "Qwen/Qwen3-7B",
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
            model_patcher = FastQwen3Model,
            tokenizer_name = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )


# Helper functions for Qwen3 Omni Patching
def qwen3_omni_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V


def qwen3_omni_apply_o(self, X):
    O = self.o_proj(X)
    return O


def _patch_qwen3_omni_model_class(model_class):
    """
    Patch Qwen3OmniMoeForConditionalGeneration with required methods for training.
    The model delegates to its thinker submodule for actual forward pass.
    """
    # Add embedding accessor methods that delegate to thinker
    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.thinker.lm_head

    def set_output_embeddings(self, value):
        self.thinker.lm_head = value

    # Add forward method that delegates to thinker for training
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits=None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ):
        """Forward method that delegates to the thinker for training."""
        return self.thinker.forward(
            input_ids=input_ids,
            input_features=input_features,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            labels=labels,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            use_audio_in_video=use_audio_in_video,
            cache_position=cache_position,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

    # Apply patches to the model class
    if not hasattr(model_class, "_unsloth_patched"):
        model_class.get_input_embeddings = get_input_embeddings
        model_class.set_input_embeddings = set_input_embeddings
        model_class.get_output_embeddings = get_output_embeddings
        model_class.set_output_embeddings = set_output_embeddings
        model_class.forward = forward
        model_class._unsloth_patched = True


def _patch_qwen3_omni_attention_modules(model):
    """
    Patch Attention and MLP modules in Qwen3-Omni for faster training.
    """
    patched_classes = set()

    for name, module in model.named_modules():
        if module.__class__ in patched_classes:
            continue

        class_name = module.__class__.__name__

        # Replace Attention with Triton Kernels
        if (
            "Attention" in class_name
            and hasattr(module, "q_norm")
            and hasattr(module, "k_norm")
        ):
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
                and hasattr(module, "o_proj")
            ):
                module.__class__.apply_qkv = qwen3_omni_apply_qkv
                module.__class__.apply_o = qwen3_omni_apply_o

                # Save original forward to allow fallback during inference
                if not hasattr(module.__class__, "_original_forward"):
                    module.__class__._original_forward = module.__class__.forward

                # Define a Safe Wrapper
                def _attention_wrapper(
                    self,
                    hidden_states,
                    position_embeddings=None,
                    attention_mask=None,
                    past_key_values=None,
                    **kwargs,
                ):
                    # FALLBACK: If caching is used (Inference), use original code
                    if past_key_values is not None:
                        return self._original_forward(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            **kwargs,
                        )

                    # If training (No Cache), use Unsloth's 2x Faster Kernel
                    return Qwen3Attention_fast_forward(
                        self,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                        past_key_value=None,
                        **kwargs,
                    )

                module.__class__.forward = _attention_wrapper
                patched_classes.add(module.__class__)

        # Replace MLP with SwiGLU Kernels
        if "MLP" in class_name and "Moe" not in class_name:
            if (
                hasattr(module, "gate_proj")
                and hasattr(module, "up_proj")
                and hasattr(module, "down_proj")
            ):
                module.__class__.forward = fast_swiglu_inference
                patched_classes.add(module.__class__)

    return patched_classes


class FastQwen3OmniMoeModel(FastQwen3MoeModel):
    @staticmethod
    def from_pretrained(
        model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        token=None,
        device_map="sequential",
        rope_scaling=None,
        fix_tokenizer=True,
        model_patcher=None,
        tokenizer_name=None,
        trust_remote_code=False,
        **kwargs,
    ):
        # Import the model class
        from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor

        # Patch the model class with required methods before loading
        _patch_qwen3_omni_model_class(Qwen3OmniMoeForConditionalGeneration)

        # Determine dtype
        if dtype is None:
            dtype = torch.bfloat16

        # Load the model using the multimodal pattern
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "token": token,
            "device_map": device_map,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Merge any additional kwargs
        model_kwargs.update(kwargs)

        # Load model
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Load processor/tokenizer
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        # Patch attention and MLP modules for faster training
        patched_classes = _patch_qwen3_omni_attention_modules(model)

        return model, processor
