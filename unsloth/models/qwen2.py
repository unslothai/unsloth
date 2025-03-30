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
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)
# For Pytorch 2.1.1
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2SdpaAttention,
        Qwen2FlashAttention2,
    )
except:
    Qwen2SdpaAttention   = Qwen2Attention
    Qwen2FlashAttention2 = Qwen2Attention
pass


class FastQwen2Model(FastLlamaModel):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "qwen2",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = Qwen2Attention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen2Attention.__init__  = eval(init_name)
        pass
        Qwen2Attention      .forward = LlamaAttention_fast_forward
        Qwen2SdpaAttention  .forward = LlamaAttention_fast_forward
        Qwen2FlashAttention2.forward = LlamaAttention_fast_forward
        Qwen2DecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        Qwen2Model          .forward = LlamaModel_fast_forward
        Qwen2ForCausalLM    .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        fix_prepare_inputs_for_generation(Qwen2ForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.qwen2.modeling_qwen2
        transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding = LlamaRotaryEmbedding
        return
    pass

    # Special handling for Qwen2 models' generate function
    @staticmethod
    def patch_qwen2_model(model):
        """Ensure Qwen2 models have proper generate method"""
        if not hasattr(model, "_old_generate") and hasattr(model, "generate"):
            # Save original generate method to restore if needed
            model._old_generate = model.generate
            
            # Create a Qwen2-specific fixed generate function
            def qwen2_fixed_generate(self, *args, **kwargs):
                try:
                    # First try the original generate method
                    return self._old_generate(*args, **kwargs)
                except TypeError as e:
                    if "str" in str(e) and "not callable" in str(e):
                        # Fallback to transformers standard generation
                        print("Unsloth: Using fallback generation for Qwen2 model")
                        from transformers.generation.utils import GenerationMixin
                        return GenerationMixin.generate(self, *args, **kwargs)
                    raise
            
            # Replace the generate method
            import types
            model.generate = types.MethodType(qwen2_fixed_generate, model)
        return model

    @staticmethod
    def from_pretrained(
        model_name        = "Qwen/Qwen2-7B",
        max_seq_length    = 4096,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None, # Qwen2 does not support RoPE scaling
        fix_tokenizer     = True,
        model_patcher     = None,
        tokenizer_name    = None,
        trust_remote_code = False,
        **kwargs,
    ):
        model, tokenizer = FastLlamaModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastQwen2Model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
        
        # Apply Qwen2-specific fixes
        model = FastQwen2Model.patch_qwen2_model(model)
        
        return model, tokenizer
    pass
pass
