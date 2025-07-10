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
import math

try:
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaDecoderLayer,
        GemmaModel,
        GemmaForCausalLM,
        GemmaRotaryEmbedding,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except:
    from packaging.version import Version
    transformers_version = Version(transformers_version)
    if not transformers_version >= Version("4.38"):
        raise ImportError(
            f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"\
            f"The minimum required version is 4.38.\n"\
            f'Try `pip install --upgrade "transformers>=4.38"`\n'\
            f"to obtain the latest transformers build, then restart this session."\
        )
    pass
pass

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
# For Pytorch 2.1.1
try:
    from transformers.models.gemma.modeling_gemma import (
        GemmaSdpaAttention,
        GemmaFlashAttention2,
    )
except:
    GemmaSdpaAttention   = GemmaAttention
    GemmaFlashAttention2 = GemmaAttention
pass


torch_nn_functional_gelu = torch.nn.functional.gelu
def fast_geglu_inference(self, X):
    # gate = self.gate_proj(X)
    # up   = self.up_proj(X)
    bsz, _, hd = X.shape
    # mlp_size = self.config.intermediate_size
    # temp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = "cuda:0")

    gate = fast_linear_forward(self.gate_proj, X)#, out = temp[0])
    up   = fast_linear_forward(self.  up_proj, X)#, out = temp[1])
    gate = torch_nn_functional_gelu(gate, approximate = "tanh")
    gate *= up

    # X = self.down_proj(gate)
    down = fast_linear_forward(self.down_proj, gate, out = up[:,:,:hd])
    return down
pass


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def GemmaDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
):
    if use_cache and hasattr(self, "_flag_for_generation"): #past_key_value is not None:
        out_weight = torch.empty(self.input_layernorm.weight.shape, dtype = torch.float32, device = "cuda:0")

        # Self Attention
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self.input_layernorm, hidden_states, out_weight)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states += residual

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(self.post_attention_layernorm, hidden_states, out_weight)
        hidden_states = fast_geglu_inference(self.mlp, hidden_states)
        hidden_states += residual
    else:
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states, gemma = True)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states, gemma = True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
    pass

    outputs = (hidden_states,)
    if output_attentions: outputs += (self_attn_weights,)
    if use_cache: outputs += (present_key_value,)
    return outputs
pass


from math import sqrt as math_sqrt

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
# @torch.inference_mode
def GemmaModel_fast_forward_inference(
    self,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask = None,
):
    out_weights = tuple(torch.empty_like(self.model.layers[0].input_layernorm.weight, dtype = torch.float32, device = torch.device(x)) for x in range(DEVICE_COUNT))
    input_ids = input_ids[:,:self.max_seq_length]
    hidden_states = self.model.embed_tokens(input_ids)
    hidden_states = hidden_states.to(self.config.torch_dtype)
    # 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
    # 2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
    hidden_states *= torch.tensor(math_sqrt(self.config.hidden_size), dtype = hidden_states.dtype)

    bsz, q_len, hd = hidden_states.shape
    seq_len = past_key_values[0][0].shape[-2]
    if bsz != 1:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            seq_len,
        )
    pass

    next_decoder_cache = []
    for idx, decoder_layer in enumerate(self.model.layers):
        device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
        hidden_states, position_ids = move_to_device(
            device_index, hidden_states, position_ids
        )

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.input_layernorm, hidden_states, out_weights[device_index])
        hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
            decoder_layer.self_attn,
            hidden_states = hidden_states,
            past_key_value = past_key_values[idx],
            position_ids = position_ids,
            attention_mask = attention_mask,
            do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
        )
        hidden_states += residual

        residual = hidden_states
        hidden_states = fast_rms_layernorm_inference_gemma(decoder_layer.post_attention_layernorm, hidden_states, out_weights[device_index])
        hidden_states = fast_geglu_inference(decoder_layer.mlp, hidden_states)
        hidden_states += residual

        next_decoder_cache.append(present_key_value)
    pass
    hidden_states = fast_rms_layernorm_inference_gemma(self.model.norm, hidden_states, out_weights[device_index])

    return BaseModelOutputWithPast(
        last_hidden_state = hidden_states,
        past_key_values = next_decoder_cache,
        hidden_states = [],
        attentions = [],
    )
pass


# Follows line by line https://github.com/google-deepmind/gemma/blob/main/gemma/positional_embeddings.py#L45
# Formulates cos and sin differently from Llama!
class GemmaFixedRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
            dim = getattr(config, "head_dim", None)
            if dim is None: dim = int((config.hidden_size // config.num_attention_heads))
            device = "cuda"
            max_position_embeddings = config.max_position_embeddings
        pass
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None]*DEVICE_COUNT
        self.multi_gpu_sin_cached = [None]*DEVICE_COUNT

        # Build here to make `torch.jit.trace` work.
        for device in range(DEVICE_COUNT):
            self._set_cos_sin_cache(seq_len=self.current_rope_size, device=torch.device(device), dtype=torch.get_default_dtype())

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.get_default_dtype())
        self.sin_cached = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        # The difference is we do division explicity instead of t * (1/x) ie we do t/x.
        freq_exponents = (2.0 / self.dim) * (
            torch.arange(self.dim // 2, dtype = torch.int64, device = "cpu").float()
        )
        timescale = self.base**freq_exponents
        positions = torch.arange(self.current_rope_size, device = "cpu", dtype = torch.int64).float()
        radians_new = positions[..., None] / timescale[None, None, :]
        radians_new = radians_new.squeeze(0)

        emb = torch.cat((radians_new, radians_new), dim = -1)
        # We must do RoPE in float32!
        cos = emb.cos().to(device = device, non_blocking = True)#, dtype = dtype)
        sin = emb.sin().to(device = device, non_blocking = True)#, dtype = dtype)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index

        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )
    pass

    def get_cached(self, seq_len = None, device_index = None):
        if device_index is None:
            device_index = torch.cuda.current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[device_index]
    pass

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size: return
        # Iteratively grow by increments of 8192
        self.current_rope_size = math.ceil(seq_len / 8192) * 8192
        for device in range(DEVICE_COUNT):
            self._set_cos_sin_cache(self.current_rope_size, device = torch.device(device), dtype = x.dtype)
    pass
pass


class GemmaFixedLinearScalingRotaryEmbedding(GemmaFixedRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim = None, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0,
        config = None, # [TODO] Hack to pass in config - need to remove later
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim = dim, max_position_embeddings = max_position_embeddings, base = base, device = device, config = config)
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        # The difference is we do division explicity instead of t * (1/x) ie we do t/x.
        freq_exponents = (2.0 / self.dim) * (
            torch.arange(self.dim // 2, dtype = torch.int64, device = "cpu").float()
        )
        timescale = self.base**freq_exponents
        positions = torch.arange(self.current_rope_size, device = "cpu", dtype = torch.int64).float()
        positions = positions /  self.scaling_factor
        radians_new = positions[..., None] / timescale[None, None, :]
        radians_new = radians_new.squeeze(0)

        emb = torch.cat((radians_new, radians_new), dim = -1)
        # We must do RoPE in float32!
        cos = emb.cos().to(device = device, non_blocking = True)#, dtype = dtype)
        sin = emb.sin().to(device = device, non_blocking = True)#, dtype = dtype)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin
    pass
pass


class FastGemmaModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "gemma",
            rope_module        = GemmaFixedRotaryEmbedding,
            scaled_rope_module = GemmaFixedLinearScalingRotaryEmbedding,
            attention_module   = GemmaAttention,
        )
        if init_name is not None:
            exec(function, globals())
            GemmaAttention.__init__  = eval(init_name)
        pass
        GemmaAttention      .forward = LlamaAttention_fast_forward
        GemmaSdpaAttention  .forward = LlamaAttention_fast_forward
        GemmaFlashAttention2.forward = LlamaAttention_fast_forward
        GemmaDecoderLayer   .forward = GemmaDecoderLayer_fast_forward
        GemmaModel          .forward = LlamaModel_fast_forward
        GemmaForCausalLM    .forward = CausalLM_fast_forward(GemmaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(GemmaForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.gemma.modeling_gemma
        transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding = GemmaFixedRotaryEmbedding
        return
    pass


    @staticmethod
    def post_patch(model, tokenizer):
        # Gemma does not downcast RoPE
        model, tokenizer = patch_model_and_tokenizer(model, tokenizer, downcast_rope = False)

        # Add 1 to weight
        # return output * (1 + self.weight)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L89
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

        # Freeze all parameters except LoRA
        # We do this first since += 1 seems to not be liked by requires_grad = True
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        pass

        # Patch RMS Layernorm
        for name, module in model.named_modules():
            if isinstance(module, GemmaRMSNorm):
                # Must be in float32
                # https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L36
                # module = module.to(torch.float32)
                # Leave + 1 to Triton kernel itself
                # module.weight += 1.0 # return output * (1 + self.weight)
                if not hasattr(module, "variance_epsilon"):
                    module.variance_epsilon = module.eps # Gemma doesn't use variance_epsilon
        pass

        # Clear deleted GPU items
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model, tokenizer
    pass
pass
