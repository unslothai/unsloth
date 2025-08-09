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
from unsloth_zoo.utils import Version, _get_dtype
from .vision import FastBaseModel
from ..kernels import DeterministicDropout

import torch
from typing import Optional, Tuple

try:
    from transformers import __version__ as transformers_version
    transformers_version = Version(transformers_version)
    from transformers.models.phi.modeling_phi import (
        PhiAttention,
        PhiDecoderLayer,
        PhiModel,
        PhiForCausalLM,
    )
    try:
        from transformers.models.phi.modeling_phi import PhiSdpaAttention, PhiFlashAttention2
    except Exception:
        PhiSdpaAttention = PhiAttention
        PhiFlashAttention2 = PhiAttention
except Exception as error:
    # We only import when actually used; loader will guard by AutoConfig
    PhiAttention = None
    PhiDecoderLayer = None
    PhiModel = None
    PhiForCausalLM = None
    PhiSdpaAttention = None
    PhiFlashAttention2 = None


def _phi_get_rotary_dims(attn_module) -> int:
    head_dim: int = attn_module.head_dim
    # Prefer explicit rotary_dim if provided by config
    rotary_dim = getattr(attn_module.config, "rotary_dim", None)
    if isinstance(rotary_dim, int) and 0 < rotary_dim <= head_dim:
        # Ensure even for half-rotate math
        return (rotary_dim // 2) * 2
    # Else use partial_rotary_factor if present
    fraction = getattr(attn_module.config, "partial_rotary_factor", None)
    if isinstance(fraction, (float, int)) and 0 < fraction <= 1:
        rotary_dim = int(head_dim * float(fraction))
        return (rotary_dim // 2) * 2
    # Default: full rotation
    return (head_dim // 2) * 2 if head_dim % 2 != 0 else head_dim


def PhiAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Clear inference caches if any (mirrors other model fastpaths)
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads: int = self.config.num_attention_heads
    n_kv_heads: int = getattr(self.config, "num_key_value_heads", n_heads)
    n_groups_attr = getattr(self, "num_key_value_groups", None)
    n_groups: int = n_groups_attr if isinstance(n_groups_attr, int) and n_groups_attr > 0 else max(1, n_heads // max(1, n_kv_heads))
    head_dim: int = self.head_dim
    assert (n_kv_heads * n_groups == n_heads)

    # Q, K, V projections
    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    # Sequence lengths with KV cache
    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Apply (partial) RoPE on first rotary_dim dims of Q, K
    rotary_dim: int = _phi_get_rotary_dims(self)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if rotary_dim < head_dim:
            cos = cos[..., :rotary_dim]
            sin = sin[..., :rotary_dim]
        # Fast path when position_ids provided handled below via inplace op
        if position_ids is None:
            Q_rot = Q[..., :rotary_dim]
            K_rot = K[..., :rotary_dim]
            Q_rot, K_rot = inplace_rope_embedding(Q_rot, K_rot, cos, sin, position_ids)
            Q[..., :rotary_dim] = Q_rot
            K[..., :rotary_dim] = K_rot
        else:
            Q_rot = Q[..., :rotary_dim]
            K_rot = K[..., :rotary_dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seqlen, dim]
            sin = sin[position_ids].unsqueeze(1)
            Q_rot, K_rot = inplace_rope_embedding(Q_rot, K_rot, cos, sin, position_ids)
            Q[..., :rotary_dim] = Q_rot
            K[..., :rotary_dim] = K_rot
    else:
        # Compute cos/sin from available rotary embedding; if none, create a local one
        rope_module = None
        if hasattr(self, "rotary_emb"):
            rope_module = self.rotary_emb
            rope_module.extend_rope_embedding(V, seq_len=kv_seq_len)
            if position_ids is None:
                cos = rope_module.cos_cached
                sin = rope_module.sin_cached
            else:
                cos, sin = rope_module(V, seq_len=kv_seq_len)
        else:
            rope_module = getattr(self, "_unsloth_phi_rope", None)
            if rope_module is None:
                # Build Llama-style rotary embedding configured for Phi
                try:
                    base = getattr(self.config, "rope_theta", 10000)
                    max_pos = getattr(self.config, "max_position_embeddings", 2048)
                except Exception:
                    base = 10000
                    max_pos = 2048
                rope_module = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=max_pos, base=base, device=V.device)
                # Keep for reuse
                self._unsloth_phi_rope = rope_module
            # Ensure buffers sized appropriately
            rope_module.extend_rope_embedding(V, seq_len=kv_seq_len)
            if position_ids is None:
                cos = rope_module.cos_cached
                sin = rope_module.sin_cached
            else:
                cos, sin = rope_module(V, seq_len=kv_seq_len)

        # Apply (partial) RoPE
        if rotary_dim < head_dim:
            cos = cos[..., :rotary_dim]
            sin = sin[..., :rotary_dim]
        Q_rot = Q[..., :rotary_dim]
        K_rot = K[..., :rotary_dim]
        Q_rot, K_rot = inplace_rope_embedding(Q_rot, K_rot, cos, sin, position_ids)
        Q[..., :rotary_dim] = Q_rot
        K[..., :rotary_dim] = K_rot

    # KV cache update
    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    past_key_value = (K, V) if use_cache else None

    # Attention computation (dispatch as in other models)
    if (not HAS_FLASH_ATTENTION and HAS_XFORMERS and attention_mask is None):
        # Xformers memory efficient attention with (bsz, seqlen, heads, dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Grouped Query Attention (expand KV across groups)
        if n_groups != 1:
            K = K.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
            V = V.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
            K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            if hidden_states.requires_grad:
                K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        A = xformers_attention(Q, K, V, attn_bias=causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        A = flash_attn_func(Q, K, V, causal=True)
    else:
        # SDPA fallback, support GQA if available
        if SDPA_HAS_GQA:
            A = scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask, is_causal=False, enable_gqa=n_groups != 1)
            A = A.transpose(1, 2)
        else:
            if n_groups != 1:
                K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
                K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
                V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
            Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
            A = scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask, is_causal=False)
            A = A.transpose(1, 2).contiguous()

    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = self.apply_o(self, attn_output)
    # Optional deterministic residual dropout after attention projection
    resid_attn_dropout = getattr(self, "_unsloth_resid_attn_dropout", None)
    if resid_attn_dropout is not None and self.training:
        attn_output = resid_attn_dropout(attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


class FastPhiModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        if PhiAttention is None:
            return
        # Patch attention forward for partial RoPE support and Unsloth compute path
        PhiAttention     .forward = PhiAttention_fast_forward
        try:
            PhiSdpaAttention  .forward = PhiAttention_fast_forward
            PhiFlashAttention2.forward = PhiAttention_fast_forward
        except Exception:
            pass
        # Patch CausalLM for Unsloth fastpath when compatible
        try:
            PhiForCausalLM   .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
            PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
            fix_prepare_inputs_for_generation(PhiForCausalLM)
        except Exception:
            pass
        return

    @staticmethod
    def post_patch(model, tokenizer):
        # Ensure Phi-2 defaults for partial RoPE if missing in config
        try:
            if getattr(model.config, "model_type", None) == "phi":
                if not hasattr(model.config, "partial_rotary_factor") and not hasattr(model.config, "rotary_dim"):
                    # Empirically common fraction for Phi-2 partial RoPE
                    model.config.partial_rotary_factor = 0.4
                # Attach deterministic dropout layers if dropout > 0 for residuals
                p_attn = float(getattr(model.config, "attention_dropout", 0.0))
                p_mlp  = float(getattr(model.config, "hidden_dropout", 0.0))
                if p_attn > 0.0 or p_mlp > 0.0:
                    import os
                    seed = int(os.environ.get("UNSLOTH_DROPOUT_SEED", 3407))
                    for layer in model.model.layers:
                        if p_attn > 0.0:
                            # Attach to attention module so it can be used in patched attention forward
                            layer.self_attn._unsloth_resid_attn_dropout = DeterministicDropout(p_attn, seed)
                        if p_mlp > 0.0:
                            # Apply dropout to MLP outputs via a forward hook (post-MLP, pre-residual add)
                            layer._unsloth_resid_mlp_dropout  = DeterministicDropout(p_mlp,  seed)
                            def _mlp_hook(mod, inputs, output, _layer=layer):
                                if _layer.training and _layer._unsloth_resid_mlp_dropout is not None:
                                    return _layer._unsloth_resid_mlp_dropout(output)
                                return output
                            # Keep handle to prevent GC
                            layer._unsloth_mlp_hook = layer.mlp.register_forward_hook(_mlp_hook)
        except Exception:
            pass
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules = "all-linear",
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = True,
        random_state: int = 3407,
        max_seq_length: int = 2048,
        use_rslora: bool = False,
        modules_to_save = None,
        init_lora_weights: bool = True,
        loftq_config: dict = {},
        temporary_location: str = "_unsloth_temporary_saved_buffers",
        **kwargs,
    ):
        return FastBaseModel.get_peft_model(
            model                      = model,
            r                          = r,
            target_modules             = target_modules,
            lora_alpha                 = lora_alpha,
            lora_dropout               = lora_dropout,
            bias                       = bias,
            layers_to_transform        = layers_to_transform,
            layers_pattern             = layers_pattern,
            use_gradient_checkpointing = use_gradient_checkpointing,
            random_state               = random_state,
            max_seq_length             = max_seq_length,
            use_rslora                 = use_rslora,
            modules_to_save            = modules_to_save,
            init_lora_weights          = init_lora_weights,
            loftq_config               = loftq_config,
            temporary_location         = temporary_location,
            **kwargs,
        )

    @staticmethod
    def from_pretrained(
        model_name: str = "microsoft/Phi-2",
        max_seq_length: Optional[int] = None,
        dtype = None,
        load_in_4bit: bool = True,
        token: Optional[str] = None,
        device_map: str = "sequential",
        rope_scaling = None,
        fix_tokenizer: bool = True,
        model_patcher = None,
        tokenizer_name: Optional[str] = None,
        trust_remote_code: bool = False,
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
            model_patcher     = FastPhiModel,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            **kwargs,
        )
    pass
pass


