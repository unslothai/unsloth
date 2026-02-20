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

"""Pure MLX Llama model implementation for Apple Silicon training.

This module provides a native MLX implementation of the Llama architecture,
designed to run entirely on Apple Silicon without PyTorch conversion overhead.
"""

from __future__ import annotations

from typing import Optional, Tuple
import mlx.core as mx
from mlx import nn as mnn

from .base import (
    MLXModelConfig,
    MLXLinear,
    MLXEmbedding,
    MLXRMSNorm,
    LoRAConfig,
    DEFAULT_LORA_TARGET_MODULES,
)


def silu(x: mx.array) -> mx.array:
    """SiLU activation function: x * sigmoid(x)."""
    return x * mx.sigmoid(x)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """Apply rotary positional embeddings to query and key tensors."""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MLXRotaryEmbedding:
    """MLX-native Rotary Positional Embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        dtype: type = mx.float32,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype
        self.inv_freq = None
        self._cos_cached = None
        self._sin_cached = None

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        if self._cos_cached is not None and seq_len <= self._cos_cached.shape[0]:
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        self.inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=self.dtype) / self.dim))
        t = mx.arange(seq_len, dtype=self.dtype)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)

        self._cos_cached = mx.cos(emb)
        self._sin_cached = mx.sin(emb)
        return self._cos_cached, self._sin_cached


class MLXAttention:
    """MLX-native Multi-Head Attention with optional LoRA."""

    def __init__(
        self,
        config: MLXModelConfig,
        layer_idx: Optional[int] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = 0.0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = self.hidden_size // self.num_heads

        if (self.hidden_size % self.num_heads) != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )

        self.q_proj = MLXLinear(self.hidden_size, self.hidden_size, bias=False, dtype=mx.float32)
        self.k_proj = MLXLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=mx.float32)
        self.v_proj = MLXLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=mx.float32)
        self.o_proj = MLXLinear(self.hidden_size, self.hidden_size, bias=False, dtype=mx.float32)

        self.lora_config = lora_config
        self.use_lora = lora_config is not None and lora_config.r > 0

        if self.use_lora:
            target_modules = lora_config.target_modules or DEFAULT_LORA_TARGET_MODULES.get("llama", [])
            self.lora_q_A = None
            self.lora_q_B = None
            self.lora_k_A = None
            self.lora_k_B = None
            self.lora_v_A = None
            self.lora_v_B = None
            self.lora_o_A = None
            self.lora_o_B = None

            r = lora_config.r
            alpha = lora_config.lora_alpha
            dropout = lora_config.lora_dropout

            if "q_proj" in target_modules:
                self.lora_q_A = MLXLinear(self.hidden_size, r, bias=False, dtype=mx.float32)
                self.lora_q_B = MLXLinear(r, self.hidden_size, bias=False, dtype=mx.float32)
            if "k_proj" in target_modules:
                self.lora_k_A = MLXLinear(self.hidden_size, r, bias=False, dtype=mx.float32)
                self.lora_k_B = MLXLinear(r, self.num_key_value_heads * self.head_dim, bias=False, dtype=mx.float32)
            if "v_proj" in target_modules:
                self.lora_v_A = MLXLinear(self.hidden_size, r, bias=False, dtype=mx.float32)
                self.lora_v_B = MLXLinear(r, self.num_key_value_heads * self.head_dim, bias=False, dtype=mx.float32)
            if "o_proj" in target_modules:
                self.lora_o_A = MLXLinear(self.hidden_size, r, bias=False, dtype=mx.float32)
                self.lora_o_B = MLXLinear(r, self.hidden_size, bias=False, dtype=mx.float32)

            self.lora_scaling = alpha / r

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Tuple[mx.array, mx.array]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_lora:
            if self.lora_q_A is not None:
                lora_q_out = mx.matmul(hidden_states, self.lora_q_A.weight.T)
                lora_q_out = mx.matmul(lora_q_out, self.lora_q_B.weight.T)
                query_states = query_states + self.lora_scaling * lora_q_out
            if self.lora_k_A is not None:
                lora_k_out = mx.matmul(hidden_states, self.lora_k_A.weight.T)
                lora_k_out = mx.matmul(lora_k_out, self.lora_k_B.weight.T)
                key_states = key_states + self.lora_scaling * lora_k_out
            if self.lora_v_A is not None:
                lora_v_out = mx.matmul(hidden_states, self.lora_v_A.weight.T)
                lora_v_out = mx.matmul(lora_v_out, self.lora_v_B.weight.T)
                value_states = value_states + self.lora_scaling * lora_v_out

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(kv_seq_len)
        position_ids = position_ids if position_ids is not None else mx.arange(q_len)
        cos = cos[position_ids]
        sin = sin[position_ids]

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = mx.concatenate([past_key_value[0], key_states], axis=2)
            value_states = mx.concatenate([past_key_value[1], value_states], axis=2)

        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(query_states.dtype)
            attention_mask = (1.0 - attention_mask) * float("-inf")

        attn_weights = (query_states @ key_states.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ value_states

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.use_lora and hasattr(self, 'lora_o_A') and self.lora_o_A is not None:
            attn_before_proj = attn_output
            attn_output = attn_output + self.lora_scaling * mx.matmul(
                mx.matmul(attn_before_proj, self.lora_o_A.weight.T),
                self.lora_o_B.weight.T
            )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, (key_states, value_states)

    def _repeat_kv(self, x: mx.array, n_rep: int) -> mx.array:
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        x = mx.broadcast_to(x[:, :, None, :, :], (bsz, num_kv_heads, n_rep, seq_len, head_dim))
        return x.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class MLXMLP:
    """MLX-native MLP with SwiGLU activation."""

    def __init__(
        self,
        config: MLXModelConfig,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_proj = MLXLinear(self.hidden_size, self.intermediate_size, bias=False, dtype=mx.float32)
        self.up_proj = MLXLinear(self.hidden_size, self.intermediate_size, bias=False, dtype=mx.float32)
        self.down_proj = MLXLinear(self.intermediate_size, self.hidden_size, bias=False, dtype=mx.float32)

        self.lora_config = lora_config
        self.use_lora = lora_config is not None and lora_config.r > 0

        if self.use_lora:
            target_modules = lora_config.target_modules or DEFAULT_LORA_TARGET_MODULES.get("llama", [])
            r = lora_config.r
            alpha = lora_config.lora_alpha
            self.lora_scaling = alpha / r

            self.lora_gate = None
            self.lora_up = None
            self.lora_down = None

            if "gate_proj" in target_modules:
                self.lora_gate = MLXLinear(self.intermediate_size, r, bias=False, dtype=mx.float32)
            if "up_proj" in target_modules:
                self.lora_up = MLXLinear(self.intermediate_size, r, bias=False, dtype=mx.float32)
            if "down_proj" in target_modules:
                self.lora_down_A = MLXLinear(self.hidden_size, r, bias=False, dtype=mx.float32)
                self.lora_down_B = MLXLinear(r, self.hidden_size, bias=False, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        if self.use_lora:
            if self.lora_gate is not None:
                gate = gate + self.lora_scaling * mx.matmul(x, self.lora_gate.weight.T)
            if self.lora_up is not None:
                up = up + self.lora_scaling * mx.matmul(x, self.lora_up.weight.T)

        gate = silu(gate)
        down = self.down_proj(gate * up)

        if self.use_lora and hasattr(self, 'lora_down_A') and self.lora_down_A is not None:
            lora_out = self.lora_scaling * mx.matmul(
                mx.matmul(x, self.lora_down_A.weight.T),
                self.lora_down_B.weight.T
            )
            down = down + lora_out

        return down


class MLXLlamaDecoderLayer:
    """MLX-native Llama decoder layer."""

    def __init__(
        self,
        config: MLXModelConfig,
        layer_idx: int,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = MLXAttention(config, layer_idx, lora_config)
        self.mlp = MLXMLP(config, lora_config)
        self.input_layernorm = MLXRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MLXRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[Tuple[mx.array, mx.array]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MLXLlamaModel:
    """MLX-native Llama model (without language model head)."""

    def __init__(
        self,
        config: MLXModelConfig,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.config = config
        self.lora_config = lora_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = MLXEmbedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MLXLlamaDecoderLayer(config, idx, lora_config)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = MLXRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        for layer in self.layers:
            layer.self_attn.rotary_emb = MLXRotaryEmbedding(
                dim=layer.self_attn.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mx.array, list]:
        bsz, seq_len = input_ids.shape

        past_key_values = past_key_values or [None] * len(self.layers)

        hidden_states = self.embed_tokens(input_ids)

        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if idx < len(past_key_values) else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class MLXLlamaForCausalLM:
    """MLX-native Llama model with language model head."""

    def __init__(
        self,
        config: MLXModelConfig,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.config = config
        self.lora_config = lora_config

        self.model = MLXLlamaModel(config, lora_config)
        self.vocab_size = config.vocab_size
        self.lm_head = MLXLinear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, self.vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)
            loss = mx.nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")

        return logits, loss

    def parameters(self) -> dict:
        """Return all model parameters as a dictionary."""
        params = {}
        for name, value in vars(self.model).items():
            if isinstance(value, mx.array):
                params[name] = value
            elif hasattr(value, "weight"):
                params[f"{name}.weight"] = value.weight
                if hasattr(value, "bias") and value.bias is not None:
                    params[f"{name}.bias"] = value.bias
        params["lm_head.weight"] = self.lm_head.weight
        return params

    def trainable_parameters(self) -> dict:
        """Return only trainable parameters (LoRA params if LoRA enabled)."""
        trainable = {}
        for name, module in self.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                trainable[f"{name}.lora_A"] = module.lora_A
                trainable[f"{name}.lora_B"] = module.lora_B
        if not trainable:
            return self.parameters()
        return trainable

    def update(self, params: dict):
        """Update model parameters from a dictionary."""
        for key, value in params.items():
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif hasattr(obj, "model") and hasattr(obj.model, part):
                    obj = getattr(obj.model, part)
                else:
                    break
            else:
                final_key = parts[-1]
                if hasattr(obj, final_key):
                    setattr(obj, final_key, value)

    def named_modules(self) -> list:
        """Yield (name, module) pairs for all submodules."""
        modules = []
        for name, value in vars(self).items():
            if hasattr(value, "weight") or hasattr(value, "lora_A"):
                modules.append((name, value))
        for name, value in vars(self.model).items():
            if hasattr(value, "weight") or hasattr(value, "lora_A"):
                modules.append((name, value))
            elif hasattr(value, "self_attn"):
                modules.append((f"{name}.self_attn", value.self_attn))
            elif hasattr(value, "mlp"):
                modules.append((f"{name}.mlp", value.mlp))
        return modules

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> mx.array:
        """Generate text using the model."""
        self.model.eval()
        bsz, seq_len = input_ids.shape

        for _ in range(max_new_tokens):
            logits, _ = self.model(input_ids=input_ids)
            next_token_logits = logits[:, -1, :] / temperature

            if top_p < 1.0:
                sorted_indices = mx.argsort(next_token_logits, axis=-1, descending=True)
                cumsum = mx.cumsum(mx.softmax(next_token_logits, axis=-1), axis=-1)
                mask = cumsum > top_p
                next_token_logits = mx.where(mask, -float("inf"), next_token_logits)
                next_token_logits = sorted_indices[0:1]

            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
            input_ids = mx.concatenate([input_ids, next_token], axis=-1)

            if next_token.item() == self.config.eos_token_id:
                break

        return input_ids


def create_llama_model(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
    max_position_embeddings: int = 2048,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    hidden_act: str = "silu",
    pad_token_id: Optional[int] = None,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    lora_config: Optional[LoRAConfig] = None,
) -> MLXLlamaForCausalLM:
    """Create a Llama model with the given configuration."""
    config = MLXModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act=hidden_act,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    return MLXLlamaForCausalLM(config, lora_config)
