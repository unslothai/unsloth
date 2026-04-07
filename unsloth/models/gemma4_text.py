# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope


def _gelu_pytorch_tanh(x: mx.array) -> mx.array:
    return nn.gelu_approx(x)


ACT2FN = {
    "gelu": nn.gelu,
    "gelu_pytorch_tanh": _gelu_pytorch_tanh,
    "silu": nn.silu,
}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int = 131072
    layer_types: Optional[list[str]] = None
    sliding_window: int = 512
    rope_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    num_global_key_value_heads: Optional[int] = None
    global_head_dim: Optional[int] = None
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    final_logit_softcapping: Optional[float] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_bidirectional_attention: Optional[str] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 6 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"
        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads
        if self.global_head_dim is None:
            self.global_head_dim = self.head_dim
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        mean_squared = mx.mean(y * y, axis = -1, keepdims = True) + self.eps
        y = y * mx.rsqrt(mean_squared)
        if self.with_scale:
            y = y * self.weight.astype(mx.float32)
        return y.astype(x.dtype)


class Float32RoPE(nn.Module):
    def __init__(self, rope: nn.Module):
        super().__init__()
        self.rope = rope

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> mx.array:
        y = self.rope(x.astype(mx.float32), offset = offset)
        return y.astype(x.dtype)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = args.num_hidden_layers - args.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = args.use_double_wide_mlp and is_kv_shared_layer
        hidden_dim = args.intermediate_size * (2 if use_double_wide_mlp else 1)

        self.gate_proj = nn.Linear(args.hidden_size, hidden_dim, bias = False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_dim, bias = False)
        self.down_proj = nn.Linear(hidden_dim, args.hidden_size, bias = False)
        self.act = ACT2FN[args.hidden_activation]

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm = Gemma4RMSNorm(
            args.hidden_size, eps = args.rms_norm_eps, with_scale = False
        )
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias = False)
        self.scale = mx.ones((args.hidden_size,))
        self.per_expert_scale = mx.ones((args.num_experts,))
        self._root_size = args.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = self.norm(x)
        x = x * self._root_size
        x = x * self.scale

        expert_scores = self.proj(x)
        router_probs = mx.softmax(expert_scores, axis = -1)

        top_k_indices = mx.argpartition(
            -expert_scores, kth = self.args.top_k_experts - 1, axis = -1
        )[..., : self.args.top_k_experts]

        top_k_weights = mx.take_along_axis(router_probs, top_k_indices, axis = -1)
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis = -1, keepdims = True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]
        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    def __call__(self, x, gate):
        return nn.gelu_approx(gate) * x


class Experts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        try:
            from .switch_layers import SwitchGLU
        except ImportError:
            raise ImportError(
                "Gemma4 MoE requires mlx-lm >= 0.31. Please upgrade: pip install -U mlx-lm"
            )

        self.switch_glu = SwitchGLU(
            input_dims = args.hidden_size,
            hidden_dims = args.moe_intermediate_size,
            num_experts = args.num_experts,
            activation = GeGLU(),
            bias = False,
        )

    def __call__(
        self,
        x: mx.array,
        top_k_indices: mx.array,
        top_k_weights: mx.array,
    ) -> mx.array:
        B, S, H = x.shape
        K = top_k_indices.shape[-1]

        x_flat = x.reshape(B * S, H)
        indices_flat = top_k_indices.reshape(B * S, K)

        expert_out = self.switch_glu(x_flat, indices_flat)

        weights = top_k_weights.reshape(B * S, K)[..., None]
        return (expert_out * weights).sum(axis = -2).reshape(B, S, H)


def build_rope(args: ModelArgs, layer_type: str, head_dim: int):
    rope_config = args.rope_parameters[layer_type]
    rope_type = rope_config.get("rope_type", "default")
    rope_theta = rope_config.get("rope_theta", 10_000.0)

    if rope_type == "proportional":
        partial_rotary_factor = rope_config.get("partial_rotary_factor", 1.0)
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        # Use full head_dim RoPE but with zero inv_freq for NoPE dimensions,
        # matching HF's rotate_half pairing: (0, head_dim//2), (1, head_dim//2+1), ...
        return Float32RoPE(ProportionalRoPE(head_dim, rope_angles, base = rope_theta))

    return Float32RoPE(
        initialize_rope(
            dims = head_dim,
            base = rope_theta,
            traditional = False,
            scaling_config = rope_config,
            max_position_embeddings = args.max_position_embeddings,
        )
    )


class ProportionalRoPE(nn.Module):
    """RoPE with partial rotation matching HF's rotate_half pairing.

    Rotates `rope_angles` pairs out of `head_dim // 2` total pairs.
    Non-rotated pairs get cos=1, sin=0 (identity).
    Pairing follows HF convention: (i, i + head_dim//2).
    """

    def __init__(self, head_dim: int, rope_angles: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_angles = rope_angles

        inv_freq_rotated = 1.0 / (
            base ** (mx.arange(0, 2 * rope_angles, 2, dtype = mx.float32) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            self._inv_freq = mx.concatenate(
                [inv_freq_rotated, mx.zeros(nope_angles, dtype = mx.float32)]
            )
        else:
            self._inv_freq = inv_freq_rotated

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x shape: (B, n_heads, L, head_dim)
        seq_len = x.shape[-2]
        positions = mx.arange(int(offset), int(offset) + seq_len, dtype = mx.float32)

        # (L, head_dim//2)
        freqs = mx.outer(positions, self._inv_freq)
        # (L, head_dim) — interleaved cos/sin
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # HF-style rotate_half: split at head_dim//2
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis = -1)
        return out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = args.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_kv_shared_layer = (
            layer_idx >= (args.num_hidden_layers - args.num_kv_shared_layers) > 0
        )

        self.n_heads = args.num_attention_heads
        self.n_kv_heads = (
            args.num_key_value_heads
            if self.is_sliding or not args.attention_k_eq_v
            else args.num_global_key_value_heads
        )
        self.head_dim = (
            args.head_dim
            if self.is_sliding or not args.global_head_dim
            else args.global_head_dim
        )
        self.scale = 1.0
        self.use_alternative_attention = args.attention_k_eq_v and not self.is_sliding

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias = args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias = args.attention_bias,
        )
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(
                args.hidden_size,
                self.n_kv_heads * self.head_dim,
                bias = args.attention_bias,
            )
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias = args.attention_bias,
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps = args.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps = args.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps = args.rms_norm_eps, with_scale = False
        )
        self.rope = build_rope(args, self.layer_type, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_offset: int = 0,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        offset = position_offset

        queries = self.q_proj(x).reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset = offset)

        _cache_not_empty = cache is not None and not (
            cache.empty() if hasattr(cache, "empty") else len(cache) == 0
        )
        if self.is_kv_shared_layer and _cache_not_empty:
            keys, values = cache.state
        else:
            raw_keys = self.k_proj(x).reshape(
                batch_size, seq_len, self.n_kv_heads, self.head_dim
            )
            raw_values = (
                raw_keys
                if self.v_proj is None
                else self.v_proj(x).reshape(
                    batch_size, seq_len, self.n_kv_heads, self.head_dim
                )
            )

            keys = self.k_norm(raw_keys).transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset = offset)

            values = self.v_norm(raw_values).transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache = cache,
            scale = self.scale,
            mask = mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size_per_layer_input = args.hidden_size_per_layer_input
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args, layer_idx)

        self.input_layernorm = Gemma4RMSNorm(args.hidden_size, eps = args.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            args.hidden_size, eps = args.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            args.hidden_size, eps = args.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            args.hidden_size, eps = args.rms_norm_eps
        )
        self.layer_scalar = mx.ones((1,))

        # MoE
        self.enable_moe = args.enable_moe_block
        if self.enable_moe:
            self.router = Router(args)
            self.experts = Experts(args)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                args.hidden_size, eps = args.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                args.hidden_size, eps = args.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                args.hidden_size, eps = args.rms_norm_eps
            )

        if self.hidden_size_per_layer_input:
            self.act = ACT2FN[args.hidden_activation]
            self.per_layer_input_gate = nn.Linear(
                args.hidden_size, args.hidden_size_per_layer_input, bias = False
            )
            self.per_layer_projection = nn.Linear(
                args.hidden_size_per_layer_input, args.hidden_size, bias = False
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                args.hidden_size, eps = args.rms_norm_eps
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        position_offset: int = 0,
    ) -> mx.array:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache, position_offset = position_offset)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        h = self.post_feedforward_layernorm(h)
        h = residual + h

        if self.hidden_size_per_layer_input:
            residual = h
            h = self.per_layer_input_gate(h)
            h = self.act(h)
            h = h * per_layer_input
            h = self.per_layer_projection(h)
            h = self.post_per_layer_input_norm(h)
            h = residual + h

        return h * self.layer_scalar


@partial(mx.compile, shapeless = True)
def logit_softcap(softcap, x):
    out = mx.tanh(x / softcap)
    out = out * softcap
    return out


class Gemma4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_size_per_layer_input = args.hidden_size_per_layer_input
        self.first_kv_shared_layer_idx = (
            args.num_hidden_layers - args.num_kv_shared_layers
        )
        self.embed_scale = args.hidden_size**0.5
        self.per_layer_embed_scale = args.hidden_size_per_layer_input**0.5
        self.per_layer_projection_scale = args.hidden_size**-0.5
        self.per_layer_input_scale = 2.0**-0.5

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args = args, layer_idx = i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = Gemma4RMSNorm(args.hidden_size, eps = args.rms_norm_eps)

        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                args.vocab_size_per_layer_input,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = nn.Linear(
                args.hidden_size,
                args.num_hidden_layers * args.hidden_size_per_layer_input,
                bias = False,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                args.hidden_size_per_layer_input,
                eps = args.rms_norm_eps,
            )

        concrete_layers = args.layer_types[: self.first_kv_shared_layer_idx]
        concrete_layer_types = set(concrete_layers)
        for layer_type in args.layer_types[self.first_kv_shared_layer_idx :]:
            if layer_type not in concrete_layer_types:
                raise ValueError(
                    "num_kv_shared_layers requires at least one earlier "
                    f"{layer_type!r} layer before the shared suffix."
                )
        self.layer_idx_to_cache_idx = []
        for i, layer_type in enumerate(args.layer_types):
            if i < self.first_kv_shared_layer_idx:
                self.layer_idx_to_cache_idx.append(i)
                continue

            shared_idx = (
                len(concrete_layers) - 1 - concrete_layers[::-1].index(layer_type)
            )
            self.layer_idx_to_cache_idx.append(shared_idx)

        self.first_full_idx = next(
            (
                self.layer_idx_to_cache_idx[i]
                for i, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ),
            None,
        )
        self.first_sliding_idx = next(
            (
                self.layer_idx_to_cache_idx[i]
                for i, layer_type in enumerate(args.layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )

    def get_input_embeddings(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids) * self.embed_scale

    def get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array],
    ) -> mx.array:
        if input_ids is None:
            if input_embeddings is None:
                raise ValueError(
                    "Either input ids or input embeddings are required for Gemma4 per-layer inputs."
                )
            exact_matches = mx.all(
                input_embeddings[:, :, None, :]
                == self.embed_tokens.weight[None, None, :, :] * self.embed_scale,
                axis = -1,
            )
            if not mx.all(mx.sum(exact_matches, axis = -1) == 1):
                raise ValueError(
                    "Gemma4 input embeddings must exactly match embed_tokens when "
                    "input ids are omitted."
                )
            input_ids = mx.argmax(exact_matches, axis = -1).astype(mx.int32)

        tokens = mx.where(
            input_ids < self.args.vocab_size_per_layer_input,
            input_ids,
            mx.zeros_like(input_ids),
        )
        result = self.embed_tokens_per_layer(tokens) * self.per_layer_embed_scale
        return result.reshape(
            *input_ids.shape,
            self.args.num_hidden_layers,
            self.args.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: mx.array,
        per_layer_inputs: mx.array,
    ) -> mx.array:
        per_layer_projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_projection_scale
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.args.num_hidden_layers,
            self.args.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is None:
            h = self.get_input_embeddings(inputs)
        else:
            h = input_embeddings

        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.get_per_layer_inputs(inputs, h)
            per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            if self.first_kv_shared_layer_idx < self.num_hidden_layers:
                # Must create real caches so template layers store KV
                # for shared layers to reuse — even without external cache
                cache = []
                for layer_type in self.args.layer_types[
                    : self.first_kv_shared_layer_idx
                ]:
                    if layer_type == "full_attention":
                        cache.append(KVCache())
                    else:
                        cache.append(
                            RotatingKVCache(max_size = self.args.sliding_window, keep = 0)
                        )
            else:
                cache = [None] * self.num_hidden_layers

        global_mask = (
            None
            if self.first_full_idx is None
            else create_attention_mask(h, cache[self.first_full_idx])
        )
        sliding_mask = (
            None
            if self.first_sliding_idx is None
            else create_attention_mask(
                h,
                cache[self.first_sliding_idx],
                window_size = self.args.sliding_window,
            )
        )
        global_offset = (
            0
            if self.first_full_idx is None or cache[self.first_full_idx] is None
            else cache[self.first_full_idx].offset
        )
        sliding_offset = (
            0
            if self.first_sliding_idx is None or cache[self.first_sliding_idx] is None
            else cache[self.first_sliding_idx].offset
        )

        for i, layer in enumerate(self.layers):
            layer_type = self.args.layer_types[i]
            mask = global_mask if layer_type == "full_attention" else sliding_mask
            position_offset = (
                global_offset if layer_type == "full_attention" else sliding_offset
            )
            per_layer_input = (
                None if per_layer_inputs is None else per_layer_inputs[:, :, i, :]
            )
            cache_entry = (
                None if cache is None else cache[self.layer_idx_to_cache_idx[i]]
            )
            h = layer(
                h,
                mask = mask,
                cache = cache_entry,
                per_layer_input = per_layer_input,
                position_offset = position_offset,
            )

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4Model(args)
        self.tie_word_embeddings = False
        self.final_logit_softcapping = args.final_logit_softcapping
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias = False)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache = cache, input_embeddings = input_embeddings)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
            self.pop("lm_head")

        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb" in k:
                continue

            if k.endswith(".experts.down_proj"):
                k = k.replace(
                    ".experts.down_proj", ".experts.switch_glu.down_proj.weight"
                )
                sanitized[k] = v
                continue

            if k.endswith(".experts.gate_up_proj"):
                gate_key = k.replace(
                    ".experts.gate_up_proj", ".experts.switch_glu.gate_proj.weight"
                )
                up_key = k.replace(
                    ".experts.gate_up_proj", ".experts.switch_glu.up_proj.weight"
                )
                v = v.swapaxes(-1, -2)
                mid_dim = v.shape[-1] // 2
                sanitized[gate_key] = v[..., :mid_dim].swapaxes(-1, -2)
                sanitized[up_key] = v[..., mid_dim:].swapaxes(-1, -2)
                continue

            sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types[: self.model.first_kv_shared_layer_idx]:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size = self.args.sliding_window, keep = 0)
                )
        return caches
