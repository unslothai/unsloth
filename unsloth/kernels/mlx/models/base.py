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

"""Pure MLX model implementations for Apple Silicon training.

This module provides native MLX model definitions that run entirely on the GPU
without PyTorch conversion overhead, maximizing Apple Silicon performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
import mlx.nn as mnn


@dataclass
class MLXModelConfig:
    """Configuration for MLX models."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    hidden_act: str = "silu"
    sliding_window: Optional[int] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 8
    lora_alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = None
    bias: str = "none"
    inference_mode: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


DEFAULT_LORA_TARGET_MODULES = {
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class MLXLinear(mnn.Module):
    """MLX-native linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: type = mx.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        scale = 1.0 / (in_features ** 0.5)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features, in_features),
            dtype=dtype,
        )

        if bias:
            self.bias = mx.zeros(out_features, dtype=dtype)
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.bias is not None:
            return mx.matmul(x, self.weight.T) + self.bias
        return mx.matmul(x, self.weight.T)

    def to_dict(self) -> dict:
        state = {"weight": self.weight}
        if self.bias is not None:
            state["bias"] = self.bias
        return state

    @classmethod
    def from_dict(cls, state: dict, bias: bool = True):
        layer = cls(
            in_features=state["weight"].shape[1],
            out_features=state["weight"].shape[0],
            bias=bias,
            dtype=state["weight"].dtype,
        )
        layer.weight = state["weight"]
        if "bias" in state:
            layer.bias = state["bias"]
        return layer


class MLXEmbedding(mnn.Module):
    """MLX-native embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: type = mx.float32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        scale = 1.0 / (embedding_dim ** 0.5)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_embeddings, embedding_dim),
            dtype=dtype,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x]

    def to_dict(self) -> dict:
        return {"weight": self.weight}

    @classmethod
    def from_dict(cls, state: dict):
        return cls(
            num_embeddings=state["weight"].shape[0],
            embedding_dim=state["weight"].shape[1],
            dtype=state["weight"].dtype,
        )


class MLXRMSNorm(mnn.Module):
    """MLX-native RMSNorm layer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: type = mx.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype

        self.weight = mx.ones(hidden_size, dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            x,
            self.weight,
            self.eps,
        )

    def to_dict(self) -> dict:
        return {"weight": self.weight}

    @classmethod
    def from_dict(cls, state: dict, hidden_size: int, eps: float = 1e-6):
        layer = cls(hidden_size=hidden_size, eps=eps)
        layer.weight = state["weight"]
        return layer


class MLXLayerNorm(mnn.Module):
    """MLX-native LayerNorm layer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        bias: bool = True,
        dtype: type = mx.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype

        self.weight = mx.ones(hidden_size, dtype=dtype)
        self.bias = mx.zeros(hidden_size, dtype=dtype) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(
            x,
            self.weight,
            self.bias,
            self.eps,
        )

    def to_dict(self) -> dict:
        state = {"weight": self.weight}
        if self.bias is not None:
            state["bias"] = self.bias
        return state

    @classmethod
    def from_dict(cls, state: dict, hidden_size: int, eps: float = 1e-6, bias: bool = True):
        layer = cls(hidden_size=hidden_size, eps=eps, bias=bias)
        layer.weight = state["weight"]
        if "bias" in state:
            layer.bias = state["bias"]
        return layer
