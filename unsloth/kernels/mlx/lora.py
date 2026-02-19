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

"""MLX LoRA (Low-Rank Adaptation) for pure MLX training.

This module provides LoRA implementations that work directly with MLX arrays,
enabling efficient fine-tuning without PyTorch dependencies.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any
import math
import mlx.core as mx


class LoRALinear:
    """LoRA linear layer for efficient fine-tuning.
    
    Implements the LoRA (Low-Rank Adaptation) technique from the paper:
    "LoRA: Low-Rank Adaptation of Large Language Models"
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: Dropout probability (default: 0.0)
        merge_weights: Whether to merge weights at inference (default: False)
        bias: Whether to use bias in base layer (default: True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.merged = False
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Base layer weights (frozen)
        scale = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features, in_features),
            dtype=mx.float32,
        )
        self.weight.requires_grad = False  # Freeze base weights
        
        if bias:
            self.bias = mx.zeros(out_features, dtype=mx.float32)
            self.bias.requires_grad = False
        else:
            self.bias = None
        
        # LoRA parameters (trainable)
        if r > 0:
            # Initialize lora_A with Kaiming uniform
            lora_a_scale = 1.0 / math.sqrt(in_features)
            self.lora_A = mx.random.uniform(
                low=-lora_a_scale,
                high=lora_a_scale,
                shape=(r, in_features),
                dtype=mx.float32,
            )
            # Initialize lora_B with zeros
            self.lora_B = mx.zeros((out_features, r), dtype=mx.float32)
        else:
            self.lora_A = None
            self.lora_B = None
        
        self.eval_mode = False
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through LoRA layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base layer output
        result = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            result = result + self.bias
        
        # Add LoRA adaptation if not merged
        if self.r > 0 and not self.merged:
            # Apply dropout if training
            if self.lora_dropout > 0 and not self.eval_mode:
                x = mx.dropout(x, self.lora_dropout)
            
            # LoRA computation: x @ A.T @ B.T * scaling
            lora_out = mx.matmul(x, self.lora_A.T)
            lora_out = mx.matmul(lora_out, self.lora_B.T)
            lora_out = lora_out * self.scaling
            
            result = result + lora_out
        
        return result
    
    def merge(self) -> None:
        """Merge LoRA weights into base weights for inference.
        
        This permanently merges the low-rank adaptation into the base weights,
        eliminating the computational overhead of the adapter during inference.
        """
        if self.merged or self.r == 0:
            return
        
        # W_merged = W_base + B @ A * scaling
        lora_weight = mx.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight = self.weight + lora_weight
        self.merged = True
    
    def unmerge(self) -> None:
        """Unmerge LoRA weights from base weights.
        
        Restores the base weights to their original state and restores
        the separate LoRA parameters.
        """
        if not self.merged or self.r == 0:
            return
        
        # W_base = W_merged - B @ A * scaling
        lora_weight = mx.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight = self.weight - lora_weight
        self.merged = False
    
    def train(self) -> None:
        """Set layer to training mode."""
        self.eval_mode = False
    
    def eval(self) -> None:
        """Set layer to evaluation mode."""
        self.eval_mode = True
    
    def get_lora_params(self) -> Dict[str, mx.array]:
        """Get LoRA parameters (trainable).
        
        Returns:
            Dictionary with lora_A and lora_B
        """
        if self.r > 0:
            return {
                "lora_A": self.lora_A,
                "lora_B": self.lora_B,
            }
        return {}
    
    def set_lora_params(self, params: Dict[str, mx.array]) -> None:
        """Set LoRA parameters.
        
        Args:
            params: Dictionary with lora_A and lora_B
        """
        if "lora_A" in params and self.lora_A is not None:
            self.lora_A = params["lora_A"]
        if "lora_B" in params and self.lora_B is not None:
            self.lora_B = params["lora_B"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer to dictionary."""
        state = {
            "weight": self.weight,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }
        
        if self.bias is not None:
            state["bias"] = self.bias
        
        if self.r > 0:
            state["lora_A"] = self.lora_A
            state["lora_B"] = self.lora_B
        
        return state
    
    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "LoRALinear":
        """Deserialize layer from dictionary."""
        layer = cls(
            in_features=state["weight"].shape[1],
            out_features=state["weight"].shape[0],
            r=state.get("r", 8),
            lora_alpha=state.get("lora_alpha", 16),
            lora_dropout=state.get("lora_dropout", 0.0),
        )
        
        layer.weight = state["weight"]
        if "bias" in state:
            layer.bias = state["bias"]
        if "lora_A" in state and layer.r > 0:
            layer.lora_A = state["lora_A"]
        if "lora_B" in state and layer.r > 0:
            layer.lora_B = state["lora_B"]
        
        return layer


class LoRAEmbedding:
    """LoRA adaptation for embedding layers.
    
    This is useful when you want to fine-tune embeddings with LoRA.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / r
        
        # Base embedding weights (frozen)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_embeddings, embedding_dim),
            dtype=mx.float32,
        )
        self.weight.requires_grad = False
        
        # LoRA parameters for embeddings
        if r > 0:
            # We'll use a projection approach for embeddings
            # Instead of modifying the embedding table directly,
            # we add a learned projection to the embeddings
            self.lora_A = mx.zeros((embedding_dim, r), dtype=mx.float32)
            self.lora_B = mx.zeros((r, embedding_dim), dtype=mx.float32)
            
            # Initialize lora_A with small random values
            lora_scale = 0.01
            self.lora_A = mx.random.normal((embedding_dim, r)) * lora_scale
        else:
            self.lora_A = None
            self.lora_B = None
        
        self.eval_mode = False
        self.merged = False
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through LoRA embedding.
        
        Args:
            x: Token indices of shape (...,)
            
        Returns:
            Embeddings of shape (..., embedding_dim)
        """
        # Base embeddings
        embeddings = self.weight[x]
        
        # Add LoRA adaptation
        if self.r > 0 and not self.merged:
            # Get base embeddings for LoRA computation
            base_emb = embeddings
            
            if self.lora_dropout > 0 and not self.eval_mode:
                base_emb = mx.dropout(base_emb, self.lora_dropout)
            
            # Apply LoRA: emb @ A @ B * scaling
            lora_out = mx.matmul(base_emb, self.lora_A)
            lora_out = mx.matmul(lora_out, self.lora_B)
            lora_out = lora_out * self.scaling
            
            embeddings = embeddings + lora_out
        
        return embeddings
    
    def train(self) -> None:
        self.eval_mode = False
    
    def eval(self) -> None:
        self.eval_mode = True


class LoRAConfig:
    """Configuration for LoRA adaptation.
    
    Args:
        r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout probability (default: 0.0)
        target_modules: List of module names to apply LoRA (default: ["q_proj", "v_proj"])
        bias: Whether to train biases (default: "none")
        modules_to_save: Additional modules to train (not frozen)
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.bias = bias
        self.modules_to_save = modules_to_save or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class GradientCheckpointing:
    """Gradient checkpointing for memory-efficient training.
    
    Trades computation for memory by recomputing activations during backward pass
    instead of storing them.
    
    Usage:
        >>> checkpointing = GradientCheckpointing()
        >>> output = checkpointing(self.layer, input_tensor)
    """

    def __init__(self):
        self.enabled = True
    
    def __call__(
        self,
        layer: Callable,
        *args,
        **kwargs,
    ) -> mx.array:
        """Apply gradient checkpointing to a layer.
        
        Args:
            layer: Layer function or callable
            *args: Positional arguments to layer
            **kwargs: Keyword arguments to layer
            
        Returns:
            Layer output
        """
        if not self.enabled:
            return layer(*args, **kwargs)
        
        # In MLX, we can use a custom gradient function
        # that recomputes the forward pass during backward
        def forward_fn(*flat_args):
            # Reconstruct args from flat list
            return layer(*args, **kwargs)
        
        # Use checkpoint pattern: don't save activations
        # Instead recompute them during backward
        return mx.checkpoint(layer)(*args, **kwargs)
    
    def enable(self) -> None:
        """Enable gradient checkpointing."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable gradient checkpointing."""
        self.enabled = False


class QuantizedLoRALinear(LoRALinear):
    """LoRA linear layer with 4-bit quantized base weights.
    
    Implements QLoRA-style quantization where the base model weights
    are quantized to 4-bit while LoRA adapters remain in full precision.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank (default: 64 for QLoRA)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: Dropout probability (default: 0.0)
        quantize_bits: Number of bits for quantization (default: 4)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        quantize_bits: int = 4,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        self.quantize_bits = quantize_bits
        
        # Quantize base weights
        self._quantize_weights()
    
    def _quantize_weights(self) -> None:
        """Quantize base weights to specified bits."""
        # Simple uniform quantization
        # For production, consider using NF4 or FP4 quantization
        
        n_levels = 2 ** self.quantize_bits
        
        # Compute scale and zero point
        w_min = self.weight.min()
        w_max = self.weight.max()
        scale = (w_max - w_min) / (n_levels - 1)
        zero_point = -w_min / scale
        
        # Quantize
        quantized = mx.round((self.weight - w_min) / scale)
        quantized = mx.clip(quantized, 0, n_levels - 1)
        
        # Store quantized weights and quantization params
        self.quantized_weight = quantized.astype(mx.uint8)
        self.weight_scale = scale
        self.weight_zero_point = zero_point
        
        # Don't keep full precision weights in memory
        self.weight = None
        self._is_quantized = True
    
    def _dequantize_weights(self) -> mx.array:
        """Dequantize weights for computation."""
        if not self._is_quantized:
            return self.weight
        
        # Dequantize: w = quantized * scale + w_min
        w = self.quantized_weight.astype(mx.float32)
        w = w * self.weight_scale
        w = w - (self.weight_zero_point * self.weight_scale)
        return w
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with quantized weights."""
        # Dequantize weights for computation
        if self._is_quantized:
            weight = self._dequantize_weights()
        else:
            weight = self.weight
        
        # Base layer output
        result = mx.matmul(x, weight.T)
        if self.bias is not None:
            result = result + self.bias
        
        # Add LoRA adaptation (in full precision)
        if self.r > 0 and not self.merged:
            if self.lora_dropout > 0 and not self.eval_mode:
                x = mx.dropout(x, self.lora_dropout)
            
            lora_out = mx.matmul(x, self.lora_A.T)
            lora_out = mx.matmul(lora_out, self.lora_B.T)
            lora_out = lora_out * self.scaling
            
            result = result + lora_out
        
        return result


def mark_only_lora_as_trainable(model: Any) -> None:
    """Mark only LoRA parameters as trainable.
    
    This function freezes all parameters except LoRA weights.
    
    Args:
        model: Model with LoRA layers
    """
    def mark_module(module):
        if hasattr(module, "weight"):
            module.weight.requires_grad = False
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.requires_grad = False
        
        # Mark LoRA parameters as trainable
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                module.lora_A.requires_grad = True
            if module.lora_B is not None:
                module.lora_B.requires_grad = True
        
        # Recursively process child modules
        if hasattr(module, "__dict__"):
            for child in module.__dict__.values():
                if hasattr(child, "__dict__") or isinstance(child, (list, tuple)):
                    if isinstance(child, (list, tuple)):
                        for c in child:
                            mark_module(c)
                    else:
                        mark_module(child)
    
    mark_module(model)


def get_peft_model(model: Any, lora_config: LoRAConfig) -> Any:
    """Convert a model to PEFT model with LoRA.
    
    This function replaces target linear layers with LoRA layers.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Model with LoRA layers
    """
    def replace_modules(module, prefix=""):
        if not hasattr(module, "__dict__"):
            return
        
        for name, child in list(module.__dict__.items()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this is a target module
            is_target = any(target in full_name for target in lora_config.target_modules)
            
            if is_target and hasattr(child, "weight"):
                # Replace with LoRA layer
                in_features = child.weight.shape[1]
                out_features = child.weight.shape[0]
                has_bias = hasattr(child, "bias") and child.bias is not None
                
                lora_layer = LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    bias=has_bias,
                )
                
                # Copy base weights
                lora_layer.weight = child.weight
                if has_bias:
                    lora_layer.bias = child.bias
                
                # Replace module
                setattr(module, name, lora_layer)
            elif hasattr(child, "__dict__"):
                replace_modules(child, full_name)
            elif isinstance(child, (list, tuple)):
                for i, item in enumerate(child):
                    if hasattr(item, "__dict__"):
                        replace_modules(item, f"{full_name}[{i}]")
    
    replace_modules(model)
    mark_only_lora_as_trainable(model)
    
    return model


def get_lora_state_dict(model: Any) -> Dict[str, mx.array]:
    """Get state dict with only LoRA parameters.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Dictionary of LoRA parameters
    """
    lora_state = {}
    
    def collect_lora_params(module, prefix=""):
        if isinstance(module, LoRALinear) and module.r > 0:
            if module.lora_A is not None:
                lora_state[f"{prefix}.lora_A"] = module.lora_A
            if module.lora_B is not None:
                lora_state[f"{prefix}.lora_B"] = module.lora_B
            lora_state[f"{prefix}.scaling"] = mx.array(module.scaling)
        
        if hasattr(module, "__dict__"):
            for name, child in module.__dict__.items():
                if name.startswith("_"):
                    continue
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, LoRALinear):
                    collect_lora_params(child, full_name)
                elif hasattr(child, "__dict__"):
                    collect_lora_params(child, full_name)
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        collect_lora_params(item, f"{full_name}[{i}]")
    
    collect_lora_params(model)
    return lora_state


__all__ = [
    "LoRALinear",
    "LoRAEmbedding",
    "LoRAConfig",
    "GradientCheckpointing",
    "QuantizedLoRALinear",
    "mark_only_lora_as_trainable",
    "get_peft_model",
    "get_lora_state_dict",
]
