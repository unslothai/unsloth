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

"""
Truncated router for CGGR difficulty scoring.

Creates a lightweight router from the first N layers of a model to score
token difficulty without running the full forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

__all__ = ["TruncatedRouter", "create_truncated_router"]


class TruncatedRouter(nn.Module):
    """
    A truncated version of a language model using only the first N layers.
    
    Used for fast difficulty scoring in CGGR. Shares weights with the parent
    model, so uses zero additional memory.
    
    Args:
        model: The parent HuggingFace model
        num_layers: Number of decoder layers to use (default: 2)
    """
    
    def __init__(self, model: nn.Module, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # Get the base model (handle PEFT wrapping)
        base_model = model
        if hasattr(model, "base_model"):
            base_model = model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model
        
        # Store reference to model components (shares weights, no copy)
        self.embed_tokens = self._get_embed_tokens(base_model)
        self.layers = self._get_layers(base_model, num_layers)
        self.norm = self._get_norm(base_model)
        self.lm_head = self._get_lm_head(model, base_model)
        
        # Store config for reference
        self.config = getattr(base_model, "config", None)
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        
    def _get_embed_tokens(self, model: nn.Module) -> nn.Module:
        """Extract embedding layer from model."""
        if hasattr(model, "embed_tokens"):
            return model.embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte  # GPT-2 style
        raise ValueError(f"Cannot find embedding layer in model: {type(model)}")
    
    def _get_layers(self, model: nn.Module, num_layers: int) -> nn.ModuleList:
        """Extract first N decoder layers."""
        layers = None
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h  # GPT-2 style
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer  # BERT style
        
        if layers is None:
            raise ValueError(f"Cannot find decoder layers in model: {type(model)}")
        
        # Return reference to first N layers (shares weights)
        return nn.ModuleList([layers[i] for i in range(min(num_layers, len(layers)))])
    
    def _get_norm(self, model: nn.Module) -> Optional[nn.Module]:
        """Extract final normalization layer."""
        if hasattr(model, "norm"):
            return model.norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            return model.transformer.ln_f  # GPT-2 style
        return None
    
    def _get_lm_head(self, original_model: nn.Module, base_model: nn.Module) -> nn.Module:
        """Extract language model head."""
        if hasattr(original_model, "lm_head"):
            return original_model.lm_head
        if hasattr(base_model, "lm_head"):
            return base_model.lm_head
        if hasattr(original_model, "base_model") and hasattr(original_model.base_model, "lm_head"):
            return original_model.base_model.lm_head
        raise ValueError(f"Cannot find lm_head in model: {type(original_model)}")
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through truncated model to get logits for scoring.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Generate position_ids if not provided (needed for RoPE)
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
        
        # Simple expansion for 2D mask to 4D if needed by layers
        mask_input = attention_mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert [batch, seq] to [batch, 1, 1, seq]
            mask_input = attention_mask[:, None, None, :]
            mask_input = mask_input.to(dtype=hidden_states.dtype)
            mask_input = (1.0 - mask_input) * torch.finfo(hidden_states.dtype).min
        
        # Pass through truncated layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask_input,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # Apply final norm if available
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return logits


def create_truncated_router(
    model: nn.Module,
    num_layers: int = 2,
) -> TruncatedRouter:
    """
    Create a truncated router from a model for CGGR difficulty scoring.
    
    The router shares weights with the parent model, so uses zero additional
    GPU memory. It only runs the first N layers to quickly estimate token
    difficulty.
    
    Args:
        model: HuggingFace model (can be PEFT-wrapped)
        num_layers: Number of decoder layers to use (default: 2)
        
    Returns:
        TruncatedRouter instance
        
    Example:
        >>> from unsloth import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> router = create_truncated_router(model, num_layers=2)
    """
    router = TruncatedRouter(model, num_layers=num_layers)
    logger.info(f"Created truncated router with {num_layers} layers for CGGR scoring")
    return router
