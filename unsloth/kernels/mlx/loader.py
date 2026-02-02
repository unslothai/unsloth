# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import os
import torch
import mlx.core as mx
from .quantization import MLXQuantizedWeight
from .bridge import torch_to_mlx


# MLX Key Mapping for Llama/Mistral architectures (mlx-lm style)
MLX_KEY_MAPPING = {
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo",
    "model.layers.{}.mlp.gate_proj.weight":    "layers.{}.feed_forward.w1",
    "model.layers.{}.mlp.up_proj.weight":      "layers.{}.feed_forward.w3",
    "model.layers.{}.mlp.down_proj.weight":    "layers.{}.feed_forward.w2",
}

def load_mlx_weights(model, weights_path):
    """
    Loads pre-quantized weights from an MLX-format file (safetensors or npz)
    and populates model.param._mlx_cache for zero-copy optimized inference.
    """
    if not os.path.exists(weights_path):
        return False
        
    print(f"Unsloth: Loading pre-quantized MLX weights from {os.path.basename(weights_path)}...")
    
    # Load MLX weights
    mlx_weights = mx.load(weights_path)
    
    num_layers = 0
    while f"layers.{num_layers}.attention.wq.weight" in mlx_weights or f"layers.{num_layers}.attention.wq" in mlx_weights:
        num_layers += 1
        
    count = 0
    for i in range(num_layers):
        for hf_temp, mlx_temp in MLX_KEY_MAPPING.items():
            hf_key = hf_temp.format(i)
            mlx_base = mlx_temp.format(i)
            
            # Find the parameter in the model
            param = None
            for name, p in model.named_parameters():
                if name == hf_key:
                    param = p
                    break
            
            if param is None: continue
            
            # Extract MLX components
            # Handle both 'key.weight' and 'key' naming conventions
            w = mlx_weights.get(f"{mlx_base}.weight") or mlx_weights.get(mlx_base)
            s = mlx_weights.get(f"{mlx_base}.scales")
            b = mlx_weights.get(f"{mlx_base}.biases")
            
            if w is not None and s is not None:
                # Meta-information from the weights or model config
                # Default to 4-bit, 64 group_size if not found
                bits = 4 
                group_size = 64
                
                # Check for explicit bits/group_size metadata in the file
                bits = mlx_weights.get(f"{mlx_base}.bits", mx.array(bits)).item()
                group_size = mlx_weights.get(f"{mlx_base}.group_size", mx.array(group_size)).item()
                
                param._mlx_cache = MLXQuantizedWeight(
                    weight = w,
                    scales = s,
                    biases = b,
                    group_size = group_size,
                    bits = bits
                )
                count += 1

    print(f"Unsloth: Successfully mapped {count} MLX-native weights to model.")
    return True
