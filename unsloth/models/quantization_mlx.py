# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.quantization import quantize_4bit
from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

def quantize_model_mlx(model, group_size=64):
    """
    Iterates through the model's linear layers and quantizes them to 4-bit MLX.
    Attaches the quantized weight to the module as `weight_quant`.
    """
    print("Unsloth: Quantizing model to 4-bit MLX... this might take a minute.")
    
    # Identify linear layers to quantize
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this module should be skipped
            is_skipped = False
            for skip in SKIP_QUANTIZATION_MODULES:
                if skip in name:
                    is_skipped = True
                    break
            
            if is_skipped: continue
            
            # Key projectable layers for Unsloth
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                # Perform 4-bit quantization
                # We store it as a private attribute that the Fast LORA kernels will check
                with torch.no_grad():
                    module.weight_quant = quantize_4bit(module.weight, group_size=group_size)
                    
    print("Unsloth: 4-bit MLX quantization complete.")
    return model
