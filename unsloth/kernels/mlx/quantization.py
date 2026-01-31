# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx


class MLXQuantizedWeight:
    """
    Container for quantized weights (4-bit).
    Holds the packed weight data, scales, and biases.
    """

    def __init__(self, weight, scales, biases, group_size, bits = 4):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.shape = (
            weight.shape[0],
            weight.shape[1] * (32 // bits),
        )  # Approx original shape reconstruction logic

    @property
    def T(self):
        """
        Helper for transposed access logic.
        MLX quantized matmul expects (x, w, scales, biases).
        """
        return self


def quantize_4bit(tensor, group_size = 64):
    """
    Quantizes a PyTorch tensor to 4-bit MLX format.

    Args:
        tensor (torch.Tensor): Input tensor (Out, In).
        group_size (int): Quantization group size (default 64).

    Returns:
        MLXQuantizedWeight: Packed quantized object.
    """
    from unsloth.kernels.mlx.bridge import torch_to_mlx

    # 1. Convert to MLX (FP16/FP32)
    # Ensure it's on the right device? torch_to_mlx handles it or we ensure it.
    w_mlx = torch_to_mlx(tensor)

    # MLX quantize expects (Out, In) usually.
    # mx.quantize returns (quantized_w, scales, biases)
    # Check MLX docs: quantize(w, group_size, bits) -> (w_q, scales, biases)
    w_q, scales, biases = mx.quantize(w_mlx, group_size, 4)

    return MLXQuantizedWeight(w_q, scales, biases, group_size, bits = 4)
