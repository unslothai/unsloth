# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx


class MLXQuantizedWeight:
    """
    Container for quantized weights (4-bit).
    Holds the packed weight data, scales, and biases for MLX.
    """

    def __init__(self, weight, scales, biases, group_size, bits=4, shape=None):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def dequantize(self):
        """Dequantize back to FP16."""
        return mx.dequantize(
            self.weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
        )

    def __mx_compile_flatten__(self):
        """Allows mx.compile to trace this object."""
        return (
            [self.weight, self.scales, self.biases],
            {"group_size": self.group_size, "bits": self.bits, "shape": self._shape},
        )

    @classmethod
    def __mx_compile_unflatten__(cls, arrays, metadata):
        return cls(*arrays, **metadata)


def quantize_4bit(tensor, group_size=64):
    """
    Quantizes a PyTorch tensor to 4-bit MLX format.

    Args:
        tensor (torch.Tensor): Input tensor (Out, In).
        group_size (int): Quantization group size (default 64).

    Returns:
        MLXQuantizedWeight: Packed quantized object.
    """
    from .bridge import torch_to_mlx

    # 1. Convert to MLX
    w_mlx = torch_to_mlx(tensor)
    original_shape = w_mlx.shape

    # 2. MLX quantize
    # mx.quantize returns (quantized_w, scales, biases)
    w_q, scales, biases = mx.quantize(w_mlx, group_size, 4)

    return MLXQuantizedWeight(
        w_q, scales, biases, group_size, bits=4, shape=original_shape
    )
