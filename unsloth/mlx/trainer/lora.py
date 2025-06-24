import math

import mlx.core as mx
import mlx.nn as nn

from .switch_layers import QuantizedSwitchLinear, SwitchLinear


class LoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = linear.scales.dtype
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


class LoRASwitchLinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Module,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        lora_b = (self.scale * self.lora_b).astype(dtype)
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims).astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = fused_linear.to_quantized(linear.group_size, linear.bits)

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(r * num_experts, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

    def __call__(self, x, indices):
        shape = x.shape[:-3] + (self.num_experts, -1)

        y = self.linear(x, indices)
        z = (self.dropout(x) @ self.lora_a.T).reshape(shape)
        z = mx.take_along_axis(z, indices[..., None], axis=-2)
        z = z[..., None, :] @ self.lora_b[indices].swapaxes(-2, -1)

        return y + (self.scale * z).astype(x.dtype)


class LoRAEmbedding(nn.Module):
    @staticmethod
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        num_embeddings, dims = embedding.weight.shape
        if isinstance(embedding, nn.QuantizedEmbedding):
            dims *= 32 // embedding.bits
        lora_embedding = LoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_embedding.embedding = embedding
        return lora_embedding

    def fuse(self, de_quantize: bool = False):
        embedding = self.embedding
        weight = embedding.weight
        is_quantized = isinstance(embedding, nn.QuantizedEmbedding)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = embedding.scales.dtype
            weight = mx.dequantize(
                weight,
                embedding.scales,
                embedding.biases,
                embedding.group_size,
                embedding.bits,
            )
        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        lora_a = (self.scale * self.lora_a).astype(dtype)
        lora_b = self.lora_b.astype(dtype)
        fused_embedding.weight = weight + lora_a @ lora_b

        if is_quantized and not de_quantize:
            fused_embedding = nn.QuantizedEmbedding.from_embedding(
                fused_embedding,
                embedding.group_size,
                embedding.bits,
            )

        return fused_embedding

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        # Regular embedding layer
        self.embedding = nn.Embedding(num_embeddings, dims)
        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(num_embeddings)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros(shape=(r, dims))

    def __call__(self, x):
        y = self.embedding(x)
        z = self.dropout(self.lora_a[x] @ self.lora_b)
        out = y + (self.scale * z).astype(y.dtype)
        return out

    def as_linear(self, x):
        y = self.embedding.as_linear(x)
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        return y + (self.scale * z).astype(x.dtype)
