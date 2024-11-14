# Copyright © 2023 Apple Inc.

import inspect
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from . import mlx_utils as utils
from mlx.utils import tree_flatten, tree_unflatten

@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

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
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized:
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
        lora_rank: int = 8,
        bias: bool = False,
        scale: float = 20.0,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + self.scale * z


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache
    


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache
    
    def save_merged_model(self,args):
        model, tokenizer, config = utils.load(args.model_name)

        # Load the LoRA adapter weights which we assume should exist by this point
        if not Path(args.save_path,args.adapter_file).is_file():
            raise ValueError(
            f"Adapter file {args.adapter_file} missing. ")
        
        # Load adapters and get number of LoRA layers
        adapters = list(mx.load(args.save_path+"/"+args.adapter_file).items())
        lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

        # Freeze all layers other than LORA linears
        model.freeze()
        for l in model.model.layers[len(model.model.layers) - lora_layers :]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

        model.update(tree_unflatten(adapters))
        fused_linears = [
            (n, m.to_linear())
            for n, m in model.named_modules()
            if isinstance(m, LoRALinear)
        ]

        model.update_modules(tree_unflatten(fused_linears))

        if False:
            de_quantize_layers = []
            for n, m in model.named_modules():
                if isinstance(m, nn.QuantizedLinear):
                    bias = "bias" in m
                    weight = m.weight
                    weight = mx.dequantize(
                        weight,
                        m.scales,
                        m.biases,
                        m.group_size,
                        m.bits,
                    ).astype(mx.float16)
                    output_dims, input_dims = weight.shape
                    linear = nn.Linear(input_dims, output_dims, bias=bias)
                    linear.weight = weight
                    if bias:
                        linear.bias = m.bias
                    de_quantize_layers.append((n, linear))

            model.update_modules(tree_unflatten(de_quantize_layers))

        weights = dict(tree_flatten(model.parameters()))
        utils.save_model(args.save_path, weights, tokenizer, config)

        if args.push_model:
            from huggingface_hub import whoami
            try: 
                username = whoami(token = args.hub_token)["name"]
            except:
                raise RuntimeError(
                    "Unsloth: Please supply a token!\n"\
                    "Go to https://huggingface.co/settings/tokens"
                )
            pass
        pass

        if  args.push_model and args.hub_path is not None:
            hf_path = args.hub_path
            if not Path(args.model_name).exists():
                # If the model path doesn't exist, assume it's an HF repo
                hf_path = args.model_name
            elif hf_path is None:
                raise ValueError(
                    "Must provide original Hugging Face repo to upload local model."
                )
            utils.upload_to_hub(config['_name_or_path'],config['model_type'],username,args.save_path, args.hub_path,args.hub_token)
