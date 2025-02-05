# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "patch_vllm",
    "vllm_dynamic_quant_supported",
    "convert_vllm_to_huggingface",
    "get_vllm_state_dict",
    "assert_same_state_dict",
    "load_vllm",
    "create_batches",
]
from typing import Optional, List, Tuple, Dict, Any
import importlib.util
import re
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import math
import gc
import os
import contextlib
from unsloth_zoo.utils import _get_dtype

# Ignore logging messages
import logging
class HideLoggingMessage(logging.Filter):
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())
pass

if importlib.util.find_spec("vllm") is not None:

    # Allow unsloth dynamic quants to work
    def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules):
        # Split the prefix into its dot-separated components
        components = prefix.split('.')
        # Check if any of the skip modules exactly matches any component
        vllm_check = any(
            module_name in components
            for module_name in llm_int8_skip_modules
        )

        # Allow certain layers to not be quantized
        components = set(".".join(components[:i+1]) for i in range(len(components)))
        unsloth_check = len(set(llm_int8_skip_modules) & components) != 0
        
        return vllm_check or unsloth_check
    pass

    # Fix force using torch.bfloat16 all the time and make it dynamic
    def _apply_4bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # only load the bitsandbytes module when needed
        from bitsandbytes import matmul_4bit

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True

        qweight = layer.weight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets
        inference_dtype = quant_states[0].dtype
        bf_x = x.to(inference_dtype) # Originally used bfloat16

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                            out_dim_1,
                            dtype=inference_dtype,
                            device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]
            # It is more efficient to use out kwarg like
            # matmul_4bit(..., out = ...).  Infeasible now due to the bug
            # https://github.com/TimDettmers/bitsandbytes/issues/1235.
            # Need to change  after the bug is fixed.
            out[:, current_index:current_index + output_size] = matmul_4bit(
                bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

            current_index += output_size

        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        return out
    pass

    def patch_vllm_bitsandbytes():
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.is_layer_skipped_bnb = is_layer_skipped_bnb
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesLinearMethod._apply_4bit_weight = _apply_4bit_weight

        # Disable all not supported messages
        from vllm.config import logger as vllm_config_logger
        vllm_config_logger.addFilter(HideLoggingMessage("not supported"))
        vllm_config_logger.addFilter(HideLoggingMessage("is not tested"))
        vllm_config_logger.addFilter(HideLoggingMessage("is not fully optimized"))
        vllm_config_logger.addFilter(HideLoggingMessage("not set"))
        del vllm_config_logger
    pass

    def patch_vllm_compute_dtype(dtype = torch.float16):
        # vLLM defaults to using the model config file's compute_dtype
        # We shall fix it dynamically!
        import vllm.model_executor.layers.quantization.bitsandbytes
        old_config = vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig

        dtype = str(dtype)
        if dtype.startswith("torch."): dtype = dtype[len("torch."):]
        os.environ["UNSLOTH_bnb_4bit_compute_dtype"] = dtype

        class BitsAndBytesConfig(
            vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig
        ):
            def __init__(self, *args, **kwargs):
                dtype = os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", kwargs["bnb_4bit_compute_dtype"])
                kwargs["bnb_4bit_compute_dtype"] = dtype
                print(f"Unsloth: vLLM Bitsandbytes config using kwargs = {kwargs}")
                super().__init__(*args, **kwargs)
            pass
        pass

        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig = BitsAndBytesConfig
        return old_config
    pass

    def unpatch_vllm_compute_dtype(old_config):
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig = old_config
        del os.environ["UNSLOTH_bnb_4bit_compute_dtype"]
    pass
else:
    def patch_vllm_bitsandbytes():
        return
    pass

    def patch_vllm_compute_dtype():
        return
    pass

    def unpatch_vllm_compute_dtype(old_config):
        return
    pass
pass


if importlib.util.find_spec("bitsandbytes") is not None:
    import bitsandbytes.functional
    from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

    # Force offsets to be in float32 and not bfloat16 / float16
    @classmethod
    def from_dict(cls, qs_dict: Dict[str, Any], device: torch.device) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            # Must use float32 and disable autocasting - vLLM fails!
            # offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            with torch.autocast(device_type = "cuda", enabled = False):
                offset = torch.tensor(qs_dict["nested_offset"], dtype = torch.float32, device = "cuda")
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            # dtype=getattr(torch, qs_dict["dtype"]),
            # Patch over the compute dtype for vLLM
            dtype=getattr(torch, os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", qs_dict["dtype"])),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state
    pass

    import bitsandbytes.nn.modules
    class Linear4bit(bitsandbytes.nn.modules.Linear4bit):
        def __init__(self, *args, **kwargs):
            compute_dtype = os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", None)
            if compute_dtype is not None:
                compute_dtype = getattr(torch, compute_dtype)
                kwargs["compute_dtype"] = compute_dtype
            super().__init__(*args, **kwargs)
        pass
    pass
    
    def patch_bitsandbytes_quant_state():
        bitsandbytes.functional.QuantState.from_dict = from_dict
        bitsandbytes.nn.modules.Linear4bit = Linear4bit
    pass

    def patch_bitsandbytes_compute_dtype(dtype):
        dtype = str(dtype)
        if dtype.startswith("torch."): dtype = dtype[len("torch."):]
        os.environ["UNSLOTH_bnb_4bit_compute_dtype"] = dtype
        return
    pass

    def unpatch_bitsandbytes_compute_dtype():
        del os.environ["UNSLOTH_bnb_4bit_compute_dtype"]
        return
    pass
else:
    def patch_bitsandbytes_quant_state():
        return
    pass

    def patch_bitsandbytes_compute_dtype(dtype):
        return
    pass

    def unpatch_bitsandbytes_compute_dtype():
        return
    pass
pass


def patch_vllm():
    patch_bitsandbytes_quant_state()
    patch_vllm_bitsandbytes()
pass


def vllm_dynamic_quant_supported(
    model_name,
    config,
) -> bool:
    # All Unsloth Zoo code licensed under LGPLv3

    # Check if vLLM supports some Unsloth dynamic quants
    # Sometimes we quantize modules within a layer, but not an entire layer
    # If so, then we cannot use dynamic quants for now
    if not model_name.lower().endswith("unsloth-bnb-4bit"): return True
    if "quantization_config" not in config: return True

    llm_int8_skip_modules = config.quantization_config.get("llm_int8_skip_modules", {})
    
    # Only allow layer modules ie model.layers.1.mlp or model.layers.1.self_attn
    
    # Exclude model.layers.27.mlp.gate_proj
    parent_llm_int8_skip_modules = []
    for module in llm_int8_skip_modules:
        # $ means end of string
        if re.search(r"[\d]\.[^\.]{1,}$", module) or "." not in module:
            parent_llm_int8_skip_modules.append(module)
    pass

    parent_llm_int8_skip_modules = set(parent_llm_int8_skip_modules)
    find_regex = "|".join(re.escape(x) for x in parent_llm_int8_skip_modules)
    find_regex = re.compile(find_regex)

    for module in llm_int8_skip_modules:
        # Could not find parent
        if find_regex.search(module) is None: return False
    return True
pass


@torch.inference_mode
def get_vllm_state_dict(llm, return_state_dict = False, config = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules and returns HF equivalent state_dict
    try:
        llm_engine = getattr(llm, "llm_engine", llm)
        vllm_internals = llm_engine.model_executor.driver_worker.model_runner.model
    except:
        raise RuntimeError("Unsloth: Failed to access llm.llm_engine.model_executor.driver_worker.model_runner.model")
    pass
    assert(config is not None)
    vocab_size = config.vocab_size

    state_dict = OrderedDict()
    quant_state_dict = OrderedDict()

    def get_state_dict(prefix, kk, state_dict, proj):
        proj = getattr(proj, "base_layer", proj)
        qweight = proj.weight
        if hasattr(proj, "output_sizes"):
            dim_offsets = np.cumsum([0] + proj.output_sizes)
        else:
            dim_offsets = [0, qweight.shape[0]]
        pass

        if hasattr(qweight, "bnb_quant_state"):
            # Bitsandbytes quantizations
            quant_states = qweight.bnb_quant_state
            offsets = qweight.bnb_shard_offsets
            state_dict[prefix + ".weight"] = qweight[offsets[kk] : offsets[kk + 1]]
            quant_state_dict[prefix + ".weight.quant_state"] = quant_states[kk]
            quant_state_dict[prefix + ".weight"] = state_dict[prefix + ".weight"]
            quant_state = quant_states[kk].as_dict(packed = True)
            for k, v in quant_state.items():
                state_dict[prefix + ".weight." + k] = v
            pass
        else:
            # Normal FP16 weights
            qweight.requires_grad_(False) # Disable grad - sometimes vLLM forgets
            state_dict[prefix + ".weight"] = qweight[dim_offsets[kk] : dim_offsets[kk + 1]]
            quant_state_dict[prefix + ".weight"] = state_dict[prefix + ".weight"]
        pass

        # Check bias
        bias = getattr(proj, "bias", None)
        if bias is not None:
            bias.requires_grad_(False) # Disable grad - sometimes vLLM forgets
            state_dict[prefix + ".bias"] = bias[dim_offsets[kk] : dim_offsets[kk + 1]]
            quant_state_dict[prefix + ".bias"] = state_dict[prefix + ".bias"]
        pass
    pass

    # Embedding
    embed_tokens = vllm_internals.model.embed_tokens
    embed_tokens = getattr(embed_tokens, "base_layer", embed_tokens).weight.data

    # Counteract vLLM padding vocabs for LoRA
    if vocab_size is not None: embed_tokens = embed_tokens[:vocab_size]
    state_dict["model.embed_tokens.weight"] = embed_tokens
    quant_state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]

    # All layers
    for kk in range(len(vllm_internals.model.layers)):
        proj = vllm_internals.model.layers[kk].self_attn.qkv_proj
        get_state_dict(f"model.layers.{kk}.self_attn.q_proj", 0, state_dict, proj)
        get_state_dict(f"model.layers.{kk}.self_attn.k_proj", 1, state_dict, proj)
        get_state_dict(f"model.layers.{kk}.self_attn.v_proj", 2, state_dict, proj)

        proj = vllm_internals.model.layers[kk].self_attn.o_proj
        get_state_dict(f"model.layers.{kk}.self_attn.o_proj", 0, state_dict, proj)

        proj = vllm_internals.model.layers[kk].mlp.gate_up_proj
        get_state_dict(f"model.layers.{kk}.mlp.gate_proj", 0, state_dict, proj)
        get_state_dict(f"model.layers.{kk}.mlp.up_proj",   1, state_dict, proj)

        proj = vllm_internals.model.layers[kk].mlp.down_proj
        get_state_dict(f"model.layers.{kk}.mlp.down_proj", 0, state_dict, proj)

        state_dict[f"model.layers.{kk}.input_layernorm.weight"] = \
            vllm_internals.model.layers[kk].input_layernorm.state_dict()["weight"]
        quant_state_dict[f"model.layers.{kk}.input_layernorm.weight"] = \
            state_dict[f"model.layers.{kk}.input_layernorm.weight"]

        state_dict[f"model.layers.{kk}.post_attention_layernorm.weight"] = \
            vllm_internals.model.layers[kk].post_attention_layernorm.state_dict()["weight"]
        quant_state_dict[f"model.layers.{kk}.post_attention_layernorm.weight"] = \
            state_dict[f"model.layers.{kk}.post_attention_layernorm.weight"]
    pass

    # Norm
    state_dict["model.norm.weight"] = vllm_internals.model.norm.weight.data
    quant_state_dict["model.norm.weight"] = state_dict["model.norm.weight"]

    # LM Head
    if getattr(config, "tie_word_embeddings", True) is False:
        lm_head = vllm_internals.lm_head
        lm_head = getattr(lm_head, "base_layer", lm_head).weight.data

        # Counteract vLLM padding vocabs for LoRA
        if vocab_size is not None: lm_head = lm_head[:vocab_size]

        state_dict["lm_head.weight"] = lm_head
        quant_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    pass

    if not return_state_dict: state_dict = None
    return state_dict, quant_state_dict
pass


@torch.inference_mode
def assert_same_state_dict(old_state_dict, new_state_dict):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if state_dict are equivalent

    difference = new_state_dict.keys() ^ old_state_dict.keys()
    difference -= set(("lm_head.weight",))
    if len(difference) != 0:
        raise RuntimeError(f"Unsloth: Failed comparing state_dict with {difference}")
    pass

    for key in old_state_dict:
        try:
            torch.testing.assert_close(old_state_dict[key], new_state_dict[key], check_stride = True)
        except Exception as error:
            if key == "lm_head.weight":
                # Maybe tied embeddings?
                key1 = key if key in old_state_dict else "model.embed_tokens.weight"
                key2 = key if key in new_state_dict else "model.embed_tokens.weight"
                torch.testing.assert_close(old_state_dict[key1], new_state_dict[key2], check_stride = True)
            else:
                raise RuntimeError(f"[{key}]\n{str(error)}")
        pass
    pass
pass


@torch.inference_mode
def create_empty_causal_lm(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    # Empty model from config
    new_config = deepcopy(config)
    new_config.intermediate_size = 0
    new_config.hidden_size = 0
    new_config.vocab_size = 1
    new_config.pad_token_id = 0

    # Set attention module head_dim
    # Otherwise will get error if (head_dim)**-0.5 is seen like in Qwen
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    new_config.update({"head_dim" : head_dim})

    from transformers import AutoModelForCausalLM
    new_model = AutoModelForCausalLM.from_config(
        new_config,
        attn_implementation = "eager",
    )
    new_model = new_model.to(device = "cuda:0", dtype = dtype)
    return new_model
pass


@torch.inference_mode
def convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules to create HF compatible model
    config.update({"torch_dtype" : dtype}) # Do not use config file's dtype!
    new_model = create_empty_causal_lm(config, dtype)
    quantization_config = getattr(config, "quantization_config", {})
    kwargs = dict()
    if quantization_config != {}:
        # Get quantization_config flags
        compute_dtype = _get_dtype(quantization_config["bnb_4bit_compute_dtype"])
        compute_dtype = dtype # Do not use config file's dtype!
        kwargs["compress_statistics"] = quantization_config["bnb_4bit_use_double_quant"]
        kwargs["quant_type"] = quantization_config["bnb_4bit_quant_type"]
        kwargs["quant_storage"] = _get_dtype(quantization_config["bnb_4bit_quant_storage"])
    pass

    from bitsandbytes.nn.modules import Linear4bit, Params4bit
    from torch.nn.modules import Linear

    layer_names = [
        "model.layers.{kk}.self_attn.q_proj",
        "model.layers.{kk}.self_attn.k_proj",
        "model.layers.{kk}.self_attn.v_proj",
        "model.layers.{kk}.self_attn.o_proj",
        "model.layers.{kk}.mlp.gate_proj",
        "model.layers.{kk}.mlp.up_proj",
        "model.layers.{kk}.mlp.down_proj",
        "model.layers.{kk}.input_layernorm",
        "model.layers.{kk}.post_attention_layernorm",
    ]
    layernorm_names = [
        "input_layernorm",
        "post_attention_layernorm",
    ]

    for kk in range(config.num_hidden_layers):
        for layer_name in layer_names:
            layer_name = layer_name.format(kk = kk)
            weight = quant_state_dict[f"{layer_name}.weight"]

            if f"{layer_name}.bias" in quant_state_dict:
                # Has bias!
                has_bias = True
                bias = quant_state_dict[f"{layer_name}.bias"]
                bias = torch.nn.Parameter(bias, requires_grad = False)
            else:
                has_bias = False
                bias = None
            pass

            if f"{layer_name}.weight.quant_state" in quant_state_dict:
                # Layer is quantized!
                quant_state = quant_state_dict[f"{layer_name}.weight.quant_state"]
                n_layers = config.num_hidden_layers
                layer = Linear4bit(0, 0, device = "cuda:0", bias = has_bias, compute_dtype = compute_dtype, **kwargs)
                layer.in_features  = quant_state.shape[1]
                layer.out_features = quant_state.shape[0]
                layer.weight = Params4bit(data = weight, requires_grad = False, **kwargs)
                layer.weight.quant_state = quant_state
                layer.bias = bias
            elif not any(x in layer_name for x in layernorm_names):
                layer = Linear(0, 0, device = "cuda:0", bias = has_bias)
                layer.in_features  = weight.shape[1]
                layer.out_features = weight.shape[0]
                layer.weight = torch.nn.Parameter(weight, requires_grad = False)
                layer.bias = bias
            else:
                # Layernorms
                weight = torch.nn.Parameter(weight, requires_grad = False)
                layer_name = re.sub(r"\.([\d]{1,})\.", r"[\1].", layer_name)
                exec(f"new_model.{layer_name}.weight = None")
                exec(f"new_model.{layer_name}.weight = weight")
                continue
            pass
            
            # Convert model.layers.0.self_attn.q_proj to model.layers[0].self_attn.q_proj
            layer_name = re.sub(r"\.([\d]{1,})\.", r"[\1].", layer_name)
            exec(f"new_model.{layer_name} = layer")
        pass
    pass

    # Norm
    norm = quant_state_dict["model.norm.weight"]
    norm = torch.nn.Parameter(norm, requires_grad = False)
    new_model.model.norm.weight = norm

    # Embeddings
    new_model.model.embed_tokens = torch.nn.Embedding.from_pretrained(
        quant_state_dict["model.embed_tokens.weight"],
        freeze = True,
        padding_idx = config.pad_token_id,
    )

    # LM Head
    if getattr(config, "tie_word_embeddings", False):
        weight = quant_state_dict["model.embed_tokens.weight"]
    else:
        weight = quant_state_dict["lm_head.weight"]
    layer = Linear(0, 0, device = "cuda:0", bias = False)
    layer.in_features  = weight.shape[1]
    layer.out_features = weight.shape[0]
    layer.weight = torch.nn.Parameter(weight, requires_grad = False)
    new_model.lm_head = layer
    if getattr(config, "tie_word_embeddings", False): new_model.tie_weights()

    # Fix up config items with correct items
    config_as_dict = config.to_dict()
    for module in new_model.modules():
        for key, value in config_as_dict.items():
            if hasattr(module, key): exec(f"module.{key} = {value}")
        if hasattr(module, "config"): module.config = config
    pass
    for param in new_model.parameters():
        for key, value in config_as_dict.items():
            if hasattr(param, key): exec(f"param.{key} = {value}")
        if hasattr(param, "config"): param.config = config
    pass
    module = new_model
    for key, value in config_as_dict.items():
        if hasattr(module, key): exec(f"module.{key} = {value}")
    new_model.config = config

    # Cleanup
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return new_model
pass


def approximate_vllm_memory_usage(
    config, 
    max_seq_length = 2048,
    gpu_memory_utilization = 0.8,
    enable_lora = True,
    max_lora_rank = 16,
    max_loras = 1,
    float8_kv_cache = False,
    account_for_gradients = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Gets approximate max model length and max num sequences
    load_in_4bit = "quantization_config" in config
    free_memory, total_memory = torch.cuda.mem_get_info()
    free_memory = gpu_memory_utilization * free_memory

    vocab_size = config.vocab_size
    hd = config.hidden_size
    context_length = config.max_position_embeddings
    mlp_size = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", 1)
    n_heads    = getattr(config, "num_attention_heads", 1)
    # Group Query Attention
    kv_size = hd // n_heads * n_kv_heads

    # Modules
    qkvo = hd + kv_size + kv_size + hd
    qkvo = qkvo * hd
    mlp  = (hd * mlp_size) * 3
    layernorms = 2 * hd
    embed_tokens = vocab_size * hd
    lm_head = 0 if getattr(config, "tie_word_embeddings", True) else vocab_size * hd

    # LoRA modules on all QKVO, MLP
    qkvo_A = hd * max_lora_rank * 4
    qkvo_B = max_lora_rank * (hd + kv_size + kv_size + hd)
    mlp_A  = hd * max_lora_rank * 2 + mlp_size * max_lora_rank
    mlp_B  = max_lora_rank * (mlp_size + mlp_size) + max_lora_rank * hd
    lora_elements = qkvo_A + qkvo_B + mlp_A + mlp_B
    lora_elements = lora_elements * max_loras
    # 2 bytes = float16 for LoRA
    lora_elements = lora_elements*n_layers * 2
    if not enable_lora: lora_elements = 0

    # Get activation and gradients for LoRA
    # 8bit Adam most likely * 2 for momentum, variance
    gradient_lora_elements  = lora_elements + lora_elements
    # Parameter left in float32
    parameter_lora_elements = lora_elements*4

    # Activation memory - assume bsz=2
    bsz = 2
    activation_qkv  = max_seq_length * bsz * (hd + kv_size + kv_size)
    residual_memory = (max_seq_length * bsz)*2
    activation_mlp  = max_seq_length * bsz * (mlp_size + mlp_size)
    weights = mlp_size * hd
    maximum_activation = \
        activation_qkv + residual_memory + activation_mlp + weights
    # 2 bytes with 25% extra just in case
    maximum_activation = (maximum_activation*1.25) * 2
    if not account_for_gradients: maximum_activation = 0
    # Minus for activations
    if total_memory - free_memory < maximum_activation:
        free_memory = total_memory - maximum_activation
    actual_gpu_memory_utilization = free_memory / total_memory

    # 2 bytes = float16
    total_quantizable_elements = (qkvo + mlp)*n_layers * 2
    total_float16_elements     = (layernorms + embed_tokens + lm_head)*2
    factor = 16/5 if load_in_4bit else 1 # Should be 4.5 but use 5
    bytes_for_model = \
        total_quantizable_elements / factor + total_float16_elements + lora_elements

    # KV cache size (float16 is 2 bytes. float8 is 1.25 bytes)
    float_bytes = 1.25 if float8_kv_cache else 2
    kv_elements = (kv_size * 2 * n_layers) * float_bytes
    memory_left_for_kv_cache = free_memory - bytes_for_model
    # Approx maximum # of KV cache elements
    max_num_batched_tokens = int(0.95*(memory_left_for_kv_cache / kv_elements))
    # Round by 256
    max_num_batched_tokens = (max_num_batched_tokens // 256) * 256
    # Assuming all requests output max_seq_length, get theoretical max requests
    approx_max_num_seqs = int(max_num_batched_tokens / max_seq_length)

    # GB for KV cache
    memory_left_for_kv_cache_gb = memory_left_for_kv_cache / 1024 / 1024 / 1024

    return \
        max_num_batched_tokens, approx_max_num_seqs, \
        actual_gpu_memory_utilization, memory_left_for_kv_cache_gb
pass


def load_vllm(
    model_name             : str   = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    config                 = None,
    gpu_memory_utilization : float = 0.8,
    max_seq_length         : int   = 8192,
    dtype                  : torch.dtype = None,
    training               : bool = True,
    float8_kv_cache        : bool = False,
    random_state           : int  = 0,
    enable_lora            : bool = True,
    max_lora_rank          : int  = 16,
    max_loras              : int  = 1,
    use_async              : bool = False,
    use_engine             : bool = False,
    disable_log_stats      : bool = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Create vLLM instance
    assert(config is not None)
    max_num_batched_tokens, approx_max_num_seqs, \
    actual_gpu_memory_utilization, memory_left_for_kv_cache_gb = \
    approximate_vllm_memory_usage(
        config, 
        max_seq_length = max_seq_length,
        gpu_memory_utilization = gpu_memory_utilization,
        enable_lora = enable_lora,
        max_lora_rank = max_lora_rank,
        max_loras = max_loras,
        float8_kv_cache = float8_kv_cache,
        account_for_gradients = training,
    )

    # Check max_num_batched_tokens for max_seq_length
    # Must be >= max_num_batched_tokens
    if max_num_batched_tokens <= max_seq_length:
        print(
            f"Unsloth: Your GPU cannot handle sequence lengths of {max_seq_length} due to limited GPU memory.\n"\
            f"Unsloth: Your GPU can only handle approximately the maximum sequence length of {max_seq_length}."
        )
        max_seq_length = max_num_batched_tokens
    pass

    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version < 7: raise NotImplementedError("Unsloth: Your GPU is too old!")
    if major_version >= 8: _dtype = torch.bfloat16
    else: _dtype = torch.float16
    if dtype == torch.bfloat16 and _dtype == torch.float16:
        print("Unsloth: We switched to dtype = torch.float16 since your GPU does not support torch.bfloat16")
        dtype = torch.float16
    elif dtype is None:
        dtype = _dtype
        print(f"Unsloth: Using dtype = {dtype} for vLLM.")
    elif dtype == torch.float16 or dtype == torch.bfloat16: pass
    else:
        raise NotImplementedError(f"Unsloth: We do not support dtype = {dtype} yet!")

    free_memory, total_memory = torch.cuda.mem_get_info()
    total_memory_gb = round(total_memory / 1024 / 1024 / 1024, 2)

    print(
        f"Unsloth: vLLM loading {model_name} with actual GPU utilization = {round(actual_gpu_memory_utilization*100, 2)}%\n"\
        f"Unsloth: Your GPU has CUDA compute capability {major_version}.{minor_version} with VRAM = {total_memory_gb} GB.\n"\
        f"Unsloth: vLLM can process {approx_max_num_seqs} sequences and {max_num_batched_tokens} tokens in tandem.\n"\
        f"Unsloth: vLLM's KV Cache can use up to {round(memory_left_for_kv_cache_gb, 2)} GB."
    )
    use_bitsandbytes = model_name.lower().endswith("-bnb-4bit")

    # Fix up vLLM compute_dtype for bitsandbytes
    BitsAndBytesConfig = patch_vllm_compute_dtype(dtype)

    # Use Flashinfer if possible (doesn't seem to be faster for BnB)
    # Also seems to process 2x less sequences in 1 go so less throughput?
    # Maybe FP8 Flashinfer is much better
    # See https://docs.vllm.ai/en/latest/serving/env_vars.html
    if importlib.util.find_spec("flashinfer"):
        # Allowed: FLASHINFER, TORCH_SDPA, FLASH_ATTN, XFORMERS, ROCM_FLASH
        if not use_bitsandbytes and major_version >= 8:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        # Flashinfer sampler maybe makes it somewhat faster on newer GPUs
        # Tesla T4 is 280 tok/s vs 330 tok/s
        if major_version >= 8:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        else:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    pass

    from vllm import LLM, LLMEngine, AsyncLLMEngine, EngineArgs

    # Default vLLM max_num_seqs is 256
    approx_max_num_seqs = 256
    if   memory_left_for_kv_cache_gb <=  2: approx_max_num_seqs = 128 # - 32
    elif memory_left_for_kv_cache_gb <=  4: approx_max_num_seqs = 160 # - 32
    elif memory_left_for_kv_cache_gb <=  8: approx_max_num_seqs = 192 # - 32
    elif memory_left_for_kv_cache_gb <= 12: approx_max_num_seqs = 224 # - 32
    elif memory_left_for_kv_cache_gb <= 16: approx_max_num_seqs = 256 # Default
    elif memory_left_for_kv_cache_gb <= 24: approx_max_num_seqs = 288 # + 32
    elif memory_left_for_kv_cache_gb <= 40: approx_max_num_seqs = 320 # + 32
    elif memory_left_for_kv_cache_gb <= 48: approx_max_num_seqs = 226 # + 16
    elif memory_left_for_kv_cache_gb <= 80: approx_max_num_seqs = 368 # + 32
    else: approx_max_num_seqs = 400 # + 32

    # float8 KV cache can fit more sequences in 1 go so more throughput
    if float8_kv_cache: approx_max_num_seqs = int(approx_max_num_seqs * 1.05)

    # vLLM default max_num_batched_tokens is 2048
    chunked_prefill_tokens = 2048
    if   memory_left_for_kv_cache_gb <=  8: chunked_prefill_tokens = 1024 # + 0
    elif memory_left_for_kv_cache_gb <= 12: chunked_prefill_tokens = 1536 # + 512
    elif memory_left_for_kv_cache_gb <= 16: chunked_prefill_tokens = 2048 # + 512
    elif memory_left_for_kv_cache_gb <= 24: chunked_prefill_tokens = 3072 # + 1024
    elif memory_left_for_kv_cache_gb <= 40: chunked_prefill_tokens = 4096 # + 1024
    elif memory_left_for_kv_cache_gb <= 48: chunked_prefill_tokens = 4608 # + 512
    elif memory_left_for_kv_cache_gb <= 80: chunked_prefill_tokens = 8192 # + 4096
    else: chunked_prefill_tokens = 8192 # + 0

    # vLLM errors out from max_seq_length (2048) being bigger than chunked_prefill_tokens (1024)
    if max_seq_length > chunked_prefill_tokens:
        chunked_prefill_tokens = max_seq_length
    elif chunked_prefill_tokens > max_seq_length:
        chunked_prefill_tokens = max_seq_length

    engine_args = dict(
        model                  = model_name,
        gpu_memory_utilization = actual_gpu_memory_utilization,
        max_model_len          = max_seq_length,
        quantization           = "bitsandbytes" if use_bitsandbytes else None,
        load_format            = "bitsandbytes" if use_bitsandbytes else "auto",
        kv_cache_dtype         = "fp8" if float8_kv_cache else "auto",
        dtype                  = dtype,

        max_num_batched_tokens = chunked_prefill_tokens, # Max tokens for chunked prefill default 2048
        max_num_seqs           = approx_max_num_seqs, # vLLM default uses 256 -> reduce if OOM
        max_logprobs           = 0, # Disallow logprobs being returned
        seed                   = random_state, # Default is 0

        # lora_extra_vocab_size = 0, # Breaks vLLM so we leave it as 256
        enable_lora            = enable_lora,
        max_lora_rank          = max_lora_rank,
        max_loras              = max_loras,

        disable_log_stats      = disable_log_stats,
        # enable_prefix_caching  = True, # LoRA fails with chunked prefill as at Feb 2025
        # enable_chunked_prefill = True, # LoRA fails with chunked prefill as at Feb 2025
        max_seq_len_to_capture = 8192, # Default is 8192 for CUDAGraphs
        compilation_config     = 3, # 0, 1, 2, 3
    )

    # Keep trying until success!
    while True:
        try:
            if use_async:
                llm = AsyncLLMEngine.from_engine_args(EngineArgs(**engine_args))
            elif use_engine:
                llm = LLMEngine.from_engine_args(EngineArgs(**engine_args))
            else:
                llm = LLM(**engine_args)
            pass
            break
        except Exception as error:
            # Cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            pass
            error = str(error)
            if "gpu_memory_utilization" in error or "memory" in error:
                approx_max_num_seqs = int(approx_max_num_seqs * 0.75)
                engine_args["max_num_seqs"] = approx_max_num_seqs
                print(
                    f"Unsloth: Retrying vLLM to process {approx_max_num_seqs} sequences and {max_num_batched_tokens} tokens in tandem."
                )
            else:
                raise RuntimeError(error)
        pass
    pass
    # Save maximum requests length since llm.generate fails to partition inputs sometimes
    llm.approx_max_num_seqs = approx_max_num_seqs

    # Unpatch vLLM compute_dtype for bitsandbytes
    unpatch_vllm_compute_dtype(BitsAndBytesConfig)
    return llm
pass


def create_batches(requests, num_sequences = 64):
    # All Unsloth Zoo code licensed under LGPLv3
    # llm.generate must be batched!
    n_splits = int(math.ceil(len(requests) / num_sequences))
    offsets = np.arange(0, len(requests), num_sequences)
    if offsets[-1] != len(requests):
        offsets = np.hstack((offsets, len(requests)))
    batches = [requests[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    return batches
pass


def delete_vllm(llm):
    # From https://github.com/vllm-project/vllm/issues/1908
    import ray
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    # Delete the llm object and free the memory
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
pass


@torch.inference_mode
def _test_get_vllm_state_dict(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    dtype = torch.float16,
    gpu_memory_utilization = 0.7,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if model is allowed to be used in vLLM
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_name,
        token = None,
        revision = None,
        trust_remote_code = False,
        attn_implementation = "sdpa",
    )
    if not vllm_dynamic_quant_supported(model_name, config):
        raise NotImplementedError(f"Unsloth: Dynamic quant of {model_name} not supported in vLLM")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = None
    load_in_4bit = model_name.lower().endswith("-bnb-4bit")
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = dtype,
        )
    pass
    kwargs = dict()
    if load_in_4bit: kwargs["quantization_config"] = bnb_config
    # Must patch BnB compute_dtype since it's forced to bfloat16!
    patch_bitsandbytes_quant_state()
    patch_bitsandbytes_compute_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map          = "sequential",
        torch_dtype         = dtype,
        attn_implementation = "sdpa",
        **kwargs,
    )
    unpatch_bitsandbytes_compute_dtype()
    for param in model.parameters():
        param.requires_grad_(False)

    llm = load_vllm(
        model_name             = model_name,
        config                 = config,
        gpu_memory_utilization = 0.7,
        max_seq_length         = 2048,
        dtype                  = dtype,
    )

    state_dict, quant_state_dict = get_vllm_state_dict(
        llm,
        return_state_dict = True,
        config = config,
    )
    assert_same_state_dict(model.state_dict(), state_dict)

    new_model = convert_vllm_to_huggingface(quant_state_dict, config, dtype)
    assert_same_state_dict(model.state_dict(), new_model.state_dict())

    # Run the model as well
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        [{"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},],
        [{"role": "user", "content": "Write a long poem about the world."},],
        [{"role": "user", "content": "What is the capital of France? Describe it."},],
        [{"role": "user", "content": "Why is the sky blue?"},],
        [{"role": "user", "content": "Explain Newton's third law of motion."},],
        [{"role": "user", "content": "Why is spacetime bent?"},],
        [{"role": "user", "content": "Explain heliocentricism."},],
        [{"role": "user", "content": "Derive the formula for an infinite sum of 1, 1/2, 1/4, 1/8 and so on."},],
    ]*100
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        padding = True,
    )

    # Check hidden_states
    with torch.autocast(device_type = "cuda", dtype = dtype):
        input_ids = tokenizer(inputs[0], add_special_tokens = False, return_tensors = "pt")
        input_ids = input_ids["input_ids"].to("cuda", non_blocking = True)
        old_outputs =     model(input_ids = input_ids, output_hidden_states = True)
        new_outputs = new_model(input_ids = input_ids, output_hidden_states = True)
    pass
    for i, (a, b) in enumerate(zip(old_outputs.hidden_states, new_outputs.hidden_states)):
        try:
            torch.testing.assert_close(a, b)
        except Exception as error:
            raise RuntimeError(f"[Hidden_States[{i}]]\n{str(error)}")
        pass
    pass
    try:
        torch.testing.assert_close(old_outputs.logits, new_outputs.logits)
    except Exception as error:
        raise RuntimeError(f"[Logits]\n{str(error)}")
    pass

    # Cannot just use llm.generate or OOM - split into batches
    batches = create_batches(inputs, llm.approx_max_num_seqs)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 1.5,
        min_p = 0.1,
        logprobs = 0,
        prompt_logprobs = 0,
        max_tokens = 256,
    )
    completion_ids = []
    for batch in batches:
        outputs = llm.generate(batch, sampling_params)
        completion_ids.extend(out.token_ids for completions in outputs for out in completions.outputs)
    pass

    del completion_ids
    delete_vllm(llm)
    for module in new_module.modules():
        dir(module)
pass


def test_get_vllm_state_dict(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    dtype = torch.float16,
):
    patch_vllm()
    _test_get_vllm_state_dict(
        model_name = "unsloth/Qwen2.5-1.5B-Instruct",
        dtype = torch.float16,
        gpu_memory_utilization = 0.7,
    )
    _test_get_vllm_state_dict(
        model_name = "unsloth/Qwen2.5-1.5B-Instruct",
        dtype = torch.bfloat16,
        gpu_memory_utilization = 0.7,
    )
    _test_get_vllm_state_dict(
        model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        dtype = torch.bfloat16,
        gpu_memory_utilization = 0.8,
    )
    _test_get_vllm_state_dict(
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        dtype = torch.float16,
        gpu_memory_utilization = 0.8,
    )
    _test_get_vllm_state_dict(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        dtype = torch.bfloat16,
        gpu_memory_utilization = 0.7,
    )
    _test_get_vllm_state_dict(
        model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit",
        dtype = torch.float16,
        gpu_memory_utilization = 0.5,
    )
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
