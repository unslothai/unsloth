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
    "delete_vllm",
    "save_lora",
    "load_lora",
    "generate_batches",
    "convert_lora_modules",
    "return_lora_modules",
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
import torch
import json
import psutil
import functools
import contextlib
import inspect
from functools import partial
from .utils import _get_dtype
from .patching_utils import patch_model_and_tokenizer
global LORA_REQUEST_ID

# Ignore logging messages
import logging
class HideLoggingMessage(logging.Filter):
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())
pass

def _return_nothing(*args, **kwargs): return None
def _return_self(self, *args, **kwargs): return self


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
        # All Unsloth Zoo code licensed under LGPLv3
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
        # All Unsloth Zoo code licensed under LGPLv3
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
            # All Unsloth Zoo code licensed under LGPLv3
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
        # All Unsloth Zoo code licensed under LGPLv3
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesConfig = old_config
        del os.environ["UNSLOTH_bnb_4bit_compute_dtype"]
    pass

    def patch_vllm_lora_tokenizer():
        import vllm.transformers_utils.tokenizer
        vllm.transformers_utils.tokenizer.get_lora_tokenizer = _return_nothing
        vllm.transformers_utils.tokenizer.get_lora_tokenizer_async = _return_nothing
        
        import vllm.transformers_utils.tokenizer_group.tokenizer_group
        vllm.transformers_utils.tokenizer_group.tokenizer_group.get_lora_tokenizer = _return_nothing
        vllm.transformers_utils.tokenizer_group.tokenizer_group.get_lora_tokenizer_async = _return_nothing
    pass

    from .vllm_lora_request import LoRARequest as PatchedLoRARequest
    from .vllm_lora_worker_manager import (
        WorkerLoRAManager as PatchedWorkerLoRAManager,
        LRUCacheWorkerLoRAManager as PatchedLRUCacheWorkerLoRAManager,
    )
    def patch_vllm_lora_load_tensors():
        import vllm.lora.request
        vllm.lora.request.LoRARequest = PatchedLoRARequest
        import vllm.lora.worker_manager
        vllm.lora.worker_manager.LoRARequest = PatchedLoRARequest
        vllm.lora.worker_manager.WorkerLoRAManager = PatchedWorkerLoRAManager
        vllm.lora.worker_manager.LRUCacheWorkerLoRAManager = PatchedLRUCacheWorkerLoRAManager
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

    def patch_vllm_lora_tokenizer():
        return
    pass

    def patch_vllm_lora_load_tensors():
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
        # All Unsloth Zoo code licensed under LGPLv3
        def __init__(self, *args, **kwargs):
            compute_dtype = os.environ.get("UNSLOTH_bnb_4bit_compute_dtype", None)
            if compute_dtype is not None:
                compute_dtype = getattr(torch, compute_dtype)
                kwargs["compute_dtype"] = compute_dtype
            super().__init__(*args, **kwargs)
        pass
    pass

    def patch_bitsandbytes_quant_state():
        # All Unsloth Zoo code licensed under LGPLv3
        bitsandbytes.functional.QuantState.from_dict = from_dict
        bitsandbytes.nn.modules.Linear4bit = Linear4bit
    pass

    def patch_bitsandbytes_compute_dtype(dtype):
        # All Unsloth Zoo code licensed under LGPLv3
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
    patch_vllm_lora_tokenizer()
    patch_vllm_lora_load_tensors()
    global LORA_REQUEST_ID
    LORA_REQUEST_ID = 0
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
        llm_engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))
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
def convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float16, bnb_config = None):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules to create HF compatible model
    config.update({"torch_dtype" : dtype}) # Do not use config file's dtype!
    new_model = create_empty_causal_lm(config, dtype)
    quantization_config = getattr(config, "quantization_config", {})
    kwargs = dict()
    compute_dtype = dtype  # Do not use config file's dtype!

    if quantization_config != {} or bnb_config is not None:
        # Get quantization_config flags
        if quantization_config != {}:
            kwargs["compress_statistics"] = quantization_config["bnb_4bit_use_double_quant"]
            kwargs["quant_type"] = quantization_config["bnb_4bit_quant_type"]
            kwargs["quant_storage"] = _get_dtype(quantization_config["bnb_4bit_quant_storage"])

        # Get bnb_config flags
        elif bnb_config is not None:
            kwargs["compress_statistics"] = bnb_config.bnb_4bit_use_double_quant
            kwargs["quant_type"] = bnb_config.bnb_4bit_quant_type
            kwargs["quant_storage"] = _get_dtype(bnb_config.bnb_4bit_quant_storage)

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
    # Override .to("cuda") to disable it otherwise we'll get
    # ValueError: Blockwise quantization only supports 16/32-bit floats, but got torch.uint8
    def _override_to(self, *args, **kwargs):
        try: return self.to(*args, **kwargs)
        except: return self
    pass

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

                # Must override or else Bitsandbytes will error
                layer.to = partial(_override_to, layer)
                layer.weight.to = partial(_override_to, layer.weight)

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

    # Fix up rotary_emb by re-initing them
    for module in new_model.modules():
        if hasattr(module, "rotary_emb"):
            module.rotary_emb = module.rotary_emb.__class__(
                config = config,
                device = "cuda:0",
            )
        pass
    pass

    # Must override or else Bitsandbytes will error
    new_model.to = partial(_override_to, new_model)

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
    if memory_left_for_kv_cache <= 0: memory_left_for_kv_cache = 0

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
    enforce_eager          : bool = False, # Good for debugging
    enable_prefix_caching  : bool = True,
    compilation_config     : int  = 3, # -O3 for maximum performance
    conservativeness       : float = 1.0, # For low VRAM devices, scale batches, num_seqs
    max_logprobs           : int  = 0,
    use_bitsandbytes       : bool = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Create vLLM instance
    assert(config is not None)
    assert(type(use_bitsandbytes) is bool)
    assert(conservativeness >= 0.0 and conservativeness <= 1.0)

    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version < 7: raise NotImplementedError("Unsloth: Your GPU is too old!")

    # Float8 KV cache only works for 8.0 or higher
    if float8_kv_cache and major_version < 8:
        raise NotImplementedError("Unsloth: Your GPU is too old for float8 KV cache! Set it to False.")

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
    if max_num_batched_tokens <= 0:
        max_seq_length = 256
        max_num_batched_tokens = 256

    if max_num_batched_tokens <= max_seq_length:
        print(
            f"Unsloth: Your GPU cannot handle sequence lengths of {max_seq_length} due to limited GPU memory.\n"\
            f"Unsloth: Your GPU can only handle approximately the maximum sequence length of {max_seq_length}."
        )
        max_seq_length = max_num_batched_tokens
    pass

    # Get correct dtype
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
    use_bitsandbytes = use_bitsandbytes or \
        model_name.lower().endswith("-bnb-4bit")

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

    # Prefix Caching fails for V100, Titan X CUDA Compute Capability 7.0
    # See https://github.com/huggingface/trl/issues/2798
    major_version, minor_version = torch.cuda.get_device_capability()
    if (major_version < 7) or (major_version == 7 and minor_version < 5):
        print("Unsloth: Your GPU does not support prefix caching - will disable!")
        enable_prefix_caching = False
    pass

    # Use VLLM_USE_V1 for vllm >= 0.7.4 and CUDA >= 8.0
    # [FAILS] for bitsandbytes - https://github.com/unslothai/unsloth/issues/2102
    # if importlib.util.find_spec("vllm") and (major_version >= 8):
    #     from importlib.metadata import version as importlib_version
    #     from packaging.version import Version
    #     if Version(importlib_version("vllm")) > Version("0.7.3"):
    #         os.environ["VLLM_USE_V1"] = "1"
    # pass

    from vllm import LLM, LLMEngine, AsyncLLMEngine, EngineArgs, AsyncEngineArgs

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

    # Scale num_seqs by conservativeness
    approx_max_num_seqs = int(approx_max_num_seqs * conservativeness)

    # Check max RAM usage for vLLM (swap space) default is 4GB
    memory = psutil.virtual_memory()
    RAM_GB = memory.available / 1024 / 1024 / 1024
    swap_space = 4
    if   RAM_GB <= 4:  swap_space = 0
    elif RAM_GB <= 8:  swap_space = 1
    elif RAM_GB <= 12: swap_space = 2
    elif RAM_GB <= 16: swap_space = 3
    elif RAM_GB <= 24: swap_space = 4
    elif RAM_GB <= 48: swap_space = 5
    else: swap_space = 6

    print(
        f"Unsloth: vLLM loading {model_name} with actual GPU utilization = {round(actual_gpu_memory_utilization*100, 2)}%\n"\
        f"Unsloth: Your GPU has CUDA compute capability {major_version}.{minor_version} with VRAM = {total_memory_gb} GB.\n"\
        f"Unsloth: Using conservativeness = {conservativeness}. Chunked prefill tokens = {chunked_prefill_tokens}. Num Sequences = {approx_max_num_seqs}.\n"\
        f"Unsloth: vLLM's KV Cache can use up to {round(memory_left_for_kv_cache_gb, 2)} GB. Also swap space = {swap_space} GB."
    )

    # Get device as well
    device = "cuda:0"

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
        max_logprobs           = max_logprobs, # Disallow logprobs being returned
        seed                   = random_state, # Default is 0

        # lora_extra_vocab_size = 0, # Breaks vLLM so we leave it as 256
        enable_lora            = enable_lora,
        max_lora_rank          = max_lora_rank,
        max_loras              = max_loras,

        disable_log_stats      = disable_log_stats,
        enable_prefix_caching  = enable_prefix_caching,
        # enable_chunked_prefill = True, # LoRA fails with chunked prefill as at Feb 2025
        max_seq_len_to_capture = min(8192, max_seq_length + 256), # Default is 8192 for CUDAGraphs
        compilation_config     = compilation_config, # 0, 1, 2, 3
        enforce_eager          = enforce_eager,
        swap_space             = swap_space, # Low memory devices like Colab (13GB) default 4GB
        device                 = device,
    )
    good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs).parameters.keys()
    old_keys = engine_args.keys()
    for key in old_keys:
        if key not in good_keys:
            del engine_args[key]
            print(f"Unsloth: Not an error, but `{key}` is not supported in vLLM. Skipping.")
        pass
    pass

    # Keep trying until success (2 times)
    trials = 0
    while True:
        try:
            if use_async:
                llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
            elif use_engine:
                llm = LLMEngine.from_engine_args(EngineArgs(**engine_args))
            else:
                llm = LLM(**engine_args)
            pass
            break
        except Exception as error:
            trials += 1
            # Cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            pass
            error = str(error)
            if trials >= 2:
                raise RuntimeError(error)
            
            if "gpu_memory_utilization" in error or "memory" in error:
                approx_max_num_seqs = int(approx_max_num_seqs * 0.75)
                engine_args["max_num_seqs"] = approx_max_num_seqs
                engine_args["gpu_memory_utilization"] *= 0.85
                print(
                    f"Unsloth: Retrying vLLM to process {approx_max_num_seqs} sequences and {max_num_batched_tokens} tokens in tandem.\n"\
                    f"Error:\n{error}"
                )
            else:
                raise RuntimeError(error)
        pass
    pass
    # Save maximum requests length since llm.generate fails to partition inputs sometimes
    llm.approx_max_num_seqs = approx_max_num_seqs

    # Unpatch vLLM compute_dtype for bitsandbytes
    unpatch_vllm_compute_dtype(BitsAndBytesConfig)

    # Cleanup
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
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


@torch.inference_mode
def save_lora(model, save_directory, *args, **kwargs):
    # All Unsloth Zoo code licensed under LGPLv3
    state_dict = model.state_dict()
    dtype = model.get_input_embeddings().weight.dtype
    # Cast LoRA to float16 / bfloat16
    state_dict = {k:v.to(dtype) for k, v in state_dict.items() if ".lora_A." in k or ".lora_B." in k}
    kwargs["state_dict"] = state_dict
    model.save_pretrained(save_directory = save_directory, *args, **kwargs)
pass


@functools.cache
def get_peft_config(save_directory):
    with open(os.path.join(save_directory, "adapter_config.json")) as f:
        config = json.load(f)
    return config
pass


def vllm_lora_already_loaded(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if LoRA is loaded - if not, we should load the first one
    m = model.vllm_engine.llm_engine.model_executor.driver_worker.model_runner
    lora_cache = m.lora_manager._adapter_manager._active_adapters.cache

    layers = m.model.model.layers
    v_layer = layers[0]
    print(lora_cache, v_layer.self_attn.qkv_proj.lora_a_stacked[0].data_ptr())
    return len(lora_cache) != 0
pass


def prepare_vllm_lora_loading(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Get all vLLM LoRAs
    assert(hasattr(model, "vllm_engine"))

    # Must split into 2 lists since B is scaled in vLLM
    model_loras_A, model_loras_B = [], []
    vllm_loras_A,  vllm_loras_B  = [], []
    vllm_model = model.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    
    # Go through all layers!
    for v_layer, m_layer in zip(vllm_model .model.layers, model.model.model.layers):
        model_loras_A.append(m_layer.self_attn.q_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.self_attn.k_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.self_attn.v_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[0])
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[1])
        vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[2])

        sq = m_layer.self_attn.q_proj.scaling["default"]
        sk = m_layer.self_attn.k_proj.scaling["default"]
        sv = m_layer.self_attn.v_proj.scaling["default"]
        sq = None if sq == 1.0 else sq
        sk = None if sk == 1.0 else sk
        sv = None if sv == 1.0 else sv
        model_loras_B.append( m_layer.self_attn.q_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.self_attn.k_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.self_attn.v_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[0], sq,))
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[1], sk,))
        vllm_loras_B .append((v_layer.self_attn.qkv_proj.lora_b_stacked[2], sv,))

        so = m_layer.self_attn.o_proj.scaling["default"]
        so = None if so == 1.0 else so
        model_loras_A.append(m_layer.self_attn.o_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.self_attn.o_proj.lora_a_stacked[0])
        model_loras_B.append( m_layer.self_attn.o_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.self_attn.o_proj.lora_b_stacked[0], so,))

        model_loras_A.append(m_layer.mlp.gate_proj.lora_A.default.weight)
        model_loras_A.append(m_layer.mlp.gate_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.mlp.gate_up_proj.lora_a_stacked[0])
        vllm_loras_A .append(v_layer.mlp.gate_up_proj.lora_a_stacked[1])

        sg = m_layer.mlp.gate_proj.scaling["default"]
        su = m_layer.mlp.  up_proj.scaling["default"]
        sg = None if sg == 1.0 else sg
        su = None if su == 1.0 else su
        model_loras_B.append( m_layer.mlp.gate_proj.lora_B.default.weight)
        model_loras_B.append( m_layer.mlp.gate_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.mlp.gate_up_proj.lora_b_stacked[0], sg,))
        vllm_loras_B .append((v_layer.mlp.gate_up_proj.lora_b_stacked[1], su,))

        sd = m_layer.mlp.down_proj.scaling["default"]
        sd = None if sd == 1.0 else sd
        model_loras_A.append(m_layer.mlp.down_proj.lora_A.default.weight)
        vllm_loras_A .append(v_layer.mlp.down_proj.lora_a_stacked[0])
        model_loras_B.append( m_layer.mlp.down_proj.lora_B.default.weight)
        vllm_loras_B .append((v_layer.mlp.down_proj.lora_b_stacked[0], sd,))
    pass

    # Check all shapes
    for model_lora_A, vllm_lora_A in zip(model_loras_A, vllm_loras_A):
        assert(model_lora_A.squeeze().shape == vllm_lora_A.squeeze().shape)
    for model_lora_B, (vllm_lora_B, s,) in zip(model_loras_B, vllm_loras_B):
        assert(model_lora_B.squeeze().shape == vllm_lora_B.squeeze().shape)
    pass

    # Set model items
    model.model_loras_A = model_loras_A
    model.model_loras_B = model_loras_B
    model. vllm_loras_A = vllm_loras_A
    model. vllm_loras_B = vllm_loras_B
    return
pass


def load_lora_directly(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Load LoRAs directly from model into vLLM internal LoRAs
    model_loras_A = model.model_loras_A
    model_loras_B = model.model_loras_B
    vllm_loras_A  = model. vllm_loras_A
    vllm_loras_B  = model. vllm_loras_B

    for model_lora_A, vllm_lora_A in zip(model_loras_A, vllm_loras_A):
        vllm_lora_A.copy_(model_lora_A, non_blocking = True)
    pass

    # Must also scale B with scaling since vLLM does this
    for model_lora_B, (vllm_lora_B, s) in zip(model_loras_B, vllm_loras_B):
        vllm_lora_B.copy_(model_lora_B, non_blocking = True)
        if s is not None: vllm_lora_B *= s
    pass
    # Must block!
    torch.cuda.synchronize()
pass


from peft import PeftType

@torch.inference_mode
def convert_lora_modules(
    model,
    dtype = None,
):
    dtype = _get_dtype(model.config.torch_dtype if dtype is None else dtype)

    if (hasattr(model, "peft_config") and "default" in model.peft_config) \
        and (model.peft_config["default"].peft_type == PeftType.LORA):

        state_dict = model.state_dict().items()
        state_dict = {
            k : v.detach().clone() for k, v in state_dict \
            if (v.dtype != dtype) and \
               (".lora_A.default" in k or ".lora_B.default" in k)
        }
        if len(state_dict) == 0: return {}

        for name, module in model.named_modules():
            if name + ".default.weight" in state_dict:
                exec(f"module.to({dtype})")
        pass
        return state_dict
    return {}
pass


@torch.inference_mode
def return_lora_modules(
    model,
    state_dict = {},
    dtype = torch.float32,
):
    if state_dict == {} or state_dict is None: return
    dtype = _get_dtype(model.config.torch_dtype if dtype is None else dtype)

    if (hasattr(model, "peft_config") and "default" in model.peft_config) \
        and (model.peft_config["default"].peft_type == PeftType.LORA):

        for name, module in model.named_modules():
            old_name = name + ".default.weight"
            old_weight = state_dict.get(old_name, None)
            if old_weight is not None:
                exec(f"module.to({dtype})")
                # module.default.weight.copy_(old_weight)
        pass
        return
    return
pass


@torch.inference_mode
def load_lora(model, save_directory, load_tensors = False):
    # vllm_lora_already_loaded(model)
    # Check internally if model has hot loaded LoRAs
    # if load_tensors and hasattr(model, "saved_vllm_lora_request"):# vllm_lora_already_loaded(model):
    #     if not hasattr(model, "model_loras_A"):
    #         # Prepare vLLM for LoRA direct loading!
    #         prepare_vllm_lora_loading(model)
    #     pass
    #     load_lora_directly(model)
    #     return model.saved_vllm_lora_request
    # pass

    # All Unsloth Zoo code licensed under LGPLv3
    global LORA_REQUEST_ID
    if LORA_REQUEST_ID is None: LORA_REQUEST_ID = 0

    # Check if path exists
    if not os.path.exists(save_directory) or LORA_REQUEST_ID == 0:
        if load_tensors:
            # We need to save and load the config file once!
            model.peft_config["default"].save_pretrained(save_directory)
        elif not os.path.exists(save_directory):
            raise OSError(f"Unsloth: LoRA filepath = {save_directory} does not exist!")
    pass

    from vllm.lora.request import LoRARequest
    if load_tensors:
        # We extract it directly from the model's state_dict
        peft_config = get_peft_config(save_directory)
        state_dict = model.state_dict()
        items = state_dict.items()
        state_dict = {k.replace(".default", ""):v for k, v in items if ".lora_A." in k or ".lora_B." in k}

        # vllm_lora_already_loaded(model)
        lora_request = LoRARequest(str(LORA_REQUEST_ID), LORA_REQUEST_ID, lora_tensors = state_dict, lora_config = peft_config)
        # Warm up LoRA
        # vllm_lora_already_loaded(model)
        # outputs = model.vllm_engine.generate(["Hi!"], use_tqdm = False, lora_request = lora_request)
        # del outputs
        # vllm_lora_already_loaded(model)
        # print("###", LORA_REQUEST_ID)
        # vllm_lora_already_loaded(model)
            # model.saved_vllm_lora_request = lora_request
    else:
        lora_request = LoRARequest(str(LORA_REQUEST_ID), LORA_REQUEST_ID, save_directory)
    pass
    # vllm_lora_already_loaded(model)

    LORA_REQUEST_ID += 1
    # Set model's current LoRA adapater
    # model.vllm_engine.vllm_lora_request = lora_request
    return lora_request
pass


def generate_batches(llm, inputs, n_batches = None, lora_request = None, *args, **kwargs):
    # All Unsloth Zoo code licensed under LGPLv3
    # Cannot just use llm.generate or will OOM - split into batches
    if n_batches is None:
        if "UNSLOTH_VLLM_BATCHES" in os.environ:
            n_batches = int(os.environ["UNSLOTH_VLLM_BATCHES"])
        else:
            free_memory, total_memory = torch.cuda.mem_get_info()
            total_memory_gb = round(total_memory / 1024 / 1024 / 1024, 2)
            if   total_memory_gb <=  8: n_batches = llm.approx_max_num_seqs // 10
            elif total_memory_gb <= 16: n_batches = llm.approx_max_num_seqs // 5
            elif total_memory_gb <= 24: n_batches = llm.approx_max_num_seqs // 2
            else: n_batches = llm.approx_max_num_seqs

            os.environ["UNSLOTH_VLLM_BATCHES"] = str(n_batches)

            if n_batches != llm.approx_max_num_seqs:
                print(f"Unsloth: Will use {n_batches} batches to reduce memory usage for generation!")
        pass
    pass

    # We should disable for now since it might interfere with the reference model in RL
    # if lora_request is None:
    #     if hasattr(llm, "vllm_lora_request"): lora_request = llm.vllm_lora_request
    # pass

    batches = create_batches(inputs, n_batches)
    kwargs["lora_request"] = lora_request
    output_list = []
    for batch in batches:
        outputs = llm.generate(batch, *args, **kwargs)
        output_list += list(outputs)
    pass
    return output_list
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


def _test_same_model(model, new_model, input_ids):
    # All Unsloth Zoo code licensed under LGPLv3
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
    )
    from peft.utils.integrations import dequantize_module_weight as df

    A =     model.model.embed_tokens(input_ids)
    B = new_model.model.embed_tokens(input_ids)
    torch.testing.assert_close(model.model.embed_tokens.weight, new_model.model.embed_tokens.weight)
    torch.testing.assert_close(A, B)

    position_ids = torch.arange(input_ids.shape[1], device = "cuda")
    position_ids = position_ids.repeat((1, input_ids.shape[0]))
    rotary_A =     model.model.rotary_emb(A, position_ids)
    new_rotary = new_model.model.rotary_emb.__class__(new_model.config, device = "cuda")
    rotary_B = new_rotary(B, position_ids)
    torch.testing.assert_close(rotary_A[0], rotary_B[0])
    torch.testing.assert_close(rotary_A[1], rotary_B[1])

    for i, (old, new) in enumerate(zip(model.model.layers, new_model.model.layers)):
        print(i, end = ",")
        residualA = A
        residualB = B
        
        torch.testing.assert_close(old.input_layernorm.weight, new.input_layernorm.weight)
        A = old.input_layernorm(A)
        B = new.input_layernorm(B)

        AA, _ = old.self_attn(A.clone(), attention_mask = None, position_embeddings = rotary_A)
        BB, _ = new.self_attn(B.clone(), attention_mask = None, position_embeddings = rotary_B)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        
        torch.testing.assert_close(df(old.self_attn.q_proj), df(new.self_attn.q_proj))
        torch.testing.assert_close(df(old.self_attn.k_proj), df(new.self_attn.k_proj))
        torch.testing.assert_close(df(old.self_attn.v_proj), df(new.self_attn.v_proj))

        input_shapeA = A.shape[:-1]
        hidden_shapeA = (*input_shapeA, -1, old.self_attn.head_dim)
        QA = old.self_attn.q_proj(A).view(hidden_shapeA).transpose(1, 2)
        KA = old.self_attn.k_proj(A).view(hidden_shapeA).transpose(1, 2)
        VA = old.self_attn.v_proj(A).view(hidden_shapeA).transpose(1, 2)

        input_shapeB = B.shape[:-1]
        hidden_shapeB = (*input_shapeB, -1, new.self_attn.head_dim)
        QB = new.self_attn.q_proj(B).view(hidden_shapeB).transpose(1, 2)
        KB = new.self_attn.k_proj(B).view(hidden_shapeB).transpose(1, 2)
        VB = new.self_attn.v_proj(B).view(hidden_shapeB).transpose(1, 2)
        torch.testing.assert_close(QA, QB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(KA, KB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(VA, VB, rtol = 0.01, atol = 0.005)

        QA, KA = apply_rotary_pos_emb(QA, KA, *rotary_A)
        QB, KB = apply_rotary_pos_emb(QB, KB, *rotary_B)
        torch.testing.assert_close(QA, QB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(KA, KB, rtol = 0.01, atol = 0.005)

        f = ALL_ATTENTION_FUNCTIONS[old.self_attn.config._attn_implementation]
        attentionA, _ = f(old.self_attn, QA, KA, VA,
            attention_mask = None,
            dropout = 0.0 if not old.self_attn.training else old.self_attn.attention_dropout,
            scaling = old.self_attn.scaling,
        )
        f = ALL_ATTENTION_FUNCTIONS[new.self_attn.config._attn_implementation]
        attentionB, _ = f(new.self_attn, QB, KB, VB,
            attention_mask = None,
            dropout = 0.0 if not new.self_attn.training else new.self_attn.attention_dropout,
            scaling = new.self_attn.scaling,
        )
        torch.testing.assert_close(attentionA, attentionB)

        A = attentionA.reshape(*input_shapeA, -1).contiguous()
        A = old.self_attn.o_proj(A)
        B = attentionB.reshape(*input_shapeB, -1).contiguous()
        B = new.self_attn.o_proj(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(BB, B, rtol = 0.01, atol = 0.005)

        residualA = A
        residualB = B
        torch.testing.assert_close(old.post_attention_layernorm.weight, new.post_attention_layernorm.weight)
        A = old.post_attention_layernorm(A)
        B = new.post_attention_layernorm(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)

        AA = old.mlp(A.clone())
        BB = new.mlp(B.clone())
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        gateA = old.mlp.gate_proj(A)
        gateB = new.mlp.gate_proj(B)
        torch.testing.assert_close(gateA, gateB, rtol = 0.01, atol = 0.005)
        upA = old.mlp.up_proj(A)
        upB = new.mlp.up_proj(B)
        torch.testing.assert_close(upA, upB, rtol = 0.01, atol = 0.005)
        A = old.mlp.act_fn(gateA) * upA
        B = new.mlp.act_fn(gateB) * upB
        A = old.mlp.down_proj(A)
        B = new.mlp.down_proj(B)
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, BB, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(AA, A, rtol = 0.01, atol = 0.005)
        torch.testing.assert_close(BB, B, rtol = 0.01, atol = 0.005)

        A = residualA + A
        B = residualB + B
        torch.testing.assert_close(A, B, rtol = 0.01, atol = 0.005)

        B = A.clone()
    pass

    A =     model.model.norm(A)
    B = new_model.model.norm(B)
    torch.testing.assert_close(A, B)

    torch.testing.assert_close(model.lm_head.weight, new_model.lm_head.weight)
    A =     model.lm_head(A)
    B = new_model.lm_head(B)
    torch.testing.assert_close(A, B)
    return
pass


@torch.inference_mode
def _test_get_vllm_state_dict(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    dtype = torch.float16,
    gpu_memory_utilization = 0.7,
    counts = 100,
    conservativeness = 1.0,
    float8_kv_cache = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if model is allowed to be used in vLLM
    gc.collect()
    torch.cuda.empty_cache()

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
    # patch_bitsandbytes_compute_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map          = "sequential",
        torch_dtype         = dtype,
        attn_implementation = "sdpa",
        **kwargs,
    )
    # unpatch_bitsandbytes_compute_dtype()
    for param in model.parameters():
        param.requires_grad_(False)
    model, _ = patch_model_and_tokenizer(model, None)

    llm = load_vllm(
        model_name             = model_name,
        config                 = config,
        gpu_memory_utilization = gpu_memory_utilization,
        max_seq_length         = 2048,
        dtype                  = dtype,
        disable_log_stats      = False,
        float8_kv_cache        = float8_kv_cache,
        conservativeness       = conservativeness,
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
    ]*counts
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        padding = True,
    )

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        # temperature = 1.5,
        # min_p = 0.1,
        temperature = 0.8,
        top_p = 0.95,
        logprobs = 0,
        prompt_logprobs = 0,
        max_tokens = 256,
    )

    # Cannot just use llm.generate or OOM - split into batches
    batches = create_batches(inputs, llm.approx_max_num_seqs)
    completion_ids = []
    for batch in batches:
        outputs = llm.generate(batch, sampling_params)
        completion_ids.extend(out.token_ids for completions in outputs for out in completions.outputs)
    pass
    del completion_ids

    # Check all hidden states manually
    input_ids = tokenizer(inputs[0], add_special_tokens = False, return_tensors = "pt")
    input_ids = input_ids["input_ids"].to("cuda", non_blocking = True)
    _test_same_model(model, new_model, input_ids)

    delete_vllm(llm)

    # Delete model as well
    model.model.embed_tokens.weight = None
    new_model.model.embed_tokens.weight = None

    for i in range(len(model.model.layers)):
        model.model.layers[i] = None
        new_model.model.layers[i] = None
    pass

    model.model.norm.weight = None
    new_model.model.norm.weight = None
    model.lm_head.weight = None
    new_model.lm_head.weight = None
    model.model = None
    new_model.model = None
    del model
    del new_model

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
pass


def test_get_vllm_state_dict():
    # All Unsloth Zoo code licensed under LGPLv3
    patch_vllm()

    free_memory, total_memory = torch.cuda.mem_get_info()

    model_names = [
        ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", 100,),
        ("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", 100,),
        ("unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit", 50,),
    ]
    bfloat16_dtype = torch.float16
    if total_memory >= 40 * 1000 * 1000 * 1000:
        model_names += [
            ("unsloth/Qwen2.5-3B-Instruct", 50,),
            ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit", 100,),
            ("unsloth/meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit", 25,),
            ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", 25,),
        ]
        bfloat16_dtype = torch.bfloat16
    pass

    for i, (model_name, counts,) in enumerate(model_names):
        gc.collect()
        torch.cuda.empty_cache()
        dtype = torch.float16 if i % 2 == 0 else bfloat16_dtype
        print(f"##### Testing {model_name} with dtype = {dtype} #####")
        if bfloat16_dtype == torch.float16:
            counts = counts // 4
            conservativeness = 0.8
            float8_kv_cache = True
            gpu_memory_utilization = 0.5
        else:
            conservativeness = 1.0
            float8_kv_cache = True
            gpu_memory_utilization = 0.7
        try:
            _test_get_vllm_state_dict(
                model_name = model_name,
                dtype = dtype,
                gpu_memory_utilization = gpu_memory_utilization,
                counts = counts,
                conservativeness = conservativeness,
                float8_kv_cache = float8_kv_cache,
            )
        except Exception as error:
            error = str(error)
            raise RuntimeError(f"[{model_name}]\n{error}")
        gc.collect()
        torch.cuda.empty_cache()
    pass
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
