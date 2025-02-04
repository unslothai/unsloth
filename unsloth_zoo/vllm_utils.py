# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
from transformers.utils.import_utils import _is_package_available
import re
from collections import OrderedDict
import numpy as np
from transformers import AutoModelForCausalLM
from copy import deepcopy
import math
from .utils import _get_dtype


if _is_package_available("vllm"):

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
    pass
else:
    def patch_vllm_bitsandbytes():
        return
    pass
pass


if _is_package_available("bitsandbytes"):
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
            dtype=getattr(torch, qs_dict["dtype"]),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state
    pass

    def patch_bitsandbytes_quant_state():
        bitsandbytes.functional.QuantState.from_dict = from_dict
    pass
else:
    def patch_bitsandbytes_quant_state():
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


def create_empty_causal_lm(config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    # Empty model from config
    new_config = deepcopy(config)
    new_config.intermediate_size = 0
    new_config.hidden_size = 0
    new_config.vocab_size = 1
    new_config.pad_token_id = 0
    new_model = AutoModelForCausalLM.from_config(
        new_config,
        attn_implementation = "eager",
    )
    new_model = new_model.to(device = "cuda:0", dtype = dtype)
    return new_model
pass


def convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float16):
    # All Unsloth Zoo code licensed under LGPLv3
    # Unmerges vLLM modules to create HF compatible model
    new_model = create_empty_causal_lm(config, dtype)
    quantization_config = config.quantization_config
    kwargs = dict()
    # Get quantization_config flags
    compute_dtype = _get_dtype(quantization_config["bnb_4bit_compute_dtype"])
    kwargs["compress_statistics"] = quantization_config["bnb_4bit_use_double_quant"]
    kwargs["quant_type"] = quantization_config["bnb_4bit_quant_type"]
    kwargs["quant_storage"] = _get_dtype(quantization_config["bnb_4bit_quant_storage"])

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

            if f"{layer_name}.weight.bias" in quant_state_dict:
                # Has bias!
                has_bias = True
                bias = quant_state_dict[f"{layer_name}.weight.bias"]
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
    layer = Linear(0, 0, device = "cuda:0", bias = has_bias)
    layer.in_features  = weight.shape[1]
    layer.out_features = weight.shape[0]
    layer.weight = torch.nn.Parameter(weight, requires_grad = False)
    new_model.lm_head = layer
    if getattr(config, "tie_word_embeddings", False): new_model.tie_weights()

    # Fix up config file
    for module in new_model.modules():
        if hasattr(module, "config"):
            module.config = config
        if hasattr(module, "intermediate_size"):
            module.intermediate_size = config.intermediate_size
        if hasattr(module, "hidden_size"):
            module.hidden_size = config.hidden_size
    pass
    new_model.config = config

    # Cleanup
    import gc
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

    # 2 bytes = float16
    total_quantizable_elements = (qkvo + mlp)*n_layers * 2
    total_float16_elements     = (layernorms + embed_tokens + lm_head)*2
    factor = 16/5 if load_in_4bit else 1 # Should be 4.5 but use 5
    bytes_for_model = \
        total_quantizable_elements / factor + total_float16_elements + lora_elements

    # KV cache size (float16 is 2 bytes)
    kv_elements = (kv_size * 2 * n_layers) * 2
    memory_left_for_kv_cache = free_memory - bytes_for_model
    # Approx maximum # of KV cache elements
    max_num_batched_tokens = int(0.9*(memory_left_for_kv_cache / kv_elements))
    # Round by 256
    max_num_batched_tokens = (max_num_batched_tokens // 256) * 256
    # Assuming all requests output max_seq_length, get theoretical max requests
    approx_max_num_seqs = int(max_num_batched_tokens / max_seq_length)

    if approx_max_num_seqs <= 1:
        raise MemoryError("Unsloth: Not enough memory to load vLLM!")
    return max_num_batched_tokens, approx_max_num_seqs
pass


def load_vllm(
    model_name             : str   = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    config                 = None,
    gpu_memory_utilization : float = 0.8,
    max_seq_length         : int   = 8192,
    random_state           : int   = 0,
    enable_lora            : bool  = True,
    max_lora_rank          : int   = 16,
    max_loras              : int   = 1,
    use_async              : bool  = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Create vLLM instance
    assert(config is not None)
    max_num_batched_tokens, approx_max_num_seqs = approximate_vllm_memory_usage(
        config, 
        max_seq_length = max_seq_length,
        gpu_memory_utilization = gpu_memory_utilization,
        enable_lora = enable_lora,
        max_lora_rank = max_lora_rank,
        max_loras = max_loras,
    )
    print(
        f"Unsloth: vLLM loading {model_name} with GPU utilization = {gpu_memory_utilization*100}%\n"\
        f"Unsloth: vLLM can process {approx_max_num_seqs} sequences and {max_num_batched_tokens} tokens in tandem."
    )

    from vllm import LLM, LLMEngine, AsyncLLMEngine, EngineArgs
    use_bitsandbytes = model_name.lower().endswith("-bnb-4bit")
    engine_args = dict(
        model                  = model_name,
        gpu_memory_utilization = gpu_memory_utilization,
        max_model_len          = max_seq_length,
        quantization           = "bitsandbytes" if use_bitsandbytes else None,
        load_format            = "bitsandbytes" if use_bitsandbytes else "auto",

        max_num_batched_tokens = max_num_batched_tokens, # Max tokens for chunked prefill or else OOM
        max_num_seqs           = approx_max_num_seqs, # Force only some requests at 1 time or else OOM
        max_logprobs           = 0, # Disallow logprobs being returned
        seed                   = random_state, # Default is 0

        # lora_extra_vocab_size = 0, # Breaks vLLM so we leave it as 256
        enable_lora            = enable_lora,
        max_lora_rank          = max_lora_rank,
        max_loras              = max_loras,

        disable_log_stats      = True,
        # enable_prefix_caching  = True, # LoRA fails with chunked prefill as at Feb 2025
        # enable_chunked_prefill = True, # LoRA fails with chunked prefill as at Feb 2025
        max_seq_len_to_capture = 8192, # Default is 8192 for CUDAGraphs
        compilation_config     = 3, # 0, 1, 2, 3
    )

    if use_async:
        llm = AsyncLLMEngine.from_engine_args(EngineArgs(**engine_args))
    else:
        llm = LLM(**engine_args)
    pass

    # Save maximum requests length since llm.generate fails to partition inputs
    llm.approx_max_num_seqs = approx_max_num_seqs
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


def test_get_vllm_state_dict(
    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    dtype = torch.float16,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check if model is allowed to be used in vLLM
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_name,
        token = None,
        revision = None,
        trust_remote_code = False,
    )
    if not vllm_dynamic_quant_supported(model_name, config):
        raise NotImplementedError(f"Unsloth: Dynamic quant of {model_name} not supported in vLLM")

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        use_exact_model_name = True,
    )

    llm = load_vllm(
        model_name             = model_name,
        config                 = config,
        gpu_memory_utilization = 0.5,
        max_seq_length         = 2048,
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
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
