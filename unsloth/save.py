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

from peft import PeftModelForCausalLM
from collections import OrderedDict
import bitsandbytes as bnb
import peft
import gc
import os
from tqdm import tqdm as ProgressBar
import shutil
from typing import Optional, Callable, Union
import torch
from transformers.models.llama.modeling_llama import logger
from .kernels import fast_dequantize, QUANT_STATE, get_lora_parameters

__all__ = [
    "unsloth_save_model",
    #"colab_quantize_to_gguf",
]

LLAMA_WEIGHTS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
LLAMA_LAYERNORMS = (
    "input_layernorm", "post_attention_layernorm",
)

# From https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html
ALLOWED_QUANTS = \
{
    "q2_k"   : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l" : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m" : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s" : "Uses Q3_K for all tensors",
    "q4_0"   : "Original quant method, 4-bit.",
    "q4_1"   : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_m" : "Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q4_k_s" : "Uses Q4_K for all tensors",
    "q5_0"   : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"   : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_m" : "Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q5_k_s" : "Uses Q5_K for all tensors",
    "q6_k"   : "Uses Q8_K for all tensors",
    "q8_0"   : "Almost indistinguishable from float16. High resource use and slow. Not recommended for most users.",
}


def _merge_lora(layer, name):
    if isinstance(layer, (bnb.nn.Linear4bit, peft.tuners.lora.Linear4bit)):
        # Is LoRA so we need to merge!
        W, quant_state, A, B, s = get_lora_parameters(layer)
        dtype = quant_state.dtype if type(quant_state) is not list else quant_state[2]
        W = fast_dequantize(W, quant_state).to(torch.float32).t()
        sAB = (A.t().to(torch.float32) @ (s * B.t().to(torch.float32)))
        W += sAB
        if not torch.isfinite(W).all():
            raise ValueError(f"Unsloth: Merge failed.\n{name} has some elements = infinity.")
        W = W.t().to(dtype)
    else:
        W = layer.weight
    return W
pass


@torch.inference_mode
def unsloth_save_model(
    model,
    tokenizer,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    push_to_hub: bool = False,
    max_shard_size: Union[int, str] = "7GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    save_peft_format: bool = True,
    temporary_location = "_unsloth_temporary_saved_buffers",
    **kwargs,
):
    logger.warning_once(
        "Unsloth: `unsloth_save_model` is still in development mode.\n"\
        "If anything errors or breaks, please file a ticket on Github.\n"\
        "Also, if you used this successfully, please tell us on Discord!"
    )

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass

    assert(hasattr(model, "model"))
    assert(hasattr(model.model, "model"))
    assert(hasattr(model.model.model, "layers"))

    # HF also uses a OrderedDict
    state_dict = OrderedDict()
    state_dict["model.embed_tokens.weight"] = model.model.model.embed_tokens.weight

    print("Unsloth: Merging 4bit and LoRA weights to 16bit...")
    for j, layer in enumerate(ProgressBar(model.model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W = _merge_lora(proj, name)
            filename = os.path.join(temporary_location, f"{name}.pt")
            torch.save(W, filename)
            state_dict[name] = torch.load(filename, map_location = "cpu", mmap = True)
        pass
        for item in LLAMA_LAYERNORMS:
            state_dict[f"model.layers.{j}.{item}.weight"] = eval(f"layer.{item}.weight")
        pass
    pass

    state_dict["model.norm.weight"] = model.model.model.norm.weight
    state_dict["lm_head.weight"]    = model.model.lm_head.weight

    print("Unsloth: Saving tokenizer...")
    tokenizer.save_pretrained(
        save_directory = save_directory,
        is_main_process = is_main_process,
        state_dict = state_dict,
        save_function = save_function,
        push_to_hub = push_to_hub,
        max_shard_size = max_shard_size,
        safe_serialization = safe_serialization,
        variant = variant,
        token = token,
        save_peft_format = save_peft_format,
    )

    print("Unsloth: Saving model. This will take 5 minutes for Llama-7b...")
    model.model.save_pretrained(
        save_directory = save_directory,
        is_main_process = is_main_process,
        state_dict = state_dict,
        save_function = save_function,
        push_to_hub = push_to_hub,
        max_shard_size = max_shard_size,
        safe_serialization = safe_serialization,
        variant = variant,
        token = token,
        save_peft_format = save_peft_format,
    )

    # Remove temporary location
    shutil.rmtree(temporary_location)
pass


"""
def _colab_quantize_to_gguf(save_directory, quantization_method = "q4_k_m"):

    logger.warning_once(
        "Unsloth: `colab_quantize_to_gguf` is still in development mode.\n"\
        "If anything errors or breaks, please file a ticket on Github.\n"\
        "Also, if you used this successfully, please tell us on Discord!"
    )

    if quantization_method not in ALLOWED_QUANTS.keys():
        error = f"Unsloth: Quant method = [{quantization_method}] not supported. Choose from below:\n"
        for key, value in ALLOWED_QUANTS.items():
            error += f"[{key}] => {value}\n"
        raise RuntimeError(error)
    pass

    print_info = \
        f"==((====))==  Unsloth: Conversion from QLoRA to GGUF information\n"\
        f"   \\\   /|    [0] Installing llama.cpp will take 3 minutes.\n"\
        f"O^O/ \_/ \\    [1] Converting HF to GUUF 16bits will take 3 minutes.\n"\
        f"\        /    [2] Converting GGUF 16bits to q4_k_m will take 20 minutes.\n"\
        f' "-____-"     In total, you will have to wait around 26 minutes.\n'
    print(print_info)

    if not os.path.exists("llama.cpp"):
        print("Unsloth: [0] Installing llama.cpp. This will take 3 minutes...")
        !git clone https://github.com/ggerganov/llama.cpp
        !cd llama.cpp && make clean && LLAMA_CUBLAS=1 make -j
        !pip install gguf protobuf
    pass

    print("Unsloth: [1] Converting HF into GGUF 16bit. This will take 3 minutes...")
    !python llama.cpp/convert.py {save_directory} \
        --outfile {save_directory}-unsloth.gguf \
        --outtype f16

    print("Unsloth: [2] Converting GGUF 16bit into q4_k_m. This will take 20 minutes...")
    final_location = f"./{save_directory}-{quantization_method}-unsloth.gguf"
    !./llama.cpp/quantize ./{save_directory}-unsloth.gguf \
        {final_location} {quantization_method}

    print(f"Unsloth: Output location: {final_location}")
pass
"""
