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

from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from typing import Optional, Callable, Union, List
import torch
import os
from transformers.models.llama.modeling_llama import logger
from .kernels import fast_dequantize, QUANT_STATE, get_lora_parameters

__all__ = [
    "print_quantization_methods",
    "unsloth_save_model",
    "save_to_gguf",
    "patch_push_to_hub",
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
    "not quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
}

def print_quantization_methods():
    for key, value in ALLOWED_QUANTS.items():
        print(f'"{key}"  ==> {value}')
    pass
pass


def _merge_lora(layer, name):
    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit)):
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


def patch_push_to_hub(model):
    import inspect
    import re
    import types
    from typing import Callable, Optional, Union, List

    if hasattr(model, "_original_push_to_hub"): return

    original_push_to_hub = model.push_to_hub
    signature = str(inspect.signature(original_push_to_hub)).replace("NoneType", "None")
    signature = signature[1:]
    signature = re.sub("<function save at .+?>", "torch.save", signature)
    docs = original_push_to_hub.__doc__.encode("utf-8").decode("utf-8")
    model._original_push_to_hub = original_push_to_hub

    push_to_hub_text = f'''def unsloth_push_to_hub(self, {signature}:
    """
    {docs}
    """
    arguments = dict(locals())
    del arguments["self"]
    if arguments["tags"] is not None:
        assert(isinstance(arguments["tags"], (list, tuple)))
        arguments["tags"] = list(arguments["tags"]) + ["unsloth",]
    else:
        arguments["tags"] = ["unsloth",]
    try:
        return self._original_push_to_hub(**arguments)
    except:
        del arguments["tags"]
        return self._original_push_to_hub(**arguments)
    pass
    '''
    exec(push_to_hub_text, globals())
    model.push_to_hub = types.MethodType(unsloth_push_to_hub, model)

    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        if hasattr(original_model, "_original_push_to_hub"): continue
        
        original_model._original_push_to_hub = original_model.push_to_hub
        original_model.push_to_hub = types.MethodType(unsloth_push_to_hub, original_model)
    pass
    return
pass


@torch.inference_mode
def unsloth_save_model(
    model,
    tokenizer,
    save_directory       : Union[str, os.PathLike],
    save_method          : str = "lora", # ["lora", "merged_16bit", "merged_4bit"]
    push_to_hub          : bool = False,
    token                : Optional[Union[str, bool]] = None,
    repo_id              : str = None,
    is_main_process      : bool = True,
    state_dict           : Optional[dict] = None,
    save_function        : Callable = torch.save,
    max_shard_size       : Union[int, str] = "5GB",
    safe_serialization   : bool = True,
    variant              : Optional[str] = None,
    save_peft_format     : bool = True,
    tags                 : List[str] = None,
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.85,
    **kwargs,        
):
    import gc
    import re
    import psutil

    assert(maximum_memory_usage > 0 and maximum_memory_usage <= 0.95)

    # Clean memory up first
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    pass

    save_method = save_method.lower().replace(" ", "_")
    if save_method != "lora" and save_method != "merged_16bit" and save_method != "merged_4bit":
        raise RuntimeError(
            "Unsloth: You must select one of 3 options when saving models:\n"\
            '"lora"         ==> This is the fastest and easiet. Just saves LoRA modules.\n'\
            '"merged_16bit" ==> This merges LoRA weights and saves to float16. Needed for llama.cpp / GGUF.\n'\
            '"merged_4bit"  ==> This merges LoRA weights and saves to 4bit. Useful for DPO / inference.'
        )
    pass

    if save_method == "merged_4bit":
        print("Unsloth: Merging 4bit and LoRA weights to 4bit...")
        print("This might take 5 minutes...")
        model = model.merge_and_unload()
        print("Done.")
    pass

    if tags is not None:
        assert(isinstance(tags, (list, tuple)))
        tags = list(tags) + ["unsloth",]
    else:
        tags = ["unsloth",]
    pass

    if (save_method == "lora") and push_to_hub:
        if token is None:
            raise RuntimeError(
                "Unsloth: Pushing to HF requires a token. Pass `token = 'hf_....'`\n"\
                "Go to https://huggingface.co/settings/tokens."
            )
        pass
        if repo_id is None: repo_id = save_directory

        model.push_to_hub(
            repo_id = repo_id,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            token = token,
            tags = tags,
            **kwargs,
        )
        tokenizer.push_to_hub(
            repo_id = repo_id,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            token = token,
            tags = tags,
            **kwargs,
        )
        return
    
    elif (save_method == "merged_4bit") or (save_method == "lora") or (
        not hasattr(model, "model") or \
        not hasattr(model.model, "model") or \
        not hasattr(model.model.model, "layers")
    ):
        # Do general saving?
        print("Unsloth: Saving tokenizer...", end = "")
        tokenizer.save_pretrained(
            save_directory = save_directory,
            push_to_hub = push_to_hub,
            token = token,
            tags = tags,
        )
        print(" Done.")

        print("Unsloth: Saving model...", end = "")
        if save_method != "lora": print(" This might take 10 minutes for Llama-7b...", end = "")

        model.save_pretrained(
            save_directory = save_directory,
            is_main_process = is_main_process,
            save_function = save_function,
            push_to_hub = push_to_hub,
            max_shard_size = max_shard_size,
            safe_serialization = safe_serialization,
            variant = variant,
            token = token,
            save_peft_format = save_peft_format,
            tags = tags,
        )
        print(" Done.")
        return
    pass

    print("Unsloth: Merging 4bit and LoRA weights to 16bit...")

    # Determine max RAM usage minus sharding
    max_ram = psutil.virtual_memory().total
    sharded_ram_usage = 5 * 1024 * 1024 * 1024
    if type(max_shard_size) is str:
        gb_found = re.match("([0-9]{1,})[\s]{0,}GB", max_shard_size, flags = re.IGNORECASE)
        mb_found = re.match("([0-9]{1,})[\s]{0,}MB", max_shard_size, flags = re.IGNORECASE)
        if   gb_found: sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024
        elif mb_found: sharded_ram_usage = int(mb_found.group(1)) * 1024 * 1024 
    elif type(max_shard_size) is int:
        sharded_ram_usage = sharded_ram_usage
    pass
    max_ram -= sharded_ram_usage
    max_ram = int(max(0, max_ram) * maximum_memory_usage)

    # Max directory for disk saving
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass

    # HF also uses a OrderedDict
    from collections import OrderedDict
    state_dict = OrderedDict()
    state_dict["model.embed_tokens.weight"] = model.model.model.embed_tokens.weight

    max_vram = int(torch.cuda.get_device_properties(0).total_memory * maximum_memory_usage)

    from tqdm import tqdm as ProgressBar
    for j, layer in enumerate(ProgressBar(model.model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W = _merge_lora(proj, name)

            if (torch.cuda.memory_allocated() + W.nbytes) < max_vram:
                # Save to GPU memory
                state_dict[name] = W
            elif (max_ram - W.nbytes) > 0:
                # Save to CPU memory
                logger.warning_once(f"We will save to RAM and not VRAM now.")
                state_dict[name] = W.to("cpu", non_blocking = True)
                max_ram = max(max_ram - W.nbytes, 0)
            else:
                # Save to Disk
                logger.warning_once(f"We will save to Disk and not RAM now.")
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

    print("Unsloth: Saving tokenizer...", end = "")
    tokenizer.save_pretrained(
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        token = token,
        tags = tags,
    )
    print(" Done.")

    print("Unsloth: Saving model... This might take 10 minutes for Llama-7b...", end = "")
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
        tags = tags,
    )
    print(" Done.")

    for j, (key, value) in enumerate(state_dict.items()):
        state_dict[key] = None
        if j % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        pass
    pass
    state_dict = None
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

    # Remove temporary location
    import shutil
    shutil.rmtree(temporary_location)

    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    return
pass


def save_to_gguf(
    model_directory     : str = "finetuned_model",
    quantization_method : str = "not quantized",
):
    from transformers.models.llama.modeling_llama import logger
    import os
    import subprocess
    import psutil

    if   quantization_method == "not quantized":  quantization_method = "f16"
    elif quantization_method == "fast quantized": quantization_method = "q8_0"
    elif quantization_method == "quantized":      quantization_method = "q4_k_m"
    elif quantization_method is None:             quantization_method = "f16"

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
        f"\        /    [2] Converting GGUF 16bits to {quantization_method} will take 20 minutes.\n"\
        f' "-____-"     In total, you will have to wait around 26 minutes.\n'
    print(print_info)

    if not os.path.exists("llama.cpp"):
        print("Unsloth: [0] Installing llama.cpp. This will take 3 minutes...")

        commands = [
            "git clone https://github.com/ggerganov/llama.cpp",
            "cd llama.cpp && make clean && LLAMA_CUBLAS=1 make -j",
            "pip install gguf protobuf",
        ]
        for command in commands:
            with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
                for line in sp.stdout:
                    print(line.decode("utf-8"), flush = True, end = "")
            pass
        pass
    pass

    print("Unsloth: [1] Converting HF into GGUF format. This will take 3 minutes...")
    first_conversion = "f16"
    if   quantization_method == "f32":  first_conversion = "f32"
    elif quantization_method == "f16":  first_conversion = "f16"
    elif quantization_method == "q8_0": first_conversion = "q8_0"

    n_cpus = psutil.cpu_count()*2
    # Concurrency from https://rentry.org/llama-cpp-conversions#merging-loras-into-a-model
    
    command = f"python llama.cpp/convert.py {model_directory} "\
        f"--outfile {model_directory}-{first_conversion}-unsloth.gguf "\
        f"--outtype {first_conversion} --concurrency {n_cpus}"
    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
        for line in sp.stdout:
            print(line.decode("utf-8"), flush = True, end = "")
    pass

    final_location = f"./{model_directory}-{first_conversion}-unsloth.gguf"
    print(f"Unsloth: Conversion completed! Output location: {final_location}")

    if quantization_method != first_conversion:
        print(f"Unsloth: [2] Converting GGUF 16bit into {quantization_method}. This will take 20 minutes...")
        final_location = f"./{model_directory}-{quantization_method}-unsloth.gguf"

        command = f"./llama.cpp/quantize ./{model_directory}-{first_conversion}-unsloth.gguf "\
            f"{final_location} {quantization_method} {n_cpus}"
        
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stdout:
                print(line.decode("utf-8"), flush = True, end = "")
        pass
        print(f"Unsloth: Conversion completed! Output location: {final_location}")
    pass
pass
