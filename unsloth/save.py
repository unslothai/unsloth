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
import pickle
import gc
from transformers.models.llama.modeling_llama import logger
from .kernels import fast_dequantize, QUANT_STATE, get_lora_parameters
import subprocess
import psutil

__all__ = [
    "print_quantization_methods",
    "unsloth_save_model",
    "save_to_gguf",
    "patch_saving_functions",
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
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
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


def fast_save_pickle(shard, name):
    # Use this if # CPUs is <= 2
    print(f"Unsloth: Saving {name}...")
    torch.save(
        shard,
        name,
        pickle_module   = pickle,
        pickle_protocol = pickle.HIGHEST_PROTOCOL,
    )
    return
pass


@torch.inference_mode
def unsloth_save_model(
    model,
    tokenizer,
    save_directory       : Union[str, os.PathLike],
    merge_method         : str = "lora", # ["lora", "16bit", "4bit"]
    push_to_hub          : bool = False,
    token                : Optional[Union[str, bool]] = None,
    is_main_process      : bool = True,
    state_dict           : Optional[dict] = None,
    save_function        : Callable = torch.save,
    max_shard_size       : Union[int, str] = "5GB",
    safe_serialization   : bool = True,
    variant              : Optional[str] = None,
    save_peft_format     : bool = True,

    # Push to hub
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = None,
    private              : Optional[bool] = None,
    create_pr            : bool = False,
    revision             : str = None,
    commit_description   : str = None,
    tags                 : List[str] = None,

    # Our functions
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.9,
):
    save_pretrained_settings = dict(locals())
    for deletion in ("model", "tokenizer", "merge_method", "temporary_location", "maximum_memory_usage"):
        del save_pretrained_settings[deletion]
    pass
    import re

    assert(maximum_memory_usage > 0 and maximum_memory_usage <= 0.95)

    # Clean memory up first
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    pass

    merge_method = merge_method.lower().replace(" ", "_")
    if merge_method != "lora" and merge_method != "16bit" and merge_method != "4bit":
        raise RuntimeError(
            "Unsloth: You must select one of 3 options when saving models:\n"\
            '"lora"         ==> This is the fastest and easiet. Just saves LoRA modules.\n'\
            '"merged_16bit" ==> This merges LoRA weights and saves to float16. Needed for llama.cpp / GGUF.\n'\
            '"merged_4bit"  ==> This merges LoRA weights and saves to 4bit. Useful for DPO / inference.'
        )
    pass

    if merge_method == "4bit":
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
    save_pretrained_settings["tags"] = tags

    if (merge_method == "lora") and push_to_hub:
        if token is None:
            raise RuntimeError(
                "Unsloth: Pushing to HF requires a token. Pass `token = 'hf_....'`\n"\
                "Go to https://huggingface.co/settings/tokens."
            )
        pass

        model.push_to_hub(
            repo_id            = save_directory,
            use_temp_dir       = use_temp_dir,
            commit_message     = commit_message,
            private            = private,
            token              = token,
            max_shard_size     = max_shard_size,
            create_pr          = create_pr,
            safe_serialization = safe_serialization,
            revision           = revision,
            commit_description = commit_description,
            tags               = tags,
        )
        if tokenizer is not None:
            tokenizer.push_to_hub(
                repo_id            = save_directory,
                use_temp_dir       = use_temp_dir,
                commit_message     = commit_message,
                private            = private,
                token              = token,
                max_shard_size     = max_shard_size,
                create_pr          = create_pr,
                safe_serialization = safe_serialization,
                revision           = revision,
                commit_description = commit_description,
                tags               = tags,
            )
        pass
        return save_directory
    pass

    # If push_to_hub, we must remove the .../ part of a repo
    if push_to_hub and "/" in save_directory:

        new_save_directory = save_directory[save_directory.find("/"):]

        logger.warning_once(
            f"Unsloth: You are pushing to hub, but you passed your HF username.\n"\
            f"We shall truncate {save_directory} to {new_save_directory}"
        )

        save_pretrained_settings["save_directory"] = new_save_directory
        save_directory = new_save_directory
    pass
    
    if (merge_method == "4bit") or (merge_method == "lora") or (
        not hasattr(model, "model") or \
        not hasattr(model.model, "model") or \
        not hasattr(model.model.model, "layers")
    ):
        # Do general saving
        
        # Edit save_pretrained_settings
        # [TODO] _create_repo has errors due to **kwargs getting accepted
        for deletion in \
            ("use_temp_dir", "commit_message", "create_pr", "revision", "commit_description", "tags",):
            del save_pretrained_settings[deletion]
        pass
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["unsloth",])

        if tokenizer is not None:
            print("Unsloth: Saving tokenizer...", end = "")
            tokenizer.save_pretrained(**save_pretrained_settings)
            print(" Done.")
        else:
            print()

        print("Unsloth: Saving model...", end = "")
        if merge_method != "lora": print(" This might take 10 minutes for Llama-7b...", end = "")

        model.save_pretrained(**save_pretrained_settings)
        print(" Done.")
        return save_directory
    pass

    print("Unsloth: Merging 4bit and LoRA weights to 16bit...")

    # Determine max RAM usage minus sharding
    max_ram = psutil.virtual_memory().available
    sharded_ram_usage = 5 * 1024 * 1024 * 1024
    if type(max_shard_size) is str:
        gb_found = re.match("([0-9]{1,})[\s]{0,}GB", max_shard_size, flags = re.IGNORECASE)
        mb_found = re.match("([0-9]{1,})[\s]{0,}MB", max_shard_size, flags = re.IGNORECASE)
        if   gb_found: sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024
        elif mb_found: sharded_ram_usage = int(mb_found.group(1)) * 1024 * 1024 
    elif type(max_shard_size) is int:
        sharded_ram_usage = sharded_ram_usage
    pass

    # Switch to our fast saving modules if it's a slow PC!
    n_cpus = psutil.cpu_count(logical = False)

    if safe_serialization is None:
        safe_serialization = True
        save_pretrained_settings["safe_serialization"] = safe_serialization

    elif safe_serialization and (n_cpus <= 2):
        logger.warning_once(
            f"Unsloth: You have {n_cpus} CPUs. Using `safe_serialization` is 10x slower.\n"\
            f"We shall switch to Pytorch saving, which will take 3 minutes and not 30 minutes.\n"\
            f"To force `safe_serialization`, set it to None instead.",
        )
        safe_serialization = False
        save_function = fast_save_pickle
        save_pretrained_settings["safe_serialization"] = safe_serialization
        save_pretrained_settings["save_function"]      = save_function
    pass

    # Only safe_serialization uses more RAM
    if safe_serialization:
        max_ram -= sharded_ram_usage
    else:
        max_ram -= sharded_ram_usage*0.25 # Uses much less
    pass

    max_ram = int(max(0, max_ram) * maximum_memory_usage)
    print(f"Unsloth: Will use up to "\
          f"{round(max_ram/1024/1024/1024, 2)} out of "\
          f"{round(psutil.virtual_memory().total/1024/1024/1024, 2)} RAM for saving.")

    # Max directory for disk saving
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass

    # HF also uses a OrderedDict
    from collections import OrderedDict
    state_dict = OrderedDict()
    state_dict["model.embed_tokens.weight"] = model.model.model.embed_tokens.weight.data

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
            # elif (max_ram - W.nbytes) > 0:
            #     # Save to CPU memory
            #     logger.warning_once(f"We will save to RAM and not VRAM now.")
            #     state_dict[name] = W.to("cpu", non_blocking = True)
            #     max_ram = max(max_ram - W.nbytes, 0)
            else:
                # Save to Disk
                logger.warning_once(f"We will save to Disk and not RAM now.")
                filename = os.path.join(temporary_location, f"{name}.pt")
                torch.save(W, filename, pickle_module = pickle, pickle_protocol = pickle.HIGHEST_PROTOCOL,)
                state_dict[name] = torch.load(filename, map_location = "cpu", mmap = True)
        pass
        for item in LLAMA_LAYERNORMS:
            state_dict[f"model.layers.{j}.{item}.weight"] = eval(f"layer.{item}.weight.data")
        pass
    pass

    state_dict["model.norm.weight"] = model.model.model.norm.weight.data
    state_dict["lm_head.weight"]    = model.model.lm_head.weight.data

    # All tensors MUST be type torch.Tensor and not torch.nn.parameter.Parameter
    for key, value in state_dict.items():
        if hasattr(value, "data"): state_dict[key] = value = value.data
        if type(value) is not torch.Tensor:
            logger.warning_once(f"Unsloth: {key} is not a Tensor but a {type(value)}.")
        pass
    pass

    # Edit save_pretrained_settings
    # [TODO] _create_repo has errors due to **kwargs getting accepted
    save_pretrained_settings["state_dict"] = state_dict
    for deletion in \
        ("use_temp_dir", "commit_message", "create_pr", "revision", "commit_description", "tags",):
        del save_pretrained_settings[deletion]
    pass
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth",])

    if tokenizer is not None:
        print("Unsloth: Saving tokenizer...", end = "")
        tokenizer.save_pretrained(**save_pretrained_settings)
        print(" Done.")
    else:
        print()

    print("Unsloth: Saving model... This might take 5 minutes for Llama-7b...")
    model.model.save_pretrained(**save_pretrained_settings)
    print("Done.")

    save_pretrained_settings["state_dict"] = None

    # for j, (key, value) in enumerate(state_dict.items()):
    #     state_dict[key] = None
    #     if j % 10 == 0:
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #     pass
    # pass
    # state_dict = None
    # del state_dict
    # torch.cuda.empty_cache()
    # gc.collect()

    # Remove temporary location
    import shutil
    shutil.rmtree(temporary_location)

    # for _ in range(3):
    #     torch.cuda.empty_cache()
    #     gc.collect()
    return save_directory
pass


def install_llama_cpp_clone_non_blocking():
    full_command = ["git", "clone", "https://github.com/ggerganov/llama.cpp"]
    run_installer = subprocess.Popen(full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_llama_cpp_make_non_blocking():
    env = { **os.environ, "LLAMA_CUBLAS": "1", }
    n_jobs = max(int(psutil.cpu_count()*1.5), 1)
    full_command = ["make", "-j", str(n_jobs), "-C", "llama.cpp"]
    run_installer = subprocess.Popen(full_command, env = env, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_python_non_blocking(packages = []):
    full_command = ["pip", "install"] + packages
    run_installer = subprocess.Popen(full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_llama_cpp_blocking():
    commands = [
        "git clone https://github.com/ggerganov/llama.cpp",
        f"cd llama.cpp && make clean && LLAMA_CUBLAS=1 make -j {psutil.cpu_count()*2}",
        "pip install gguf protobuf",
    ]
    if os.path.exists("llama.cpp"): return
    for command in commands:
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stdout:
                print(line.decode("utf-8"), flush = True, end = "")
        pass
    pass
pass


def save_to_gguf(
    model_directory : str = "unsloth_finetuned_model",
    quantization    : str = "fast_quantized",
    _run_installer = None, # Non blocking install of llama.cpp
):
    from transformers.models.llama.modeling_llama import logger

    if   quantization == "not_quantized":  quantization = "f16"
    elif quantization == "fast_quantized": quantization = "q8_0"
    elif quantization == "quantized":      quantization = "q4_k_m"
    elif quantization is None:             quantization = "q8_0"

    if quantization not in ALLOWED_QUANTS.keys():
        error = f"Unsloth: Quant method = [{quantization}] not supported. Choose from below:\n"
        for key, value in ALLOWED_QUANTS.items():
            error += f"[{key}] => {value}\n"
        raise RuntimeError(error)
    pass

    print_info = \
        f"==((====))==  Unsloth: Conversion from QLoRA to GGUF information\n"\
        f"   \\\   /|    [0] Installing llama.cpp will take 3 minutes.\n"\
        f"O^O/ \_/ \\    [1] Converting HF to GUUF 16bits will take 3 minutes.\n"\
        f"\        /    [2] Converting GGUF 16bits to {quantization} will take 20 minutes.\n"\
        f' "-____-"     In total, you will have to wait around 26 minutes.\n'
    print(print_info)

    print("Unsloth: [0] Installing llama.cpp. This will take 3 minutes...")
    if _run_installer is not None:
        _run_installer.wait()
    else:
        install_llama_cpp_blocking()
    pass

    print("Unsloth: [1] Converting HF into GGUF format. This will take 3 minutes...")
    first_conversion = "f16"
    if   quantization == "f32":  first_conversion = "f32"
    elif quantization == "f16":  first_conversion = "f16"
    elif quantization == "q8_0": first_conversion = "q8_0"

    n_cpus = psutil.cpu_count()*2
    # Concurrency from https://rentry.org/llama-cpp-conversions#merging-loras-into-a-model
    
    final_location = f"./{model_directory}-unsloth.{first_conversion.upper()}.gguf"

    command = f"python llama.cpp/convert.py {model_directory} "\
        f"--outfile {final_location} "\
        f"--outtype {first_conversion} --concurrency {n_cpus}"

    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
        for line in sp.stdout:
            print(line.decode("utf-8"), flush = True, end = "")
    pass

    print(f"Unsloth: Conversion completed! Output location: {final_location}")

    if quantization != first_conversion:
        old_location = final_location
        print(f"Unsloth: [2] Converting GGUF 16bit into {quantization}. This will take 20 minutes...")
        final_location = f"./{model_directory}-unsloth.{quantization.upper()}.gguf"

        command = f"./llama.cpp/quantize {old_location} "\
            f"{final_location} {quantization} {n_cpus}"
        
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stdout:
                print(line.decode("utf-8"), flush = True, end = "")
        pass
        print(f"Unsloth: Conversion completed! Output location: {final_location}")
    pass

    return final_location
pass


def unsloth_save_pretrained_merged(
    self,
    save_directory       : Union[str, os.PathLike],
    tokenizer            = None,
    merge_method         : str = "16bit", # ["lora", "16bit", "4bit"]
    push_to_hub          : bool = False,
    token                : Optional[Union[str, bool]] = None,
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
):
    """
        Same as .save_pretrained(...) except 4bit weights are auto
        converted to float16 with as few overhead as possible.

        Choose for `merge_method` to be either:
        1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
        2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
        3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"\
            "You can do it separately via `tokenizer.save_pretrained(...)`"
        )
    pass

    arguments = dict(locals())
    arguments["model"] = self
    del arguments["self"]
    unsloth_save_model(**arguments)
    for _ in range(3):
        gc.collect()
pass


def unsloth_push_to_hub_merged(
    self,
    repo_id              : str,
    tokenizer            = None,
    merge_method         : str = "16bit", # ["lora", "16bit", "4bit"]
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = None,
    private              : Optional[bool] = None,
    token                : Union[bool, str, None] = None,
    max_shard_size       : Union[int, str, None] = "5GB",
    create_pr            : bool = False,
    safe_serialization   : bool = True,
    revision             : str = None,
    commit_description   : str = None,
    tags                 : Optional[List[str]] = None,
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.85,
):
    """
        Same as .push_to_hub(...) except 4bit weights are auto
        converted to float16 with as few overhead as possible.

        Choose for `merge_method` to be either:
        1. `16bit`: Merge LoRA into float16 weights. Useful for GGUF / llama.cpp.
        2.  `4bit`: Merge LoRA into int4 weights. Useful for DPO / HF inference.
        3.  `lora`: Save LoRA adapters with no merging. Useful for HF inference.
    """
    if tokenizer is None:
        logger.warning_once(
            "Unsloth: You're not saving a tokenizer as well?\n"\
            "You can do it separately via `tokenizer.push_to_hub(...)`"
        )
    pass

    arguments = dict(locals())
    arguments["model"]          = self
    arguments["save_directory"] = repo_id
    arguments["push_to_hub"]    = True
    del arguments["self"]
    del arguments["repo_id"]
    unsloth_save_model(**arguments)
    for _ in range(3):
        gc.collect()
pass


def unsloth_save_pretrained_gguf(
    self,
    save_directory       : Union[str, os.PathLike],
    tokenizer            = None,
    quantization         : str = "fast_quantized",
    push_to_hub          : bool = False,
    token                : Optional[Union[str, bool]] = None,
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
):
    """
        Same as .save_pretrained(...) except 4bit weights are auto
        converted to float16 then converted to GGUF / llama.cpp format.

        Choose for `quantization` to be:
        "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
        "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
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
    """
    if tokenizer is None:
        raise ValueError("Unsloth: Saving to GGUF must have a tokenizer.")

    arguments = dict(locals())
    arguments["model"]        = self
    arguments["tokenizer"]    = tokenizer
    arguments["push_to_hub"]  = False # We save ourselves
    arguments["merge_method"] = "16bit" # Must be 16bit
    del arguments["self"]
    del arguments["quantization"]

    # Non blocking install GGUF first
    git_clone = install_llama_cpp_clone_non_blocking()
    python_install = install_python_non_blocking(["gguf", "protobuf"])
    git_clone.wait()
    makefile  = install_llama_cpp_make_non_blocking()
    new_save_directory = unsloth_save_model(**arguments)
    python_install.wait()

    for _ in range(3):
        gc.collect()

    file_location = save_to_gguf(new_save_directory, quantization, makefile)

    # And save to HF
    if push_to_hub:
        print("Unsloth: Uploading GGUF to Huggingface Hub...")

        from huggingface_hub import create_repo
        create_repo(
            repo_id   = save_directory,
            token     = token,
            repo_type = "model",
            exist_ok  = True,
        )

        from huggingface_hub import HfApi
        hf_api = HfApi(token = token)

        if "/" in file_location:
            uploaded_location = file_location[file_location.rfind("/")+1:]
        else:
            uploaded_location = file_location
        pass

        hf_api.upload_file(
            path_or_fileobj = file_location,
            path_in_repo    = uploaded_location,
            repo_id         = save_directory,
            repo_type       = "model",
        )
    pass
pass


def unsloth_push_to_hub_gguf(
    self,
    repo_id              : str,
    tokenizer            = None,
    quantization         : str = "fast_quantized",
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = None,
    private              : Optional[bool] = None,
    token                : Union[bool, str, None] = None,
    max_shard_size       : Union[int, str, None] = "5GB",
    create_pr            : bool = False,
    safe_serialization   : bool = True,
    revision             : str = None,
    commit_description   : str = None,
    tags                 : Optional[List[str]] = None,
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.85,
):
    """
        Same as .push_to_hub(...) except 4bit weights are auto
        converted to float16 then converted to GGUF / llama.cpp format.

        Choose for `quantization` to be:
        "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
        "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
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
    """
    if tokenizer is None:
        raise ValueError("Unsloth: Saving to GGUF must have a tokenizer.")

    arguments = dict(locals())
    arguments["model"]          = self
    arguments["tokenizer"]      = tokenizer
    arguments["save_directory"] = repo_id
    arguments["push_to_hub"]    = False # We save ourselves
    arguments["merge_method"]   = "16bit" # Must be 16bit
    del arguments["self"]
    del arguments["repo_id"]
    del arguments["quantization"]

    # Non blocking install GGUF first
    git_clone = install_llama_cpp_clone_non_blocking()
    python_install = install_python_non_blocking(["gguf", "protobuf"])
    git_clone.wait()
    makefile  = install_llama_cpp_make_non_blocking()
    new_save_directory = unsloth_save_model(**arguments)

    for _ in range(3):
        gc.collect()

    python_install.wait()
    file_location = save_to_gguf(new_save_directory, quantization, makefile)

    # Save to hub
    print("Unsloth: Uploading GGUF to Huggingface Hub...")

    from huggingface_hub import create_repo
    create_repo(
        repo_id   = save_directory,
        private   = private,
        token     = token,
        repo_type = "model",
        exist_ok  = True,
    )

    from huggingface_hub import HfApi
    hf_api = HfApi(token = token)

    if "/" in file_location:
        uploaded_location = file_location[file_location.rfind("/")+1:]
    else:
        uploaded_location = file_location
    pass

    hf_api.upload_file(
        path_or_fileobj = file_location,
        path_in_repo    = uploaded_location,
        repo_id         = save_directory,
        repo_type       = "model",
    )
pass


def patch_saving_functions(model):
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
    if "tags" in arguments and arguments["tags"] is not None:
        assert(isinstance(arguments["tags"], (list, tuple)))
        arguments["tags"] = list(arguments["tags"]) + ["unsloth",]
    elif "tags" in arguments:
        arguments["tags"] = ["unsloth",]
    elif hasattr(self, "add_model_tags"):
        self.add_model_tags(["unsloth",])
    try:
        return self._original_push_to_hub(**arguments)
    except:
        del arguments["tags"]
        return self._original_push_to_hub(**arguments)
    pass
    '''
    exec(push_to_hub_text, globals())
    model.push_to_hub = types.MethodType(unsloth_push_to_hub, model)

    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth",])

    if hasattr(model, "config"):
        # Counteract tokenizers
        model.push_to_hub_merged     = types.MethodType(unsloth_push_to_hub_merged,     model)
        model.save_pretrained_merged = types.MethodType(unsloth_save_pretrained_merged, model)
        model.push_to_hub_gguf       = types.MethodType(unsloth_push_to_hub_gguf,       model)
        model.save_pretrained_gguf   = types.MethodType(unsloth_save_pretrained_gguf,   model)
    else:
        model.push_to_hub_merged     = model.push_to_hub
        model.save_pretrained_merged = model.save_pretrained
        model.push_to_hub_gguf       = model.push_to_hub
        model.save_pretrained_gguf   = model.save_pretrained
    pass

    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        if hasattr(original_model, "_original_push_to_hub"): continue
        
        original_model._original_push_to_hub = original_model.push_to_hub
        original_model.push_to_hub = types.MethodType(unsloth_push_to_hub, original_model)

        if hasattr(original_model, "add_model_tags"):
            original_model.add_model_tags(["unsloth",])

        if hasattr(original_model, "config"):
            # Counteract tokenizers
            original_model.push_to_hub_merged     = \
                types.MethodType(unsloth_push_to_hub_merged,     original_model)

            original_model.save_pretrained_merged = \
                types.MethodType(unsloth_save_pretrained_merged, original_model)

            original_model.push_to_hub_gguf       = \
                types.MethodType(unsloth_push_to_hub_gguf,       original_model)

            original_model.save_pretrained_gguf   = \
                types.MethodType(unsloth_save_pretrained_gguf,   original_model)
        pass
    pass
    return
pass
