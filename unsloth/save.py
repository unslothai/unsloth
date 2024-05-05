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
from peft.tuners.lora import Linear as Peft_Linear
from typing import Optional, Callable, Union, List
import torch
import os
import pickle
import gc
from transformers.models.llama.modeling_llama import logger
from .kernels import fast_dequantize, QUANT_STATE, get_lora_parameters
import subprocess
import psutil
import re
from transformers.models.llama.modeling_llama import logger

__all__ = [
    "print_quantization_methods",
    "unsloth_save_model",
    "save_to_gguf",
    "patch_saving_functions",
]

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT  = "\nCOLAB_"  in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
del keynames

# Weights
LLAMA_WEIGHTS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
LLAMA_LAYERNORMS = (
    "input_layernorm", "post_attention_layernorm",
)

# https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
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
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    # "iq2_xxs" : "2.06 bpw quantization", # Not supported sadly
    # "iq2_xs"  : "2.31 bpw quantization",
    # "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}

def print_quantization_methods():
    for key, value in ALLOWED_QUANTS.items():
        print(f'"{key}"  ==> {value}')
    pass
pass


def _free_cached_model(model):
    from huggingface_hub import scan_cache_dir
    cached_repos = list(scan_cache_dir().repos)

    # Go through every cached repo, and delete the one that matches the model we want to save.
    # Can save 4GB of disk space - useful for Kaggle systems.
    for cached_repo in cached_repos:
        if cached_repo.repo_id == model.config._name_or_path:
            remove_cache_commit = list(cached_repo.revisions)[0].commit_hash
            delete_strategy = scan_cache_dir().delete_revisions(remove_cache_commit,)

            logger.warning_once(
                "Unsloth: Will remove a cached repo with size " + \
                delete_strategy.expected_freed_size_str,
            )

            delete_strategy.execute()
        pass
    pass
pass


def _merge_lora(layer, name):

    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
        # Is LoRA so we need to merge!
        W, quant_state, A, B, s = get_lora_parameters(layer)
        if quant_state is not None:
            dtype = quant_state.dtype if type(quant_state) is not list else quant_state[2]
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype
        W = W.to(torch.float32).t()
        # W = W.t()

        if A is not None:
            # sAB = (A.t().to(torch.float32) @ (s * B.t().to(torch.float32)))
            # W += sAB
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha = s)
            # W.addmm_(A.t().to(W.dtype), B.t().to(W.dtype), alpha = s)
            # if not torch.isfinite(W).all():
            maximum_element = torch.max(W.min().abs(), W.max())
            if not torch.isfinite(maximum_element).item():
                raise ValueError(f"Unsloth: Merge failed.\n{name} has some elements = infinity.")
        pass
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
        # HIGHEST_PROTOCOL seems to not work with Pytorch!
        # pickle_module   = pickle,
        # pickle_protocol = pickle.HIGHEST_PROTOCOL,
    )
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
    is_main_process      : bool = True,
    state_dict           : Optional[dict] = None,
    save_function        : Callable = torch.save,
    max_shard_size       : Union[int, str] = "5GB",
    safe_serialization   : bool = True,
    variant              : Optional[str] = None,
    save_peft_format     : bool = True,

    # Push to hub
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = "Trained with Unsloth",
    private              : Optional[bool] = None,
    create_pr            : bool = False,
    revision             : str = None,
    commit_description   : str = "Upload model trained with Unsloth 2x faster",
    tags                 : List[str] = None,

    # Our functions
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.9,
):
    if token is None and "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]

    if token is None and "HUGGINGFACE_TOKEN" in os.environ:
        token = os.environ["HUGGINGFACE_TOKEN"]

    if commit_message is None: commit_message = ""
    if "Unsloth" not in commit_message:
        commit_message += " (Trained with Unsloth)"
    commit_message = commit_message.lstrip()

    if commit_description is None:
        commit_description = "Upload model trained with Unsloth 2x faster"
    elif "Unsloth 2x faster" not in commit_description:
        commit_description += " (Trained with Unsloth 2x faster)"
    pass

    if save_method == "merged_4bit":
        raise RuntimeError(
            "Unsloth: Merging into 4bit will cause your model to lose accuracy if you plan\n"\
            "to merge to GGUF or others later on. I suggest you to do this as a final step\n"\
            "if you're planning to do multiple saves.\n"\
            "If you are certain, change `save_method` to `merged_4bit_forced`."
        )
    elif save_method == "merged_4bit_forced":
        save_method = "merged_4bit"
    pass

    save_pretrained_settings = dict(locals())
    for deletion in ("model", "tokenizer", "save_method", "temporary_location", "maximum_memory_usage"):
        del save_pretrained_settings[deletion]
    pass

    # First check for a token!
    if push_to_hub:
        from huggingface_hub import whoami
        try: 
            username = whoami(token = token)["name"]
        except:
            raise RuntimeError(
                "Unsloth: Please supply a token!\n"\
                "Go to https://huggingface.co/settings/tokens"
            )
        pass
    pass

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

        # Counteract no LoRA adapters!
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        pass
        print("Done.")
    pass

    if tags is not None:
        assert(isinstance(tags, (list, tuple)))
        tags = list(tags) + ["unsloth",]
    else:
        tags = ["unsloth",]
    pass
    save_pretrained_settings["tags"] = tags

    if ((save_method == "lora") or (save_method == "merged_4bit")) and push_to_hub:
        if token is None:
            raise RuntimeError(
                "Unsloth: Pushing to HF requires a token. Pass `token = 'hf_....'`\n"\
                "Go to https://huggingface.co/settings/tokens."
            )
        pass

        if save_method == "lora":
            print("Unsloth: Saving LoRA adapters. Please wait...")
        elif save_method == "merged_4bit":
            print("Unsloth: Saving 4bit Bitsandbytes model. Please wait...")
        pass

        # Update model tag
        _ = upload_to_huggingface(
            model, save_directory, token,
            "finetuned", "trl", file_location = None,
            old_username = None, private = private,
        )

        getattr(model, "original_push_to_hub", tokenizer.push_to_hub)\
        (
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
            # Set padding side to left for inference
            old_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

            getattr(tokenizer, "original_push_to_hub", tokenizer.push_to_hub)\
            (
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

            # Revert back padding side
            tokenizer.padding_side = old_padding_side
        pass

        if hasattr(model, "config"):
            print(f"Saved {save_method} model to https://huggingface.co/" + save_directory)
        pass
        return save_directory, None
    pass

    # Tokenizer has different saving arguments
    tokenizer_save_settings = \
    {
        "save_directory"  : save_pretrained_settings["save_directory"],
        "legacy_format"   : None,
        "filename_prefix" : None,
        "push_to_hub"     : save_pretrained_settings["push_to_hub"],
        "private"         : save_pretrained_settings["private"],
        "token"           : save_pretrained_settings["token"],
    }

    # Check if PEFT Model or not - if yes, 3 levels. If not 2 levels.
    from peft import PeftModelForCausalLM
    if isinstance(model, PeftModelForCausalLM):
        internal_model = model.model
    else:
        internal_model = model
    pass
        
    # Cannot be converted properly!
    if (save_method == "merged_4bit") or (save_method == "lora") or (
        not hasattr(model, "model") or \
        not hasattr(internal_model.model, "layers")
    ):
        # Do general saving
        # Edit save_pretrained_settings
        # [TODO] _create_repo has errors due to **kwargs getting accepted
        # commit_description does not seem to work?
        what_to_delete = ("use_temp_dir", "commit_message", "create_pr", "revision", "commit_description", "tags",) \
            if save_pretrained_settings["push_to_hub"] is False else \
            ("use_temp_dir", "create_pr", "revision", "tags", "commit_description",)
        for deletion in what_to_delete:
            del save_pretrained_settings[deletion]
        pass
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["unsloth",])

        # Update model tag
        if push_to_hub:
             _ = upload_to_huggingface(
                model, save_pretrained_settings["save_directory"], token,
                "finetuned", "trl", file_location = None,
                old_username = None, private = private,
            )
        pass

        if tokenizer is not None:
            print("Unsloth: Saving tokenizer...", end = "")

            # Set padding side to left for inference
            old_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

            tokenizer.save_pretrained(**tokenizer_save_settings)

            # Revert back padding side
            tokenizer.padding_side = old_padding_side

            print(" Done.")
        else:
            print()

        print("Unsloth: Saving model...", end = "")
        if save_method != "lora": print(" This might take 10 minutes for Llama-7b...", end = "")

        model.save_pretrained(**save_pretrained_settings)

        if push_to_hub and hasattr(model, "config"):
            print("Saved to https://huggingface.co/" + save_pretrained_settings["save_directory"])
        pass

        print(" Done.")
        return save_directory, None
    pass

    # If push_to_hub, we must remove the .../ part of a repo
    username = None
    if push_to_hub and "/" in save_directory:

        # +1 solves absolute path issues
        username = save_directory[:save_directory.find("/")]
        new_save_directory = save_directory[save_directory.find("/")+1:]

        logger.warning_once(
            f"Unsloth: You are pushing to hub, but you passed your HF username = {username}.\n"\
            f"We shall truncate {save_directory} to {new_save_directory}"
        )

        save_pretrained_settings["save_directory"] = new_save_directory
        tokenizer_save_settings ["save_directory"] = new_save_directory
        save_directory = new_save_directory
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
    if n_cpus is None: n_cpus = psutil.cpu_count()
    if n_cpus is None: n_cpus = 1

    if safe_serialization is None:
        safe_serialization = True
        save_pretrained_settings["safe_serialization"] = safe_serialization

    elif safe_serialization and (n_cpus <= 2):
        logger.warning_once(
            f"Unsloth: You have {n_cpus} CPUs. Using `safe_serialization` is 10x slower.\n"\
            f"We shall switch to Pytorch saving, which will take 3 minutes and not 30 minutes.\n"\
            f"To force `safe_serialization`, set it to `None` instead.",
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

    # Check if Kaggle or Colab, since only 20GB of Disk space allowed.
    if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
        # We free up 4GB of space
        logger.warning_once(
            "Unsloth: Kaggle/Colab has limited disk space. We need to delete the downloaded\n"\
            "model which will save 4-16GB of disk space, allowing you to save on Kaggle/Colab."
        )
        _free_cached_model(internal_model)
    pass

    # HF also uses a OrderedDict
    from collections import OrderedDict
    state_dict = OrderedDict()

    torch_dtype = internal_model.config.torch_dtype
    if type(torch_dtype) is str:
        if   torch_dtype ==  "float16": torch_dtype = torch.float16
        elif torch_dtype == "bfloat16": torch_dtype = torch.bfloat16
    pass

    # Check modules to save float32 dtype
    state_dict["model.embed_tokens.weight"] = internal_model.model.embed_tokens.weight.data.to(torch_dtype)

    max_vram = int(torch.cuda.get_device_properties(0).total_memory * maximum_memory_usage)

    from tqdm import tqdm as ProgressBar
    for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W = _merge_lora(proj, name)

            if (torch.cuda.memory_allocated() + W.nbytes) < max_vram:
                # Save to GPU memory
                state_dict[name] = W
            # [TODO] Saving to RAM seems to leak memory???
            # elif (max_ram - W.nbytes) > 0:
            #     # Save to CPU memory
            #     logger.warning_once(f"We will save to RAM and not VRAM now.")
            #     state_dict[name] = W.to("cpu", non_blocking = True, copy = True)
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

    state_dict["model.norm.weight"] = internal_model.model.norm.weight.data
    # Check for modules_to_save float32 dtype

    # Check for tied weights
    if internal_model.model.embed_tokens.weight.data_ptr() != internal_model.lm_head.weight.data_ptr():
        state_dict["lm_head.weight"] = internal_model.lm_head.weight.data.to(torch_dtype)
    pass

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
    
    # commit_description does not seem to work?
    what_to_delete = ("use_temp_dir", "commit_message", "create_pr", "revision", "commit_description", "tags",) \
        if not push_to_hub else \
        ("use_temp_dir", "create_pr", "revision", "tags", "commit_description",)
    for deletion in what_to_delete:
        del save_pretrained_settings[deletion]
    pass
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth",])

    # Update model tag
    if push_to_hub:
        _ = upload_to_huggingface(
            model, save_pretrained_settings["save_directory"], token,
            "finetuned", "trl", file_location = None,
            old_username = username, private = private,
        )
    pass

    # First check if we're pushing to an organization!
    save_directory = save_pretrained_settings["save_directory"]

    if save_pretrained_settings["push_to_hub"]:
        new_save_directory, new_username = _determine_username(save_directory, username, token)

        if token is not None:
            from huggingface_hub import whoami
            actual_username = whoami(token = token)["name"]
        else:
            actual_username = username
    pass

    # Check if pushing to an organization
    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        # We upload everything at the end!
        tokenizer_save_settings["push_to_hub"] = False
        tokenizer_save_settings["save_directory"] = new_save_directory
    pass

    # Save tokenizer
    if tokenizer is not None:
        print("Unsloth: Saving tokenizer...", end = "")

        # Set padding side to left for inference
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        tokenizer.save_pretrained(**tokenizer_save_settings)

        # Revert back padding side
        tokenizer.padding_side = old_padding_side
            
        print(" Done.")
    else:
        print()
    pass

    print("Unsloth: Saving model... This might take 5 minutes for Llama-7b...")

    # Since merged, edit quantization_config
    old_config = model.config
    new_config = model.config.to_dict()
    if "quantization_config" in new_config:
        del new_config["quantization_config"]
    original_model = model
    new_config = type(model.config).from_dict(new_config)
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = new_config
    model.config = new_config

    # Save!

    # Check if pushing to an organization
    if save_pretrained_settings["push_to_hub"] and (username != actual_username):
        print(f"Unsloth: Saving to organization with address {new_save_directory}")
        # Pushing to organization!
        # Sadly .save_pretrained doesn't work :(
        # We first save it via .save_pretrained, then upload manually!
        save_pretrained_settings["save_directory"] = new_save_directory
        save_pretrained_settings["push_to_hub"] = False
        internal_model.save_pretrained(**save_pretrained_settings)

        # Now manually go through each file and upload them manually!
        filenames = os.listdir(new_save_directory)

        from huggingface_hub import HfApi
        hf_api = HfApi(token = save_pretrained_settings["token"])

        print("Unsloth: Uploading all files... Please wait...")
        hf_api.upload_folder(
            folder_path = new_save_directory,
            path_in_repo = ".",
            repo_id = new_save_directory,
            repo_type = "model",
            commit_message  = "(Trained with Unsloth)",
            ignore_patterns = "*.md",
        )
    else:
        internal_model.save_pretrained(**save_pretrained_settings)
    pass

    # Revert config back
    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = old_config
    model.config = old_config
    print("Done.")

    if push_to_hub and hasattr(model, "config"):
        print(f"Saved merged model to https://huggingface.co/{username}/{save_directory.lstrip('/')}")
    pass

    save_pretrained_settings["state_dict"] = None

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
    return save_directory, username
pass


def install_llama_cpp_clone_non_blocking():
    full_command = ["git", "clone", "--recursive", "https://github.com/ggerganov/llama.cpp"]
    run_installer = subprocess.Popen(full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_llama_cpp_make_non_blocking():
    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # env = { **os.environ, "LLAMA_CUDA": "1", }
    n_jobs = max(int(psutil.cpu_count()*1.5), 1)
    # Force make clean
    os.system("make clean -C llama.cpp")
    full_command = ["make", "all", "-j"+str(n_jobs), "-C", "llama.cpp"]

    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # run_installer = subprocess.Popen(full_command, env = env, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    run_installer = subprocess.Popen(full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_python_non_blocking(packages = []):
    full_command = ["pip", "install"] + packages
    run_installer = subprocess.Popen(full_command, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    return run_installer
pass


def install_llama_cpp_old(version = -10):
    # Download the 10th latest release since the latest might be broken!
    # FALLBACK mechanism
    releases = subprocess.check_output(["git", "ls-remote", "--tags", "https://github.com/ggerganov/llama.cpp.git"])
    releases = releases.decode("utf-8").replace("\t", " ").split("\n")
    for i, x in enumerate(releases):
        if "refs/tags/b" not in x: break
    releases = releases[:i]
    latest = releases[-1]
    version = releases[version].split(" ")[0]

    # Check if the llama.cpp exists
    if os.path.exists("llama.cpp"):
        print(
            "**[WARNING]** You have a llama.cpp old directory which is broken.\n"\
            "Unsloth will DELETE the broken directory and install a new one.\n"\
            "Press CTRL + C / cancel this if this is wrong. We shall wait 10 seconds.\n"
        )
        import time
        for i in range(10):
            print(f"**[WARNING]** Deleting llama.cpp directory... {10-i} seconds left.")
            time.sleep(1)
        import shutil
        shutil.rmtree("llama.cpp")
    pass

    # Clone a specific commit
    # Also don't use the GPU!
    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        f"cd llama.cpp && git reset --hard {version} && git clean -df",
        "make clean -C llama.cpp",
        f"make all -j{psutil.cpu_count()*2} -C llama.cpp",
    ]
    for command in commands:
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stdout:
                print(line.decode("utf-8", errors = "replace"), flush = True, end = "")
        pass
    pass
    # Check if successful
    if not os.path.exists("llama.cpp/quantize"):
        raise RuntimeError(
            "Unsloth: llama.cpp GGUF seems to be too buggy to install.\n"\
            "File a report to llama.cpp's main repo since this is not an Unsloth issue."
        )
    pass
pass


def install_llama_cpp_blocking(use_cuda = True):
    # https://github.com/ggerganov/llama.cpp/issues/7062
    # Weirdly GPU conversion for GGUF breaks??
    # use_cuda = "LLAMA_CUDA=1" if use_cuda else ""

    commands = [
        "git clone --recursive https://github.com/ggerganov/llama.cpp",
        "make clean -C llama.cpp",
        # https://github.com/ggerganov/llama.cpp/issues/7062
        # Weirdly GPU conversion for GGUF breaks??
        # f"{use_cuda} make all -j{psutil.cpu_count()*2} -C llama.cpp",
        f"make all -j{psutil.cpu_count()*2} -C llama.cpp",
        "pip install gguf protobuf",
    ]
    if os.path.exists("llama.cpp"): return

    for command in commands:
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stdout:
                print(line.decode("utf-8", errors = "replace"), flush = True, end = "")
        pass
    pass
pass


def _fix_gemma_gguf():
    # Fixes Gemma saving to GGUF to float32 instead of float16!
    with open("llama.cpp/convert-hf-to-gguf.py", "rb") as file:
        text = file.read()
    pass

    gemma_start = text.find(b"class GemmaModel(Model):")
    if gemma_start == -1: return

    gemma_end   = text.find(b"self.gguf_writer.add_tensor(new_name, data)", gemma_start)
    if gemma_end == -1: return

    gemma_text = text[gemma_start : gemma_end]
    bad_text = \
b"""         data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)"""
    good_text = \
b"""         # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)"""
    find_bad = gemma_text.find(bad_text)
    if find_bad == -1: return

    gemma_text = gemma_text[:find_bad] + good_text + gemma_text[find_bad + len(bad_text):]
    text = text[:gemma_start] + gemma_text + text[gemma_end:]

    with open("llama.cpp/convert-hf-to-gguf.py", "w+b") as file:
        file.write(text)
    pass
pass


def save_to_gguf(
    model_type           : str,
    model_directory      : str = "unsloth_finetuned_model",
    quantization_method  : str = "fast_quantized",
    first_conversion     : str = "f16",
    _run_installer = None, # Non blocking install of llama.cpp
):
    logger.warning(
        "NOTICE: llama.cpp GGUF conversion is currently unstable, since llama.cpp is\n"\
        "undergoing some major bug fixes as at 5th of May 2024. This is not an Unsloth issue.\n"\
        "Please be patient - GGUF saving should still work, but might not work as well."
    )

    if quantization_method.startswith("iq2"):
        raise RuntimeError("Unsloth: Currently iq2 type quantizations aren't supported yet - sorry!")

    # Careful convert.py is only for Llama / Mistral based archs
    use_fast_convert = False
    if   model_type == "llama":   use_fast_convert = True
    elif model_type == "mistral": use_fast_convert = True
    pass
    logger.warning_once(f"Unsloth: Converting {model_type} model. Can use fast conversion = {use_fast_convert}.")

    if   quantization_method == "not_quantized":  quantization_method = "f16"
    elif quantization_method == "fast_quantized": quantization_method = "q8_0"
    elif quantization_method == "quantized":      quantization_method = "q4_k_m"
    elif quantization_method is None:             quantization_method = "q8_0"
    pass

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

    # Check first_conversion format
    if   first_conversion == "f16" : pass
    elif first_conversion == "f32" : pass
    elif first_conversion == "q8_0": pass
    else:
        raise RuntimeError(
            f"Unsloth: `first_conversion` can only be one of ['f16', 'f32', 'q8_0'] and not `{first_conversion}`."
        )
    pass

    print("Unsloth: [0] Installing llama.cpp. This will take 3 minutes...")
    if _run_installer is not None:
        error = _run_installer.wait()
    else:
        error = 0
        install_llama_cpp_blocking()
    pass
    # Check if successful. If not install 10th latest release
    if error != 0 or not os.path.exists("llama.cpp/quantize"):
        print(f"Unsloth: llama.cpp error code = {error}.")
        install_llama_cpp_old(-10)
    pass

    if   quantization_method == "f32":  first_conversion = "f32"
    elif quantization_method == "f16":  first_conversion = "f16"
    elif quantization_method == "q8_0": first_conversion = "q8_0"
    else:
        # Quantized models must have f16 as the default argument
        if   first_conversion == "f32" : pass
        elif first_conversion == "f16" : pass
        elif first_conversion == "q8_0":
            logger.warning_once(
                "Unsloth: Using q8_0 for the `first_conversion` will lose a bit of accuracy, "\
                "but saves disk space!"
            )
            # first_conversion = "f16"
        pass
    pass

    # Non llama/mistral needs can only use f32 or f16
    if not use_fast_convert and (first_conversion != "f16" or first_conversion != "f32"):
        logger.warning_once("Unsloth: We must use f16 for non Llama and Mistral models.")
        first_conversion = "f16"
    pass

    n_cpus = psutil.cpu_count()
    if n_cpus is None: n_cpus = 1
    n_cpus *= 2
    # Concurrency from https://rentry.org/llama-cpp-conversions#merging-loras-into-a-model
    
    final_location = f"./{model_directory}-unsloth.{first_conversion.upper()}.gguf"

    print(f"Unsloth: [1] Converting model at {model_directory} into {first_conversion} GGUF format.\n"\
          f"The output location will be {final_location}\n"\
          "This will take 3 minutes...")

    # We first check if tokenizer.model exists in the model_directory
    if os.path.exists(f"{model_directory}/tokenizer.model"):
        vocab_type = "spm,hfft,bpe"
    else:
        vocab_type = "bpe"
    pass

    if use_fast_convert:
        command = f"python llama.cpp/convert.py {model_directory} "\
            f"--outfile {final_location} --vocab-type {vocab_type} "\
            f"--outtype {first_conversion} --concurrency {n_cpus}"
    else:
        # Need to fix convert-hf-to-gguf.py for some models!
        _fix_gemma_gguf()

        command = f"python llama.cpp/convert-hf-to-gguf.py {model_directory} "\
            f"--outfile {final_location} "\
            f"--outtype {first_conversion}"
    pass

    with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE, bufsize = 1) as sp:
        for line in sp.stdout:
            print(line.decode("utf-8", errors = "replace"), flush = True, end = "")
        if sp.returncode is not None and sp.returncode != 0:
            raise subprocess.CalledProcessError(sp.returncode, sp.args)
    pass

    # Check if quantization succeeded!
    if not os.path.isfile(final_location):
        if IS_KAGGLE_ENVIRONMENT:
            raise RuntimeError(
                f"Unsloth: Quantization failed for {final_location}\n"\
                "You are in a Kaggle environment, which might be the reason this is failing.\n"\
                "Kaggle only provides 20GB of disk space. Merging to 16bit for 7b models use 16GB of space.\n"\
                "This means using `model.{save_pretrained/push_to_hub}_merged` works, but\n"\
                "`model.{save_pretrained/push_to_hub}_gguf will use too much disk space.\n"\
                "I suggest you to save the 16bit model first, then use manual llama.cpp conversion."
            )
        else:
            raise RuntimeError(
                f"Unsloth: Quantization failed for {final_location}\n"\
                "You might have to compile llama.cpp yourself, then run this again.\n"\
                "You do not need to close this Python program. Run the following commands in a new terminal:\n"\
                "You must run this in the same folder as you're saving your model.\n"\
                "git clone --recursive https://github.com/ggerganov/llama.cpp\n"\
                "cd llama.cpp && make clean && make all -j\n"\
                "Once that's done, redo the quantization."
            )
        pass
    pass
    print(f"Unsloth: Conversion completed! Output location: {final_location}")

    if quantization_method != first_conversion:
        old_location = final_location
        print(f"Unsloth: [2] Converting GGUF 16bit into {quantization_method}. This will take 20 minutes...")
        final_location = f"./{model_directory}-unsloth.{quantization_method.upper()}.gguf"

        command = f"./llama.cpp/quantize {old_location} "\
            f"{final_location} {quantization_method} {n_cpus}"
        
        # quantize uses stderr
        with subprocess.Popen(command, shell = True, stderr = subprocess.PIPE, bufsize = 1) as sp:
            for line in sp.stderr:
                print(line.decode("utf-8", errors = "replace"), flush = True, end = "")
            if sp.returncode is not None and sp.returncode != 0:
                raise subprocess.CalledProcessError(sp.returncode, sp.args)
        pass

        # Check if quantization succeeded!
        if not os.path.isfile(final_location):
            if IS_KAGGLE_ENVIRONMENT:
                raise RuntimeError(
                    f"Unsloth: Quantization failed for {final_location}\n"\
                    "You are in a Kaggle environment, which might be the reason this is failing.\n"\
                    "Kaggle only provides 20GB of disk space. Merging to 16bit for 7b models use 16GB of space.\n"\
                    "This means using `model.{save_pretrained/push_to_hub}_merged` works, but\n"\
                    "`model.{save_pretrained/push_to_hub}_gguf will use too much disk space.\n"\
                    "I suggest you to save the 16bit model first, then use manual llama.cpp conversion."
                )
            else:
                raise RuntimeError(
                    "Unsloth: Quantization failed! You might have to compile llama.cpp yourself, then run this again.\n"\
                    "You do not need to close this Python program. Run the following commands in a new terminal:\n"\
                    "You must run this in the same folder as you're saving your model.\n"\
                    "git clone --recursive https://github.com/ggerganov/llama.cpp\n"\
                    "cd llama.cpp && make clean && make all -j\n"\
                    "Once that's done, redo the quantization."
                )
            pass
        pass

        print(f"Unsloth: Conversion completed! Output location: {final_location}")
    pass

    return final_location
pass


def unsloth_save_pretrained_merged(
    self,
    save_directory       : Union[str, os.PathLike],
    tokenizer            = None,
    save_method          : str = "merged_16bit", # ["lora", "merged_16bit", "merged_4bit"]
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

        Choose for `save_method` to be either:
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
    save_method          : str = "merged_16bit", # ["lora", "merged_16bit", "merged_4bit"]
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = "Trained with Unsloth",
    private              : Optional[bool] = None,
    token                : Union[bool, str, None] = None,
    max_shard_size       : Union[int, str, None] = "5GB",
    create_pr            : bool = False,
    safe_serialization   : bool = True,
    revision             : str = None,
    commit_description   : str = "Upload model trained with Unsloth 2x faster",
    tags                 : Optional[List[str]] = None,
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.85,
):
    """
        Same as .push_to_hub(...) except 4bit weights are auto
        converted to float16 with as few overhead as possible.

        Choose for `save_method` to be either:
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


MODEL_CARD = \
"""---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""


def _determine_username(save_directory, old_username, token):
    username = ""
    save_directory = save_directory.lstrip("./")
    if "/" not in save_directory:
        from huggingface_hub import whoami
        try: 
            username = whoami(token = token)["name"]
            if type(old_username) is str and username != old_username:
                username = old_username
            pass
            save_directory = f"{username}/{save_directory}"
        except:
            raise RuntimeError(f"Unsloth: {save_directory} is not a Huggingface directory.")
    else:
        username = save_directory.split("/")[0]
    pass
    return save_directory, username
pass


def upload_to_huggingface(
    model,
    save_directory,
    token,
    method,
    extra = "",
    file_location = None,
    old_username = None,
    private = None,
):
    save_directory, username = _determine_username(save_directory, old_username, token)

    from huggingface_hub import create_repo
    try:
        create_repo(
            repo_id   = save_directory,
            token     = token,
            repo_type = "model",
            exist_ok  = False,
            private   = private,
        ) 

        # Create model card
        from huggingface_hub import ModelCard
        content = MODEL_CARD.format(
            username   = username,
            base_model = model.config._name_or_path,
            model_type = model.config.model_type,
            method     = "",
            extra      = extra,
        )
        card = ModelCard(content)
        card.push_to_hub(save_directory, token = token)
    except:
        pass

    if file_location is not None:
        # Now upload file
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
            commit_message  = "(Trained with Unsloth)",
        )

        # We also upload a config.json file
        import json
        with open("_temporary_unsloth_config.json", "w") as file:
            json.dump({"model_type" : model.config.model_type}, file, indent = 4)
        pass
        hf_api.upload_file(
            path_or_fileobj = "_temporary_unsloth_config.json",
            path_in_repo    = "config.json",
            repo_id         = save_directory,
            repo_type       = "model",
            commit_message  = "(Trained with Unsloth)",
        )
        os.remove("_temporary_unsloth_config.json")
    pass
    return username
pass


def unsloth_save_pretrained_gguf(
    self,
    save_directory       : Union[str, os.PathLike],
    tokenizer            = None,
    quantization_method  : str = "fast_quantized",
    first_conversion     : str = "f16",
    push_to_hub          : bool = False,
    token                : Optional[Union[str, bool]] = None,
    private              : Optional[bool] = None,
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

        Choose for `quantization_method` to be:
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
        "q4_k"    : "alias for q4_k_m",
        "q5_k"    : "alias for q5_k_m",
        "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
        "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
        "q5_k_s"  : "Uses Q5_K for all tensors",
        "q6_k"    : "Uses Q8_K for all tensors",
        "iq2_xxs" : "2.06 bpw quantization",
        "iq2_xs"  : "2.31 bpw quantization",
        "iq3_xxs" : "3.06 bpw quantization",
        "q3_k_xs" : "3-bit extra small quantization",
    """
    if tokenizer is None:
        raise ValueError("Unsloth: Saving to GGUF must have a tokenizer.")

    arguments = dict(locals())
    arguments["model"]        = self
    arguments["tokenizer"]    = tokenizer
    arguments["push_to_hub"]  = False # We save ourselves
    arguments["save_method"] = "merged_16bit" # Must be 16bit
    del arguments["self"]
    del arguments["quantization_method"]
    del arguments["first_conversion"]

    # Non blocking install GGUF first
    if not os.path.exists("llama.cpp"):

        if IS_KAGGLE_ENVIRONMENT:
            # Kaggle is weird - no blocking installs, and no CUDA?
            python_install = install_python_non_blocking(["gguf", "protobuf"])
            python_install.wait()
            install_llama_cpp_blocking(use_cuda = False)
            new_save_directory, old_username = unsloth_save_model(**arguments)
            makefile = None
        else:
            git_clone = install_llama_cpp_clone_non_blocking()
            python_install = install_python_non_blocking(["gguf", "protobuf"])
            git_clone.wait()
            makefile  = install_llama_cpp_make_non_blocking()
            new_save_directory, old_username = unsloth_save_model(**arguments)
            python_install.wait()
        pass
    else:
        try:
            new_save_directory, old_username = unsloth_save_model(**arguments)
            makefile = None
        except:
            # Retry by recloning llama.cpp
            if IS_KAGGLE_ENVIRONMENT:
                # Kaggle is weird - no blocking installs, and no CUDA?
                python_install = install_python_non_blocking(["gguf", "protobuf"])
                python_install.wait()
                install_llama_cpp_blocking(use_cuda = False)
                new_save_directory, old_username = unsloth_save_model(**arguments)
                makefile = None
            else:
                git_clone = install_llama_cpp_clone_non_blocking()
                python_install = install_python_non_blocking(["gguf", "protobuf"])
                git_clone.wait()
                makefile  = install_llama_cpp_make_non_blocking()
                new_save_directory, old_username = unsloth_save_model(**arguments)
                python_install.wait()
            pass
        pass
    pass

    for _ in range(3):
        gc.collect()

    model_type = self.config.model_type
    file_location = save_to_gguf(model_type, new_save_directory, quantization_method, first_conversion, makefile)

    if push_to_hub:
        print("Unsloth: Uploading GGUF to Huggingface Hub...")
        username = upload_to_huggingface(
            self, save_directory, token,
            "GGUF converted", "gguf", file_location, old_username, private,
        )
        link = f"{username}/{new_save_directory.lstrip('/.')}" \
            if username not in new_save_directory else \
            new_save_directory.lstrip('/.')
        print(f"Saved GGUF to https://huggingface.co/{link}")
    pass
pass


def unsloth_push_to_hub_gguf(
    self,
    repo_id              : str,
    tokenizer            = None,
    quantization_method  : str = "fast_quantized",
    first_conversion     : str = "f16",
    use_temp_dir         : Optional[bool] = None,
    commit_message       : Optional[str] = "Trained with Unsloth",
    private              : Optional[bool] = None,
    token                : Union[bool, str, None] = None,
    max_shard_size       : Union[int, str, None] = "5GB",
    create_pr            : bool = False,
    safe_serialization   : bool = True,
    revision             : str = None,
    commit_description   : str = "Upload model trained with Unsloth 2x faster",
    tags                 : Optional[List[str]] = None,
    temporary_location   : str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage : float = 0.85,
):
    """
        Same as .push_to_hub(...) except 4bit weights are auto
        converted to float16 then converted to GGUF / llama.cpp format.

        Choose for `quantization_method` to be:
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
    arguments["save_method"]   = "merged_16bit" # Must be 16bit
    del arguments["self"]
    del arguments["repo_id"]
    del arguments["quantization_method"]
    del arguments["first_conversion"]

    # Non blocking install GGUF first
    if not os.path.exists("llama.cpp"):

        if IS_KAGGLE_ENVIRONMENT:
            # Kaggle is weird - no blocking installs, and no CUDA?
            python_install = install_python_non_blocking(["gguf", "protobuf"])
            python_install.wait()
            install_llama_cpp_blocking(use_cuda = False)
            new_save_directory, old_username = unsloth_save_model(**arguments)
            makefile = None
        else:
            git_clone = install_llama_cpp_clone_non_blocking()
            python_install = install_python_non_blocking(["gguf", "protobuf"])
            git_clone.wait()
            makefile  = install_llama_cpp_make_non_blocking()
            new_save_directory, old_username = unsloth_save_model(**arguments)
            python_install.wait()
        pass
    else:
        try:
            new_save_directory, old_username = unsloth_save_model(**arguments)
            makefile = None
        except:
            # Retry by recloning llama.cpp
            if IS_KAGGLE_ENVIRONMENT:
                # Kaggle is weird - no blocking installs, and no CUDA?
                python_install = install_python_non_blocking(["gguf", "protobuf"])
                python_install.wait()
                install_llama_cpp_blocking(use_cuda = False)
                new_save_directory, old_username = unsloth_save_model(**arguments)
                makefile = None
            else:
                git_clone = install_llama_cpp_clone_non_blocking()
                python_install = install_python_non_blocking(["gguf", "protobuf"])
                git_clone.wait()
                makefile  = install_llama_cpp_make_non_blocking()
                new_save_directory, old_username = unsloth_save_model(**arguments)
                python_install.wait()
            pass
        pass
    pass

    for _ in range(3):
        gc.collect()

    model_type = self.config.model_type
    file_location = save_to_gguf(model_type, new_save_directory, quantization_method, first_conversion, makefile)

    print("Unsloth: Uploading GGUF to Huggingface Hub...")
    username = upload_to_huggingface(
        self, repo_id, token,
        "GGUF converted", "gguf", file_location, old_username, private,
    )
    link = f"{username}/{new_save_directory.lstrip('/.')}" \
        if username not in new_save_directory else \
        new_save_directory.lstrip('/.')
    print(f"Saved GGUF to https://huggingface.co/{link}")
pass


def patch_saving_functions(model):
    import inspect
    import re
    import types
    from typing import Callable, Optional, Union, List

    # And now re add our saving methods!
    if model.push_to_hub.__name__ == "unsloth_push_to_hub":
        original_push_to_hub = model.original_push_to_hub
    else:
        original_push_to_hub = model.push_to_hub
    pass

    signature = str(inspect.signature(original_push_to_hub)).replace("NoneType", "None")
    signature = signature[1:]
    signature = re.sub("<function save at .+?>", "torch.save", signature)
    docs = original_push_to_hub.__doc__.encode("utf-8").decode("utf-8")

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

    if "commit_message" in arguments:
        commit_message = arguments["commit_message"]
        if commit_message is not None:
            if not commit_message.endswith(" "): commit_message += " "
            if "Unsloth" not in commit_message:
                commit_message += "(Trained with Unsloth)"
        else:
            commit_message = "Upload model trained with Unsloth"
        arguments["commit_message"] = commit_message

    if "commit_description" in arguments:
        commit_description = arguments["commit_description"]
        if commit_description is not None:
            if not commit_description.endswith(" "): commit_description += " "
            if "Unsloth" not in commit_description:
                commit_description += "(Trained with Unsloth 2x faster)"
        else:
            commit_description = "Upload model trained with Unsloth 2x faster"
        arguments["commit_description"] = commit_description

    # Update model tag
    if hasattr(self, "config"):
        _ = upload_to_huggingface(
            self, arguments["repo_id"], arguments["token"],
            "finetuned", "trl", file_location = None,
            old_username = None, private = arguments["private"],
        )
    pass

    try:
        self.original_push_to_hub(**arguments)
    except:
        del arguments["tags"]
        self.original_push_to_hub(**arguments)
    pass

    if hasattr(self, "config"):
        print("Saved model to https://huggingface.co/" + arguments["repo_id"])
    pass
    '''
    exec(push_to_hub_text, globals())

    original_model = model
    while True:

        if original_model.push_to_hub.__name__ != "unsloth_push_to_hub":
            original_model.original_push_to_hub = original_model.push_to_hub
            original_model.push_to_hub = types.MethodType(unsloth_push_to_hub, original_model)
            if hasattr(original_model, "add_model_tags"):
                original_model.add_model_tags(["unsloth",])
            pass
        pass

        if hasattr(original_model, "model"): original_model = original_model.model
        else: break
    pass

    # Add saving methods to top level model
    if hasattr(model, "config"):
        # Counteract tokenizers
        model.push_to_hub_merged     = types.MethodType(unsloth_push_to_hub_merged,     model)
        model.save_pretrained_merged = types.MethodType(unsloth_save_pretrained_merged, model)
        model.push_to_hub_gguf       = types.MethodType(unsloth_push_to_hub_gguf,       model)
        model.save_pretrained_gguf   = types.MethodType(unsloth_save_pretrained_gguf,   model)
    pass
    return model
pass
