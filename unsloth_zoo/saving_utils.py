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
    "create_huggingface_repo",
    "merge_and_dequantize_lora",
    "merge_and_overwrite_lora",
]

from .peft_utils import get_lora_layer_modules
from .utils import _get_dtype

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

# Uploaded finetuned {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""

import torch
import bitsandbytes as bnb
try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        # For older versions of huggingface_hub
        from huggingface_hub.utils._token import get_token
    pass
pass
from transformers.modeling_utils import PushToHubMixin
import json
import os
from pathlib import Path
import tempfile
from peft import PeftModelForCausalLM

def find_skipped_quantized_modules(model):
    skipped_modules = []
    quantized_modules = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
                quantized_modules.append(name)
            else:
                skipped_modules.append(name)
        elif isinstance(module, torch.nn.Linear):
            skipped_modules.append(name)
    return skipped_modules, quantized_modules

def create_huggingface_repo(
    model,
    repo_id,
    private = False,
    token = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(repo_id) is str)
    if repo_id.count("/") != 1:
        raise TypeError(f"Unsloth: You are pushing to Hugging Face, but {repo_id} is not a valid repo.")

    from huggingface_hub import ModelCard
    if token is None: token = get_token()
    repo_id = PushToHubMixin._create_repo(
        PushToHubMixin,
        repo_id = repo_id,
        private = private,
        token = token,
    )
    username = repo_id.split("/")[0]
    # Create model card
    content = MODEL_CARD.format(
        username   = username,
        base_model = model.config._name_or_path,
        model_type = model.config.model_type,
        method     = "",
        extra      = "unsloth",
    )
    card = ModelCard(content)
    card.push_to_hub(repo_id, token = token, commit_message = "Unsloth Model Card")

    from huggingface_hub import HfApi
    hf_api = HfApi(token = token)
    return username, repo_id, hf_api
pass


from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    HfFileSystem,
)
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from tqdm import tqdm as ProgressBar
import os, shutil, re, functools


def _merge_lora(W, lora_stats, name):
    if lora_stats.lora_A is None or lora_stats.lora_B is None: return W
    W = W.to("cuda", dtype = torch.float32, non_blocking = True)
    W = W.addmm_(
        lora_stats.lora_B.to("cuda", dtype = torch.float32, non_blocking = True),
        lora_stats.lora_A.to("cuda", dtype = torch.float32, non_blocking = True),
        alpha = lora_stats.alpha,
    )
    if not torch.isfinite(torch.amax(W)).item():
        raise ValueError('Unsloth: Merge failed as there are infinite elements in ' + name)
    return W
pass


def check_if_quantized(module: torch.nn.Module) -> bool:
    # All Unsloth Zoo code licensed under LGPLv3
    # Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/integrations.py
    if not hasattr(module, "weight"): return False

    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        # weight = module.dequantize()
        # return weight
        return True
    elif type(module.weight).__module__.startswith("torchao."):
        # check for torchao without requiring any torchao imports
        # weight = module.weight.dequantize()
        # return weight
        return True

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            # this is an FSDP-specific edge case
            # return weight  # type: ignore
            return False
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        # return weight
        return False

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    return True
    # weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    # if is_cpu:
    #     # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
    #     module.weight = module.weight.to(device)
    # return weight
pass


def expand_module_keys(name, module, original_keys):
    # All Unsloth Zoo code licensed under LGPLv3
    keys = module.state_dict().keys()
    for key in keys: original_keys.add(name + "." + key)
    return original_keys
pass


from peft.utils.integrations import dequantize_module_weight
import collections
import numpy as np
import inspect
from tqdm import tqdm as ProgressBar
from dataclasses import dataclass

@dataclass
class LoraStats:
    module : torch.nn.Module
    lora_A : torch.Tensor
    lora_B : torch.Tensor
    alpha  : float
pass


def assert_same_keys(model, new_state_dict):
    # All Unsloth Zoo code licensed under LGPLv3
    inner_model = model.base_model.model if hasattr(model, "base_model") else model
    original_keys = inner_model.state_dict().keys()
    all_original_keys = set()
    for x in original_keys:
        where_weight = x.rfind(".weight")
        where_bias   = x.rfind(".bias")
        if where_weight != -1: x = x[:where_weight + len(".weight")]
        elif where_bias != -1: x = x[:where_bias   + len(".bias")  ]
        else: pass

        # Remove LoRA and base_layer
        j = max(x.rfind(".lora_"), x.rfind(".base_layer"))
        if j != -1: x = x[:j] + x[x.rfind("."):]

        all_original_keys.add(x)
    pass
    difference = all_original_keys ^ set(new_state_dict)
    if len(difference) != 0:
        raise RuntimeError(f"Unsloth: Extracted keys = {difference} do not match!")
    pass
pass


@torch.inference_mode
def create_lora_statistics(model, merge_into_original = False, return_state_dict = True):
    # All Unsloth Zoo code licensed under LGPLv3
    # merge_into_original is merging directly into 16bit downloaded model
    # without dequantizing
    Linear_LoRA_Layers = get_lora_layer_modules()
    Linear_LoRA_Layers = tuple(x[0] for x in Linear_LoRA_Layers)

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    module_count, lora_A_count, lora_B_count, scaling_count = 0, 0, 0, 0

    remove_keys = set()
    keep_keys   = set()

    inner_model = model.base_model.model if hasattr(model, "base_model") else model
    for name, module in inner_model.named_modules():
        if name == "": continue

        elif name.endswith(".lora_A.default"):
            lora_weights[name[:-len(".lora_A.default")]].lora_A = module.weight
            lora_A_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".lora_B.default"):
            lora_weights[name[:-len(".lora_B.default")]].lora_B = module.weight
            lora_B_count += 1
            expand_module_keys(name, module, remove_keys)

        elif isinstance(module, Linear_LoRA_Layers):
            active_adapter = module.active_adapters[0] if \
                hasattr(module, "active_adapters") else module.active_adapter
            lora_weights[name].alpha = module.scaling[active_adapter]
            scaling_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".base_layer"):
            lora_weights[name[:-len(".base_layer")]].module = module
            module_count += 1
            remove_keys.add(name)
            remove_keys.add(name[:-len(".base_layer")])

        elif (not merge_into_original) and check_if_quantized(module):
            lora_weights[name].module = module
            keep_keys.add(name + ".weight")
            if getattr(module, "bias", None) is not None: keep_keys.add(name + ".bias")
            expand_module_keys(name, module, remove_keys)
            remove_keys.add(name)

        elif ".lora_" in name: continue

        else:
            new_keys = expand_module_keys(name, module, set())
            for key in new_keys:
                if not key.endswith((".weight", ".bias")):
                    # Check if quantized item exactly which has ".weight"
                    if ".weight." in key:
                        remove_keys.add(key)
                    else:
                        # Keep gate_tanh, embedding etc
                        pass
            remove_keys.add(name)
        pass
    pass
    assert(module_count == lora_A_count == lora_B_count == scaling_count)

    # Also return state_dict if needed
    if return_state_dict:
        old_state_dict = inner_model.state_dict()
        state_dict     = collections.OrderedDict()
        for name, param in old_state_dict.items():

            if name.endswith(".base_layer.weight"):
                name = name[:-len(".base_layer.weight")]

            if name in lora_weights:
                state_dict[name + ".weight"]   = lora_weights[name]
                if getattr(lora_weights[name].module, "bias", None) is not None:
                    state_dict[name + ".bias"] = lora_weights[name].module.bias
                continue
            elif name in keep_keys:
                # Quantized modules with no LoRA adapters
                lora_name = name[:-len(".weight")]
                if lora_name in lora_weights:
                    param = lora_weights[lora_name]
                else:
                    # Bias term
                    pass
            elif name in remove_keys: continue

            state_dict[name] = param
        pass
    else:
        state_dict = None
    pass

    if return_state_dict: assert_same_keys(model, state_dict)
    return lora_weights, state_dict
pass


@torch.inference_mode
def _merge_and_overwrite_lora(save_directory, filename, lora_weights, output_dtype):
    # All Unsloth Zoo code licensed under LGPLv3
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename = os.path.join(save_directory, filename)
    tensors = OrderedDict()
    count = 0
    import psutil
    import tempfile
    import pickle
    limit = 700 * 1024 * 1024 # 700MB
    with safe_open(filename, framework = "pt", device = "cpu") as file:
        for key in file.keys():
            W = file.get_tensor(key)
            lora_stats = lora_weights.get(key[:-len(".weight")], None)
            if lora_stats is not None:
                count += 1
                W = _merge_lora(W, lora_stats, key)
                if psutil.virtual_memory().available <= limit:
                    filename = tempfile.NamedTemporaryFile(suffix = ".pt")
                    torch.save(W.to(output_dtype), filename, pickle_module = pickle, pickle_protocol = pickle.HIGHEST_PROTOCOL)
                    W = torch.load(filename, map_location = "cpu", mmap = True, weights_only = False)
                else:
                    W = W.to(device = "cpu", dtype = output_dtype, non_blocking = True)
            pass
            tensors[key] = W
        pass
    pass
    save_file(tensors, filename, metadata = {"format": "pt"})
    return count
pass


from huggingface_hub import (
    split_state_dict_into_shards_factory,
    get_torch_storage_size,
    get_torch_storage_id,
)

def get_torch_storage_size_new(x, element_size):
    if isinstance(x, LoraStats):
        shape = (x.module.in_features, x.module.out_features)
        return int(np.prod(shape)) * element_size
    else:
        return get_torch_storage_size(x)
pass


def get_torch_storage_id_new(x):
    if isinstance(x, LoraStats):
        return None
    else:
        return get_torch_storage_id(x)
pass


def prepare_saving(
    model,
    save_directory,
    push_to_hub = False,
    max_shard_size = "5GB",
    private = True,
    token = None,
    output_dtype = None,
    merge_into_original = False,
    low_disk_space_usage = False,
    min_size_in_bytes = 100_000_000, # Must be of this size - 100MB default
    use_temp_file = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check size
    from huggingface_hub.serialization._base import parse_size_to_int
    max_shard_size_in_bytes = max_shard_size
    if type(max_shard_size_in_bytes) is not int:
        max_shard_size_in_bytes = parse_size_to_int(max_shard_size)
    pass

    temp_file = None
    username, repo_id, hf_api = None, None, None

    if push_to_hub:
        if token is None: token = get_token()
        username, repo_id, hf_api = create_huggingface_repo(
            model = model,
            repo_id = save_directory,
            private = private,
            token = token,
        )
        # Check if temporary folder is needed
        if os.path.isdir(save_directory) or use_temp_file:
            temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
            save_directory = temp_file.name
            use_temp_file = True
        pass
    pass

    if output_dtype is None: output_dtype = _get_dtype(model.config.torch_dtype)
    assert(output_dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16))
    assert(type(torch.bfloat16) is torch.dtype)
    element_size = torch.tensor([], dtype = output_dtype).element_size()

    # Get state_dict
    lora_weights, state_dict = create_lora_statistics(
        model,
        merge_into_original = merge_into_original,
        return_state_dict = True,
    )
    # Total save size in bytes
    save_size = sum(get_torch_storage_size_new(x, element_size) for x in state_dict.values())

    # Create folder if it does not exist
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory, exist_ok = True)
        except Exception as error:
            raise RuntimeError(f"Unsloth: Error creating directory {save_directory} with error = {str(error)}")
    pass

    # Check if directory has enough space
    total, used, free = shutil.disk_usage(save_directory)
    free = int(free*0.95)

    def raise_upload_works():
        # Works with individual shard uploading
        raise RuntimeError(
            "Unsloth: Failed saving locally - no disk space left. "\
            "Uploading can work luckily! Use .push_to_hub instead."
        )
    pass

    if free < save_size:
        # Fail if already using temp folder except if individual portions work!
        if use_temp_file:
            if merge_into_original:
                if free > min_size_in_bytes:
                    # Downloading safetensor shards must be min shard size
                    low_disk_space_usage = True
                else: raise_upload_works()
            elif free > 100_000_000:
                if push_to_hub:
                    # Instead we form shards on the fly and push them!
                    low_disk_space_usage = True
                    max_shard_size_in_bytes = free
                else: raise_upload_works()
            else:
                raise RuntimeError("Failed saving - no disk space left!")
        pass

        # Too small - try using the temporary file system (sometimes large like Kaggle)
        try_temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
        try_save_directory = temp_file.name

        total, used, free = shutil.disk_usage(save_directory)
        free = int(free*0.95)
        if not push_to_hub and free > save_size: raise_upload_works()
        elif push_to_hub and free < save_size:
            raise RuntimeError("Unsloth: Failed uploading - no disk space left.")
        elif push_to_hub:
            print(
                f"Unsloth: Saving to {save_directory} will fail, but using a temp folder works! "\
                "Switching to a temp folder then uploading!"
            )
            # Switch to temp directory
            temp_file = try_temp_file
            save_directory = try_save_directory
            use_temp_file = True
        else:
            raise RuntimeError("Failed saving - no disk space left!")
        pass
    pass

    return (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    )
pass


def _remove_quantization_config(config_path: Path):
    assert config_path.exists(), "Given config does not exist"
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        # Remove the quantization_config field
        del config["quantization_config"]
    else:
        return
    # Overwrite the config file
    with open(config_path, "w") as f:
        json.dump(config, f, indent = 4)
    pass
pass


@torch.inference_mode
def merge_and_overwrite_lora(
    get_model_name,
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    private              = False,
    token                = None,
    save_method          = "lora",
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    cleanup_temp_file    = True,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Directly downloads 16bit original weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModelForCausalLM) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model

    base_model = model.base_model if isinstance(model, PeftModelForCausalLM) else model

    try:
        model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
    except:
        model_name = model.config._name_or_path

    # Find repository's max shard size and total size of everything
    try:
        file_list = HfFileSystem(token = token).ls(model_name, detail = True)
    except:
        original_model_id = get_original_model_id(model_name)
        model_name = original_model_id
        file_list = HfFileSystem(token = token).ls(model_name, detail = True)

    safetensors_list = []
    max_size_in_bytes = 0
    total_size_in_bytes = 0
    for x in file_list:
        if not x["name"].endswith(".safetensors"): continue
        safetensors_list.append(os.path.split(x["name"])[-1])
        max_size_in_bytes = max(max_size_in_bytes, x["size"])
        total_size_in_bytes += x["size"]
    pass
    assert(max_size_in_bytes != 0 and total_size_in_bytes != 0)

    (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, new_use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    ) = prepare_saving(
        model = model,
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        max_shard_size = "5GB",
        private = private,
        token = token,
        output_dtype = output_dtype,
        low_disk_space_usage = low_disk_space_usage,
        merge_into_original = True,
        min_size_in_bytes = max_size_in_bytes,
        use_temp_file = use_temp_file,
    )
    use_temp_file = use_temp_file or new_use_temp_file

    n_saved_modules = 0
    def upload_items(filename = None):
        extras = {"repo_id" : repo_id, "repo_type" : "model", "commit_message" : "(Trained with Unsloth)", }
        if filename is None:
            hf_api.upload_folder(folder_path = save_directory, **extras,)
        else:
            hf_api.upload_file(
                path_or_fileobj = os.path.join(save_directory, filename),
                path_in_repo = filename,
                **extras,
            )
        pass
    pass

    # Save config / generation_config via no state_dict and tokenizer
    if tokenizer is not None: tokenizer.save_pretrained(save_directory = save_directory,)

    if save_method == "merged_16bit":
        inner_model.save_pretrained(
            save_directory = save_directory,
            state_dict = {},
        )
        _remove_quantization_config(config_path = Path(save_directory) / "config.json")
    elif save_method == "merged_4bit":
        print(f"Unsloth: Saving model 4bit...")
        base_model = base_model.merge_and_unload()
        skipped_modules, quantized_modules = find_skipped_quantized_modules(base_model)
        if len(skipped_modules) > 0:
            # Reconstruct skipped modules so that it can be loaded
            base_model.config.quantization_config["llm_int8_skip_modules"] = skipped_modules

        base_model.save_pretrained(
            save_directory = save_directory,
        )
    # Remove the quantization_config in the config.json file if it exists,
    # as we are exporting the model in 16-bit format.

    if push_to_hub: upload_items()

    if save_method == "merged_16bit":
        safe_tensor_index_files = ["model.safetensors.index.json"] if len(safetensors_list) > 1 else []
        if not low_disk_space_usage:
            # Download all safetensors in 1 go!
            print(f"Downloading safetensors for {model_name}...")
            snapshot_download(
                repo_id = model_name,
                local_dir = save_directory,
                allow_patterns = safe_tensor_index_files + safetensors_list,
            )
        elif safe_tensor_index_files:
            print(f"Downloading safetensors index for {model_name}...")
            snapshot_download(
                repo_id = model_name,
                local_dir = save_directory,
                allow_patterns = ["model.safetensors.index.json"],
            )

        for filename in ProgressBar(safetensors_list, desc = "Unsloth: Merging weights into 16bit"):
            if low_disk_space_usage:
                hf_hub_download(
                    repo_id = model_name,
                    filename = filename,
                    repo_type = "model",
                    local_dir = save_directory,
                )
            pass
            n_saved_modules += _merge_and_overwrite_lora(
                save_directory = save_directory,
                filename = filename,
                lora_weights = lora_weights,
                output_dtype = output_dtype,
            )
            torch.cuda.empty_cache()
            if low_disk_space_usage and push_to_hub:
                upload_items(filename)
                os.remove(os.path.join(save_directory, filename)) # Remove to conserve disk space
            pass
        pass

        # Check for errors
        if len(lora_weights) != n_saved_modules:
            raise RuntimeError(
                f"Unsloth: Saving LoRA finetune failed since # of LoRAs = {len(lora_weights)} "\
                f"does not match # of saved modules = {n_saved_modules}. Please file a bug report!"
            )
        pass
    if not low_disk_space_usage and push_to_hub: upload_items()

    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass

    return save_directory
pass


_PUSHING_CODE = \
"""
PushToHubMixin._upload_modified_files(
    PushToHubMixin,
    working_dir = save_directory,
    repo_id = '{repo_id}',
    files_timestamps = files_timestamps,
    commit_message = "Upload Unsloth finetuned model",
    token = token,
    create_pr = False,
    revision = {revision},
    commit_description = "Upload Unsloth finetuned model",
)
if {use_temp_file} and temp_file is not None: temp_file.cleanup()
else:
    shutil.rmtree(save_directory)
    os.makedirs(save_directory, exist_ok = True)
if {use_temp_file}:
    temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
    save_directory = temp_file.name
files_timestamps = PushToHubMixin._get_files_timestamps(PushToHubMixin, save_directory)
"""

def incremental_save_pretrained(
    save_pretrained,
    low_disk_space_usage = True,
    use_temp_file = True,
    repo_id = "",
    revision = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Move file timestamps out
    makedir = re.search(r"os\.makedirs\(save_directory.+?\n", save_pretrained)
    assert(makedir is not None)
    span = makedir.span(0)
    save_pretrained = save_pretrained[:span[1]-1] + \
        "; files_timestamps = self._get_files_timestamps(save_directory); temp_file = None;\n" + \
        save_pretrained[span[1]:]
    pass

    # Find the main loop
    if "for shard_file, tensors in filename_to_tensors" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `for shard_file, tensors in filename_to_tensors`")
    for_loop = re.search(
        r"for shard_file, tensors in filename_to_tensors\:"\
        r".*?[\n]{1,}[ ]{4}[a-zA-Z0-9\_\#]",
        save_pretrained,
        flags = re.DOTALL | re.MULTILINE,
    )
    assert(for_loop is not None)

    span = for_loop.span(0)
    for_loop = save_pretrained[max(span[0], span[1]-8) : span[1]-1]
    where = re.search(r"[\n]{1,}", for_loop[::-1]).span(0)[0]
    for_loop = save_pretrained[span[0] : span[1]-where-1]
    spaces = len(re.findall(r"\n([ ]{4,})", for_loop)[0])

    first_newline = for_loop.find("\n") + 1
    for_loop = for_loop.rstrip()

    if low_disk_space_usage:
        new_for_loop = for_loop[:first_newline] + \
            for_loop[first_newline:] + \
            " "*spaces + \
            re.sub(r"[ ]{8,}", "",
                   _PUSHING_CODE.format(
                       repo_id = repo_id,
                       revision = revision,
                       use_temp_file = use_temp_file,
                    ).rstrip()
            ).replace("\n", "\n" + " "*spaces)
    else:
        new_for_loop = for_loop
    pass

    new_for_loop = new_for_loop + \
        "\n" + \
        " "*spaces + \
        "for tensor in shard:\n" + \
        " "*(spaces+4) + \
        "if tensor in DEQUANTIZED_KEYS: shard[tensor] = None\n"

    if low_disk_space_usage:
        new_for_loop = new_for_loop + \
            "\n" + \
            " "*(spaces-4) + \
            f"if {use_temp_file}:\n" + \
            " "*(spaces) + \
            "temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)\n" + \
            " "*(spaces) + \
            "save_directory = temp_file.name\n" + \
            " "*(spaces) + \
            f"repo_id = '{repo_id}'\n"
    pass
    save_pretrained = save_pretrained.replace(for_loop, new_for_loop)
    
    if not low_disk_space_usage:
        save_pretrained = save_pretrained.replace(
            "for shard_file, tensors in filename_to_tensors",
            "for shard_file, tensors in ProgressBar(filename_to_tensors, desc = 'Unsloth: Saving ' + str(len(filename_to_tensors)) + ' safetensor(s)')",
            1,
        )
    pass
    return save_pretrained
pass


def merge_and_dequantize_lora(
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    max_shard_size       = "5GB",
    safe_serialization   = True,
    token                = None,
    private              = False,
    revision             = None,
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Dequantizes model to 16bit weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModelForCausalLM) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model

    (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    ) = prepare_saving(
        model = model,
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        max_shard_size = max_shard_size,
        private = private,
        token = token,
        output_dtype = output_dtype,
        low_disk_space_usage = low_disk_space_usage,
        merge_into_original = False,
        min_size_in_bytes = 100_000_000, # 100MB default
        use_temp_file = use_temp_file,
    )

    import transformers.modeling_utils
    save_pretrained = inspect.getsource(transformers.modeling_utils.PreTrainedModel.save_pretrained)
    spaces = save_pretrained.find("def")
    save_pretrained = save_pretrained.split("\n")
    save_pretrained = "\n".join(x[spaces:] for x in save_pretrained)

    # Now patch for incremental pushing to hub
    if push_to_hub:
        save_pretrained = incremental_save_pretrained(
            save_pretrained = save_pretrained,
            low_disk_space_usage = low_disk_space_usage,
            use_temp_file = use_temp_file,
            repo_id = repo_id,
            revision = revision,
        )
    pass

    functions = dir(transformers.modeling_utils)
    # functions = [x for x in functions if (f"{x}." in save_pretrained or f"{x}(" in save_pretrained) and x != "PreTrainedModel"]
    exec(f"from transformers.modeling_utils import ({', '.join(functions)})", locals(), globals())

    replace_state_dict = f"""
    DEQUANTIZED_KEYS = []

    def merge_lora_weights(state_dict, name):
        x = state_dict[name]
        if type(x) is LoraStats:
            DEQUANTIZED_KEYS.append(name)
            W = dequantize_module_weight(x.module)
            W = _merge_lora(W, x, name)
            x = W.to(device = 'cpu', dtype = {str(output_dtype)}, non_blocking = True)
        # Remove memory leak
        state_dict[name] = None
        return x
    pass
    state_dict_split = split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size   = {max_shard_size_in_bytes},
        filename_pattern = filename_pattern,
        get_storage_size = functools.partial(get_torch_storage_size_new, element_size = {element_size}),
        get_storage_id   = get_torch_storage_id_new,
    )
    """
    left  = save_pretrained.find("state_dict_split = split_torch_state_dict_into_shards")
    if left == -1: raise RuntimeError("Unsloth: Failed to find `state_dict_split`")
    right = save_pretrained.find(")", left) + 1
    save_pretrained = save_pretrained[:left] + replace_state_dict + save_pretrained[right:]

    if "state_dict[tensor].contiguous()" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `state_dict[tensor].contiguous()`")
    save_pretrained = save_pretrained.replace(
        "state_dict[tensor].contiguous()",
        "merge_lora_weights(state_dict, tensor).contiguous()",
        1,
    )

    if "def save_pretrained" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `def save_pretrained`")
    save_pretrained = save_pretrained.replace(
        "def save_pretrained",
        "def save_pretrained_dequantized",
        1,
    )

    functions = {}
    exec(save_pretrained, globals(), functions)
    save_pretrained_dequantized = functions["save_pretrained_dequantized"]
    save_pretrained_dequantized = torch.inference_mode(save_pretrained_dequantized)

    files_timestamps = PushToHubMixin._get_files_timestamps(
        PushToHubMixin,
        save_directory,
    )
    save_pretrained_dequantized(
        inner_model,
        save_directory     = save_directory,
        push_to_hub        = False,
        max_shard_size     = max_shard_size_in_bytes,
        safe_serialization = safe_serialization,
        token              = token,
        private            = private,
        state_dict         = state_dict,
        **kwargs,
    )

    # Save tokenizer
    if tokenizer is not None: tokenizer.save_pretrained(save_directory = save_directory,)

    if push_to_hub:
        commit = PushToHubMixin._upload_modified_files(
            PushToHubMixin,
            working_dir = save_directory,
            repo_id = repo_id,
            files_timestamps = files_timestamps,
            commit_message = "Upload Unsloth finetuned model",
            token = token,
            create_pr = False,
            revision = revision,
            commit_description = "Upload Unsloth finetuned model",
        )
        print(f"Unsloth: Uploaded model to https://huggingface.co/{repo_id}")
        return commit
    pass
    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass
pass

def get_original_model_id(local_path: str):
    import json
    import os
    
    config_path = os.path.join(local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Check for _name_or_path that's not a local path
        # When we load using AutoConfig, the _name_or_path changed into the local path instead
        if "_name_or_path" in config:
            return config["_name_or_path"]
        
    config_path = os.path.join(local_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        if "base_model_name_or_path" in config:
            return config["base_model_name_or_path"]
    
    return None

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
