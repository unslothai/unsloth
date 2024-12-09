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
    "get_peft_regex",
    "merge_and_overwrite_lora",
    "merge_and_dequantize_lora",
    "SKIP_QUANTIZATION_MODULES",
]

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

# Skip some modules sensitive to quantization
SKIP_QUANTIZATION_MODULES = [
    "lm_head",
    "multi_modal_projector", # Llama 3.2 Vision, Pixtral, Llava
    "merger",                # Qwen2 VL
    "modality_projection",   # Idefics, SmolVLM
]

def get_peft_regex(
    model,
    finetune_vision_layers     : bool = True,
    finetune_language_layers   : bool = True,
    finetune_attention_modules : bool = True,
    finetune_mlp_modules       : bool = True,
    target_modules             : list[str] = None,
    vision_tags                : list[str] = ["vision", "image", "visual", "patch",],
    language_tags              : list[str] = ["language", "text",],
    attention_tags             : list[str] = ["self_attn", "attention", "attn",],
    mlp_tags                   : list[str] = ["mlp", "feed_forward", "ffn", "dense",],
) -> str:
    """
    Create a regex pattern to apply LoRA to only select layers of a model.
    """
    # Code licensed under LGPL
    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError(
            "Unsloth: No layers to finetune - please select to finetune the vision and/or the language layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
            "Unsloth: No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )
    pass

    import re
    from collections import Counter
    # Get only linear layers
    modules = model.named_modules()
    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]
    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices if count == 1
    if target_modules is None:
        only_linear_modules = []
        projection_modules  = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
        pass
    else:
        assert(type(target_modules) is list)
        only_linear_modules = list(target_modules)
    pass

    # Create regex matcher
    regex_model_parts = []
    if finetune_vision_layers:     regex_model_parts += vision_tags
    if finetune_language_layers:   regex_model_parts += language_tags
    regex_components  = []
    if finetune_attention_modules: regex_components  += attention_tags
    if finetune_mlp_modules:       regex_components  += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components  = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    regex_matcher = \
        r".*?(?:"  + regex_model_parts + \
        r").*?(?:" + regex_components + \
        r").*?"    + match_linear_modules + ".*?"

    # Also account for model.layers.0.self_attn/mlp type modules like Qwen
    if finetune_language_layers:
        regex_matcher = r"(?:" + regex_matcher + \
        r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
        r")\.(?:" + match_linear_modules + r"))"
    pass

    # Check if regex is wrong since model does not have vision parts
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check:
        regex_matcher = \
            r".*?(?:" + regex_components + \
            r").*?"   + match_linear_modules + ".*?"
    pass

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check and target_modules is not None:
        raise RuntimeError(
            f"Unsloth: No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
            f"Unsloth: No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )
    pass
    return regex_matcher
pass


from huggingface_hub import (
    HfFileSystem,
    snapshot_download,
    hf_hub_download,
)
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from tqdm import tqdm as ProgressBar
import os, shutil


def get_lora_layer_modules():
    # Code licensed under LGPL
    import peft.tuners.lora
    path = os.path.split(peft.tuners.lora.__file__)[0]
    files = os.listdir(path)

    Linear_LoRA_Layers = []
    for file in files:
        if file == "__init__.py" or not file.endswith(".py"): continue
        item = f"peft.tuners.lora.{file[:-len('.py')]}"
        exec(f"import {item}", locals(), globals())
        modules = dir(eval(item))
        modules = [x for x in modules if x.startswith("Linear") or x.endswith("Linear")]
        if len(modules) == 0: continue
        exec(f"from {item} import ({', '.join(modules)})", locals(), globals())
        Linear_LoRA_Layers += [eval(x) for x in modules]
    pass
    return tuple(Linear_LoRA_Layers)
pass


def check_if_quantized(module: torch.nn.Module) -> bool:
    # Code licensed under LGPL
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
    # Code licensed under LGPL
    keys = module.state_dict().keys()
    for key in keys: original_keys.add(name + "." + key)
    return original_keys
pass


from peft.utils.integrations import dequantize_module_weight
import collections
from dataclasses import dataclass
import numpy as np
import inspect
from huggingface_hub import (
    split_state_dict_into_shards_factory,
    get_torch_storage_size,
    get_torch_storage_id,
)


@dataclass
class LoraStats:
    module : torch.nn.Module
    lora_A : torch.Tensor
    lora_B : torch.Tensor
    alpha  : float
pass

@torch.inference_mode
def create_lora_statistics(model, merge_into_original = False, return_state_dict = True):
    # Code licensed under LGPL
    # merge_into_original is merging directly into 16bit downloaded model
    # without dequantizing
    Linear_LoRA_Layers = get_lora_layer_modules()

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    module_count, lora_A_count, lora_B_count, scaling_count = 0, 0, 0, 0

    remove_keys = set()
    keep_keys   = set()
    assert(hasattr(model, "base_model"))
    assert(hasattr(model.base_model, "model"))
    for name, module in model.base_model.model.named_modules():
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
                if not key.endswith((".weight", ".bias")): remove_keys.add(key)
            remove_keys.add(name)
        pass
    pass
    assert(module_count == lora_A_count == lora_B_count == scaling_count)

    # Also return state_dict if needed
    if return_state_dict:
        old_state_dict = model.base_model.model.state_dict()
        state_dict     = collections.OrderedDict()
        for name, param in old_state_dict.items():
            if name.endswith(".base_layer.weight"):
                name = name[:-len(".base_layer.weight")]
            
            if name in lora_weights:
                state_dict[name + ".weight"]   = lora_weights[name]
                if getattr(lora_weights[name].module, "bias", None) is not None:
                    state_dict[name + ".bias"] = lora_weights[name].module.bias
                continue
            elif name in keep_keys: pass
            elif name in remove_keys: continue

            state_dict[name] = param
        pass
    else:
        state_dict = None
    return lora_weights, state_dict
pass


@torch.inference_mode
def _merge_and_overwrite_lora(save_directory, filename, lora_weights,):
    # Code licensed under LGPL
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename = os.path.join(save_directory, filename)
    tensors = OrderedDict()
    count = 0
    with safe_open(filename, framework = "pt", device = "cpu") as file:
        for key in file.keys():
            W = file.get_tensor(key)
            if key.endswith(".weight") and key[:-len(".weight")] in lora_weights:
                count += 1
                lora_stats = lora_weights[key[:-len(".weight")]]
                A, B, scaling = lora_stats.lora_A, lora_stats.lora_B, lora_stats.alpha
                old_dtype = W.dtype
                W = W.to("cuda", dtype = torch.float32, non_blocking = True)

                W = W.addmm_(B.to(torch.float32), A.to(torch.float32), alpha = scaling)

                maximum_element = torch.max(W.min().abs(), W.max())
                if not torch.isfinite(maximum_element).item():
                    raise ValueError(f"Unsloth: Merge failed.\n{key} has some elements = infinity.")
                W = W.to("cpu", dtype = old_dtype, non_blocking = True)
            pass
            tensors[key] = W
        pass
    pass
    save_file(tensors, filename, metadata = {"format": "pt"})
    return count
pass


@torch.inference_mode
def merge_and_overwrite_lora(
    get_model_name,
    create_huggingface_repo,
    model,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    token                = None,
    upload_location      = None,
    low_disk_space_usage = True,
    private              = False,
):
    # Code licensed under LGPL
    # Directly downloads 16bit original weights and merges LoRA
    ignore_files = [
        "*.gitattributes",
        "*.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    try:
        model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
    except:
        model_name = model.config._name_or_path
    print(f"Unsloth: Merging QLoRA weights directly to the 16bit version of {model_name}.")

    if push_to_hub and upload_location is None:
        raise RuntimeError(
            "Unsloth: You're trying to upload to a HuggingFace repo, but did not provide an `upload_location`. Please do!"
        )
    pass

    if upload_location is not None:
        upload_location, hf_api = create_huggingface_repo(
            model = model,
            save_directory = upload_location,
            token = token,
            private = private,
        )
    pass

    lora_weights, _ = create_lora_statistics(model, merge_into_original = True, return_state_dict = False)
    n_saved_modules = 0

    # Only enable low_disk_space_usage for uploading
    if upload_location is not None and low_disk_space_usage:
        file_list = HfFileSystem().ls(model_name, detail = False)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        file_list = [x[len(model_name):].strip("/\\") for x in file_list if x.startswith(model_name)]

        # Download other items that are not .safetensors
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            ignore_patterns = ["*.safetensors"] + ignore_files,
        )

        for filename in ProgressBar(file_list, desc = "Unsloth: Merging weights into 16bit"):
            hf_hub_download(
                repo_id = model_name,
                filename = filename,
                repo_type = "model",
                local_dir = save_directory,
            )
            n_saved_modules += _merge_and_overwrite_lora(
                save_directory = save_directory,
                filename = filename,
                lora_weights = lora_weights,
            )

            if upload_location is not None:
                location_to_file = os.path.join(save_directory, filename)
                hf_api.upload_file(
                    path_or_fileobj = location_to_file,
                    path_in_repo = filename,
                    repo_id = upload_location,
                    repo_type = "model",
                    commit_message  = "(Trained with Unsloth)",
                )
                # Remove safetensors file
                os.remove(location_to_file)
            pass
        pass

        # Upload rest of files that are not safetensors
        if upload_location is not None:
            hf_api.upload_folder(
                folder_path = save_directory,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ["*.safetensors"] + ignore_files,
            )
            # Delete entire repo at the end!
            shutil.rmtree(save_directory, ignore_errors = True)
        pass
    else:
        # Download entire repo in 1 call
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            ignore_patterns = ignore_files,
        )

        file_list = os.listdir(save_directory)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        for filename in ProgressBar(file_list, desc = "Unsloth: Merging weights into 16bit"):
            n_saved_modules += _merge_and_overwrite_lora(
                save_directory = save_directory,
                filename = filename,
                lora_weights = lora_weights,
            )
        pass

        # Upload repo
        if upload_location is not None:
            hf_api.upload_folder(
                folder_path = save_directory,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ignore_files,
            )
        pass
    pass

    # Check for errors
    if len(lora_weights) != n_saved_modules:
        raise RuntimeError(
            f"Unsloth: Saving LoRA finetune failed since # of LoRAs = {len(lora_weights)} "\
            f"does not match # of saved modules = {n_saved_modules}. Please file a bug report!"
        )
    pass
pass


def merge_and_dequantize_lora(
    create_huggingface_repo,
    model,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    upload_location      = None,
    max_shard_size       = "5GB",
    safe_serialization   = True,
    token                = None,
    private              = False,
    output_dtype         = None,
    **kwargs,
):
    # Code licensed under LGPL
    # Dequantizes model to 16bit weights and merges LoRA
    if push_to_hub and upload_location is None:
        raise RuntimeError(
            "Unsloth: You're trying to upload to a HuggingFace repo, but did not provide an `upload_location`. Please do!"
        )
    pass

    if upload_location is not None:
        upload_location, hf_api = create_huggingface_repo(
            model = model,
            save_directory = upload_location,
            token = token,
            private = private,
        )
    pass

    if output_dtype is None: output_dtype = model.config.torch_dtype

    assert(output_dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16))
    assert(type(torch.bfloat16) is torch.dtype)

    import transformers.modeling_utils
    save_pretrained = inspect.getsource(transformers.modeling_utils.PreTrainedModel.save_pretrained)
    spaces = save_pretrained.find("def")
    save_pretrained = save_pretrained.split("\n")
    save_pretrained = "\n".join(x[spaces:] for x in save_pretrained)

    functions = dir(transformers.modeling_utils)
    # functions = [x for x in functions if (f"{x}." in save_pretrained or f"{x}(" in save_pretrained) and x != "PreTrainedModel"]
    exec(f"from transformers.modeling_utils import ({', '.join(functions)})", locals(), globals())

    element_size = torch.tensor([], dtype = output_dtype).element_size()

    replace_state_dict = f"""
    DEQUANTIZED_KEYS = set()
    
    def get_torch_storage_size_new(x):
        if isinstance(x, LoraStats):
            shape = (x.module.in_features, x.module.out_features)
            return int(np.product(shape)) * {element_size}
        else:
            return get_torch_storage_size(x)
    pass

    def merge_lora_weights(x, name):
        if type(x) is LoraStats:
            DEQUANTIZED_KEYS.add(name)
            W = dequantize_module_weight(x.module)
            if x.lora_A is not None and x.lora_B is not None:
                W = W.to('cuda', dtype = torch.float32, non_blocking = True)
                W = W.addmm_(
                    x.lora_B.to('cuda', dtype = torch.float32, non_blocking = True),
                    x.lora_A.to('cuda', dtype = torch.float32, non_blocking = True),
                    alpha = x.alpha,
                )
                maximum_element = torch.max(W.min().abs(), W.max())
                if not torch.isfinite(maximum_element).item():
                    raise ValueError('Unsloth: Merge failed as there are infinite elements in' + name)
            pass
            W = W.to(device = 'cpu', dtype = {str(output_dtype)}, non_blocking = True)
            x = W
        return x
    pass

    def get_torch_storage_id_new(x):
        if isinstance(x, LoraStats):
            return None
        else:
            return get_torch_storage_id(x)
    pass

    state_dict_split = split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size   = max_shard_size,
        filename_pattern = filename_pattern,
        get_storage_size = get_torch_storage_size_new,
        get_storage_id   = get_torch_storage_id_new,
    )
    """
    left  = save_pretrained.find("state_dict_split = split_torch_state_dict_into_shards")
    if left == -1: raise
    right = save_pretrained.find(")", left) + 1
    save_pretrained = save_pretrained[:left] + replace_state_dict + save_pretrained[right:]
    save_pretrained = save_pretrained.replace(
        "state_dict[tensor].contiguous()",
        "merge_lora_weights(state_dict[tensor], tensor).contiguous()",
        1,
    )

    # Stop CPU RAM overuse by resizing to 0
    free_cpu = """
        for tensor in shard:
            if tensor in DEQUANTIZED_KEYS:
                shard[tensor] = None
    if index is None:
    """
    save_pretrained = save_pretrained.replace(
        "if index is None:",
        free_cpu,
        1,
    )

    save_pretrained = save_pretrained.replace(
        "def save_pretrained",
        "def unsloth_save_pretrained_dequantized",
        1,
    )
    exec(save_pretrained, globals())
    unsloth_save_pretrained_dequantized = torch.inference_mode(unsloth_save_pretrained_dequantized)

    unsloth_save_pretrained_dequantized(
        model.base_model.model,
        save_directory     = save_directory if upload_location is None else upload_location,
        push_to_hub        = push_to_hub,
        max_shard_size     = max_shard_size,
        safe_serialization = safe_serialization,
        token              = token,
        private            = private,
        state_dict         = state_dict,
        **kwargs,
    )
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
