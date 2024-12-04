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
    "SKIP_QUANTIZATION_MODULES",
]

import torch

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


@torch.inference_mode
def _merge_and_overwrite_lora(save_location, filename, lora_weights,):
    # Code licensed under LGPL
    # Merges LoRA and overwrites the safetensors file it was merged to
    filename = os.path.join(save_location, filename)
    tensors = OrderedDict()
    with safe_open(filename, framework = "pt", device = "cpu") as file:
        for key in file.keys():
            W = file.get_tensor(key)
            if key in lora_weights:
                A, B, scaling = lora_weights[key]
                old_dtype = W.dtype
                W = W.to("cuda", dtype = torch.float32, non_blocking = True)

                W = W.addmm_(B.to(torch.float32), A.to(torch.float32), alpha = scaling)

                maximum_element = torch.max(W.min().abs(), W.max())
                if not torch.isfinite(maximum_element).item():
                    raise ValueError(f"Unsloth: Merge failed.\n{key} has some elements = infinity.")
                W = W.to(old_dtype)
            pass
            tensors[key] = W
        pass
    pass
    save_file(tensors, filename, metadata = {"format": "pt"})
pass


@torch.inference_mode
def merge_and_overwrite_lora(
    get_model_name,
    create_huggingface_repo,
    model,
    save_location        = "unsloth_finetuned_merge",
    push_to_hub          = False,
    token                = None,
    upload_location      = None,
    low_disk_space_usage = True,
    private              = False,
):
    # Code licensed under LGPL
    ignore_files = [
        "*.gitattributes",
        "*.md",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
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

    # Find all LoRA A and B matrices
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            assert(name.startswith("base_model."))
            name = name[len("base_model."):]
            name = name.replace(".lora_A.default", "")
            lora_weights[name] = [param, None, None,]
        elif "lora_B" in name:
            assert(name.startswith("base_model."))
            name = name[len("base_model."):]
            name = name.replace(".lora_B.default", "")
            lora_weights[name][1] = param
        pass
    pass

    # Confirm count
    parameters = model.named_parameters()
    total_counted = sum(".lora_A." in name or ".lora_B." in name for name, x in parameters)
    if total_counted//2 != len(lora_weights):
        raise RuntimeError("Unsloth: The number of LoRA adapaters was not calculated correctly!")

    # Get LoRA scalings
    import peft.tuners.lora.bnb
    import peft.tuners.lora
    peft_items = dir(peft.tuners.lora.bnb)
    peft_items = [x for x in peft_items if x.startswith("Linear")]
    exec(f"from peft.tuners.lora import ({', '.join(peft_items)})", locals(), globals())

    Linear_LoRA_Layers = tuple([peft.tuners.lora.Linear,] + [eval(x) for x in peft_items])
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Linear_LoRA_Layers):
            assert(name.startswith("base_model."))
            name = name[len("base_model."):]
            active_adapter = module.active_adapters[0] if \
                hasattr(module, "active_adapters") else module.active_adapter
            scaling = module.scaling[active_adapter]
            lora_weights[name + ".weight"][2] = scaling
            count += 1
        pass
    pass
    if total_counted//2 != count:
        raise RuntimeError("Unsloth: The number of LoRA adapaters was not calculated correctly!")

    # Also model. might be repeated - remove them!
    original_keys = list(lora_weights.keys())
    for original_key in original_keys:
        if original_key.startswith("model."):
            lora_weights[original_key[len("model."):]] = lora_weights[original_key]
    pass

    # Only enable low_disk_space_usage for uploading
    if upload_location is not None and low_disk_space_usage:
        file_list = HfFileSystem().ls(model_name, detail = False)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        file_list = [x[len(model_name):].strip("/\\") for x in file_list if x.startswith(model_name)]

        # Download other items that are not .safetensors
        snapshot_download(
            repo_id = model_name,
            local_dir = save_location,
            ignore_patterns = ["*.safetensors"] + ignore_files,
        )

        for filename in ProgressBar(file_list, desc = "Unsloth: Merging weights into 16bit"):
            hf_hub_download(
                repo_id = model_name,
                filename = filename,
                repo_type = "model",
                local_dir = save_location,
            )
            _merge_and_overwrite_lora(
                save_location = save_location,
                filename = filename,
                lora_weights = lora_weights,
            )

            if upload_location is not None:
                location_to_file = os.path.join(save_location, filename)
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
                folder_path = save_location,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ["*.safetensors"] + ignore_files,
            )
            # Delete entire repo at the end!
            shutil.rmtree(save_location, ignore_errors = True)
        pass
    else:
        # Download entire repo in 1 call
        snapshot_download(
            repo_id = model_name,
            local_dir = save_location,
            ignore_patterns = ignore_files,
        )

        file_list = os.listdir(save_location)
        file_list = [x for x in file_list if x.endswith(".safetensors")]
        for filename in ProgressBar(file_list, desc = "Unsloth: Merging weights into 16bit"):
            _merge_and_overwrite_lora(
                save_location = save_location,
                filename = filename,
                lora_weights = lora_weights,
            )
        pass

        # Upload repo
        if upload_location is not None:
            hf_api.upload_folder(
                folder_path = save_location,
                repo_id = upload_location,
                repo_type = "model",
                commit_message  = "(Trained with Unsloth)",
                ignore_patterns = ignore_files,
            )
        pass
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
