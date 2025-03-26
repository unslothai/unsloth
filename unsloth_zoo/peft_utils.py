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
    "get_peft_regex",
    "merge_and_overwrite_lora",
    "merge_and_dequantize_lora",
    "SKIP_QUANTIZATION_MODULES",
    "get_lora_layer_modules",
    "requires_grad_for_gradient_checkpointing",
]

import inspect
import torch
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from collections import OrderedDict
import re

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
    # All Unsloth Zoo code licensed under LGPLv3
    if not finetune_vision_layers and not finetune_language_layers:
        raise RuntimeError(
            "Unsloth: No layers to finetune - please select to finetune the vision and/or the language layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
            "Unsloth: No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )
    pass

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


def get_lora_layer_modules():
    # All Unsloth Zoo code licensed under LGPLv3
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
        Linear_LoRA_Layers += [(eval(x), item, x,) for x in modules]
    pass
    return tuple(Linear_LoRA_Layers)
pass


def requires_grad_for_gradient_checkpointing(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Enables requires_grad to make gradient checkpointing work on
    # non language models that don't just use .embed_tokens
    def register_other_hooks(name1, name2, module, _hooks):
        old_hooks = eval(f"module.{_hooks}")
        other_hooks = []
        for value in old_hooks.values():
            qualname = getattr(value, "__qualname__", "")
            name     = getattr(value, "__name__", "")
            if name1 in qualname or name2 in qualname: pass
            elif name2 in name or name2 in name: pass
            else: other_hooks.append(value)
        pass
        # Keep none input requires grad hooks
        exec(f"module.{_hooks} = OrderedDict()")
        for hook in other_hooks:
            exec(f"module.register{_hooks[:-1]}(hook)")
        pass
    pass

    # Remove all previous forward hooks for gradient checkpointing
    for name, module in model.named_modules():
        if len(module._forward_hooks) != 0:
            register_other_hooks(
                "enable_input_require_grads",
                "make_inputs_require_grad",
                module,
                "_forward_hooks",
            )
        pass
    pass

    # Add post forward hook
    def requires_grad_post_hook(module, input, output):
        type_output = type(output)
        if type_output is torch.Tensor:
            output.requires_grad_(True)
        else:
            try:
                # Output in huggingface generally a dataclass with loss, try to add to that
                output.loss.requires_grad_(True)
            except Exception as _:
                raise RuntimeError("Unsloth: Failed to make output require gradients!")
    pass

    def requires_grad_pre_hook(module, input):
        type_input = type(input)
        if type_input is torch.Tensor:
            input.requires_grad_(True)
        elif type_input is tuple or type_input is list:
            if len(input) == 0:
                raise RuntimeError("Unsloth: Failed to make input require gradients!")
                # print(f"  WARNING: Empty list input to {module.__class__.__name__}!") # 
                # return
            if torch.is_floating_point(input[0]):
                input[0].requires_grad_(True)
        else:
            raise RuntimeError("Unsloth: Failed to make input require gradients!")
    pass

    # Find 1st ever item which requires grad
    param = None
    for name, param in model.named_parameters():
        if param.requires_grad: break
    if param is None: return

    name = re.sub("\.([\d]{1,})\.", r"[\1].", name)
    name_components = name.split(".")

    if len(name_components) == 0:
        raise RuntimeError("Unsloth: Model has 0 layers?")

    final_where = None
    # Try getting previous parent module
    for j in range(len(name_components)-1, 0, -1):
        name_curr = name_components[j]
        name_pre  = "model." + ".".join(name_components[:j])
        # Disable [\d] since it fails in gradient checkpointing
        if re.search(r"\[[\d]{1,}\]", name_pre): continue
        module = eval(name_pre)
        if hasattr(module, "forward"):
            try: forward = inspect.getsource(module.forward)
            except: continue

            # Normal self.language_model(...)
            if f"self.{name_curr}(" in forward:
                final_where = j + 1
                break

            # Fix self.blocks[0] like in Qwen
            module_list = re.sub(r"\[[\d]{1,}\]", "", name_curr)
            if f"in self.{module_list}:" in forward:
                final_where = j
                break
            elif re.search(r"for [^\s]{3,} in self\." + module_list, forward) is not None:
                # Might have failed finding self.layers: like self.layers[...]:
                final_where = j
                break
            pass
        pass
    pass

    if final_where is None:
        # Find all input embeddings and just set them all as a fallback!
        # Add other hooks first
        register_other_hooks(
            "requires_grad_post_hook",
            "requires_grad_post_hook",
            module,
            "_forward_hooks",
        )
        module.register_forward_hook(requires_grad_post_hook)
        return
    pass

    module_name = "model." + ".".join(name_components[:final_where])
    module = eval(module_name)

    if hasattr(module, "config") and (module.config.__class__.__name__ in ("CLIPVisionConfig", "SiglipVisionConfig",)):
        # CLIP - backtrack to get_input_embeddings since requires_grad fails!
        old_module = model
        for module_name, module in model.named_modules():
            if not hasattr(module, "get_input_embeddings"): break
            old_module = module
        module = old_module
    pass
    print(f"Unsloth: Making `{module_name}` require gradients")

    still_need_patching = True
    # Check if input_embeddings exists
    if hasattr(module, "get_input_embeddings"):
        # Use forward hook after Embedding() is called
        try:
            module = module.get_input_embeddings()
            # Add other hooks first
            register_other_hooks(
                "requires_grad_post_hook",
                "requires_grad_post_hook",
                module,
                "_forward_hooks",
            )
            module.register_forward_hook(requires_grad_post_hook)
            still_need_patching = False
        except:
            # Not Implemented probably?
            still_need_patching = True
    pass

    if still_need_patching:
        # Use forward pre hook before module is called
        register_other_hooks(
            "requires_grad_pre_hook",
            "requires_grad_pre_hook",
            module,
            "_forward_pre_hooks",
        )
        module.register_forward_pre_hook(requires_grad_pre_hook)
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
