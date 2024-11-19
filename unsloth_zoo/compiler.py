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
    "UNSLOTH_COMPILE_LOCATION",
    "get_transformers_model_type",
    "unsloth_compile_transformers",
]

import inspect
import re
import importlib
import numpy as np
import os
import torch
import subprocess

global COMBINED_UNSLOTH_NAME
global UNSLOTH_COMPILE_LOCATION
global UNSLOTH_CREATED_FUNCTIONS
COMBINED_UNSLOTH_NAME = "unsloth_compiled_module"
UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache"
UNSLOTH_CREATED_FUNCTIONS = []


_disabled_sdpa_code = """
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

import torch
from unsloth_zoo.loss_utils import fused_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass

"""

# Patch Layernorm, Conv
_patch_functions = [
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "LayerNorm", "RMSNorm",
]


def get_transformers_model_type(
    model_name,
    token = None,
    revision = None,
    trust_remote_code = False,
):
    # Code licensed under LGPL
    from transformers import AutoConfig
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()

    config = AutoConfig.from_pretrained(
        model_name,
        token = token,
        revision = revision,
        trust_remote_code = trust_remote_code,
    )
    if not was_disabled: enable_progress_bars()

    model_types = []
    config = str(config.to_dict())
    model_types = re.findall(r"'model_type': '([^\s\']{1,})'", config)
    model_types = [x.replace("-", "_").lower() for x in model_types]

    from transformers import models
    models = dir(models)
    all_model_types = set()
    for name in models:
        for model_type in model_types:
            if model_type in name.lower():
                all_model_types.add(model_type)
                break
    pass

    all_model_types = list(all_model_types)
    return all_model_types
pass


# Empty causal mask
def no_update_causal_mask(*args, **kwargs): return None

# Patch SDPA
def replace_with_grouped_query_attention(module, source):
    # Code licensed under LGPL
    if "enable_gqa" not in torch.nn.functional.scaled_dot_product_attention.__doc__: return source

    grouped_query_attention_finder = \
        r"(key_states \= repeat_kv[^\n]{1,}\n[\s]{1,}"\
        r"value_states \= repeat_kv[^\n]{1,}\n[\s]{1,}"\
        r"(.+?)"\
        r"query_states \= query_states\.contiguous\(\)\n[\s]{1,}"\
        r"key_states \= key_states\.contiguous\(\)\n[\s]{1,}"\
        r"value_states \= value_states\.contiguous\(\))"

    found = re.findall(grouped_query_attention_finder, source, flags = re.DOTALL | re.MULTILINE,)
    if len(found) == 1:
        found = found[0]
        # Should be == 2, but Llama has key_states = self.k_norm(key_states)
        if found[0].count("key_states = ") >= 2 and found[0].count("value_states = ") >= 2:
            print(f"Unsloth: Transforming {module}.")
            all_source = source
            source = re.sub(
                grouped_query_attention_finder,
                r"\2pass\n",
                source,
                flags = re.DOTALL | re.MULTILINE,
            )
            source = source\
                .replace(
                    "dropout_p=self.dropout if self.training else 0.0,",
                    "dropout_p=self.dropout if self.training else 0.0, "\
                    "enable_gqa=self.num_key_value_groups != 1,",
                ).replace(
                    "dropout_p=self.attention_dropout if self.training else 0.0,",
                    "dropout_p=self.attention_dropout if self.training else 0.0, "\
                    "enable_gqa=self.num_key_value_groups != 1,",
                )
        pass
    pass

    source = re.sub(
        r"if output_attentions\:.+?return super\(\)\.forward.+?\)",
        "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
        source,
        flags = re.DOTALL | re.MULTILINE,
    )
    return source
pass


def create_new_function(name, new_source, model_location, functions, prepend = "", append = "", overwrite = True):
    # Code licensed under LGPL
    global UNSLOTH_CREATED_FUNCTIONS
    global UNSLOTH_COMPILE_LOCATION
    if new_source[0] == " ":
        spaces = new_source.find("def")
        new_source = new_source.split("\n")
        new_source = "\n".join(x[spaces:] for x in new_source)
    pass

    # Import items to make the function executable
    items = [x for x in functions if ((x in new_source) and (x != name) and not (f"def {x}" in new_source))]
    imports = "from torch import Tensor\n"
    imports += f"from {model_location} import (" + ", ".join(x for x in items) + ")" if len(items) != 0 else ""
    new_source = imports + "\n\n" + new_source
    new_source = prepend + new_source + append

    # Fix super() Not necessary anymore!
    # new_source = new_source.replace("super()", "super(type(self), self)")

    # Check location
    if not os.path.exists(UNSLOTH_COMPILE_LOCATION): os.makedirs(UNSLOTH_COMPILE_LOCATION)

    location = os.path.join(UNSLOTH_COMPILE_LOCATION, f"{name}.py")
    if overwrite or not os.path.isfile(location):
        with open(location, "w") as file:
            file.write(new_source)
            file.flush()
            os.fsync(file)
        pass
    pass

    new_module = importlib.import_module(UNSLOTH_COMPILE_LOCATION + "." + name)
    # Must save to global state or else temp file closes
    UNSLOTH_CREATED_FUNCTIONS.append(location)
    return new_module
pass


def create_standalone_class(
    module,
    model_location,
    functions,
    fullgraph = False,
    forward_source = None,
    disable = False,
    add_loss_kwargs = False,
) -> str:
    # Code licensed under LGPL
    # Create optimized standalone forward function
    f = eval(f"{model_location}.{module}")
    full_class = inspect.getsource(f)
    old_source = inspect.getsource(f.forward)
    if forward_source is None: forward_source = old_source

    source = re.sub(
        "def forward",
        f"def {module}_forward",
        forward_source,
    )
    spaces = re.search(r"[^\s\n]", source).span(0)[0]
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)

    compile = \
        f"torch.compile(fullgraph = {fullgraph}, dynamic = True, options = torch_compile_options)" \
        if not disable else \
        "torch.compiler.disable(recursive = False)"

    # Create new forward calling optimized function
    parameters = inspect.signature(f.forward).parameters
    # .parameters removes **kwargs and *args so we get it back!
    keys = list(parameters.keys())
    values = list(parameters.values())
    for j, value in enumerate(values):
        value = str(value)
        if   value.startswith("**"): keys[j] = "**" + keys[j]
        elif value.startswith("*"):  keys[j] = "*"  + keys[j]
    pass
    parameters = ", ".join(keys)

    # Now create the forward function!
    definition = re.findall(r"[\s\n]{1,}def[^\(]{1,}\([^\)]{1,}\)[^\:]{0,}\:", old_source, flags = re.MULTILINE)[0]
    leftover = full_class[full_class.find(definition) + len(definition):]

    # Add **loss_kwargs
    if add_loss_kwargs and "**" not in parameters:
        parameters += ", **loss_kwargs"
        definition = re.sub(r"(\,[\n][\s]{1,}\))", r",**loss_kwargs\1", definition)
        source = re.sub(r"(\,[\n][\s]{1,}\) \-\>)", r",**loss_kwargs\1", source)
    pass

    source = f"@{compile}\n{source}\n"

    left = re.match("[\s\n]{4,}", leftover).span()[1]
    new_forward = definition + leftover[:left] + \
        f"return {module}_forward({parameters})\n"
    full_class = full_class.replace(old_source, new_forward)

    # Combine all into file
    source = source + full_class
    return source
pass


# Replace Cross Entropy cells with fused linear lm heads
cross_entropy_find_1 = """
logits = self.lm_head(hidden_states%
loss = None
if labels is not None:$logits = logits.float()
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss_fct = CrossEntropyLoss()
shift_logits = shift_logits.view(-1, self.config.vocab_size)
shift_labels = shift_labels.view(-1)
shift_labels = shift_labels.to(shift_logits.device)
loss = loss_fct(shift_logits, shift_labels)
"""

cross_entropy_replacement_1 = """
n_items = loss_kwargs.get("num_items_in_batch", None) or loss_kwargs.get("n_items", None)
loss = fused_linear_cross_entropy(
    hidden_states      = hidden_states,
    lm_weight          = self.lm_head.weight,
    labels             = labels,
    num_items_in_batch = n_items,
    logit_softcapping  = getattr(self.config, "final_logit_softcapping", 0),
)
"""

cross_entropy_find_2 = """
logits = self.lm_head(hidden_states%
loss = None
if labels is not None:$loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
"""

cross_entropy_replacement_2 = """
if self.training and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
    n_items = loss_kwargs.get("num_items_in_batch", None) or loss_kwargs.get("n_items", None)
    loss = fused_linear_cross_entropy(
        hidden_states      = hidden_states,
        lm_weight          = lm_head.weight,
        labels             = labels,
        num_items_in_batch = n_items,
        logit_softcapping  = getattr(self.config, "final_logit_softcapping", 0),
    )
else:
    logits = self.lm_head(hidden_states)
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
"""

ce_finders = [
    (cross_entropy_find_1, cross_entropy_replacement_1,),
    (cross_entropy_find_2, cross_entropy_replacement_2,),
]


def apply_fused_lm_head(forward):
    for cross_entropy_find, cross_entropy_replacement in ce_finders:
        cross_entropy_find = cross_entropy_find.strip()\
            .replace("*", "\*").replace("^", "\^")\
            .replace("-", "\-").replace("_", "\_")\
            .replace(":", "\:").replace("+", "\+")\
            .replace(".", "\.").replace(",", "\,")\
            .replace("(", "\(").replace(")", "\)")\
            .replace("[", "\[").replace("]", "\]")\
            .replace("\n", r"[\s\n]{1,}(?:\#[^\n]{1,}[\n][\s\n]{1,})?")

        # Find indentation
        cross_entropy_find = cross_entropy_find\
            .replace("$", r"[\n]([\s]{1,})(?:\#[^\n]{1,}[\n][\s\n]{1,})?")\
            .replace("%", r"(?:\[\:\,[\s]{0,}\-num_logits_to_keep\:\,[\s]{0,}\:\])?\)")

        spaces = re.findall(cross_entropy_find, forward, flags = re.DOTALL | re.MULTILINE)
        if len(spaces) == 0: continue
        spaces = spaces[0]

        replacement = cross_entropy_replacement.strip().split("\n")
        replacement = "\n".join(spaces + x for x in replacement)
        replacement = \
            "logits = None\n" + \
            (len(spaces)-4)*" " + "loss = None\n" + \
            (len(spaces)-4)*" " + "if labels is not None:\n" + \
            replacement

        forward = re.sub(
            cross_entropy_find,
            replacement,
            forward,
            flags = re.DOTALL | re.MULTILINE,
        )
    pass
    return forward
pass


def check_nvidia():
    # Unsloth doesn't work yet on AMD devices - we're working on it!
    output = np.array([0,])
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv", shell = True)
        output = re.findall(rb'([\d]{1,})[\s]{1,}M', output)
        output = np.array([int(x.decode('utf-8'))/1024 for x in output])
    except:
        if not torch.cuda.is_available():
            raise RuntimeError("Unsloth: We do not support AMD / Intel machines yet - it is a work in progress!")    
    return output
pass
PRE_CHECK = check_nvidia()


# Patch remaining functions
def convert_attention_masks_to_bool(module, old_source):
    # Code licensed under LGPL
    # Convert attention mask creation functions to boolean
    source = re.sub(r"\([\s]{0,}", "(", old_source)
    source = re.sub(r"[\s]{0,}\)", ")", source)
    all_splits = source.strip().split("\n")
    splits = all_splits[-1].strip()
    if "return" not in splits: return old_source
    vars = re.findall(r"return[\s]{1,}(?:([^\,]{1,})\,[\s]{0,}){0,}([^\s]{1,})", splits)
    if len(vars) != 1: return old_source
    vars = vars[0]

    good_vars = []
    for var in vars:
        for split in all_splits:
            if re.search(re.escape(var) + ".+?" + r"torch\.finfo\(.+?\)\.min", split):
                good_vars.append(var)
    pass
    if len(good_vars) == 0: return old_source
    good_vars = set(good_vars)
    final = all_splits[-1]
    for var in good_vars:
        if len(var) == 0: continue
        final = final.replace(var, var + f"!=torch.finfo({var}.dtype).min")
    pass
    all_splits[-1] = final
    new_source = "\n".join(all_splits)
    print(f"Unsloth: Boolean mask for {module}")
    return new_source
pass


def unsloth_compile_transformers(
    model_type           : str = "llama",
    sdpa_causal_only     : bool = False,
    sdap_bool_masks      : bool = True,
    sdpa_gqa_replace     : bool = True,
    sdpa_disable_compile : bool = True,
    disable_causal_masks : bool = True,
    import_from_cache    : bool = False,
    compile_functions    : bool = True,
    fuse_lm_head         : bool = True,
):
    # Code licensed under LGPL
    model_location = f"transformers.models.{model_type}.modeling_{model_type}"
    exec(f"import {model_location}", globals())
    modeling_file = eval(model_location)
    if hasattr(modeling_file, "__UNSLOTH_PATCHED__"): return

    # torch_compile_options
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    torch_compile_options = {
        "epilogue_fusion"   : True,
        "max_autotune"      : True,
        "shape_padding"     : True,
        "trace.enabled"     : UNSLOTH_COMPILE_DEBUG,
        "triton.cudagraphs" : False,
    }

    modeling_file.__UNSLOTH_PATCHED__ = True
    functions = dir(modeling_file)
    full_source = inspect.getsource(modeling_file)
    # Order functions by ascending order
    functions = list(np.array(functions)[np.argsort([full_source.find(x) for x in functions])])
    ordered_functions = functions.copy()

    # Get class LlamaAttention(nn.Module)
    torch_modules = re.findall(r"class ([^\s]{1,})\(.+?\.Module\)", full_source)
    # Also get class LlamaSdpaAttention(LlamaAttention)
    inherited_class = "(?:" + "|".join(re.findall(r"class ([^\s]{1,})\(.+?\.Module\)", full_source)) + ")"
    inherited_modules = re.findall(r"class ([^\s]{1,})\(" + inherited_class + "\)", full_source)
    # OrderedSet
    torch_modules = list(dict.fromkeys(torch_modules + inherited_modules))
    # Get all functions as well
    functions = [x for x in functions if x not in torch_modules]

    # Remove if no forward function
    final_torch_modules = []
    for module in torch_modules:
        source = eval(f"modeling_file.{module}")
        if hasattr(source, "forward"): final_torch_modules.append(module)
    pass
    torch_modules = final_torch_modules

    # Remove functions which have gradient checkpointing in them
    # Also check if it's an attention module
    gradient_checkpointed_modules = []
    scaled_dot_product_attention_modules = []
    full_attention_modules = []
    for module in torch_modules:
        source = eval(f"modeling_file.{module}")
        try: source = inspect.getsource(source)
        except: continue
        if "_gradient_checkpointing_func" in source:
            gradient_checkpointed_modules.append(module)
        elif "scaled_dot_product_attention" in source:
            scaled_dot_product_attention_modules.append(module)
        elif "nn.functional.softmax" in source or "flash_attn_varlen_func" in source or "_flash_attention_forward" in source:
            full_attention_modules.append(module)
    pass
    removal = set(
        scaled_dot_product_attention_modules + \
        full_attention_modules + \
        gradient_checkpointed_modules
    )
    torch_modules = [x for x in torch_modules if x not in removal]

    # Get functions which are called
    called_functions = []
    for function in functions:
        # Start of text
        defined = re.findall(r"\bdef[\s]{1,}" + re.escape(function),full_source, flags = re.DOTALL)
        # Disable self.
        called = re.findall(r"[\s]{1,}" + re.escape(function) + "\(.+?\)", full_source, flags = re.DOTALL)
        if len(defined) != 0 and len(called) != 0:
            called_functions.append(function)
    pass

    # Check if fullgraph can be used
    torch_modules = {x : True for x in torch_modules}
    for module in torch_modules.keys():
        source = eval(f"modeling_file.{module}")
        try: source = inspect.getsource(source.__init__)
        except: continue
        fullgraph = not ("nn.Linear" in source or "nn.ModuleList" in source)

        # Check if other modules is used as well
        for another_module in torch_modules:
            if another_module in source:
                fullgraph = fullgraph and torch_modules[another_module]
        pass
        torch_modules[module] = fullgraph
    pass

    # Get other classes
    other_classes = re.findall(r"class ([^\s]{1,})\(.+?\)", full_source)
    other_classes = [x for x in other_classes if x not in torch_modules and x not in removal]

    # Fix scaled dot product attention up if possible
    scaled_dot_product_attention_modules = {x:None for x in scaled_dot_product_attention_modules}
    disabled_scaled_dot_product_attention_modules = []

    for module in scaled_dot_product_attention_modules.keys():
        source = eval(f"{model_location}.{module}")
        try: source = inspect.getsource(source.forward)
        except: continue

        causal_mask_find = \
            r"(is_causal \= True if (.+?\_mask) is None and q_len \> 1 else False[\n\s]{1,})"\
            r"([A-Za-z0-9\_]{1,}[\s]{1,}\=[\s]{1,}[A-Za-z\.]{1,}scaled\_dot\_product\_attention)"\
            r"(.+?attn\_mask[\s]{0,}\=[\s]{0,})\2"\
            r"(.+?is\_causal[\s]{0,}\=[\s]{0,})is\_causal"

        scaled_dot_product_attention_find = \
            r"(\=[\s]{1,}[A-Za-z\.]{1,}scaled\_dot\_product\_attention)"

        new_source = source
        if sdpa_causal_only:
            if len(re.findall(causal_mask_find, source, flags = re.DOTALL)) == 1:
                new_source = re.sub(
                    causal_mask_find,
                    r"\1\3\4None\5True",
                    source,
                    flags = re.DOTALL,
                )
                new_source = source
            else:
                new_source = re.sub(
                    scaled_dot_product_attention_find,
                    "= disable_compile_scaled_dot_product_attention",
                    source,
                    flags = re.DOTALL,
                )
                disabled_scaled_dot_product_attention_modules.append(module)
        else:
            new_source = re.sub(
                r"if output_attentions\:.+?return super\(\)\.forward.+?\)",
                "if output_attentions: raise RuntimeError('Unsloth: Not supported')",
                new_source,
                flags = re.DOTALL | re.MULTILINE,
            )
        pass
        scaled_dot_product_attention_modules[module] = new_source
    pass

    # Fix modules with _update_causal_mask if SDPA can be used with causal masks
    remove_causal_masks = []
    if disable_causal_masks:
        for module in other_classes:
            source = eval(f"{model_location}.{module}")
            if not hasattr(source, "_update_causal_mask"): continue

            try: source = inspect.getsource(source.__init__)
            except: continue

            can_remove = True
            for x in disabled_scaled_dot_product_attention_modules:
                if x in source:
                    can_remove = False
                    break
            pass
            if can_remove: remove_causal_masks.append(module)
        pass
    pass

    # Remove modules which have attention mechanisms
    # since torch.compile will compile too many kernels
    bad_torch_modules = set()
    for module, fullgraph in torch_modules.items():
        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "forward"): continue
        try: source = inspect.getsource(source.forward)
        except: continue

        if "attn_weights" in source or "self.self_attn" in source:
            print(f"Unsloth: Will not compile {module}.")
            bad_torch_modules.add(module)
        pass

        # Check if creating arrays in inside the function
        # Error: DataDependentOutputException: aten._local_scalar_dense.default
        if "torch.arange(" in source or "torch.zeros(" in source or "torch.ones(" in source:
            print(f"Unsloth: Failed compiling function {module} since array creations are done.")
            bad_torch_modules.add(module)
        pass
    pass

    # Now patch modules
    all_standalone_classes = {}
    for module, fullgraph in torch_modules.items():
        if module in bad_torch_modules: continue
        try:
            new_module = create_standalone_class(
                module,
                model_location,
                functions,
                fullgraph = fullgraph,
            )
            print(f"Unsloth: Compiled module {module}.")
            all_standalone_classes[module] = new_module
        except:
            continue
    pass

    # SDPA
    for module, forward_source in scaled_dot_product_attention_modules.items():
        if sdpa_gqa_replace:
            forward_source = replace_with_grouped_query_attention(
                module,
                forward_source,
            )
        pass
        try:
            new_module = create_standalone_class(
                module,
                model_location,
                functions,
                fullgraph = fullgraph,
                disable = sdpa_disable_compile,
                forward_source = forward_source,
            )
            print(f"Unsloth: Fast Attention patch for {module}.")
            all_standalone_classes[module] = new_module
        except:
            continue
    pass

    # Patch full attention modules
    for module in full_attention_modules:
        try:
            new_module = create_standalone_class(
                module,
                model_location,
                functions,
                fullgraph = False,
                disable = True,
            )
            print(f"Unsloth: Slow Attention patch for {module}.")
            all_standalone_classes[module] = new_module
        except:
            continue
    pass

    # Remove causal masks
    for module in remove_causal_masks:
        source = eval(f"{model_location}.{module}")
        if not hasattr(source, "_update_causal_mask"): continue

        exec(f"{model_location}.{module}._update_causal_mask = no_update_causal_mask", globals())
        print(f"Unsloth: Removed causal mask for {module} to reduce memory usage.")
    pass

    # Patch LM Head
    if fuse_lm_head:
        from transformers.generation import GenerationMixin
        modules = dir(modeling_file)

        for module in modules:
            module_class = eval(f"modeling_file.{module}")
            if hasattr(module_class, "forward") and issubclass(module_class, GenerationMixin):
                try:
                    source = inspect.getsource(module_class.forward)
                except:
                    continue
                new_source = apply_fused_lm_head(source)
                if new_source != source:
                    new_module = create_standalone_class(
                        module,
                        model_location,
                        functions,
                        fullgraph = False,
                        disable = True,
                        forward_source = new_source,
                        add_loss_kwargs = True,
                    )
                    print(f"Unsloth: Fast fused linear cross entropy patch for {module}.")
                    all_standalone_classes[module] = new_module
                pass
            pass
        pass
    pass

    # Patch Trainer
    from transformers.trainer import Trainer
    try:
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
            Trainer._original_training_loop = inner_training_loop
        else:
            inner_training_loop = Trainer._original_training_loop
    except:
        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
    pass

    import transformers.trainer
    items_in_trainer = dir(transformers.trainer)
    good_items = []
    for item in items_in_trainer:
        # TODO: Support Deepspeed
        if item.startswith(("deepspeed", "xm", "met", "smp")): continue
        if item in inner_training_loop: good_items.append(item)
    pass
    exec("from transformers.trainer import (" + ", ".join(x for x in good_items) + ")", globals())

    start = re.search('logger\.info\([\"\'].+?Running training', inner_training_loop).span(0)[0]
    end = inner_training_loop.find("\n\n", start)
    original_debug = inner_training_loop[start:end]
    spaces = re.search('\n([\s\t]{1,})', original_debug).group(0)[1:]
    front_spaces = re.match('([\s\t]{1,})', inner_training_loop).group(0)

    debug_info = """debug_info = \\
        f"==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = {args.world_size}\\n"\\
        f"   \\\\\\   /|    Num examples = {num_examples:,} | Num Epochs = {num_train_epochs:,}\\n"\\
        f"O^O/ \\_/ \\    Batch size per device = {self._train_batch_size:,} | Gradient Accumulation steps = {args.gradient_accumulation_steps}\\n"\\
        f"\\        /    Total batch size = {total_train_batch_size:,} | Total steps = {max_steps:,}\\n"\\
        f' "-____-"     Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}\\n'\\
        f"🦥 Unsloth's automatic compiler will take 1-2 minutes to process - please wait!"
        logger.warning(debug_info)
        import subprocess, re, gc, numpy as np
        a = np.array([0,])
        try:
            a = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell = True)
            a = re.findall(rb'([\\d]{1,})[\\s]{1,}M', a)
            a = np.array([int(x.decode('utf-8'))/1024 for x in a])
        except:
            if not torch.cuda.is_available():
                raise RuntimeError('Unsloth: We do not support AMD / Intel machines yet - it is a work in progress!')
        if ((a - PRE_CHECK) >= 1).sum() > 1:
            raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()"""

    debug_info = debug_info.split('\n')
    debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
    inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

    debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once('Unsloth currently does not support multi GPU setups - but we are working on it!')
        debug_info ="""
    debug_info = debug_info.split('\n')
    debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
    inner_training_loop = inner_training_loop.replace("debug_info =", debug_info, 1)

    front_spaces = re.match(r"[\t\s]{1,}", inner_training_loop).group(0)
    inner_training_loop = re.sub(r"^" + front_spaces, "", inner_training_loop, flags = re.MULTILINE)
    inner_training_loop = inner_training_loop.replace(
        "train_dataloader = tpu_spmd_dataloader(train_dataloader)",
        "raise RuntimeError('Unsloth: TPUs are not yet supported!')"
    )
    inner_training_loop = inner_training_loop.replace(
        "self.accelerator.free_memory()",
        "self.accelerator.free_memory()\n" + \
        front_spaces + "if self.is_deepspeed_enabled:"\
        "raise RuntimeError('Unsloth: Deepspeed is not yet supported!')\n", 1,
    )

    check_batches = """train_dataloader = self.get_train_dataloader()
        ga  = args.gradient_accumulation_steps
        bsz = self._train_batch_size
        total_batches = bsz * ga * args.world_size
        n_total_devices = total_batches // ga // bsz
        if n_total_devices > 1:
            logger.warning_once('Unsloth currently does not support multi GPU setups - but we are working on it!')
            divisor = n_total_devices / 1
            bsz = self._train_batch_size = max(int(bsz / divisor), 1)
            if total_batches // ga // bsz > 1:
                divisor = n_total_devices / 1
                ga = args.gradient_accumulation_steps = max(int(ga / divisor), 1)"""
    check_batches = check_batches.split('\n')
    check_batches = "\n".join([check_batches[0]] + [front_spaces + x[8:] for x in check_batches[1:]])
    inner_training_loop = inner_training_loop.replace(
        "train_dataloader = self.get_train_dataloader()",
        check_batches, 1,
    )
    inner_training_loop = inner_training_loop.replace(
        "_inner_training_loop",
        "_fast_inner_training_loop", 1,
    )
    exec(inner_training_loop, globals())

    Trainer._inner_training_loop = _fast_inner_training_loop
    inner_training_loop = inner_training_loop.replace(
        "is_torch_tpu_available()",
        "False",
    )
    if "n_total_devices >" not in inner_training_loop:
        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
    pass
    inner_training_loop = inner_training_loop.replace(
        "is_sagemaker_mp_enabled()",
        "False",
    )
    exec(inner_training_loop, globals())
    Trainer._inner_training_loop = _fast_inner_training_loop

    # Fix up function signatures
    for module in called_functions:
        function = eval(f"{model_location}.{module}")

        parameters = inspect.signature(function)
        params = list(parameters.parameters.keys())
        source = inspect.getsource(function)

        where = source.find(str(parameters))
        if where == -1: where = source.find("\n") + 1
        else: where = where + len(str(parameters))
        code_section = source[where:]
        cleaned_code_section = re.sub(r'\"\"\".+?\"\"\"', "", code_section, flags = re.DOTALL)

        bad_params = []
        for param in params:
            if not param in cleaned_code_section:
                bad_params.append(param)
        pass
        if len(bad_params) == 0: continue

        for bad_param in bad_params:
            parameters = re.sub(
                re.escape(bad_param) + r"[\s]{0,}\=[\s]{0,}None[\s]{0,}\,",
                "", # Remove them entirely
                str(parameters),
                flags = re.DOTALL,
            )
        pass
        parameters = f"def {module}" + parameters + code_section
        print(f"Unsloth: Fixed up function {module}.")

        parameters = \
            f"@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n{parameters}"
        all_standalone_classes[module] = parameters
    pass

    # All other functions
    for module in called_functions:
        if module in all_standalone_classes: continue
        function = eval(f"{model_location}.{module}")
        source = inspect.getsource(function)

        if sdap_bool_masks:
            source = convert_attention_masks_to_bool(module, source)

        source = f"@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)\n{source}"
        all_standalone_classes[module] = source
        print(f"Unsloth: Compiled function {module}.")
    pass

    # Order all components
    final_all_standalone_classes = []
    for module in ordered_functions:
        if module in all_standalone_classes:
            final_all_standalone_classes.append(all_standalone_classes[module])
        pass
    pass

    all_code = "\n\n".join(final_all_standalone_classes)

    if import_from_cache:
        try:
            combined_module = importlib.import_module(f"{UNSLOTH_COMPILE_LOCATION}.{COMBINED_UNSLOTH_NAME}__{model_type}")
            import_from_cache = True
        except:
            import_from_cache = False
    else:
        import_from_cache = False
    pass
    if not import_from_cache:
        combined_module = create_new_function(
            f"{COMBINED_UNSLOTH_NAME}_{model_type}",
            all_code,
            model_location,
            functions,
            prepend = \
                _disabled_sdpa_code + \
                f"\ntorch_compile_options = {torch_compile_options}\n"
        )
    pass

    if compile_functions:
        for module in _patch_functions:
            try: source = eval(f"{model_location}.torch")
            except: continue
            if not hasattr(source, "nn"): continue
            if not hasattr(source.nn, module): continue
            function = eval(f"source.nn.{module}")
            if not hasattr(function, "forward"): continue
            if hasattr(function.forward, "get_compiler_config"): continue

            source = inspect.getsource(function.forward).rstrip()
            forward = create_new_function(module, source, model_location, functions, append = ".to(input.dtype)\n", overwrite = False).forward
            exec(f"{model_location}.torch.nn.{module}.forward = forward", globals(), locals())
            try:  exec(f"{model_location}.nn.{module}.forward = forward", globals(), locals())
            except: pass
            exec( f"combined_module.torch.nn.{module}.forward = forward", globals(), locals())
            try:  exec( f"combined_module.nn.{module}.forward = forward", globals(), locals())
            except: pass
        pass
    pass

    # Import and replace with new module
    for module in all_standalone_classes.keys():
        exec(f"{model_location}.{module} = combined_module.{module}", globals(), locals())
    pass

    # Finally edit dictionary items inside the target file
    replaced_classes = all_standalone_classes.keys()
    check_dicts = dir(eval(f"{model_location}"))
    for check in check_dicts:
        item = eval(f"{model_location}.{check}")
        if type(item) is not dict: continue

        for key, value in item.items():
            value = str(value)
            found = False
            for replaced_class in replaced_classes:
                if replaced_class in value:
                    exec(f"{model_location}.{check}['{key}'] = combined_module.{replaced_class}", globals(), locals())
                    # print(f"Unsloth: Replacing {check} with {replaced_class}")
                    break
                pass
            pass
        pass
    pass
    return
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
