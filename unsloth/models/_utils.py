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

__version__ = "2024.11.7"

__all__ = [
    "prepare_model_for_kbit_training",
    "xformers",
    "xformers_attention",
    "xformers_version",
    "__version__",
    "HAS_FLASH_ATTENTION",
    "HAS_FLASH_ATTENTION_SOFTCAPPING",
    "PRE_CHECK",
    "platform_system",
    "patch_tokenizer",
    "get_statistics",
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "offload_to_disk",
    "offload_input_embeddings",
    "offload_output_embeddings",
    "is_bfloat16_supported",
    "unsloth_offloaded_gradient_checkpoint",
    "torch_compile_options",
    "patch_linear_scaling",
    "patch_llama_rope_scaling",
    "check_nvidia",
    "create_boolean_mask",
    "torch_amp_custom_fwd",
    "torch_amp_custom_bwd",
    "accelerate_old_send_to_device",
    "accelerate_new_send_to_device",
    "patch_gradient_accumulation_fix",
    "patch_compiling_bitsandbytes",
    "patch_regional_compilation",
    "patch_layernorm",
    "patch_torch_compile",
    "patch_model_and_tokenizer",

    "patch_unsloth_gradient_checkpointing",
    "unpatch_unsloth_gradient_checkpointing",
    "patch_gradient_checkpointing",
    "unpatch_gradient_checkpointing",
]

import torch
from typing import Union, Optional, List, Any, Callable, Tuple
from platform import system as platform_system
platform_system = platform_system()
import numpy as np
import warnings, subprocess, re, inspect, psutil, os, math
from packaging.version import Version

from unsloth_zoo.tokenizer_utils import (
    patch_tokenizer as _patch_tokenizer,
)
from unsloth_zoo.patching_utils import (
    patch_compiling_bitsandbytes,
    patch_layernorm,
    patch_torch_compile,
    patch_model_and_tokenizer,
)
from unsloth_zoo.gradient_checkpointing import (
    Unsloth_Offloaded_Gradient_Checkpointer,
    unsloth_offloaded_gradient_checkpoint,
    patch_unsloth_gradient_checkpointing,
    unpatch_unsloth_gradient_checkpointing,

    Unsloth_Gradient_Checkpointer,
    unsloth_gradient_checkpoint,
    patch_gradient_checkpointing,
    unpatch_gradient_checkpointing,
)

# =============================================
# Disable some warnings which can get annoying
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "torch")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "trl")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "trl")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "xformers")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")
warnings.filterwarnings(action = "ignore", category = UserWarning,    module = "transformers")
warnings.filterwarnings(action = "ignore", category = FutureWarning,  module = "accelerate")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocessing")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocess")

# Stop "Special tokens have been added in the vocabulary, ..."
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL+1)

# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    def __init__(self, text): self.text = text
    def filter(self, x): return not (self.text in x.getMessage())
pass

# The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here.
from transformers.training_args import logger as transformers_training_args_logger
transformers_training_args_logger.addFilter(HideLoggingMessage("The speedups"))
del transformers_training_args_logger

# Using the default loss: `ForCausalLMLoss`.
try:
    from transformers.modeling_utils import logger as transformers_modeling_utils_logger
    transformers_modeling_utils_logger.addFilter(HideLoggingMessage("ForCausalLMLoss"))
    del transformers_modeling_utils_logger
except:
    pass

# =============================================

# =============================================
# Edits all Config files to enable RoPE Scaling for all models

# Transformers had to update for Mistral Nemo 12b since Attention is (5120, 4096) now.
def patch_mistral_nemo_config(config):
    if "head_dim (" not in config:
        add_head_dim = "If it is not specified, will default to `8`.\n"\
            "        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):\n"\
            "            The attention head dimension."
        config = config.replace("If it is not specified, will default to `8`.", add_head_dim)

        add_head_dim = "num_key_value_heads=8,\n        head_dim=None,"
        config = config.replace("num_key_value_heads=8,", add_head_dim)

        add_head_dim = "self.sliding_window = sliding_window\n        self.head_dim = head_dim or hidden_size // num_attention_heads\n"
        config = config.replace("self.sliding_window = sliding_window", add_head_dim)
    pass
    return config
pass

from transformers import __version__ as transformers_version
from transformers import PretrainedConfig
model_architectures = ["llama", "mistral", "gemma", "gemma2", "qwen2",]

for model_name in model_architectures:
    config_filepath = f"transformers.models.{model_name}.configuration_{model_name}"
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    config_filename = f"{model_name.title()}Config"
    exec(f"from {config_filepath} import {config_filename}", globals())

    try:
        config = inspect.getsource(eval(config_filename))
    except:
        continue
    if "rope_scaling" in config: continue
    config = re.sub(
        r"(\*\*kwargs)[\s]{0,}\,[\s]{0,}\)[\s]{0,}\:",
        r"rope_scaling=None,"\
        r"\n        **kwargs):\n"\
        r"\n        self.rope_scaling = rope_scaling\n",
        config,
    )

    # Just for Mistral Nemo
    if model_name == "mistral":
        if Version(transformers_version) <= Version("4.42.4"):
            config = patch_mistral_nemo_config(config)
    pass

    exec(config, globals())
    exec(f"import {config_filepath}", globals())
    exec(f"{config_filepath}.{config_filename} = {config_filename}", globals())
pass
# =============================================

# =============================================
# torch.cuda.amp.custom_fwd is deprecated >= 2.4
torch_version = torch.__version__
if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass
# =============================================

# =============================================
# Fix KeyError: 'Cache only has 0 layers, attempted to access layer with index 0'
import transformers.cache_utils
if hasattr(transformers.cache_utils, "DynamicCache") and \
    transformers.cache_utils.DynamicCache.__getitem__.__name__ != "__cache_utils_getitem__":

    source = inspect.getsource(transformers.cache_utils.DynamicCache.__getitem__)
    start = source.find("def")
    spaces = start*" "
    source = source.split("\n")
    source = "\n".join(x[start:] for x in source)
    where = source.find("raise KeyError")
    source = source[:where] + \
        f"if len(self) == 0:\n{spaces}{spaces}"\
        "    raise RuntimeError('Unsloth: You must call `FastLanguageModel.for_inference(model)` before doing inference for Unsloth models.')\n" + \
        f"{spaces}{spaces}else:\n{spaces}{spaces}{spaces}" + source[where:]
    source = source.replace("__getitem__", "__cache_utils_getitem__", 1)
    exec(source)
    transformers.cache_utils.DynamicCache.__getitem__ = __cache_utils_getitem__
pass
# =============================================

# =============================================
# Weird Databricks errors
from transformers.utils import is_openai_available
if is_openai_available():
    try:
        from openai import OpenAI
    except:
        print("Unsloth: OpenAI failed to import - ignoring for now.")
        import transformers.utils
        def _is_openai_available(): return False
        transformers.utils.is_openai_available = _is_openai_available
    pass
pass 

# =============================================
# Get Flash Attention v2 if Ampere (RTX 30xx, A100)
import bitsandbytes as bnb
from transformers import AutoTokenizer
from transformers.utils.import_utils import _is_package_available

major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = False
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False

if major_version >= 8:
    SUPPORTS_BFLOAT16 = True
    if _is_package_available("flash_attn"):
        # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
        try:
            from flash_attn.flash_attn_interface import flash_attn_cuda
            HAS_FLASH_ATTENTION = True

            # Also check for softcapping
            from flash_attn import __version__ as flash_attn_version
            HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
            if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
        except:
            print(
                "Unsloth: Your Flash Attention 2 installation seems to be broken?\n"\
                "A possible explanation is you have a new CUDA version which isn't\n"\
                "yet compatible with FA2? Please file a ticket to Unsloth or FA2.\n"\
                "We shall now use Xformers instead, which does not have any performance hits!\n"\
                "We found this negligible impact by benchmarking on 1x A100."
            )

            # Stop Flash Attention from importing!
            import transformers.utils.import_utils
            transformers.utils.import_utils.is_flash_attn_2_available = lambda *args, **kwargs: False
            import transformers.utils
            transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

            HAS_FLASH_ATTENTION = False
        pass
    else:
        HAS_FLASH_ATTENTION = False
else:
    # Tri Dao's benchmark shows xformers is faster for now.
    HAS_FLASH_ATTENTION = False
pass

from transformers.models.llama.modeling_llama import logger

# =============================================
# Get Xformers
from xformers import __version__ as xformers_version
# Temporarily disable 0.0.27 and higher - inference issues
if False: #Version(xformers_version) >= Version("0.0.27"):
    raise ImportError(
        "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "\
        "then press Disconnect Runtime and then Restart it.\n"\
        "\n"\
        "%%capture\n"
        "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
        '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'\
        '\n'\
        f"Otherwise in local machines, your xformers version of {xformers_version} is too new.\n"\
        'Please downgrade xformers via `pip install --force-reinstall "xformers<=0.0.27"'
    )
pass

if   Version(torch_version) < Version("2.2.0") and Version(xformers_version) >= Version("0.0.24"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers < 0.0.24 for torch = {torch_version}."
    )
elif Version(torch_version) < Version("2.3.0") and Version(xformers_version) >= Version("0.0.26"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers < 0.0.26 for torch = {torch_version}."
    )
elif Version(torch_version) < Version("2.4.0") and Version(xformers_version) > Version("0.0.27"):
    raise ImportError(
        f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"\
        f"Please install xformers <= 0.0.27 for torch = {torch_version}."
    )
pass

from xformers._cpp_lib import _register_extensions
try:
    _register_extensions() # Check if C++ modules are loaded correctly
except Exception as error:
    raise ImportError(
        "Unsloth: Xformers was not installed correctly.\n"\
        "Please install xformers separately first.\n"\
        "Then confirm if it's correctly installed by running:\n"\
        "python -m xformers.info\n\n"
        "Longer error message:\n" + str(error)
    )
pass
import xformers.ops.fmha as xformers
xformers_attention = xformers.memory_efficient_attention

# Check TRL version
from trl import __version__ as trl_version
# Unsloth now supports all TRL versions!
if False:#Version(trl_version) >= Version("0.9.0"):
    raise ImportError(
        "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "\
        "then press Disconnect Runtime and then Restart it.\n"\
        "\n"\
        "%%capture\n"
        "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
        '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'\
        '\n'\
        f"Otherwise in local machines, your TRL version of {trl_version} is too new.\n"\
        'Please downgrade TRL via `pip install --force-reinstall trl'
    )
pass

# =============================================
# Fix new Xformers versions TypeError: Multiple dispatch failed for 'torch._ops.aten.to.dtype_layout'
accelerate_old_send_to_device = None
accelerate_new_send_to_device = None
if Version(xformers_version) >= Version("0.0.27"):
    import accelerate.utils.operations
    if hasattr(accelerate.utils.operations, "send_to_device") and \
        accelerate.utils.operations.send_to_device.__name__ != "_fixed_send_to_device":
        accelerate_old_send_to_device = accelerate.utils.operations.send_to_device
        from accelerate.utils.operations import *
        send_to_device = inspect.getsource(accelerate.utils.operations.send_to_device)
        send_to_device = re.sub(
            r"([ ]{4,})return tensor\.to\(device\)",
            r"\1try: return tensor.to(device)\n\1except: return tensor",
            send_to_device,
        ).replace("def send_to_device", "def _fixed_send_to_device")
        exec(send_to_device)
        # accelerate.utils.operations.send_to_device = _fixed_send_to_device
        accelerate_new_send_to_device = _fixed_send_to_device
    pass
pass

# Transformers 4.46 breaks dynamic caching. This is a hack
import transformers.generation.configuration_utils
if hasattr(transformers.generation.configuration_utils, "ALL_CACHE_IMPLEMENTATIONS"):
    if type(transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS) is list:
        transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS.append("dynamic")
    pass
pass
# =============================================

# =============================================
# Torch compile settings
UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "1") == "1"
# Just remove max_autotune_gemm warning
import functools
@functools.lru_cache(None)
def is_big_gpu(index):
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    if sms < 80:  # V100
        # log.warning("not enough SMs to use max_autotune_gemm mode")
        return False
    return True
import torch._inductor.utils
torch._inductor.utils.is_big_gpu = is_big_gpu
patch_torch_compile(
    debug = UNSLOTH_COMPILE_DEBUG,
    O3 = UNSLOTH_COMPILE_MAXIMUM,
    ignore_errors = UNSLOTH_COMPILE_IGNORE_ERRORS,
)

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : UNSLOTH_COMPILE_DEBUG,
    "triton.cudagraphs" : False,
}

import accelerate
def torch_compile_kwargs(*args, **kwargs):
    print("Unsloth: Enabled auto compiling")
    return {"dynamic" : True, "fullgraph" : False, "options" : torch_compile_options,}
pass

accelerate.utils.dataclasses.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
accelerate.utils.TorchDynamoPlugin.to_kwargs             = torch_compile_kwargs
accelerate.accelerator.TorchDynamoPlugin.to_kwargs       = torch_compile_kwargs
del accelerate

def patch_regional_compilation():
    # Regional torch 2.5 Recompilation - weirdly very slow??
    if torch.nn.ModuleList.__name__ == "UnslothModuleList": return
    # Only works for torch 2.5
    if Version(torch.__version__) < Version("2.5.0"): return

    old_module_list = torch.nn.ModuleList
    os.environ["UNSLOTH_PATCHED"] = "1"

    def UnslothModuleList(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) is list:
            args = [old_module_list([torch.compile(x, dynamic = True, options = torch_compile_options, fullgraph = False) for x in args[0]])]
        return old_module_list(*args, **kwargs)
    pass
    UnslothModuleList.__doc__ = old_module_list.__doc__

    torch.nn.ModuleList = UnslothModuleList
    return
pass

# =============================================

def prepare_model_for_kbit_training(
    model                      : Any,
    use_gradient_checkpointing : Optional = True,
    use_reentrant              : Optional[bool] = True,
) -> Any:
    """
    Calculates where to place the gradient checkpoints given n_layers.
    We also freeze all other layers's gradients

    Args:
        model: Any LlamaModel with layers.
        use_gradient_checkpointing (`bool`, *optional*):
            Default enabled. Provides memory savings by not saving all activations,
            but only some.
        use_reentrant (`bool`, *optional*):
            https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L354
            Optimal gradient checkpointing algorithm which will be the default in
            future Pytorch versions.
    """

    # Freeze all parameters except LoRA
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
                param.requires_grad_(True)
                # Also must be in float32!
                if param.dtype != torch.float32:
                    name = name.replace("base_model", "model", 1)
                    layer_number = re.search(r"\.[\d]{1,}\.", name).group(0)
                    name = name.replace(layer_number, f"[{layer_number[1:-1]}].")
                    name = name.replace(".weight", "", 1)
                    exec(f"{name}.to(torch.float32)")
                pass
            else:
                param.requires_grad_(False)
        pass
    pass

    # Gradient checkpointing!
    if use_gradient_checkpointing == "unsloth":

        # Saves VRAM!
        original_model = model
        while hasattr(original_model, "model"):
            original_model._offloaded_gradient_checkpointing = True
            original_model = original_model.model
        pass
        original_model._offloaded_gradient_checkpointing = True
        
        model.gradient_checkpointing_enable()

    elif use_gradient_checkpointing == True:
        model.gradient_checkpointing_enable()
    pass

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model
pass

# =============================================
# Weirdly LoraLayer.update_layer downcasts PEFT layers to float16??
# For mixed precision, we need it to be in float32 not float16.
from peft import __version__ as peft_version
if Version(peft_version) < Version("0.12.0"):
    from peft.tuners.lora.layer import LoraLayer
    try:
        source = inspect.getsource(LoraLayer.update_layer)
        text = "if weight is not None:\n"
        start = source.find(text) + len(text)
        end = source.find("self.to(weight.device)", start)
        spaces = re.findall(r"^([ ]{1,})break", source, flags = re.MULTILINE)[0]
        source = source.replace(source[start : end], spaces)
        spaces = len(re.match(r"[\s]{1,}", source).group(0))
        lines = source.split("\n")
        source = "\n".join(x[spaces:] for x in lines)
        source = re.sub("([^\.])nn\.", r"\1torch.nn.", source)
        source = source.replace("def update_layer", "def LoraLayer_update_layer")
        exec(source, globals())

        # Fix up incorrect downcasting of LoRA weights
        from peft.tuners.lora.layer import LoraLayer
        LoraLayer.update_layer = LoraLayer_update_layer
        from peft.tuners.lora import LoraLayer
        LoraLayer.update_layer = LoraLayer_update_layer
    except:
        logger.warning_once(
            "Unsloth unsuccessfully patched LoraLayer.update_layer. Please file a bug report.\n"\
            "Luckily, your training run will still work in the meantime!"
        )
    pass
pass

# =============================================

import psutil
def _get_statistics(statistics = None, force_download = True):
    # We log some basic stats about which environment is being used.
    # We simply download a README.md file from HF - all data is made public.
    # This is simply so we can check if some envs are broken or not.
    # You can disable this by commenting the below out
    try:
        n_cpus = psutil.cpu_count(logical = False)
        keynames = "\n" + "\n".join(os.environ.keys())
        if statistics is not None: pass
        elif "\nCOLAB_"  in keynames and n_cpus == 1: statistics = "colab"
        elif "\nCOLAB_"  in keynames: statistics = "colabpro"
        elif "\nKAGGLE_" in keynames: statistics = "kaggle"
        elif "\nRUNPOD_" in keynames: statistics = "runpod"
        elif "\nAWS_"    in keynames: statistics = "aws"
        elif "\nAZURE_"  in keynames: statistics = "azure"
        # elif "\nK_" in keynames or "\nFUNCTION_" in keynames: statistics = "gcp"
        elif "\nINVOCATION_ID" in keynames: statistics = "lambda"
        # else: statistics = "other"
        else:
            def try_vllm_check():
                vendor_files = (
                    "/sys/class/dmi/id/product_version",
                    "/sys/class/dmi/id/bios_vendor",
                    "/sys/class/dmi/id/product_name",
                    "/sys/class/dmi/id/chassis_asset_tag",
                    "/sys/class/dmi/id/sys_vendor",
                )
                from pathlib import Path
                for vendor_file in vendor_files:
                    path = Path(vendor_file)
                    if path.is_file():
                        file_content = path.read_text().lower()
                        if   "amazon"                in file_content: return "aws"
                        elif "microsoft corporation" in file_content: return "azure"
                        elif "google"                in file_content: return "gcp"
                return "other"
            pass
            try:    statistics = try_vllm_check()
            except: statistics = "other"
        pass
        if statistics is not None:
            from transformers import AutoModelForCausalLM
            stats_model = AutoModelForCausalLM.from_pretrained(
                f"unslothai/{statistics}",
                force_download = force_download,
            )
            del stats_model
        pass
    except:
        pass
pass


def get_statistics():
    # We log some basic stats about which environment is being used.
    # We simply download a README.md file from HF - all data is made public.
    # This is simply so we can check if some envs are broken or not.
    # You can disable this by setting UNSLOTH_DISABLE_STATISTICS
    import os
    if "UNSLOTH_DISABLE_STATISTICS" in os.environ: return
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
    disabled = False
    if not are_progress_bars_disabled():
        disable_progress_bars()
        disabled = True
    pass
    _get_statistics(None)
    _get_statistics("repeat", force_download = False)
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        if   vram <= 8 : vram = 8
        elif vram <= 16: vram = 16
        elif vram <= 20: vram = 20
        elif vram <= 24: vram = 24
        elif vram <= 40: vram = 40
        elif vram <= 48: vram = 48
        elif vram <= 80: vram = 80
        else: vram = 96
        _get_statistics(f"vram-{vram}")
    except:
        pass
    pass
    try:
        devices = torch.cuda.device_count()
        _get_statistics(f"{devices if devices <= 8 else 9}")
    except:
        pass
    if disabled: enable_progress_bars()
pass


# =============================================
# Fixes Bitsandbytes to remove missing warnings
from transformers.utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
from inspect import getsource
from accelerate.utils.dataclasses import DistributedType
BitsAndBytesConfig__init__ = getsource(BitsAndBytesConfig.__init__)
BitsAndBytesConfig__init__ = re.sub(
    r"if[\s]{1,}kwargs\:[\s]{1,}.+?\n",
    "",
    BitsAndBytesConfig__init__,
    flags = re.MULTILINE,
)
BitsAndBytesConfig__init__ = BitsAndBytesConfig__init__.split("\n")
length_spaces = len(re.match(r"[\s]{1,}", BitsAndBytesConfig__init__[0]).group(0))
BitsAndBytesConfig__init__ = "\n".join(x[length_spaces:] for x in BitsAndBytesConfig__init__)
BitsAndBytesConfig__init__ = BitsAndBytesConfig__init__.replace(
    "__init__",
    "_BitsAndBytesConfig__init__",
)

def _prepare_backend(
    self, cpu: bool = False, sagemaker_dp = False, backend: str = None,
) -> tuple[str, DistributedType]:
    return None, DistributedType.NO
pass
import accelerate.state
accelerate.state.PartialState._prepare_backend = _prepare_backend

import accelerate.accelerator
prepare = inspect.getsource(accelerate.accelerator.Accelerator.prepare)
prepare = prepare.split("\n")
spaces = prepare[0].find("def")
prepare = "\n".join(x[spaces:] for x in prepare)
x = "for obj in args:"
s = " "*spaces
prepare = prepare.replace(x, f'self.state.distributed_type = DistributedType.NO\n{s}{x}', 1)
exec(prepare, globals())
accelerate.accelerator.Accelerator.prepare = prepare

exec(BitsAndBytesConfig__init__, globals())

import transformers.utils.quantization_config
transformers.utils.quantization_config.BitsAndBytesConfig.__init__ = _BitsAndBytesConfig__init__
# =============================================

# Offloading to disk for modules (lm_head, embed_tokens)
import pickle

def offload_to_disk(W, model, name, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    file_location = os.path.join(temporary_location, model.config._name_or_path)
    if not os.path.exists(file_location):
        os.makedirs(file_location)
    pass

    filename = os.path.join(file_location, f"{name}.pt")
    W = W.weight if hasattr(W, "weight") else W
    torch.save(W, filename, pickle_module = pickle, pickle_protocol = pickle.HIGHEST_PROTOCOL,)
    offloaded_W = torch.load(filename, map_location = "cpu", mmap = True)
    offloaded_W._offloaded_file_location = filename
    return offloaded_W
pass


def offload_input_embeddings(model, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(model.get_input_embeddings(), model, "input_embeddings", temporary_location)
    new_input_embeddings = torch.nn.Embedding.from_pretrained(offloaded_W)
    new_input_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_input_embeddings(new_input_embeddings)
    return
pass


def offload_output_embeddings(model, temporary_location : str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(model.get_output_embeddings(), model, "output_embeddings", temporary_location)

    new_output_embeddings = torch.nn.Linear(1, 1, bias = None)
    del new_output_embeddings.weight
    new_output_embeddings.weight = offloaded_W
    new_output_embeddings.in_features  = offloaded_W.shape[1]
    new_output_embeddings.out_features = offloaded_W.shape[0]

    new_output_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_output_embeddings(new_output_embeddings)
    return
pass


# Fixes a weird Torch 2.3 bug which says T4s have bfloat16
def is_bfloat16_supported():
    return SUPPORTS_BFLOAT16
pass


# Patches models to add RoPE Scaling
def patch_linear_scaling(
    model_name = "gemma2",
    rope_module = None,
    scaled_rope_module = None,
    attention_module = None,
):
    assert(rope_module is not None and scaled_rope_module is not None)
    assert(attention_module is not None)

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = \
        f"import torch.nn as nn\n"\
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"\
        f"from {model_filepath} import logger, "\
        f"{model_name.title()}Attention, {model_name.title()}Config"

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        # Most likely already patched!
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """
    fix_rope_function = fix_rope_function.format(
        rope_function        = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
    )
    rotary_emb = re.findall(
        "self.rotary_emb = .+?\)", function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0: return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function
pass


# Patches for Llama-3 LlamaExtendedRotaryEmbedding
def patch_llama_rope_scaling(
    model_name = "llama",
    rope_module = None,
    scaled_rope_module = None,
    extended_rope_module = None,
    attention_module = None,
    longrope_module = None,
):
    assert(\
        rope_module is not None and \
        scaled_rope_module is not None and \
        extended_rope_module is not None
    )
    assert(attention_module is not None)

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = \
        f"import torch.nn as nn\n"\
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"\
        f"from {model_filepath} import logger, "\
        f"{model_name.title()}Attention, {model_name.title()}Config"

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        # Most likely already patched!
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type1 = self.config.rope_scaling.get("type", None)
        scaling_type2 = self.config.rope_scaling.get("rope_type", None)
        scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2
        scaling_factor = self.config.rope_scaling.get("factor")

        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "llama3":
            self.rotary_emb = {extended_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        elif scaling_type == "longrope":
            self.rotary_emb = {longrope_rope_function}(
                dim = self.head_dim,
                max_position_embeddings = self.max_position_embeddings,
                original_max_position_embeddings = self.config.original_max_position_embeddings,
                base = self.rope_theta,
                short_factor = self.config.rope_scaling['short_factor'],
                long_factor  = self.config.rope_scaling['long_factor' ],
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """

    fix_rope_function = fix_rope_function.format(
        rope_function          = rope_module.__name__,
        scaled_rope_function   = scaled_rope_module.__name__,
        extended_rope_function = extended_rope_module.__name__,
        longrope_rope_function = \
            (longrope_module if longrope_module is not None else rope_module).__name__
    )
    rotary_emb = re.findall(
        "self.rotary_emb = .+?\)", function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0: return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function
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


def create_boolean_mask(n = 4096, sliding_window = 2048):
    # Creates a boolean mask for attention
    mask = torch.ones(n, n, dtype = torch.bool)
    if sliding_window == 0:
        return torch.triu(mask, diagonal = 1, out = mask)
    pass
    torch.triu(mask, diagonal = 0, out = mask)
    torch.triu(mask.T, diagonal = -sliding_window, out = mask.T)
    mask = mask.T
    torch.logical_not(mask, out = mask)
    return mask
pass


def test_mask_creation():
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    for n in range(2, 23):
        for s in range(1, 23):
            correct_mask = AttentionMaskConverter(
                is_causal = True,
                sliding_window = s,
            ).to_causal_4d(1, n, n, dtype = torch.float16,).squeeze(0).squeeze(0)
            correct_mask = (correct_mask == correct_mask.min())
            our_mask = create_boolean_mask(n = n, sliding_window = s)
            assert(torch.all(correct_mask == our_mask))
        pass
        correct_mask = AttentionMaskConverter(
            is_causal = True,
            sliding_window = None,
        ).to_causal_4d(1, n, n, dtype = torch.float16,).squeeze(0).squeeze(0)
        correct_mask = (correct_mask == correct_mask.min())
        our_mask = create_boolean_mask(n = n, sliding_window = 0)
        assert(torch.all(correct_mask == our_mask))
    pass
pass


def _unsloth_get_batch_samples(self, epoch_iterator, num_batches):
    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
        try:
            batch_samples += [next(epoch_iterator)]
        except StopIteration:
            break
    if len(batch_samples) > 0 and "labels" in batch_samples[0]:
        try:
            num_items_in_batch = sum(
                [torch.count_nonzero(x["labels"][..., 1:] != -100) for x in batch_samples]
            )
        except TypeError:
            pass
    return batch_samples, num_items_in_batch
pass


def _unsloth_pre_compute_loss(self, model, inputs, *args, **kwargs):
    if "num_items_in_batch" in kwargs:
        if "num_items_in_batch" not in inputs:
            inputs["num_items_in_batch"] = kwargs["num_items_in_batch"]
        pass
    pass
    return self._old_compute_loss(model, inputs, *args, **kwargs)
pass


def patch_gradient_accumulation_fix(Trainer):
    # Fixes gradient accumulation 
    import inspect
    if hasattr(Trainer, "get_batch_samples"):
        if Trainer.get_batch_samples.__name__ == "_unsloth_get_batch_samples": return
        if \
            not inspect.getsource(Trainer.get_batch_samples).strip()\
            .endswith("return batch_samples, num_items_in_batch"):

            raise NotImplementedError("Unsloth: Please make a Github issue immediately!!")
        else:
            if Trainer.get_batch_samples.__name__ != "_unsloth_get_batch_samples":
                Trainer.get_batch_samples = _unsloth_get_batch_samples
            pass

            # Also fix passing in num_items_in_batch
            if not hasattr(Trainer, "_old_compute_loss"):
                Trainer._old_compute_loss = Trainer.compute_loss
                Trainer.compute_loss = _unsloth_pre_compute_loss
            pass
        pass
    else:
        logger.warning_once(
            "Unsloth: We fixed a gradient accumulation bug, "\
            "but it seems like you don't have the latest transformers version!\n"\
            "Please update transformers, TRL and unsloth via:\n"\
            '`pip install --upgrade --no-cache-dir --no-deps unsloth transformers git+https://github.com/huggingface/trl.git`'
        )
    pass

    # Also fix up loss scaling ie negate loss *= self.args.gradient_accumulation_steps
    if Trainer.training_step.__name__ == "_unsloth_training_step": return
    if "num_items_in_batch" not in inspect.signature(Trainer.training_step).parameters: return

    function = inspect.getsource(Trainer.training_step)
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)

    # Import all variables that need importing
    import transformers.trainer
    items_in_trainer = dir(transformers.trainer)
    good_items = []
    for item in items_in_trainer:
        # TODO: Support Deepspeed
        if item.startswith(("deepspeed", "xm", "met", "smp")): continue
        if item in function: good_items.append(item)
    pass
    exec("from transformers.trainer import (" + ", ".join(x for x in good_items) + ")", globals())

    # Accelerate does / self.args.gradient_accumulation_steps internally, so if we already
    # summed it up and did the division before hand, we have to negate it.
    function = function.replace(
        "loss *= self.args.gradient_accumulation_steps",
        "if num_items_in_batch is not None: loss *= self.args.gradient_accumulation_steps",
    )
    function = function.replace("def training_step", "def _unsloth_training_step", 1)
    exec(function, globals())
    Trainer.training_step = _unsloth_training_step
pass


def patch_tokenizer(model, tokenizer):
    model, tokenizer = _patch_tokenizer(model, tokenizer)
    if model is not None:
        model.config.update({"unsloth_version" : __version__})
    return model, tokenizer
pass
