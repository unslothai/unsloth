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

from ._utils import is_bfloat16_supported, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING
from .llama   import FastLlamaModel, logger
from .mistral import FastMistralModel
from .qwen2   import FastQwen2Model
from .cohere  import FastCohereModel
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from .mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit
import os
try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
pass
from huggingface_hub import HfFileSystem

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from packaging.version import Version
transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA   = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2  = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32 = transformers_version  > Version("4.45.0")
if SUPPORTS_GEMMA:
    from .gemma  import FastGemmaModel
if SUPPORTS_GEMMA2:
    from .gemma2 import FastGemma2Model
pass
import torch

def _get_dtype(dtype):
    __DTYPE_MAP = {
        "float32": torch.float32,
        torch.float32: torch.float32,
        "float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
    }
    if   dtype is None or dtype == None: return None
    elif dtype in __DTYPE_MAP: return __DTYPE_MAP[dtype]
    else:
        print(f"Unsloth: {dtype} is not recognized, so we'll default to None")
        return None
    pass
pass


def __get_model_name(
    model_name,
    load_in_4bit = True,
    INT_TO_FLOAT_MAPPER  = None,
    FLOAT_TO_INT_MAPPER  = None,
    MAP_TO_UNSLOTH_16bit = None,
):
    model_name = str(model_name)
    lower_model_name = model_name.lower()

    if not SUPPORTS_FOURBIT and lower_model_name in INT_TO_FLOAT_MAPPER:

        model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        logger.warning_once(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "\
            f"4bit loading.\nThe minimum required version is 4.37.\n"\
            f'Try `pip install --upgrade "transformers>=4.37"`\n'\
            f"to obtain the latest transformers build, then restart this session.\n"\
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
        return model_name
    
    elif not load_in_4bit and lower_model_name in INT_TO_FLOAT_MAPPER:

        new_model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
        #     f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        # )
        return new_model_name

    elif not load_in_4bit and lower_model_name in MAP_TO_UNSLOTH_16bit:

        new_model_name = MAP_TO_UNSLOTH_16bit[lower_model_name]
        return new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and lower_model_name in FLOAT_TO_INT_MAPPER:

        new_model_name = FLOAT_TO_INT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
        #     f"We shall load `{new_model_name}` for 4x faster loading."
        # )
        return new_model_name
    pass

    return None
pass


def _get_new_mapper():
    try:
        import requests
        new_mapper = "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
        with requests.get(new_mapper, timeout = 3) as new_mapper: new_mapper = new_mapper.text
        new_mapper = new_mapper[new_mapper.find("__INT_TO_FLOAT_MAPPER"):]
        new_mapper = new_mapper\
            .replace("INT_TO_FLOAT_MAPPER",  "NEW_INT_TO_FLOAT_MAPPER")\
            .replace("FLOAT_TO_INT_MAPPER",  "NEW_FLOAT_TO_INT_MAPPER")\
            .replace("MAP_TO_UNSLOTH_16bit", "NEW_MAP_TO_UNSLOTH_16bit")

        exec(new_mapper, globals())
        return NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit
    except:
        return {}, {}, {}
    pass
pass


def get_model_name(model_name, load_in_4bit = True):
    new_model_name = __get_model_name(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        INT_TO_FLOAT_MAPPER  = INT_TO_FLOAT_MAPPER,
        FLOAT_TO_INT_MAPPER  = FLOAT_TO_INT_MAPPER,
        MAP_TO_UNSLOTH_16bit = MAP_TO_UNSLOTH_16bit,
    )
    if new_model_name is None and model_name.count("/") == 1 and model_name[0].isalnum():
        # Try checking if a new Unsloth version allows it!
        NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit = _get_new_mapper()
        upgraded_model_name = __get_model_name(
            model_name = model_name,
            load_in_4bit = load_in_4bit,
            INT_TO_FLOAT_MAPPER  = NEW_INT_TO_FLOAT_MAPPER,
            FLOAT_TO_INT_MAPPER  = NEW_FLOAT_TO_INT_MAPPER,
            MAP_TO_UNSLOTH_16bit = NEW_MAP_TO_UNSLOTH_16bit,
        )
        if upgraded_model_name is not None:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version! Please update Unsloth via:\n\n"\
                'pip uninstall unsloth -y\n'\
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
            )
        pass
    pass
    return new_model_name if new_model_name is not None else model_name
pass


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name                 = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length             = None,
        dtype                      = None,
        load_in_4bit               = True,
        token                      = None,
        device_map                 = "sequential",
        rope_scaling               = None,
        fix_tokenizer              = True,
        trust_remote_code          = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab         = None,
        revision                   = None,
        *args, **kwargs,
    ):
        if token is None: token = get_token()
        
        old_model_name = model_name
        model_name = get_model_name(model_name, load_in_4bit)

        # First check if it's a normal model via AutoConfig
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()

        autoconfig_error = None
        peft_error = None
        try:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_model = True
        except Exception as error:
            autoconfig_error = str(error)
            is_model = False
        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_peft = True
        except Exception as error:
            peft_error = str(error)
            is_peft = False
        pass

        # Both config.json and adapter_config.json should not exist!

        # Old transformers versions check
        both_exist = (is_model and is_peft) and not SUPPORTS_LLAMA32
        
        # New transformers need to check manually.
        if SUPPORTS_LLAMA32:
            # Check if folder exists locally
            if os.path.isdir(model_name):
                exist_adapter_config = os.path.exists(os.path.join(model_name, "adapter_config.json"))
                exist_config         = os.path.exists(os.path.join(model_name, "config.json"))
                both_exist = exist_adapter_config and exist_config
            else:
                files = HfFileSystem(token = token).glob(os.path.join(model_name, "*.json"))
                files = (os.path.split(x)[-1] for x in files)
                if sum(x == "adapter_config.json" or x == "config.json" for x in files) >= 2:
                    both_exist = True
                pass
            pass
        pass

        # Error out if both LoRA and normal model config exists.
        if both_exist:
            raise RuntimeError(
                "Unsloth: Your repo has a LoRA adapter and a base model.\n"\
                "You have 2 files `config.json` and `adapter_config.json`.\n"\
                "We must only allow one config file.\n"\
                "Please separate the LoRA and base models to 2 repos."
            )

        elif not is_model and not is_peft:
            error = autoconfig_error or peft_error
            # Old transformers version
            if "rope_scaling" in error.lower() and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support new RoPE scaling methods.\n"\
                    f"This includes Llama 3.1. The minimum required version is 4.43.2\n"\
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                ) 
            raise RuntimeError(autoconfig_error or peft_error)
        pass

        # Get base model for PEFT:
        if is_peft:
            # Check base model again for PEFT
            model_name = get_model_name(peft_config.base_model_name_or_path, load_in_4bit)
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
        pass

        if not was_disabled: enable_progress_bars()

        model_type = model_config.model_type

        if   model_type == "llama":
            scaling_type = None
            if getattr(model_config, "rope_scaling", None) is not None:
                scaling_type1 = model_config.rope_scaling.get("type", None)
                scaling_type2 = model_config.rope_scaling.get("rope_type", None)
                scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2
            pass

            if scaling_type == "llama3" and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Llama 3.1.\n"\
                    f"The minimum required version is 4.43.2\n"\
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )

            dispatch_model = FastLlamaModel

        elif model_type == "mistral": dispatch_model = FastMistralModel
        elif model_type == "gemma":
            if not SUPPORTS_GEMMA:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"\
                    f"The minimum required version is 4.38.\n"\
                    f'Try `pip install --upgrade "transformers>=4.38"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            dispatch_model = FastGemmaModel
        elif model_type == "gemma2":
            if not SUPPORTS_GEMMA2:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"\
                    f"The minimum required version is 4.42.3.\n"\
                    f'Try `pip install --upgrade "transformers>=4.42.3"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            # Also check for softcapping support in flash-attn which is faster!
            if is_bfloat16_supported() and not HAS_FLASH_ATTENTION:
                print(
                    "Unsloth: If you want to finetune Gemma 2, install flash-attn to make it faster!\n"\
                    "To install flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
            elif HAS_FLASH_ATTENTION and not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
            
            dispatch_model = FastGemma2Model
        elif model_type == "qwen2":
            dispatch_model = FastQwen2Model
        elif model_type == "cohere":
            dispatch_model = FastCohereModel
        else:
            raise NotImplementedError(
                f"Unsloth: {model_name} not supported yet!\n"\
                "Make an issue to https://github.com/unslothai/unsloth!",
            )
        pass

        # Check if this is local model since the tokenizer gets overwritten
        if  os.path.exists(os.path.join(old_model_name, "tokenizer_config.json")) and \
            os.path.exists(os.path.join(old_model_name, "tokenizer.json")) and \
            os.path.exists(os.path.join(old_model_name, "special_tokens_map.json")):

            tokenizer_name = old_model_name
        else:
            tokenizer_name = None
        pass

        model, tokenizer = dispatch_model.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = _get_dtype(dtype),
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = dispatch_model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            revision          = revision if not is_peft else None,
            *args, **kwargs,
        )
        
        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)
        pass

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["unsloth",])
        pass
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(["unsloth",])
        pass

        if load_in_4bit:
            # Fix up bitsandbytes config
            quantization_config = \
            {
                # Sometimes torch_dtype is not a string!!
                "bnb_4bit_compute_dtype"           : model.config.to_dict()["torch_dtype"],
                "bnb_4bit_quant_type"              : "nf4",
                "bnb_4bit_use_double_quant"        : True,
                "llm_int8_enable_fp32_cpu_offload" : False,
                "llm_int8_has_fp16_weight"         : False,
                "llm_int8_skip_modules"            : None,
                "llm_int8_threshold"               : 6.0,
                "load_in_4bit"                     : True,
                "load_in_8bit"                     : False,
                "quant_method"                     : "bitsandbytes",
            }
            model.config.update({"quantization_config" : quantization_config})
        pass

        if is_peft:
            # From https://github.com/huggingface/peft/issues/184
            # Now add PEFT adapters
            model.enable_input_require_grads()
            model = PeftModel.from_pretrained(
                model,
                old_model_name,
                token = token,
                revision = revision,
                is_trainable = True,
                trust_remote_code = trust_remote_code,
            )
            # Patch it as well!
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer
    pass
pass