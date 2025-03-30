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

from ._utils import (
    is_bfloat16_supported,
    HAS_FLASH_ATTENTION,
    HAS_FLASH_ATTENTION_SOFTCAPPING,
    USE_MODELSCOPE,
    get_transformers_model_type,
)
from .granite import FastGraniteModel
from .llama   import FastLlamaModel, logger
from .mistral import FastMistralModel
from .qwen2   import FastQwen2Model
from .cohere  import FastCohereModel
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from peft import PeftConfig, PeftModel
from .loader_utils import get_model_name
import os, contextlib, sys
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
from huggingface_hub import HfFileSystem
import importlib.util

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from unsloth_zoo.utils import Version, _get_dtype
transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA   = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2  = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32 = transformers_version  > Version("4.45.0")
SUPPORTS_GRANITE = transformers_version >= Version("4.46.0")
if SUPPORTS_GEMMA:
    from .gemma  import FastGemmaModel
if SUPPORTS_GEMMA2:
    from .gemma2 import FastGemma2Model
pass
import torch
from ._utils import (
    patch_compiling_bitsandbytes,
    patch_model_and_tokenizer,
    prepare_model_for_kbit_training,
    patch_unsloth_smart_gradient_checkpointing,
    patch_compiled_autograd,
    process_vision_info,
    unsloth_compile_transformers,
)

global FORCE_FLOAT32
FORCE_FLOAT32 = [
    "gemma3",
]

class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name                 = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length             = 2048,
        dtype                      = None,
        load_in_4bit               = True,
        load_in_8bit               = False,
        full_finetuning            = False,
        token                      = None,
        device_map                 = "sequential",
        rope_scaling               = None,
        fix_tokenizer              = True,
        trust_remote_code          = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab         = None,
        revision                   = None,
        use_exact_model_name       = False,

        fast_inference             = False, # uses vLLM
        gpu_memory_utilization     = 0.5,
        float8_kv_cache            = False,
        random_state               = 3407,
        max_lora_rank              = 64,
        disable_log_stats          = True,
        *args, **kwargs,
    ):
        if load_in_8bit or full_finetuning:
            return FastModel.from_pretrained(
                model_name                 = model_name,
                max_seq_length             = max_seq_length,
                dtype                      = dtype,
                load_in_4bit               = load_in_4bit,
                load_in_8bit               = load_in_8bit,
                full_finetuning            = full_finetuning,
                token                      = token,
                device_map                 = device_map,
                rope_scaling               = rope_scaling, # [TODO] No effect
                fix_tokenizer              = fix_tokenizer, # [TODO] No effect
                trust_remote_code          = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab         = resize_model_vocab, # [TODO] No effect
                revision                   = revision,
                return_logits              = False, # Return logits
                fullgraph                  = True, # No graph breaks
                use_exact_model_name       = use_exact_model_name,
                *args, **kwargs,
            )
        pass

        if token is None: token = get_token()
        assert (dtype is None or dtype == torch.float16 or dtype == torch.bfloat16)

        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(dtype = dtype)

        if fast_inference:
            if importlib.util.find_spec("vllm") is None:
                raise ImportError(
                    "Unsloth: Please install vLLM before enabling `fast_inference`!\n"\
                    "You can do this in a terminal via `pip install vllm`"
                )
            pass
        pass
        
        old_model_name = model_name
        if not use_exact_model_name:
            model_name = get_model_name(model_name, load_in_4bit)

        if USE_MODELSCOPE and not os.path.exists(model_name):
            from modelscope import snapshot_download
            model_name = snapshot_download(model_name)
        pass

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
                # Because HfFileSystem assumes linux paths, we need to set the path with forward slashes, even on Windows.
                files = HfFileSystem(token = token).glob(f"{model_name}/*.json")
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
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            raise RuntimeError(combined_error)
        pass

        # Get base model for PEFT:
        if is_peft:
            # Check base model again for PEFT
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(model_name, load_in_4bit)
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
            )
        pass

        if not was_disabled: enable_progress_bars()

        model_type = model_config.model_type

        if model_type == "llama":
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
        # Temporary disable optimized Cohere until errors match
        # elif model_type == "cohere":
        #     dispatch_model = FastCohereModel
        # Temporary disable optimized Granite until errors match
        # elif model_type == "granite":
        #     dispatch_model = FastGraniteModel
        else:
            return FastModel.from_pretrained(
                model_name                 = model_name,
                max_seq_length             = max_seq_length,
                dtype                      = dtype,
                load_in_4bit               = load_in_4bit,
                load_in_8bit               = load_in_8bit,
                full_finetuning            = full_finetuning,
                token                      = token,
                device_map                 = device_map,
                rope_scaling               = rope_scaling, # [TODO] No effect
                fix_tokenizer              = fix_tokenizer, # [TODO] No effect
                trust_remote_code          = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab         = resize_model_vocab, # [TODO] No effect
                revision                   = revision,
                return_logits              = False, # Return logits
                fullgraph                  = True, # No graph breaks
                use_exact_model_name       = use_exact_model_name,
                *args, **kwargs,
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

        if fast_inference:
            import platform
            if platform.system().lower() == 'windows':
                print("Unsloth: vLLM does not work in Windows! Will use Unsloth inference!")
                fast_inference = False
            pass
            from unsloth_zoo.vllm_utils import (
                patch_vllm, 
                vllm_dynamic_quant_supported,
            )
            patch_vllm()
            if model_name.endswith("unsloth-bnb-4bit"):
                if not vllm_dynamic_quant_supported(model_name, model_config):
                    # Instead use -bnb-4bit variant
                    print(
                        f"Unsloth: Switching from Unsloth dynamic quant to normal quant since\n"\
                        f"we do not yet support fast inference for {model_name}"
                    )
                    model_name = model_name[:-len("unsloth-bnb-4bit")] + "bnb-4bit"
                pass
            pass
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

            fast_inference    = fast_inference,
            gpu_memory_utilization = gpu_memory_utilization,
            float8_kv_cache   = float8_kv_cache,
            random_state      = random_state,
            max_lora_rank     = max_lora_rank,
            disable_log_stats = disable_log_stats,
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


from ..kernels import (
    patch_loss_functions,
    post_patch_loss_function,
)
from .vision import FastBaseModel
from transformers import (
    AutoModelForCausalLM,
)
try:
    from transformers import AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText
except:
    from transformers import AutoModelForVision2Seq
pass


class FastModel(FastBaseModel):
    @staticmethod
    def from_pretrained(
        model_name                 = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length             = 2048,
        dtype                      = None,
        load_in_4bit               = True,
        load_in_8bit               = False,
        full_finetuning            = False,
        token                      = None,
        device_map                 = "sequential",
        rope_scaling               = None, # [TODO] No effect
        fix_tokenizer              = True, # [TODO] No effect
        trust_remote_code          = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab         = None, # [TODO] No effect
        revision                   = None,
        return_logits              = False, # Return logits
        fullgraph                  = True, # No graph breaks
        use_exact_model_name       = False,
        *args, **kwargs,
    ):
        if token is None: token = get_token()

        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        assert(dtype in (torch.float16, torch.bfloat16, torch.float32))

        patch_compiled_autograd()
        patch_compiling_bitsandbytes()

        if full_finetuning and (load_in_4bit or load_in_8bit):
            print("Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA.")
            load_in_4bit = False
            load_in_8bit = False
        pass

        if load_in_4bit and load_in_8bit:
            raise RuntimeError(
                "Unsloth: Can only load in 4bit or 8bit, not both!\n"\
                "Also, we by default set `load_in_4bit = True`.\n"\
                "If you want 8bit finetuning, set both `load_in_4bit = False` and `load_in_8bit = True`"
            )
        pass

        old_model_name = model_name
        if not use_exact_model_name:
            model_name = get_model_name(model_name, load_in_4bit)

        # Check versions
        LATEST  = '\nPlease use transformers via `pip install --no-deps git+https://github.com/huggingface/transformers.git`'
        NIGHTLY = '\nPlease use nightly transformers via pip install --upgrade "transformers>=4.49.0"`'
        if "pixtral" in model_name.lower() and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Pixtral only works on transformers >= 4.49.0." + LATEST)
        elif "qwen2.5" in model_name.lower() and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Qwen 2.5 only works on transformers >= 4.49.0." + LATEST)
        elif "aya-vision" in model_name.lower():
            # Disable compiling for now - errors out!
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
            if transformers_version < Version("4.50.0.dev0"):
                raise RuntimeError("Unsloth: Aya Vision only works on transformers >= 4.50.0." + NIGHTLY)
        elif "gemma-3" in model_name.lower() and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: Gemma 3 only works on transformers >= 4.50.0." + NIGHTLY)
        elif "c4ai-command-a-03-2025" in model_name.lower() and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: Cohere's Command model only works on transformers >= 4.50.0." + NIGHTLY)
        elif "granite-vision" in model_name.lower():
            # Disable compiling for now - errors out!
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
            if transformers_version < Version("4.50.0.dev0"):
                raise RuntimeError("Unsloth: Granite Vision only works on transformers >= 4.50.0." + NIGHTLY)
        elif "olmo-2" in model_name.lower() and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: OLMo-2 only works on transformers >= 4.50.0." + NIGHTLY)
        pass

        if USE_MODELSCOPE and not os.path.exists(model_name):
            from modelscope import snapshot_download
            model_name = snapshot_download(model_name)
        pass

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
                files = HfFileSystem(token = token).glob(f"{model_name}/*.json")
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
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            raise RuntimeError(combined_error)
        pass

        # Get base model for PEFT:
        if is_peft:
            # Check base model again for PEFT
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(model_name, load_in_4bit)
            
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
            )
        pass

        if not was_disabled: enable_progress_bars()

        do_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
        if do_logging:
            redirector = contextlib.nullcontext()
        else:
            redirector = contextlib.redirect_stdout(open(os.devnull, "w"))

        # Get model types like Gemma3 etc
        model_types = get_transformers_model_type(
            model_name        = model_name,
            token             = token,
            revision          = revision,
            trust_remote_code = trust_remote_code,
        )
        model_types = ["siglip"] + model_types

        # Set forced float32 env flag
        os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"
        do_forced_float32 = False
        for model_type_arch in model_types:
            if model_type_arch != "siglip": break
        global FORCE_FLOAT32
        for disable_name in FORCE_FLOAT32:
            if (disable_name.lower() == model_type_arch.lower() or \
                disable_name.lower() in model_name.lower()) and \
                ((dtype == torch.float16) or not SUPPORTS_BFLOAT16):
                os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
                dtype = torch.bfloat16 # Change to bfloat16 loading
                break
        pass
        # Patch gradient checkpointing
        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(dtype = dtype)

        with redirector:
            patch_loss_functions(torch_compile = False)
            model_types, supports_sdpa = unsloth_compile_transformers(
                dtype                   = dtype,
                model_name              = model_name,
                model_types             = model_types,
                token                   = token,
                sdpa_dynamic_mask       = True,
                sdpa_bool_masks         = True,
                sdpa_gqa_replace        = True,
                sdpa_dynamic_compile    = True,
                compile_attention       = True,
                disable_causal_masks    = True,
                compile_torch_modules   = True,
                compile_custom_modules  = True,
                compile_function_calls  = True,
                fuse_lm_head            = True,
                gradient_checkpointing  = True,
                manual_replacements     = True,
                fast_lora_forwards      = True,
                fast_residual_stream    = False,
                accurate_accumulation   = True,
                epilogue_fusion         = True,
                max_autotune            = False,
                shape_padding           = True,
                cudagraphs              = False,
                debug                   = False,
                fullgraph               = fullgraph,
                import_from_cache       = False,
                disable                 = False,
                return_logits           = return_logits,
                trust_remote_code       = trust_remote_code,
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

        # Check if VLM
        is_vlm = any(x.endswith("ForConditionalGeneration") for x in model_config.architectures)
        is_vlm = is_vlm or hasattr(model_config, "vision_config")
        auto_model = AutoModelForVision2Seq if is_vlm else AutoModelForCausalLM

        model, tokenizer = FastBaseModel.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = _get_dtype(dtype),
            load_in_4bit      = load_in_4bit,
            load_in_8bit      = load_in_8bit,
            full_finetuning   = full_finetuning,
            token             = token,
            device_map        = device_map,
            trust_remote_code = trust_remote_code,
            revision          = revision if not is_peft else None,
            model_types       = model_types,
            tokenizer_name    = tokenizer_name,
            auto_model        = auto_model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            supports_sdpa     = supports_sdpa,
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
            model = PeftModel.from_pretrained(
                model,
                old_model_name,
                token = token,
                revision = revision,
                is_trainable = True,
                trust_remote_code = trust_remote_code,
            )
            # Patch it as well!
            model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer
    pass
pass

class FastVisionModel(FastModel):
    pass

class FastTextModel(FastModel):
    pass
