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

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
try:
    from transformers import AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText
except:
    from transformers import AutoModelForVision2Seq
pass
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import __version__, importlib_version
from ._utils import *
from ..save import patch_saving_functions
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from peft import PeftModelForCausalLM
from transformers import set_seed as transformers_set_seed
from unsloth_zoo.peft_utils import (
    get_peft_regex,
    SKIP_QUANTIZATION_MODULES,
    requires_grad_for_gradient_checkpointing,
)
from transformers.models.llama.modeling_llama import logger
from transformers import __version__ as transformers_version
from triton import __version__ as triton_version
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.hf_utils import (
    dtype_from_config,
    add_dtype_kwargs,
    fix_lora_auto_mapping,
    get_auto_processor,
)
from unsloth_zoo.patching_utils import patch_model_and_tokenizer
from unsloth_zoo.training_utils import prepare_model_for_training

from unsloth_zoo.utils import Version
from transformers import __version__ as transformers_version

import types
import functools
import os
import gc
import math
import functools
from typing import Optional, Tuple, List, Union
import re, inspect, sys
import contextlib
import types
try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
pass
from unsloth import DEVICE_TYPE, DEVICE_COUNT

__all__ = [
    "FastBaseModel",
]

global NUM_LOGITS_TO_KEEP
NUM_LOGITS_TO_KEEP = dict()

VLLM_SUPPORTED_VLM = [
    "qwen2_5_vl",
    "gemma3",
]
VLLM_NON_LORA_VLM = [
    "mllama"
]

from transformers import GenerationConfig, CompileConfig, HybridCache, AutoConfig, PretrainedConfig
HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__
from transformers import GenerationConfig, CompileConfig, HybridCache

_compile_config = CompileConfig(
    fullgraph = False,
    dynamic = None,
    mode = "reduce-overhead",
)
_compile_config.disable = True # Must set manually

from unsloth_zoo.vllm_utils import (
    convert_lora_modules,
    return_lora_modules,
)

try:
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_compiler_set_stance = None
pass

def unsloth_base_fast_generate(
    self,
    *args,
    **kwargs,
):
    if len(args) != 0:
        input_ids = args[0]
    elif "input_ids" in kwargs:
        input_ids = kwargs["input_ids"]
    elif "input" in kwargs:
        input_ids = kwargs["input_ids"]
    elif "input_features" in kwargs:
        input_ids = kwargs["input_features"]
    elif "input_embeds" in kwargs:
        input_ids = kwargs["input_embeds"]
    elif "inputs" in kwargs:
        input_ids = kwargs["inputs"]
    else:
        key = next(iter(kwargs.keys()))
        if type(kwargs["key"]) is not torch.Tensor:
            raise TypeError("Unsloth: You need to pass in input_ids to .generate!")
        input_ids = kwargs[key]
    pass
    assert(type(input_ids) is torch.Tensor)
    bsz = input_ids.shape[0]

    FastBaseModel.for_inference(self)
    dtype = _get_dtype(dtype_from_config(self.config))

    # Check if VLM
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in self.config.architectures
    )
    is_vlm = is_vlm or hasattr(self.config, "vision_config")
    arch = self.config.architectures[0]

    # Remove token_type_ids - WRONG for Gemma 3 since bidirectional attention
    if hasattr(self, "generate") and hasattr(self, "forward"):
        # did not combine with below since self might not have model
        keys = inspect.signature(self.forward).parameters.keys()
        if "token_type_ids" not in keys:
            kwargs.pop("token_type_ids", None)
    # kwargs.pop("token_type_ids", None)

    # VLMs do not allow logits_to_keep
    global NUM_LOGITS_TO_KEEP
    if arch not in NUM_LOGITS_TO_KEEP:
        m = self
        # Find which is needed ie
        # num_logits_to_keep or logits_to_keep
        while hasattr(m, "model"):
            if hasattr(m, "forward"):
                keys = inspect.signature(m.forward).parameters.keys()
                if "num_logits_to_keep" in keys:
                    NUM_LOGITS_TO_KEEP[arch] = "num_logits_to_keep"
                    break
                elif "logits_to_keep" in keys:
                    NUM_LOGITS_TO_KEEP[arch] = "logits_to_keep"
                    break
            m = m.model
        pass
        if arch not in NUM_LOGITS_TO_KEEP:
            NUM_LOGITS_TO_KEEP[arch] = None
        pass
    pass
    key = NUM_LOGITS_TO_KEEP[arch]
    if key is not None and key not in kwargs:
        kwargs[key] = 1

    # Check pad_token
    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Get pixel values for VLMs
    try: kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype)
    except: pass

    # Mixed precision autocast
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        autocaster = torch.autocast(device_type = "cuda", dtype = torch.float16)
        dtype = torch.float16
    else:
        autocaster = torch.autocast(device_type = "cuda", dtype = dtype)

    # Prepare LoRA
    # state_dict = convert_lora_modules(self, dtype = dtype)

    # Set compile dynamic shapes
    torch._dynamo.mark_static(input_ids, 0)
    torch._dynamo.mark_dynamic(input_ids, 1)
    if "attention_mask" in kwargs:
        torch._dynamo.mark_static(kwargs["attention_mask"], 0)
        torch._dynamo.mark_dynamic(kwargs["attention_mask"], 1)
    if "token_type_ids" in kwargs:
        torch._dynamo.mark_static(kwargs["token_type_ids"], 0)
        torch._dynamo.mark_dynamic(kwargs["token_type_ids"], 1)

    # Fix generation_config
    # Use hybrid if sliding window seen, otherwise try static
    cache_implementation = getattr(self.config, "cache_implementation", None)
    if getattr(self, "_supports_static_cache", getattr(self, "_can_compile_fullgraph", True)):
        if os.environ.get("UNSLOTH_DISABLE_STATIC_GENERATION", "0") == "0":
            cache_implementation = "static"
        else:
            cache_implementation = None
    else:
        cache_implementation = None
    if cache_implementation is not None:
        swa = getattr(getattr(self.config, "text_config", self.config), "sliding_window", None)
        if (swa == 0 or type(swa) is not int) \
            and (getattr(self, "_can_compile_fullgraph", True) is True):
            cache_implementation = "static"
        else:
            if Version(transformers_version) < Version("4.56.0.dev0"):
                cache_implementation = "hybrid"
            else:
                cache_implementation = "static"

    if "generation_config" in kwargs:
        kwargs["generation_config"].cache_implementation = cache_implementation
        if cache_implementation is not None:
            kwargs["generation_config"].compile_config = _compile_config
    else:
        kwargs["cache_implementation"] = cache_implementation
        if cache_implementation is not None:
            kwargs["compile_config"] = _compile_config
    pass

    with torch.inference_mode(), autocaster:
        output = self._old_generate(*args, **kwargs)

    FastBaseModel.for_training(self)
    return output
pass

class FastBaseModel:

    @staticmethod
    def from_pretrained(
        model_name        = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length    = 2048,
        dtype             = None,
        load_in_4bit      = True,
        load_in_8bit      = False,
        full_finetuning   = False,
        token             = None,
        device_map        = "sequential",
        trust_remote_code = False,
        model_types       = None,
        tokenizer_name    = None,
        auto_model        = AutoModelForVision2Seq,
        use_gradient_checkpointing = "unsloth",
        supports_sdpa     = True,
        whisper_language  = None,
        whisper_task      = None,
        fast_inference   = False,
        gpu_memory_utilization = 0.5,
        float8_kv_cache   = False,
        random_state      = 3407,
        max_lora_rank     = 64,
        disable_log_stats = False,
        unsloth_vllm_standby = False,
        **kwargs,
    ):
        if unsloth_vllm_standby and os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1":
            raise RuntimeError("Unsloth: UNSLOTH_VLLM_STANDBY is True, but UNSLOTH_VLLM_STANDBY is not set to 1!")
        pass

        if model_types is None:
            raise RuntimeError(
                "Unsloth: Please use FastModel or FastVisionModel and not use FastBaseModel directly!"
            )
        if os.environ.get("UNSLOTH_MODEL_NAME", "") == "":
            os.environ["UNSLOTH_MODEL_NAME"] = model_name.lower()

        is_vlm = (auto_model in [AutoModelForVision2Seq, AutoModelForImageTextToText])
        is_whisper = (whisper_language is not None and whisper_task is not None)
        auto_processor = AutoProcessor if (is_vlm or is_whisper) else AutoTokenizer

        model_type_arch = model_types[0]
        if model_type_arch == "siglip":
            for model_type_arch in model_types:
                if model_type_arch != "siglip": break

        vllm_enable_lora = True

        if is_vlm and fast_inference:
            if not any(arch in VLLM_SUPPORTED_VLM for arch in model_types):
                raise RuntimeError(
                    f"Unsloth: Fast inference is only supported for Language models and Qwen2.5-VL, Gemma3 among vision models. "
                    f"Found architectures: {', '.join(model_types)}!"
                )

        if any(arch in VLLM_NON_LORA_VLM for arch in model_types):
            # mllama is still only in vllm v0 https://arc.net/l/quote/llwkfgmu
            # https://docs.vllm.ai/en/stable/models/supported_models.html#text-generation_1
            # vLLM V0 does not support LoRA on multi modal models.
            # TODO: Update this once vLLM V1 supports Llama 3.2 aka mllama
            vllm_enable_lora = False

        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()

        if DEVICE_TYPE == "cuda":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_version = torch.version.cuda
            gpu_stats_snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."
            try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
            except: vllm_version = ""
        elif DEVICE_TYPE == "hip":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_version = torch.version.hip
            gpu_stats_snippet = f"ROCm Toolkit: {gpu_version}."
            try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
            except: vllm_version = ""
        elif DEVICE_TYPE == "xpu":
            gpu_stats = torch.xpu.get_device_properties(0)
            gpu_version = torch.version.xpu
            gpu_stats_snippet = f"Intel Toolkit: {gpu_version}."
            # TODO: After adding vLLM support for XPU, changed this
            vllm_version = ""
        else:
            raise ValueError(f"Unsloth: Unsupported device type: {DEVICE_TYPE}")

        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
        f"==((====))==  Unsloth {__version__}: Fast {model_type_arch.title()} patching. Transformers: {transformers_version}.{vllm_version}\n"\
        f"   {chr(92)}{chr(92)}   /|    {gpu_stats.name}. Num GPUs = {DEVICE_COUNT}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
        f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"\
        f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
        f' "-____-"     Free license: http://github.com/unslothai/unsloth'

        print(statistics)

        # Warn about fast transfers
        if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
            old_hf_transfer = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]
            if old_hf_transfer in ("False", "false"): old_hf_transfer = "0"
            if old_hf_transfer in ("True",  "true" ): old_hf_transfer = "1"
        else:
            old_hf_transfer = "0"
        if old_hf_transfer == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        if old_hf_transfer != "0": os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        get_statistics() # For debugging - we use a download counter to see if environments are not breaking

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            if dtype == torch.float16: dtype = torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        pass
        assert(dtype in (torch.float16, torch.bfloat16, torch.float32))

        bnb_compute_dtype = dtype
        do_forced_float32 = False
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            print(f"Unsloth: Using float16 precision for {model_type_arch} won't work! Using float32.")
            bnb_compute_dtype = torch.float16
            do_forced_float32 = True
        pass

        # Check for custom data-types
        custom_datatype = None
        correct_dtype = None
        if os.environ.get("UNSLOTH_FORCE_CUSTOM_DTYPE", "") != "":
            custom_datatype = os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"]
            assert custom_datatype.count(";") >= 4
            checker, _dtype, _bnb_compute_dtype, _custom_datatype, execute_code = custom_datatype.split(";", 4)
            # Allow custom dtypes on all runs
            allow_all_runs = (checker == "all")
            # Allow only on float16 datatypes
            allow_float16_runs = (
                (checker == "float16" or checker == "torch.float16") and \
                (dtype == torch.float16 or os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1")
            )
            if allow_all_runs or allow_float16_runs:
                if eval(_dtype) is not None:
                    dtype = eval(_dtype)
                if eval(_bnb_compute_dtype) is not None:
                    bnb_compute_dtype = eval(_bnb_compute_dtype)
                correct_dtype = bnb_compute_dtype
                custom_datatype = _custom_datatype
                # Execute code as well
                if len(execute_code.strip()) != 0:
                    exec(execute_code)
            else:
                custom_datatype = None
                correct_dtype = None
        pass

        # Stop SDPA for some archs like Pixtral / Mistral3
        if not ("attn_implementation" in kwargs):
            kwargs["attn_implementation"] = "sdpa"
        if not supports_sdpa:
            print(f"Unsloth: {model_type_arch.title()} does not support SDPA - switching to fast eager.")
            del kwargs["attn_implementation"]
        pass

        bnb_config = None
        if full_finetuning and (load_in_4bit or load_in_8bit):
            print("Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA.")
            load_in_4bit = False
            load_in_8bit = False
        pass

        if load_in_4bit and load_in_8bit:
            raise RuntimeError("Unsloth: Can only load in 4bit or 8bit, not both!")
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = bnb_compute_dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES.copy(),
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit              = True,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES.copy(),
            )
        elif not load_in_4bit and not load_in_8bit and not full_finetuning:
            print("Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.")
        pass

        if full_finetuning:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
            if dtype == torch.bfloat16:
                print("Unsloth: Using bfloat16 full finetuning which cuts memory usage by 50%.")
            else:
                print("Unsloth: Float16 full finetuning uses more memory since we upcast weights to float32.")
        else:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"
        pass

        # Fix AttributeError: 'BitsAndBytesConfig' object has no attribute 'get_loading_attributes'
        if bnb_config is not None and not hasattr(bnb_config, "get_loading_attributes"):
            bnb_config.get_loading_attributes = lambda *args, **kwargs: {}

        # Cannot be None, since HF now checks for the config
        if load_in_4bit:
            # Ignore load_in_4bit / load_in_8bit for MXFP4 - best to get config file
            if "gpt-oss" in model_name.lower():
                pass
            else:
                kwargs["quantization_config"] = bnb_config
        pass

        # Check if using forced float32 - we load it in bfloat16, then cast to float16!
        torch_dtype = dtype
        if do_forced_float32: torch_dtype = torch.bfloat16

        kwargs = add_dtype_kwargs(torch_dtype, kwargs)

        raise_handler = RaiseUninitialized()
        if not fast_inference:
            model = auto_model.from_pretrained(
                model_name,
                device_map              = device_map,
                # torch_dtype           = torch_dtype, # Transformers removed torch_dtype
                # quantization_config   = bnb_config,
                token                   = token,
                trust_remote_code       = trust_remote_code,
                # attn_implementation   = attn_implementation,
                **kwargs,
            )
            if hasattr(model, 'generate'):
                model.fast_generate = model.generate
                model.fast_generate_batches = error_out_no_vllm
        else:
            from unsloth_zoo.vllm_utils import (
                load_vllm,
                get_vllm_state_dict,
                convert_vllm_to_huggingface,
                generate_batches,
            )
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                attn_implementation = "sdpa" if supports_sdpa else "eager",
            )

            if fast_inference:
                fast_inference, model_name = fast_inference_setup(model_name, model_config)

            allowed_args = inspect.getfullargspec(load_vllm).args
            load_vllm_kwargs = dict(
                model_name             = model_name,
                config                 = model_config,
                gpu_memory_utilization = gpu_memory_utilization,
                max_seq_length         = max_seq_length,
                dtype                  = dtype,
                float8_kv_cache        = float8_kv_cache,
                enable_lora            = vllm_enable_lora,
                max_lora_rank          = max_lora_rank,
                disable_log_stats      = disable_log_stats,
                use_bitsandbytes       = load_in_4bit,
                unsloth_vllm_standby   = unsloth_vllm_standby,
                is_vision_model        = is_vlm,
            )
            for allowed_arg in allowed_args:
                if allowed_arg not in load_vllm_kwargs and allowed_arg in kwargs:
                    load_vllm_kwargs[allowed_arg] = kwargs[allowed_arg]
            pass

            # Load vLLM first
            llm = load_vllm(**load_vllm_kwargs)

            # Convert to HF format
            _, quant_state_dict = get_vllm_state_dict(llm, config = model_config, is_vision_model = True)
            model = convert_vllm_to_huggingface(quant_state_dict, model_config, dtype, bnb_config, is_vision_model = True)
            model.vllm_engine = llm
            model.fast_generate = model.vllm_engine.generate
            model.fast_generate_batches = functools.partial(generate_batches, model.vllm_engine)
        pass

        raise_handler.remove()

        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Check float32 norm weights
        if os.environ.get("UNSLOTH_HIGH_PRECISION_LAYERNORM", "0") == "1":
            for jj, (name, module) in enumerate(model.named_modules()):
                if name.endswith("norm") and hasattr(module, "weight"):
                    module._pre_set_compute_dtype = torch.float32
        pass
        # Edit data-types
        if custom_datatype is not None:
            with torch.no_grad():
                for jj, (name, module) in enumerate(model.named_modules()):
                    exec(custom_datatype)
                pass
            pass
        pass
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):  torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu": torch.xpu.empty_cache()
        pass

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        if (whisper_language and whisper_task) or auto_model.__name__.endswith("ForConditionalGeneration"):
           tokenizer = auto_processor.from_pretrained(
                tokenizer_name,
                padding_side = "right",
                token        = token,
                language     = whisper_language,
                task         = whisper_task,
            )
        else:
            try:
                tokenizer = auto_processor.from_pretrained(
                    tokenizer_name,
                    padding_side = "right",
                    token        = token,
                )
            except:
                tokenizer = get_auto_processor(
                    tokenizer_name,
                    padding_side = "right",
                    token        = token,
                )
        if hasattr(tokenizer, "tokenizer"):
            __tokenizer = tokenizer.tokenizer
            # Add padding side as well
            __tokenizer.padding_side = "right"
            # Check bos, eos, pad tokens
            if hasattr(__tokenizer, "bos_token"):
                tokenizer.bos_token    = __tokenizer.bos_token
                tokenizer.bos_token_id = __tokenizer.bos_token_id
            if hasattr(__tokenizer, "eos_token"):
                tokenizer.eos_token    = __tokenizer.eos_token
                tokenizer.eos_token_id = __tokenizer.eos_token_id
            if hasattr(__tokenizer, "pad_token"):
                tokenizer.pad_token    = __tokenizer.pad_token
                tokenizer.pad_token_id = __tokenizer.pad_token_id
        pass
        # Fix other stuff like BnB compute data types
        model, tokenizer = patch_model_and_tokenizer(
            model,
            tokenizer,
            downcast_rope = False,
            fix_embeddings = False,
            do_forced_float32 = do_forced_float32,
            correct_dtype = correct_dtype,
        )
        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)

        # Log Unsloth version for future fastpaths for inference
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})
        pass
        patch_saving_functions(model, vision = True)
        patch_saving_functions(tokenizer, vision = True)

        # Fix gradient accumulation
        from transformers.trainer import Trainer
        patch_gradient_accumulation_fix(Trainer)

        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left" # Force inference
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "left" # Force inference
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m._saved_temp_tokenizer = tokenizer
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        m.max_seq_length = max_seq_length
        m._saved_temp_tokenizer = tokenizer
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Patch generate
        if os.environ.get("UNSLOTH_DISABLE_FAST_GENERATION", "0") == "0" and hasattr(model, 'generate'):
            if model.generate.__name__ != "unsloth_base_fast_generate":
                model._old_generate = model.generate
                unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
                model.generate = types.MethodType(unsloth_base_fast_generate, model)
        pass
        model._unsloth_trust_remote_code = trust_remote_code
        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            trust_remote_code  = trust_remote_code,
        )
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        pass
        return model, tokenizer
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                          = 16,
        target_modules             = None,
        lora_alpha                 = 16,
        lora_dropout               = 0.0,
        bias                       = "none",
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        layers_to_transform        = None,
        layers_pattern             = None,
        use_gradient_checkpointing = "unsloth",
        random_state               = 3407,
        max_seq_length             = 2048, # not used anymore
        use_rslora                 = False,
        modules_to_save            = None,
        init_lora_weights          = True,
        loftq_config               = {},
        task_type                  = TaskType.CAUSAL_LM,
        temporary_location         = "_unsloth_temporary_saved_buffers",
        **kwargs
    ):
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        pass
        transformers_set_seed(random_state)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM):
            raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")

        if target_modules == "all-linear":
            finetune_vision_layers     = True
            finetune_language_layers   = True
            finetune_attention_modules = True
            finetune_mlp_modules       = True
        pass
        if target_modules is None or target_modules == "all-linear":
            target_modules = get_peft_regex(
                model,
                finetune_vision_layers     = finetune_vision_layers,
                finetune_language_layers   = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules       = finetune_mlp_modules,
            )
        else:
            assert(type(target_modules) in (list, tuple, str,))
        pass

        if hasattr(model, "vllm_engine"):
            if hasattr(model.vllm_engine, "llm_engine") and hasattr(model.vllm_engine.llm_engine, "vllm_config") and getattr(model.vllm_engine.llm_engine.vllm_config, "lora_config", None) is None:
                # If vLLM is being used but lora is not enabled, throw an error
                # Ref https://github.com/vllm-project/vllm/blob/51ba839555a5d122eadd91e9c16463ac288f5fa1/vllm/v1/engine/processor.py#L148-L151
                raise RuntimeError("Unsloth: LoRA is not enabled for this model!")
            if finetune_vision_layers:
                # vLLM does not support LoRA on vision layers
                # https://github.com/vllm-project/vllm/blob/main/vllm/lora/models.py#L471-L477
                # TODO: Update this once vLLM V1 supports LoRA on vision layers (possibly not happening)
                raise RuntimeError("Unsloth: Finetuning vision layers is not supported for fast_inference. Only text layers are supported!")
            if model.config.model_type in VLLM_NON_LORA_VLM:
                # mllama is still only in vllm v0 https://arc.net/l/quote/llwkfgmu
                # https://docs.vllm.ai/en/stable/models/supported_models.html#text-generation_1
                # vLLM V0 does not support LoRA on multi modal models.
                # TODO: Update this once vLLM V1 supports Llama 3.2 aka mllama
                raise RuntimeError("Unsloth: LoRA finetuning for Llama 3.2 aka mllama models is not supported with fast_inference!")

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        pass
        max_seq_length = model.max_seq_length
        # If we pass loftq_config = None we will get an error
        loftq_config = validate_loftq_config(loftq_config, lora_dropout, bias, init_lora_weights, model)

        # Get only allowed parameters for LoraConfig
        local_variables = { **locals(), **kwargs, }
        del local_variables["kwargs"]
        allowed_parameters = inspect.signature(LoraConfig).parameters.keys()
        lora_config = LoraConfig(
            **{ k : v for k, v in local_variables.items() if k in allowed_parameters },
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        model = _get_peft_model(model, lora_config)
        # Fix LoraConfig.auto_mapping is None
        fix_lora_auto_mapping(model)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)
        trust_remote_code = getattr(model, "_unsloth_trust_remote_code", False)
        model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing, trust_remote_code = trust_remote_code)
        model.max_seq_length = max_seq_length
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        pass
        patch_saving_functions(model, vision = True)
        patch_peft_fast_inference(model)

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def post_patch_model(
        model,
        use_gradient_checkpointing = True,
        trust_remote_code = False,
    ):
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"

        float32_mixed_precision = True
        if _get_dtype(dtype_from_config(model.config)) == torch.bfloat16 and full_finetuning:
            # Use bfloat16 precision for full finetuning
            float32_mixed_precision = False

        model = prepare_model_for_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant              = True,
            full_finetuning            = full_finetuning,
            train_layernorms           = full_finetuning,
            train_embedding            = full_finetuning,
            train_lm_head              = full_finetuning,
            float32_mixed_precision    = float32_mixed_precision,
            patch_modules_to_save      = True,
        )

        from transformers.trainer import Trainer
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop" and trust_remote_code == False:
            raise RuntimeError('Unsloth: Unsuccessfully patched inner_training_loop')
        pass
        patch_saving_functions(model, vision = True)

        # Patch tokenizer to pad to the right
        m = model
        while hasattr(m, "model"):
            if hasattr(m, "_saved_temp_tokenizer"):
                if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                    m._saved_temp_tokenizer.tokenizer.padding_side = "right"
            pass
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        if hasattr(m, "_saved_temp_tokenizer"):
            if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                m._saved_temp_tokenizer.tokenizer.padding_side = "right"
        pass
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        pass
        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def for_inference(model):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_inference!")

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = False
            if hasattr(m, "training"): m.training = False
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "left"
            # Set a flag for generation!
            m._flag_for_generation = True
        pass
        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)
        model.eval() # to turn off training on modules deeper in

        # Since transformers 4.53, must turn off explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
        pass

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        # Must disable returning hidden states in the case for GRPO
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        # Must enable returning logits
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        # Turn off skip guards and set stance to default
        if torch_compiler_set_stance is not None:
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_training!")

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"): m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"): del m._flag_for_generation
        pass
        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)
        model.train() # to turn on training on modules deeper in

        # Since transformers 4.53, must turn on explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
        pass

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        # Can re-enable not returning logits
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        # Turn off skip guards and set stance to default
        if torch_compiler_set_stance is not None:
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
        return model
    pass
pass
