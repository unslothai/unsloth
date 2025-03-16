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
from ._utils import __version__
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

from PIL import Image
import json

from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.patching_utils import patch_model_and_tokenizer
from unsloth_zoo.training_utils import prepare_model_for_training
import types
import functools
import os
import gc
import math
import functools
from typing import Optional, Tuple, List, Union
import re, inspect, sys
import types
try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
pass


__all__ = [
    "FastBaseModel",
]

global FORCE_FLOAT32
FORCE_FLOAT32 = [
    "gemma3",
]

global FORCE_EAGER_ATTENTION
FORCE_EAGER_ATTENTION = [
    "pixtral",    # Pixtral SDPA not implemented
]

global NUM_LOGITS_TO_KEEP
NUM_LOGITS_TO_KEEP = dict()

def unsloth_base_fast_generate(
    self,
    *args,
    **kwargs,
):
    FastBaseModel.for_inference(self)
    dtype = _get_dtype(self.config.torch_dtype)

    # Check if VLM
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in self.config.architectures
    )
    is_vlm = is_vlm or hasattr(self.config, "vision_config")
    arch = self.config.architectures[0]

    # Remove token_type_ids
    kwargs.pop("token_type_ids", None)

    # VLMs do not allow logits_to_keep
    if not is_vlm:
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
    else:
        pass
        # kwargs.pop("logits_to_keep", None)
        # kwargs.pop("num_logits_to_keep", None)

    # Check pad_token
    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Get pixel values for VLMs
    try: kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype)
    except: pass

    # Mixed precision autocast
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1": dtype = torch.float32
    with torch.inference_mode(), torch.autocast(device_type = "cuda", dtype = dtype):
        output = self._old_generate(*args, **kwargs)
    pass

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
        max_image_width   = None,
        max_image_height  = None,
        maintain_image_aspect_ratio = True,
        auto_model        = AutoModelForVision2Seq,
        use_gradient_checkpointing = "unsloth",
        **kwargs,
    ):
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        from importlib.metadata import version as importlib_version
        try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
        except: vllm_version = ""

        model_type_arch = model_types[0]
        if model_type_arch == "siglip" and len(model_types) != 1:
            model_type_arch = model_types[1]

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_type_arch.title()} patching. Transformers: {transformers_version}.{vllm_version}\n"\
           f"   {chr(92)}{chr(92)}   /|    {gpu_stats.name}. Num GPUs = {torch.cuda.device_count()}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
           f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {triton_version}\n"\
           f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        print(statistics)

        # Warn about fast transfers
        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        get_statistics() # For debugging - we use a download counter to see if environments are not breaking 

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        global FORCE_FLOAT32
        os.environ["UNSLOTH_FORCE_FLOAT32"] = "0"
        bnb_compute_dtype = dtype
        for disable_name in FORCE_FLOAT32:
            if (disable_name.lower() == model_type_arch.lower() or \
                disable_name.lower() in model_name.lower()) and \
                dtype == torch.float16:

                print(f"Unsloth: Using float16 precision for {model_type_arch} won't work! Using float32.")
                os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
                bnb_compute_dtype = torch.float32
                break
        pass

        global FORCE_EAGER_ATTENTION
        attn_implementation = "sdpa"
        for disable_name in FORCE_EAGER_ATTENTION:
            if (disable_name.lower() == model_type_arch.lower() or \
                disable_name.lower() in model_name.lower()):

                print(f"Unsloth: {model_type_arch} does not support SDPA - switching to eager!")
                attn_implementation = "eager"
                break
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
            print("Unsloth: LoRA, QLoRA and full finetuning all not selected. Switching to QLoRA.")
            load_in_4bit = True
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = bnb_compute_dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES.copy(),
            )
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

        kwargs.pop("attn_implementation", None); # No need since we auto call it

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config

        model = auto_model.from_pretrained(
            model_name,
            device_map              = device_map,
            torch_dtype             = dtype,
            # quantization_config   = bnb_config,
            token                   = token,
            trust_remote_code       = trust_remote_code,
            attn_implementation     = attn_implementation,
            **kwargs,
        )
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        auto_processor = AutoProcessor if auto_model is AutoModelForVision2Seq else AutoTokenizer
        tokenizer = auto_processor.from_pretrained(
            tokenizer_name,
            padding_side = "right",
            token        = token,
        )

        # Add padding side as well
        tokenizer.tokenizer.padding_side = "right"

        # Check for image size configuration in model config
        if max_image_width is None or max_image_height is None:
            try:
                # Try to get model configuration path
                config_path = None
                if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                    model_path = model.config._name_or_path
                    if os.path.isdir(model_path):
                        config_path = os.path.join(model_path, "config.json")
                
                # If we couldn't get a local path, try to get from the cache
                if config_path is None or not os.path.isfile(config_path):
                    from huggingface_hub import cached_file
                    try:
                        config_path = cached_file(model_name, "config.json", token=token)
                    except:
                        # Failed to get config path, will use defaults
                        pass
                
                # If we have a config path, try to get image size from it
                if config_path is not None and os.path.isfile(config_path):
                    config_width, config_height = FastBaseVisionModel.get_max_image_size_from_config(config_path)
                    if max_image_width is None and config_width is not None:
                        max_image_width = config_width
                        logger.warning_once(f"Unsloth: Using maximum image width of {max_image_width} from model config")
                    if max_image_height is None and config_height is not None:
                        max_image_height = config_height
                        logger.warning_once(f"Unsloth: Using maximum image height of {max_image_height} from model config")
            except Exception as e:
                logger.warning(f"Failed to extract image size from model config: {e}")
        
        # Apply image resizing if dimensions are specified
        if max_image_width is not None or max_image_height is not None:
            # Patch the processor to use our image resizing
            tokenizer = FastBaseVisionModel.patch_processor_with_image_resizing(
                tokenizer, 
                max_image_width, 
                max_image_height, 
                maintain_image_aspect_ratio
            )
            
            # Print information about image resizing
            width_info = f"{max_image_width}" if max_image_width is not None else "default"
            height_info = f"{max_image_height}" if max_image_height is not None else "default"
            aspect_ratio = "maintaining aspect ratio" if maintain_image_aspect_ratio else "ignoring aspect ratio"
            logger.warning_once(f"Unsloth: Image resizing enabled with max dimensions {width_info}x{height_info}, {aspect_ratio}")

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

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)
        # Fix other stuff like BnB compute data types
        model, tokenizer = patch_model_and_tokenizer(
            model,
            tokenizer,
            downcast_rope = False,
            fix_embeddings = False,
        )

        # Log Unsloth version for future fastpaths for inference
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})
            # Store image resizing configuration in model config
            if max_image_width is not None or max_image_height is not None:
                model.config.update({
                    "unsloth_max_image_width": max_image_width,
                    "unsloth_max_image_height": max_image_height,
                    "unsloth_maintain_image_aspect_ratio": maintain_image_aspect_ratio,
                })
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
        if model.generate.__name__ != "unsloth_base_fast_generate":
            model._old_generate = model.generate
            unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_base_fast_generate, model)

        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        return model, tokenizer
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                          = 16,
        target_modules             = None,
        lora_alpha                 = 16,
        lora_dropout               = 0,
        bias                       = "none",
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        layers_to_transform        = None,
        layers_pattern             = None,
        use_gradient_checkpointing = True,
        random_state               = 3407,
        max_seq_length             = 2048, # not used anymore
        use_rslora                 = False,
        modules_to_save            = None,
        init_lora_weights          = True,
        loftq_config               = {},
        temporary_location         = "_unsloth_temporary_saved_buffers",
        **kwargs,
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
        if target_modules is None:
            target_modules = get_peft_regex(
                model,
                finetune_vision_layers     = finetune_vision_layers,
                finetune_language_layers   = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules       = finetune_mlp_modules,
            )
        else:
            assert(type(target_modules) in (list, tuple,))
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        max_seq_length = model.max_seq_length
        lora_config = LoraConfig(
            r               = r,
            lora_alpha      = lora_alpha,
            target_modules  = target_modules,
            lora_dropout    = lora_dropout,
            bias            = bias,
            task_type       = TaskType.CAUSAL_LM,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        model = _get_peft_model(model, lora_config)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)

        model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing)
        model.max_seq_length = max_seq_length

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        patch_saving_functions(model, vision = True)

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def post_patch_model(
        model,
        use_gradient_checkpointing = True,
    ):
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"

        float32_mixed_precision = True
        if _get_dtype(model.config.torch_dtype) == torch.bfloat16 and full_finetuning:
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
        )

        from transformers.trainer import Trainer 
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
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
            torch.cuda.empty_cache()
        pass
        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)

        # Patch generate
        if model.generate.__name__ != "unsloth_base_fast_generate":
            model._old_generate = model.generate
            unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_base_fast_generate, model)
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

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
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

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        return model
    pass

    @staticmethod
    def resize_image(image, max_width, max_height, maintain_aspect_ratio=True):
        """Basic image resizing function that maintains aspect ratio by default."""
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                return image
            raise ValueError(f"Expected PIL Image or Tensor, got {type(image)}")
        
        # If both dimensions are None, return the original image
        if max_width is None and max_height is None:
            return image
        
        # Convert dimensions to integers
        max_width = int(max_width) if max_width is not None else None
        max_height = int(max_height) if max_height is not None else None
        
        # Get current dimensions
        width, height = image.size
        
        # If one dimension is None, use the other dimension to maintain aspect ratio
        if max_width is None:
            max_width = int(width * (max_height / height)) if max_height < height else width
        if max_height is None:
            max_height = int(height * (max_width / width)) if max_width < width else height
        
        # If the image is already smaller than the maximum dimensions, return it as is
        if width <= max_width and height <= max_height:
            return image
        
        if maintain_aspect_ratio:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
        else:
            new_width, new_height = max_width, max_height
        
        # Ensure dimensions are at least 1 pixel
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        return image.resize((new_width, new_height), Image.LANCZOS)

    @staticmethod
    def get_max_image_size_from_config(config_path):
        """Extract image dimensions from config file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check vision_config first
            if "vision_config" in config and "image_size" in config["vision_config"]:
                size = config["vision_config"]["image_size"]
                if isinstance(size, int):
                    return size, size
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    return size[0], size[1]
            
            # Check other common fields
            for field in ["image_size", "max_image_size", "visual_image_size"]:
                if field in config:
                    size = config[field]
                    if isinstance(size, int):
                        return size, size
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        return size[0], size[1]
            
            return None, None
        except Exception as e:
            print(f"Error reading config file at {config_path}: {e}")
            return None, None

    @staticmethod
    def patch_processor_with_image_resizing(processor, max_width=None, max_height=None, maintain_aspect_ratio=True):
        """Patch an image processor to resize images before processing."""
        if not max_width and not max_height:
            return processor
        
        # Create a wrapper class to handle resizing
        class ResizingProcessorWrapper:
            def __init__(self, processor, max_width, max_height, maintain_aspect_ratio):
                self.processor = processor
                self.max_width = max_width
                self.max_height = max_height
                self.maintain_aspect_ratio = maintain_aspect_ratio
                self._image_resizing_config = {
                    'max_width': max_width,
                    'max_height': max_height,
                    'maintain_aspect_ratio': maintain_aspect_ratio
                }
                
                # Copy processor attributes
                for attr in dir(processor):
                    if not attr.startswith('_') and not hasattr(self, attr):
                        setattr(self, attr, getattr(processor, attr))
            
            def __call__(self, images=None, **kwargs):
                """Process images with resizing."""
                # Make a copy of kwargs to avoid modifying the original
                kwargs_copy = kwargs.copy()
                
                if images is not None:
                    # Handle images passed as a positional argument
                    if isinstance(images, (list, tuple)):
                        images = [FastBaseVisionModel.resize_image(img, self.max_width, self.max_height, self.maintain_aspect_ratio) for img in images]
                    else:
                        images = FastBaseVisionModel.resize_image(images, self.max_width, self.max_height, self.maintain_aspect_ratio)
                    return self.processor(images=images, **kwargs_copy)
                elif 'images' in kwargs_copy:
                    # Handle images passed as a keyword argument
                    if isinstance(kwargs_copy['images'], (list, tuple)):
                        kwargs_copy['images'] = [FastBaseVisionModel.resize_image(img, self.max_width, self.max_height, self.maintain_aspect_ratio) for img in kwargs_copy['images']]
                    else:
                        kwargs_copy['images'] = FastBaseVisionModel.resize_image(kwargs_copy['images'], self.max_width, self.max_height, self.maintain_aspect_ratio)
                
                # Pass all kwargs to the processor
                return self.processor(**kwargs_copy)
        
        # Create and return a wrapper instance
        return ResizingProcessorWrapper(processor, max_width, max_height, maintain_aspect_ratio)
pass
