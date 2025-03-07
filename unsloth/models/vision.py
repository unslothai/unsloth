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
    AutoModelForVision2Seq,
    AutoProcessor,
)
from .llama import *
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import __version__
from peft import LoraConfig, TaskType, get_peft_model
from transformers import set_seed as transformers_set_seed
from unsloth_zoo.peft_utils import (
    get_peft_regex,
    SKIP_QUANTIZATION_MODULES,
    requires_grad_for_gradient_checkpointing,
)
from triton import __version__ as triton_version
from PIL import Image
import json
import os

__all__ = [
    "FastBaseVisionModel",
]

def _wrap_fast_inference(generate, device_type, dtype, model):
    # Wraps inference with bfloat16 / float16
    @torch.inference_mode
    def _fast_generate(*args, **kwargs):
        # For num_logits_to_keep
        # kwargs["num_logits_to_keep"] = 1

        # Remove token_type_ids
        kwargs.pop("token_type_ids", None)

        # Check pad_token
        model_eos_token_id = getattr(model.config, "eos_token_id", None)
        if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
            model_eos_token_id = model_eos_token_id[0]

        kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

        try:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(model.dtype)
        except:
            pass

        # Autocasted
        with torch.autocast(device_type = device_type, dtype = dtype):
            output = generate(*args, **kwargs)
        pass
        return output
    pass
    return _fast_generate
pass


class FastBaseVisionModel:

    @staticmethod
    def from_pretrained(
        model_name        = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length    = None,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        trust_remote_code = False,
        model_types       = None,
        tokenizer_name    = None,
        max_image_width   = None,
        max_image_height  = None,
        maintain_image_aspect_ratio = True,
        **kwargs,
    ):
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

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_types[0].title()} vision patching. Transformers: {transformers_version}.\n"\
           f"   {chr(92)}{chr(92)}   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
           f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {triton_version}\n"\
           f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free Apache license: http://github.com/unslothai/unsloth'
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

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES,
            )
        pass

        kwargs.pop("attn_implementation", None); # No need since we auto call it

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map              = device_map,
            torch_dtype             = dtype,
            # quantization_config   = bnb_config,
            token                   = token,
            trust_remote_code       = trust_remote_code,
            # attn_implementation   = "sdpa", [TODO] Pixtral for eg fails
            **kwargs,
        )
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        tokenizer = AutoProcessor.from_pretrained(
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

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)

        # Fix up config for transformers uploading PEFT
        # Not necessary anymore since we require transformers>=4.37!
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[:len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path" : name})
            pass
        pass

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
        tokenizer.tokenizer.padding_side = "left" # Force inference
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model._saved_temp_tokenizer = tokenizer
            internal_model = internal_model.model
        pass
        internal_model._saved_temp_tokenizer = tokenizer
        
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
        model = get_peft_model(model, lora_config)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)

        model = FastBaseVisionModel.patch_peft_model(model, use_gradient_checkpointing)

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        patch_saving_functions(model, vision = True)

        return model
    pass


    @staticmethod
    def patch_peft_model(
        model,
        use_gradient_checkpointing = True,
    ):
        if not isinstance(model, PeftModelForCausalLM):
            raise TypeError(
                "Unsloth: Your model needs to call `.get_peft_model` first!"
            )
        pass

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )

        from transformers.trainer import Trainer 
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            raise RuntimeError(
                'Unsloth currently does not work on multi GPU setups - sadly we are a 2 brother team so '\
                'enabling it will require much more work, so we have to prioritize. Please understand!\n'\
                'We do have a separate beta version, which you can contact us about!\n'\
                'Thank you for your understanding and we appreciate it immensely!'
            )
        pass
        patch_saving_functions(model, vision = True)

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.tokenizer.padding_side = "right"
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.tokenizer.padding_side = "right"
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        return model
    pass


    @staticmethod
    def for_inference(model):
        model.gradient_checkpointing = False
        model.training = False

        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            if hasattr(module, "training"):
                module.training = False
        pass

        dtype = model.config.torch_dtype
        if type(dtype) is str:
            if   dtype ==  "float16": dtype = torch.float16
            elif dtype == "bfloat16": dtype = torch.bfloat16
        pass
        device_type = model.device.type

        # Wrap model.generate
        if model.generate.__name__ != "_fast_generate":
            model._unwrapped_old_generate = model.generate
            model.generate = _wrap_fast_inference(model.generate, device_type, dtype, model)
        pass
        
        # Patch tokenizer to pad to the left
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.tokenizer.padding_side = "left"
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.tokenizer.padding_side = "left"
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

        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        model.gradient_checkpointing = use_gradient_checkpointing
        model.training = True

        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(module, "training"):
                module.training = True
        pass

        # Also revert model.generate
        if hasattr(model, "_unwrapped_old_generate"):
            model.generate = model._unwrapped_old_generate
            del model._unwrapped_old_generate
        pass

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.tokenizer.padding_side = "right"
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.tokenizer.padding_side = "right"
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
