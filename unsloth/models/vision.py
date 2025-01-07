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
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
           f"O^O/ \_/ \\    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {triton_version}\n"\
           f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
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

        # We currently only support NVIDIA GPUs - AMD / Intel is a work in progress!
        pre_check = check_nvidia()

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
        # We currently only support NVIDIA GPUs - AMD / Intel is a work in progress!
        post_check = check_nvidia()

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_name,
            padding_side = "right",
            token        = token,
        )
        # Add padding side as well
        tokenizer.tokenizer.padding_side = "right"

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
pass
