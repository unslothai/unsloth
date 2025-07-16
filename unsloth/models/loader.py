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

from typing import Optional
from ._utils import (
    is_bfloat16_supported,
    is_vLLM_available,
    HAS_FLASH_ATTENTION,
    HAS_FLASH_ATTENTION_SOFTCAPPING,
    USE_MODELSCOPE,
    get_transformers_model_type,
)
from .granite import FastGraniteModel
from .llama   import FastLlamaModel, logger
from .mistral import FastMistralModel
from .qwen2   import FastQwen2Model
from .qwen3   import FastQwen3Model
from .qwen3_moe import FastQwen3MoeModel
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
SUPPORTS_FOURBIT   = transformers_version >= Version("4.37")
SUPPORTS_GEMMA     = transformers_version >= Version("4.38")
SUPPORTS_GEMMA2    = transformers_version >= Version("4.42")
SUPPORTS_LLAMA31   = transformers_version >= Version("4.43.2")
SUPPORTS_LLAMA32   = transformers_version  > Version("4.45.0")
SUPPORTS_GRANITE   = transformers_version >= Version("4.46.0")
SUPPORTS_QWEN3     = transformers_version >= Version("4.50.3")
SUPPORTS_QWEN3_MOE = transformers_version >= Version("4.50.3")
SUPPORTS_FALCON_H1 = transformers_version >= Version("4.53.0")
SUPPORTS_GEMMA3N   = transformers_version >= Version("4.53.0")

if SUPPORTS_GEMMA:
    from .gemma  import FastGemmaModel
if SUPPORTS_GEMMA2:
    from .gemma2 import FastGemma2Model
pass
if SUPPORTS_FALCON_H1:
    from .falcon_h1 import FastFalconH1Model
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
    """
    Factory class for loading optimized language models with Unsloth acceleration.
    
    This class automatically dispatches to the appropriate optimized model implementation
    based on the model type. It provides hardware-accelerated implementations for popular
    model architectures while falling back to general optimization for unsupported models.
    
    Natively Supported Models:
        - Llama family (including Llama 3.1/3.2 with RoPE scaling)
        - Mistral/Mixtral models  
        - Gemma and Gemma 2
        - Qwen2 and Qwen3 (including MoE variants)
        
    Partial/Experimental Support:
        - Cohere models (temporarily disabled)
        - Granite models (temporarily disabled)
        
    For unsupported model architectures:
        - Falls back to FastModel.from_pretrained() with general optimizations
        - Still benefits from Unsloth's memory optimizations and training improvements
        - May not achieve the same speedups as natively supported models
    """
    
    @staticmethod
    def from_pretrained(
        model_name: str                   = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int               = 2048,
        dtype: Optional[torch.dtype]      = None,
        load_in_4bit: bool                = True,
        load_in_8bit: bool                = False,
        full_finetuning: bool             = False,
        token: Optional[str]              = None,
        device_map: str                   = "sequential",
        rope_scaling: Optional[dict]      = None,
        fix_tokenizer: bool               = True,
        trust_remote_code: bool           = False,
        use_gradient_checkpointing: str   = "unsloth",
        resize_model_vocab: Optional[int] = None,
        revision: Optional[str]           = None,
        use_exact_model_name: bool        = False,

        fast_inference: bool              = False, # uses vLLM
        gpu_memory_utilization: float     = 0.5,
        float8_kv_cache: bool             = False,
        random_state: int                 = 3407,
        max_lora_rank: int                = 64,
        disable_log_stats: bool           = True,
        *args, **kwargs,
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load a pretrained model with Unsloth optimizations.
        
        This method automatically detects the model architecture and applies the appropriate
        optimizations. For natively supported models (Llama, Mistral, Gemma, Qwen2/3), it
        uses custom kernels and memory-efficient implementations. For other models, it falls
        back to general optimizations.
        
        Native Support Behavior:
            - Custom CUDA kernels for attention and MLP layers
            - Optimized RoPE implementations
            - Memory-efficient LoRA patching
            - Support for advanced features like RoPE scaling
            
        Fallback Behavior (Unsupported Models):
            - Uses FastModel with general PyTorch optimizations
            - Still benefits from gradient checkpointing and memory management
            - May not support all advanced features (e.g., custom RoPE scaling)
            - Shows warning about potentially slower performance
            
        Args:
            model_name (`str`): Model identifier from HuggingFace Hub
            max_seq_length (`int`): Maximum sequence length
            dtype (`Optional[torch.dtype]`): Data type (auto-selected if None)
            load_in_4bit (`bool`): Enable 4-bit quantization
            load_in_8bit (`bool`): Enable 8-bit quantization
            full_finetuning (`bool`): Enable full model finetuning (disables LoRA)
            token (`Optional[str]`): HuggingFace API token
            device_map (`str`): Device mapping strategy
            rope_scaling (`Optional[dict]`): RoPE scaling configuration
            fix_tokenizer (`bool`): Fix common tokenizer issues
            trust_remote_code (`bool`): Trust remote code execution
            use_gradient_checkpointing (`str`): Gradient checkpointing mode
            resize_model_vocab (`Optional[int]`): Resize vocabulary size
            revision (`Optional[str]`): Model revision
            use_exact_model_name (`bool`): Use exact model name without modifications
            fast_inference (`bool`): Enable vLLM-based fast inference
            gpu_memory_utilization (`float`): GPU memory utilization for vLLM
            float8_kv_cache (`bool`): Use float8 KV cache
            random_state (`int`): Random seed
            max_lora_rank (`int`): Maximum LoRA rank for vLLM
            disable_log_stats (`bool`): Disable logging statistics
            
        Returns:
            tuple[PeftModel, PreTrainedTokenizer]: Model and tokenizer
            
        Raises:
            ImportError: If required dependencies are missing for specific models
            RuntimeError: If model configuration conflicts are detected
        """
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
        if isinstance(dtype, str) and dtype in ["float16", "bfloat16"]:
            dtype = getattr(torch, dtype)
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
        elif model_type == "qwen3":# or model_type == "qwen3_moe":
            if not SUPPORTS_QWEN3 or not SUPPORTS_QWEN3_MOE:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3.\n"\
                    f"The minimum required version is 4.50.3.\n"\
                    f'Try `pip install --upgrade "transformers>=4.50.3"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            dispatch_model = FastQwen3Model if model_type == "qwen3" else FastQwen3MoeModel
        elif model_type == "falcon_h1":
            dispatch_model = FastFalconH1Model
            if not SUPPORTS_FALCON_H1:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support FalconH1.\n"\
                    f"The minimum required version is 4.50.3.\n"\
                    f'Try `pip install --upgrade "transformers>=4.50.3"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
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
            if not is_vLLM_available():
                print("Unsloth: vLLM is not installed! Will use Unsloth inference!")
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

DISABLE_COMPILE_MODEL_NAMES = [
    "aya-vision",
    "modernbert",
    "granite-vision",
]


class FastModel(FastBaseModel):
    """
    General-purpose model loader with Unsloth optimizations for any transformer architecture.
    
    This class provides a fallback implementation that works with any HuggingFace transformer
    model, applying general optimizations without model-specific kernels. It's used when
    FastLanguageModel encounters an unsupported model architecture.
    
    Key Features:
        - Automatic torch.compile optimization with custom settings
        - Memory-efficient gradient checkpointing
        - Support for 4-bit/8-bit quantization
        - General attention and MLP optimizations
        - Compatible with any AutoModelForCausalLM-compatible model
        
    Limitations vs Native Support:
        - No custom CUDA kernels (uses PyTorch operations)
        - May not support model-specific features (e.g., custom RoPE)
        - Performance gains are model-dependent
        - Some features marked with [TODO] No effect
    """
    
    @staticmethod
    def from_pretrained(
        model_name: str                   = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length: int               = 2048,
        dtype: Optional[torch.dtype]      = None,
        load_in_4bit: bool                = True,
        load_in_8bit: bool                = False,
        full_finetuning: bool             = False,
        token: Optional[str]              = None,
        device_map: str                   = "sequential",
        rope_scaling: Optional[dict]      = None, # [TODO] No effect
        fix_tokenizer: bool               = True, # [TODO] No effect
        trust_remote_code: bool           = False,
        use_gradient_checkpointing: str   = "unsloth",
        resize_model_vocab: Optional[int] = None, # [TODO] No effect
        revision: Optional[str]           = None,
        return_logits: bool               = False, # Return logits
        fullgraph: bool                   = True, # No graph breaks
        use_exact_model_name: bool        = False,
        auto_model: Optional[type]        = None,
        whisper_language: Optional[str]   = None,
        whisper_task: Optional[str]       = None,
        unsloth_force_compile: bool       = False,
        *args, **kwargs,
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load any transformer model with general Unsloth optimizations.
        
        This method applies torch.compile and other general optimizations to any
        HuggingFace transformer model. It automatically detects model capabilities
        and applies the best available optimizations without model-specific kernels.
        
        Optimization Strategy:
            - Uses torch.compile with custom optimization passes
            - Applies memory-efficient attention implementations (SDPA)
            - Enables gradient checkpointing for memory savings
            - Supports quantization via bitsandbytes
            - Automatically detects and handles vision models
            
        Special Model Handling:
            - Vision models: Automatically uses AutoModelForVision2Seq
            - Whisper models: Supports language/task specification
            - Models with custom requirements: Sets appropriate environment flags
            
        Args:
            model_name (`str`): Model identifier from HuggingFace Hub
            max_seq_length (`int`): Maximum sequence length
            dtype (`Optional[torch.dtype]`): Data type (auto-selected if None)
            load_in_4bit (`bool`): Enable 4-bit quantization
            load_in_8bit (`bool`): Enable 8-bit quantization
            full_finetuning (`bool`): Enable full model finetuning
            token (`Optional[str]`): HuggingFace API token
            device_map (`str`): Device mapping strategy
            rope_scaling (`Optional[dict]`): RoPE scaling config (no effect in FastModel)
            fix_tokenizer (`bool`): Fix tokenizer issues (no effect in FastModel)
            trust_remote_code (`bool`): Trust remote code execution
            use_gradient_checkpointing (`str`): Gradient checkpointing mode
            resize_model_vocab (`Optional[int]`): Resize vocab (limited support)
            revision (`Optional[str]`): Model revision
            return_logits (`bool`): Return raw logits instead of loss
            fullgraph (`bool`): Enable fullgraph compilation (no graph breaks)
            use_exact_model_name (`bool`): Use exact model name
            auto_model (`Optional[type]`): Custom auto model class
            whisper_language (`Optional[str]`): Language for Whisper models
            whisper_task (`Optional[str]`): Task for Whisper models
            unsloth_force_compile (`bool`): Force torch.compile even if disabled
            
        Returns:
            tuple[PeftModel, PreTrainedTokenizer]: Optimized model and tokenizer
            
        Note:
            Parameters marked with [TODO] No effect are placeholders for compatibility
            but don't have implementations in the general FastModel path.
        """
        if token is None: token = get_token()
        if whisper_language is not None: assert(type(whisper_language) is str)
        if whisper_task is not None: assert(type(whisper_task) is str)
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
        lowered_model_name = model_name.lower()
        LATEST  = '\nPlease use transformers via `pip install --no-deps git+https://github.com/huggingface/transformers.git`'
        NIGHTLY = '\nPlease use nightly transformers via pip install --upgrade "transformers>=4.49.0"`'
        # Pixtral
        if "pixtral" in lowered_model_name and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Pixtral only works on transformers >= 4.49.0." + LATEST)
        # Qwen 2.5
        elif "qwen2.5" in lowered_model_name and transformers_version < Version("4.49.0"):
            raise RuntimeError("Unsloth: Qwen 2.5 only works on transformers >= 4.49.0." + LATEST)
        # Gemma 3
        elif "gemma-3" in lowered_model_name and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: Gemma 3 only works on transformers >= 4.50.0." + NIGHTLY)
        # Cohere
        elif "c4ai-command-a-03-2025" in lowered_model_name and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: Cohere's Command model only works on transformers >= 4.50.0." + NIGHTLY)
        # Sesame
        elif "csm-1b" in lowered_model_name:
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" # Inference is too slow
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1" # Sesame fails
            os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"] = \
                "all;torch.float32;torch.float16;"\
                "if name.endswith(('_proj', 'fc1', 'fc2', 'codebook', 'head')): module.to(torch.float16);"
        # Granite 4
        elif 'granite-4' in lowered_model_name:
            # granite-4 rms norms are stored as 16 bit, but we upcast
            os.environ["UNSLOTH_UPCAST_LAYERNORM"] = "1"
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
        # Olmo 2
        elif "olmo-2" in lowered_model_name and transformers_version < Version("4.50.0.dev0"):
            raise RuntimeError("Unsloth: OLMo-2 only works on transformers >= 4.50.0." + NIGHTLY)
        # Gemma 3N
        elif "gemma-3n" in lowered_model_name:
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
            os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"] = \
                "float16;torch.float16;torch.float16;"\
                "if name.endswith(('.conv')): module;"\
                "from unsloth_zoo.temporary_patches.gemma3n import patch_Gemma3nConvNormAct_forward; patch_Gemma3nConvNormAct_forward()"
            
            if transformers_version < Version("4.53.0"):
                raise RuntimeError("Unsloth: Gemma 3N only works on transformers >= 4.53.0" + LATEST)
        else:
            for check_model_name in DISABLE_COMPILE_MODEL_NAMES:
                if check_model_name in lowered_model_name:
                    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
                    os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
                    if transformers_version < Version("4.50.0.dev0"):
                        raise RuntimeError(f"Unsloth: {check_model_name} only works on transformers >= 4.50.0." + NIGHTLY)
                    break
        pass

        if auto_model is not None:
            # All other models need to disable static cache
            os.environ["UNSLOTH_DISABLE_STATIC_GENERATION"] = "1"
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
                unsloth_force_compile   = unsloth_force_compile,
            )
        pass
        # Fix SDPA
        if "gemma-3n" in lowered_model_name:
            supports_sdpa = False
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
        if auto_model is None:
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
            whisper_language  = whisper_language,
            whisper_task      = whisper_task,            
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
    """
    Specialized model loader for vision-language models (VLMs).
    
    This class inherits all functionality from FastModel and serves as a 
    semantic alias for loading vision-language models like LLaVA, CLIP-based
    models, and other multimodal architectures. It uses the same general
    optimization approach as FastModel.
    
    Examples of supported models:
        - LLaVA variants
        - Pixtral
        - Qwen-VL
        - Any AutoModelForVision2Seq compatible model
    """
    pass

class FastTextModel(FastModel):
    """
    Specialized model loader for text-only language models.
    
    This class inherits all functionality from FastModel and serves as a
    semantic alias for loading pure text generation models. It provides
    the same optimizations as FastModel but with clearer intent for
    text-only use cases.
    
    Use this when you explicitly want to load text-only models and want
    to distinguish them from vision-language models in your codebase.
    """
    pass
