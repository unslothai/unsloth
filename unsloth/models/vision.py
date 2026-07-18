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
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import (
    __version__,
    importlib_version,
    _prepare_model_for_qat,
    resolve_model_class,
    resolve_attention_implementation,
    _get_text_only_config,
    _is_family_text_decoder,
    _apply_text_only_key_mapping,
    _select_moe_detection_targets,
    set_task_config_attr,
)
from ._utils import *
from .loader_utils import (
    _exclude_rope_inv_freq_from_ddp,
    _get_fp8_mode_and_check_settings,
    _restore_dropped_fp8_scales,
)
from ..save import patch_saving_functions
from ..models.loader_utils import is_distributed
from unsloth_zoo.gradient_checkpointing import (
    unpatch_unsloth_gradient_checkpointing,
    unpatch_unsloth_smart_gradient_checkpointing,
)
import torch.utils.checkpoint as torch_checkpoint
import transformers.modeling_utils as hf_modeling_utils
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
import warnings
from typing import Optional, Tuple, List, Union
import re, inspect, sys
import contextlib

try:
    from huggingface_hub.utils import get_token
except:
    # Old HF Hub versions <= 0.0.25
    from huggingface_hub.utils._token import get_token
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

__all__ = [
    "FastBaseModel",
]


def _infer_device_map_from_loaded_model(model):
    """Build a compact device_map by inspecting actual parameter placements."""
    device_map = {}

    def _assign(module, prefix):
        params = list(module.named_parameters(remove_duplicate = False))
        if not params:
            bufs = list(module.named_buffers())
            if bufs:
                device_map[prefix] = bufs[0][1].device
            return
        devices = {p.device for _, p in params}
        if len(devices) == 1:
            device_map[prefix] = next(iter(devices))
        else:
            for child_name, child in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                _assign(child, child_prefix)
            for pname, param in module.named_parameters(remove_duplicate = False):
                if "." not in pname:
                    full = f"{prefix}.{pname}" if prefix else pname
                    if not any(full == k or full.startswith(k + ".") for k in device_map):
                        device_map[full] = param.device

    _assign(model, "")
    if "" in device_map and len(device_map) > 1:
        device_map.pop("")
    return device_map


def _attach_bnb_multidevice_hooks(
    model, load_in_4bit, load_in_8bit, offload_embedding, fast_inference
):
    """
    Attach accelerate AlignDevicesHook on a bnb model loaded across multiple
    devices (or a non-default device).  No-op for single-GPU cuda:0, non-bnb,
    vLLM, or already-dispatched models.
    """
    if fast_inference:
        return
    is_bnb = (
        load_in_4bit
        or load_in_8bit
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "quantization_method", None) == "bitsandbytes"
    )
    if not is_bnb:
        return
    if offload_embedding:
        return
    if getattr(model, "hf_device_map", None) is not None:
        return  # already dispatched

    try:
        all_devs = {p.device for p in model.parameters()}
    except Exception as exc:
        warnings.warn(
            "Unsloth: Failed to determine device placement from model parameters, "
            f"so multi-GPU hooks cannot be attached. ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel = 2,
        )
        return

    cuda_devs = {d for d in all_devs if d.type == "cuda"}
    if not cuda_devs:
        return

    default_cuda = torch.device("cuda", 0)
    if all_devs == {default_cuda}:
        return

    try:
        from accelerate import dispatch_model
    except ImportError:
        return  # accelerate not available

    try:
        inferred_map = _infer_device_map_from_loaded_model(model)
        if not inferred_map:
            return

        # bnb constructors reject _is_hf_initialized; strip before dispatch.
        _extra_keys = ("_is_hf_initialized",)
        _stripped = []
        for _, param in model.named_parameters():
            for key in _extra_keys:
                if key in param.__dict__:
                    _stripped.append((param, key, param.__dict__.pop(key)))

        try:
            # CUDA -> int index, non-CUDA -> type string ("cpu", "meta").
            device_map_int = {
                k: (v.index if v.type == "cuda" else v.type) if isinstance(v, torch.device) else v
                for k, v in inferred_map.items()
            }

            # force_hooks=True: install hooks even for single-device maps.
            main_device = device_map_int.get("")
            if main_device in (None, "cpu", "disk"):
                main_device = next(
                    (d for d in device_map_int.values() if d not in ("cpu", "disk")),
                    None,
                )
            dispatch_model(
                model,
                device_map = device_map_int,
                main_device = main_device,
                skip_keys = getattr(model, "_skip_keys_device_placement", None),
                force_hooks = True,
            )
            desc = f"{len(inferred_map)} block(s) across {len(cuda_devs)} device(s)"
        finally:
            # Restore stripped keys.
            for param, key, val in _stripped:
                param.__dict__[key] = val

        logger.info(
            f"Unsloth: Attached accelerate AlignDevicesHook ({desc}) "
            f"for bnb multi-GPU inference."
        )
    except Exception as exc:
        warnings.warn(
            f"Unsloth: Could not attach multi-device dispatch hooks automatically "
            f"({type(exc).__name__}: {exc}). "
            "Cross-device inference may fail. Consider using a single GPU or "
            "calling accelerate.dispatch_model() manually.",
            RuntimeWarning,
            stacklevel = 2,
        )


global NUM_LOGITS_TO_KEEP
NUM_LOGITS_TO_KEEP = dict()


def _unsloth_generate_accepts_kwarg(model, key):
    # True if the top level accepts this generate kwarg (some models expose it on an inner forward only).
    try:
        model_args = set(inspect.signature(model.prepare_inputs_for_generation).parameters)
    except (TypeError, ValueError, AttributeError):
        model_args = set()
    if "kwargs" in model_args or "model_kwargs" in model_args:
        try:
            model_args |= set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError, AttributeError):
            pass
    return key in model_args


def _install_offload_embedding_hooks(embed_tokens, output_embeddings, return_device):
    # Lookup runs on the weight's current device (CPU when offloaded); the output returns to the
    # decoder device read live from output_embeddings (lm_head, untied here) so it tracks
    # model.to() moves. A meta (disk-offloaded) or missing lm_head falls back to return_device.
    if embed_tokens is None:
        return False
    if getattr(embed_tokens, "_unsloth_offload_hooks_installed", False):
        return True

    def _decoder_device():
        weight = getattr(output_embeddings, "weight", None)
        if weight is not None and weight.device.type != "meta":
            return weight.device
        return return_device

    def _unsloth_offload_pre_hook(module, args):
        if not args:
            return args
        inp = args[0]
        if not hasattr(inp, "device"):
            return args
        weight = getattr(module, "weight", None)
        target = weight.device if weight is not None else _decoder_device()
        if target is None or inp.device == target:
            return args
        return (inp.to(target),) + tuple(args[1:])

    def _unsloth_offload_post_hook(module, args, output):
        target = _decoder_device()
        if target is not None and hasattr(output, "device") and output.device != target:
            return output.to(target)
        return output

    embed_tokens.register_forward_pre_hook(_unsloth_offload_pre_hook, prepend = True)
    embed_tokens.register_forward_hook(_unsloth_offload_post_hook, prepend = True)
    embed_tokens._unsloth_offload_hooks_installed = True
    return True


def _embeddings_are_tied(input_embeddings, output_embeddings):
    # Tied lm_head reuses this weight; offloading to CPU would strand the output projection.
    if input_embeddings is None or output_embeddings is None:
        return False
    w_in = getattr(input_embeddings, "weight", None)
    w_out = getattr(output_embeddings, "weight", None)
    if w_in is None or w_out is None:
        return False
    return w_in is w_out or w_in.data_ptr() == w_out.data_ptr()


VLLM_SUPPORTED_VLM = [
    "qwen2_5_vl",
    "gemma3",
    "mistral3",
    "qwen3_vl",
    "qwen3_vl_moe",
]
VLLM_NON_LORA_VLM = [
    "mllama",
]
PRE_COMPILE_INFERENCE = [
    "gpt_oss",
]

from transformers import GenerationConfig, CompileConfig, AutoConfig

try:
    from transformers import PreTrainedConfig
    PretrainedConfig = PreTrainedConfig
except:
    from transformers import PretrainedConfig

HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__

_compile_config = CompileConfig(
    fullgraph = False,
    dynamic = None,
    mode = "reduce-overhead",
)
_compile_config.disable = True  # Must set manually

try:
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_compiler_set_stance = None


def unsloth_base_fast_generate(self, *args, **kwargs):
    if len(args) != 0:
        input_ids = args[0]
    elif "input_ids" in kwargs:
        input_ids = kwargs["input_ids"]
    elif "input" in kwargs:
        input_ids = kwargs["input"]
    elif "input_features" in kwargs:
        input_ids = kwargs["input_features"]
    elif "inputs_embeds" in kwargs:
        # canonical HF name for embedding inputs (e.g. multimodal generate)
        input_ids = kwargs["inputs_embeds"]
    elif "input_embeds" in kwargs:
        input_ids = kwargs["input_embeds"]
    elif "inputs" in kwargs:
        input_ids = kwargs["inputs"]
    else:
        key = next(iter(kwargs.keys()))
        if type(kwargs[key]) is not torch.Tensor:
            raise TypeError("Unsloth: You need to pass in input_ids to .generate!")
        input_ids = kwargs[key]
    assert type(input_ids) is torch.Tensor
    bsz = input_ids.shape[0]

    FastBaseModel.for_inference(self)
    dtype = _get_dtype(dtype_from_config(self.config))
    # Handle full float32 cases as config.dtype == torch.float32!
    do_bfloat16_mixed_precision = os.environ.get("UNSLOTH_BFLOAT16_MIXED_PRECISION", "0") == "1"
    if do_bfloat16_mixed_precision:
        dtype = torch.bfloat16

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

    # Vision processors emit mm_token_type_ids that generate() rejects (Qwen3-VL); unlike
    # logits_to_keep it is an incoming kwarg, so drop it when generate does not accept it.
    if "mm_token_type_ids" in kwargs and not _unsloth_generate_accepts_kwarg(
        self, "mm_token_type_ids"
    ):
        kwargs.pop("mm_token_type_ids", None)

    # VLMs do not allow logits_to_keep. transformers >= 5.0 sets logits_to_keep=1
    # itself in GenerationMixin.generate AFTER _validate_model_kwargs, so pre-
    # injecting it makes the strict validator raise on PEFT models. Skip on v5+
    # and strip any leaked kwarg defensively.
    if Version(transformers_version) < Version("5.0.0.dev0"):
        global NUM_LOGITS_TO_KEEP
        if arch not in NUM_LOGITS_TO_KEEP:
            m = self
            # Find which is used: num_logits_to_keep or logits_to_keep
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
            if arch not in NUM_LOGITS_TO_KEEP:
                NUM_LOGITS_TO_KEEP[arch] = None
        key = NUM_LOGITS_TO_KEEP[arch]
        if key is not None and key not in kwargs and _unsloth_generate_accepts_kwarg(self, key):
            kwargs[key] = 1
    else:
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("num_logits_to_keep", None)

    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Get pixel values for VLMs
    try:
        kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype)
    except:
        pass
    try:
        kwargs["pixel_values_videos"] = kwargs["pixel_values_videos"].to(dtype)
    except:
        pass

    # Mixed precision autocast
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        autocaster = torch.autocast(device_type = DEVICE_TYPE_TORCH, dtype = torch.float16)
        dtype = torch.float16
    else:
        autocaster = torch.autocast(device_type = DEVICE_TYPE_TORCH, dtype = dtype)
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
        elif Version(transformers_version) < Version("4.56.0.dev0"):
            cache_implementation = None
        else:
            # Should work in latest transformers!
            cache_implementation = "static"
    else:
        cache_implementation = None
    if cache_implementation is not None:
        swa = getattr(getattr(self.config, "text_config", self.config), "sliding_window", None)
        if (swa == 0 or type(swa) is not int) and (
            getattr(self, "_can_compile_fullgraph", True) is True
        ):
            cache_implementation = "static"
        else:
            if Version(transformers_version) < Version("4.56.0.dev0"):
                cache_implementation = "hybrid"
            else:
                cache_implementation = "static"
    # [TODO] Unsure why static fails
    if do_bfloat16_mixed_precision:
        cache_implementation = None

    if "generation_config" in kwargs:
        kwargs["generation_config"].cache_implementation = cache_implementation
        if cache_implementation is not None:
            kwargs["generation_config"].compile_config = _compile_config
    else:
        kwargs["cache_implementation"] = cache_implementation
        if cache_implementation is not None:
            kwargs["compile_config"] = _compile_config

    # Delete cached Flex Attention masks to reset inference
    for name, module in self.named_modules():
        if hasattr(module, "_flex_attention_cache"):
            try:
                del module._flex_attention_cache
            except:
                pass
        # Solves AttributeError: 'SlidingWindowLayer' object has no attribute 'max_batch_size'
        if hasattr(module, "_cache") and "cache_utils" in str(module._cache.__class__):
            try:
                del module._cache
            except:
                pass

    with torch.inference_mode(), autocaster:
        output = self._old_generate(*args, **kwargs)

    # Delete cached Flex Attention masks to reset inference
    for name, module in self.named_modules():
        if hasattr(module, "_flex_attention_cache"):
            try:
                del module._flex_attention_cache
            except:
                pass
        # Solves AttributeError: 'SlidingWindowLayer' object has no attribute 'max_batch_size'
        if hasattr(module, "_cache") and "cache_utils" in str(module._cache.__class__):
            try:
                del module._cache
            except:
                pass

    # FastBaseModel.for_training(self)
    return output


# Offline helpers live in loader_utils.py (shared canonical source).
from .loader_utils import (
    _get_effective_local_files_only,
    _is_offline_related_error,
    _offline_aware_load,
)


def _missing_torchvision_error(error = None):
    """True if a VLM processor failed to load due to missing torchvision (#4202).

    Checks availability directly first, then only the specific torchvision-required
    error text (not any incidental "torchvision" substring like a model path)."""
    import importlib.util

    if importlib.util.find_spec("torchvision") is None:
        return True
    if error is not None:
        error_str = str(error).lower()
        return (
            "requires the torchvision" in error_str or "no module named 'torchvision'" in error_str
        )
    return False


def _construct_vlm_processor_fallback(
    tokenizer_name,
    model_type,
    token,
    trust_remote_code,
    cache_dir = None,
    local_files_only = False,
):
    """Build a VLM processor manually when AutoProcessor.from_pretrained fails (some VLMs
    have unresolvable tokenizer_class entries): load the image processor + tokenizer
    separately and combine. Returns (processor_or_None, error_or_None) so the caller can
    tell an offline failure (retry from cache) from a genuine one."""
    _fb_err = None
    try:
        from transformers import AutoImageProcessor, PreTrainedTokenizerFast, AutoConfig
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
        import json

        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(
            tokenizer_name,
            token = token,
            trust_remote_code = trust_remote_code,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        )
        # Load tokenizer via PreTrainedTokenizerFast (bypasses tokenizer_class check)
        tok = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_name,
            padding_side = "left",
            token = token,
            trust_remote_code = trust_remote_code,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        )
        # Read tokenizer_config.json for special tokens: prefer the local file (offline
        # / local checkpoint dir), else hf_hub_download with local_files_only forwarded.
        try:
            import json as _json

            tok_config = None
            _local_cfg = os.path.join(tokenizer_name, "tokenizer_config.json")
            if os.path.isdir(tokenizer_name):
                # Local dir: read directly. A missing file raises a clear FileNotFoundError
                # rather than letting hf_hub_download treat the path as a repo id.
                if os.path.exists(_local_cfg):
                    with open(_local_cfg, "r", encoding = "utf-8") as f:
                        tok_config = _json.load(f)
                else:
                    raise FileNotFoundError(
                        f"tokenizer_config.json not found in local directory: {tokenizer_name}"
                    )
            else:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(
                    tokenizer_name,
                    "tokenizer_config.json",
                    token = token,
                    cache_dir = cache_dir,
                    local_files_only = local_files_only,
                )
                with open(config_path, "r", encoding = "utf-8") as f:
                    tok_config = _json.load(f)
            # Set model-specific special tokens and their IDs
            for key in (
                "image_token",
                "image_start_token",
                "image_end_token",
                "image_thumbnail",
                "video_token",
            ):
                if key in tok_config and not hasattr(tok, key):
                    setattr(tok, key, tok_config[key])
                    id_key = key + "_id" if not key.endswith("_id") else key
                    token_id = tok.convert_tokens_to_ids(tok_config[key])
                    if not hasattr(tok, id_key):
                        setattr(tok, id_key, token_id)
        except Exception as _e:
            _fb_err = _e  # remember (non-fatal here); surfaced only if no processor is built

        # Find the processor class - try model_type first, then top-level config model_type
        proc_class_name = PROCESSOR_MAPPING_NAMES.get(model_type)
        if proc_class_name is None:
            # model_type might be a sub-model type (e.g. "lfm2" instead of "lfm2_vl").
            # Try the top-level config.model_type which often has the processor mapping.
            try:
                config = AutoConfig.from_pretrained(
                    tokenizer_name,
                    token = token,
                    trust_remote_code = trust_remote_code,
                    cache_dir = cache_dir,
                    local_files_only = local_files_only,
                )
                proc_class_name = PROCESSOR_MAPPING_NAMES.get(config.model_type)
            except Exception as _e:
                _fb_err = _e  # surface a network/cache miss so the offline retry can fire

        if proc_class_name is not None:
            import transformers
            proc_class = getattr(transformers, proc_class_name, None)
            if proc_class is not None:
                processor = proc_class(image_processor = image_processor, tokenizer = tok)
                # Copy chat_template from tokenizer to processor if needed
                if not getattr(processor, "chat_template", None) and getattr(
                    tok, "chat_template", None
                ):
                    processor.chat_template = tok.chat_template
                return processor, None
    except Exception as _e:
        _fb_err = _e
    return None, _fb_err


def _get_total_transformer_layers(model):
    """Best-effort total transformer block count across HF model shapes.
    Returns None if not determinable; caller should skip the conversion."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for name in (
        "num_hidden_layers",
        "n_layer",
        "n_layers",
        "num_layers",
    ):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for name in (
            "num_hidden_layers",
            "n_layer",
            "n_layers",
            "num_layers",
        ):
            v = getattr(text_cfg, name, None)
            if isinstance(v, int) and v > 0:
                return v
    return None


class FastBaseModel:
    @staticmethod
    @_offline_aware_load
    def from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        token = None,
        device_map = "sequential",
        trust_remote_code = False,
        model_types = None,
        tokenizer_name = None,
        auto_model = AutoModelForVision2Seq,
        use_gradient_checkpointing = "unsloth",
        supports_sdpa = True,
        whisper_language = None,
        whisper_task = None,
        auto_config = None,
        offload_embedding = False,
        float32_mixed_precision = None,  # Forces float32 mixed precision
        # vLLM parameters
        fast_inference = False,
        gpu_memory_utilization = 0.5,
        float8_kv_cache = False,
        random_state = 3407,
        max_lora_rank = 64,
        disable_log_stats = False,
        unsloth_vllm_standby = False,
        load_in_fp8 = False,  # fp8 LoRA (True, False, 'block')
        text_only = False,
        **kwargs,
    ):
        user_config = kwargs.pop("config", None)
        if auto_config is None and user_config is not None:
            auto_config = user_config

        # Offline snapshot for the loads below; not popped, so the weight load still
        # reads local_files_only from **kwargs. See _get_effective_local_files_only.
        local_files_only = _get_effective_local_files_only(kwargs)

        if unsloth_vllm_standby and os.environ.get("UNSLOTH_VLLM_STANDBY", "0") != "1":
            raise RuntimeError(
                "Unsloth: `unsloth_vllm_standby` is True, but environment variable `UNSLOTH_VLLM_STANDBY` is not set to 1!"
            )

        if model_types is None:
            raise RuntimeError(
                "Unsloth: Please use FastModel or FastVisionModel and not use FastBaseModel directly!"
            )
        if os.environ.get("UNSLOTH_MODEL_NAME", "") == "":
            os.environ["UNSLOTH_MODEL_NAME"] = model_name.lower()

        # Resolve text-only before the is_vlm / vLLM checks so is_vlm stays consistent;
        # skip the vision tower only for families with their own text decoder (Gemma 3). #5816
        if text_only and auto_config is None:
            auto_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
        if text_only and hasattr(auto_config, "vision_config"):
            parent_config = auto_config
            text_config = _get_text_only_config(parent_config, model_name)
            text_class = resolve_model_class(AutoModelForCausalLM, text_config)
            if text_class is not None and _is_family_text_decoder(
                getattr(parent_config, "model_type", ""),
                getattr(text_config, "model_type", ""),
            ):
                auto_config = text_config
                auto_model = AutoModelForCausalLM
                _apply_text_only_key_mapping(kwargs, parent_config, text_config)
        elif text_only and auto_model in [
            AutoModelForVision2Seq,
            AutoModelForImageTextToText,
        ]:
            # Pure text model requested text-only with a VLM auto class.
            auto_model = AutoModelForCausalLM
        is_vlm = auto_model in [AutoModelForVision2Seq, AutoModelForImageTextToText]
        # A repo-code VLM may register only AutoModel / AutoModelForCausalLM (e.g.
        # DeepSeek-OCR, Nemotron-VL), so auto_model is not a VLM class even though the
        # config is a vision model. Keep is_vlm (auto-class derived) for processor
        # selection below -- these repos ship no AutoProcessor -- but treat the model as a
        # VLM on the vLLM path so a vision_config model is never silently loaded/converted
        # as text-only. text_only resolves the tower away first, so honour that here.
        is_vlm_config = is_vlm or (not text_only and hasattr(auto_config, "vision_config"))
        is_whisper = whisper_language is not None and whisper_task is not None
        auto_processor = AutoProcessor if (is_vlm or is_whisper) else AutoTokenizer

        model_type_arch = model_types[0]
        if model_type_arch == "siglip":
            for model_type_arch in model_types:
                if model_type_arch != "siglip":
                    break

        vllm_enable_lora = True

        if is_vlm_config and fast_inference:
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
                "Unsloth: WARNING `trust_remote_code` is True.\n"
                "Are you certain you want to do remote code execution?"
            )
        token = hf_login(token)
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()

        if DEVICE_TYPE == "cuda":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_stats_name = (
                gpu_stats.name + ". " if gpu_stats.name != "" else "NVIDIA GPU Device. "
            )
            gpu_version = torch.version.cuda
            gpu_stats_snippet = (
                f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."
            )
            try:
                vllm_version = f" vLLM: {importlib_version('vllm')}."
            except:
                vllm_version = ""
        elif DEVICE_TYPE == "hip":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_stats_name = resolve_hip_gpu_stats_name(gpu_stats)
            gpu_version = torch.version.hip
            gpu_stats_snippet = f"ROCm Toolkit: {gpu_version}."
            try:
                vllm_version = f" vLLM: {importlib_version('vllm')}."
            except:
                vllm_version = ""
        elif DEVICE_TYPE == "xpu":
            gpu_stats = torch.xpu.get_device_properties(0)
            gpu_stats_name = gpu_stats.name + ". " if gpu_stats.name != "" else "Intel XPU Device. "
            gpu_version = torch.version.xpu
            gpu_stats_snippet = f"Intel Toolkit: {gpu_version}."
            # [TODO] After adding vLLM support for XPU, change this
            vllm_version = ""
        else:
            raise ValueError(f"Unsloth: Unsupported device type: {DEVICE_TYPE}")

        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        arch_name = model_type_arch.title()
        arch_name = arch_name.replace("_Vl_", "_VL_").replace("_Moe", "_MoE")
        statistics = (
            f"==((====))==  Unsloth {__version__}: Fast {arch_name} patching. Transformers: {transformers_version}.{vllm_version}\n"
            f"   {chr(92)}{chr(92)}   /|    {gpu_stats_name}Num GPUs = {DEVICE_COUNT}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"
            f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"
            f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"
            f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        )

        print(statistics)

        # Warn about fast transfers
        if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
            old_hf_transfer = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]
            if old_hf_transfer in ("False", "false"):
                old_hf_transfer = "0"
            if old_hf_transfer in ("True", "true"):
                old_hf_transfer = "1"
        else:
            old_hf_transfer = "0"
        if old_hf_transfer == "1":
            print(
                "Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!"
            )
        if old_hf_transfer != "0":
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # For debugging - we use a download counter to see if environments are not breaking or if HF is down
        get_statistics(kwargs.get("local_files_only", False))

        # The base + tokenizer prefetch runs AFTER the load-mode validation below, so an invalid
        # load_in_* combination fails without first downloading a snapshot.

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            if dtype == torch.float16:
                dtype = torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16
        assert dtype in (torch.float16, torch.bfloat16, torch.float32)

        bnb_compute_dtype = dtype
        do_forced_float32 = False
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            print(
                f"Unsloth: Using float16 precision for {model_type_arch} won't work! Using float32."
            )
            bnb_compute_dtype = torch.float16
            do_forced_float32 = True

        # Check for custom data-types
        custom_datatype = None
        correct_dtype = None
        if os.environ.get("UNSLOTH_FORCE_CUSTOM_DTYPE", "") != "":
            custom_datatype = os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"]
            assert custom_datatype.count(";") >= 4
            checker, _dtype, _bnb_compute_dtype, _custom_datatype, execute_code = (
                custom_datatype.split(";", 4)
            )
            # Allow custom dtypes on all runs
            allow_all_runs = checker == "all"
            # Allow only on float16 datatypes
            allow_float16_runs = (checker == "float16" or checker == "torch.float16") and (
                dtype == torch.float16 or os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
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

        if auto_config is None:
            auto_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
        model_class = resolve_model_class(auto_model, auto_config)
        attn_impl = resolve_attention_implementation(
            model_class,
            auto_config,
            requested_attn_implementation = kwargs.get("attn_implementation", None),
            supports_sdpa = supports_sdpa,
        )

        # Handle FP8 models: get_model_name has already redirected this to BF16 sibling if the model ships with
        # FP8 weights. We just need to update it here for sanity.
        auto_config.model_name = model_name
        kwargs["attn_implementation"] = attn_impl

        bnb_config = None
        user_quantization_config = kwargs.get("quantization_config", None)

        # Check if model already has a non-bitsandbytes quantization config (e.g. compressed-tensors/NVFP4)
        from .loader_utils import (
            check_and_disable_bitsandbytes_loading,
            sync_unsloth_model_name_bnb_flags,
        )

        load_in_4bit, load_in_8bit, _ = check_and_disable_bitsandbytes_loading(
            auto_config, load_in_4bit = load_in_4bit, load_in_8bit = load_in_8bit
        )
        # Correct UNSLOTH_MODEL_NAME's bnb tokens now that the effective bnb state is known
        # (the per-load env was built before remap/disable). gpt-oss only; no-op otherwise.
        sync_unsloth_model_name_bnb_flags(load_in_4bit, load_in_8bit)

        if full_finetuning and (load_in_4bit or load_in_8bit):
            print(
                "Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA."
            )
            load_in_4bit = False
            load_in_8bit = False
            load_in_16bit = False

        if int(load_in_4bit) + int(load_in_8bit) + int(load_in_16bit) >= 2:
            raise RuntimeError(
                "Unsloth: Can only load in 4bit or 8bit or 16bit, not a combination!"
            )

        # Prefetch the repo (killable child) so the in-process load below is a cache hit. vLLM owns the
        # weight download only when actually available; if fast_inference was requested but vLLM is
        # missing, the load falls through in-process, so weights must still be warmed here.
        _vllm_owns_weights = fast_inference and is_vLLM_available()
        _prefetched = maybe_prefetch_hf_snapshot(
            model_name,
            token = token,
            revision = kwargs.get("revision"),
            cache_dir = kwargs.get("cache_dir"),
            local_files_only = kwargs.get("local_files_only", False),
            fast_inference = _vllm_owns_weights,
            subfolder = kwargs.get("subfolder"),
            force_download = kwargs.get("force_download", False),
            use_safetensors = kwargs.get("use_safetensors"),
            from_tf = kwargs.get("from_tf", False),
            from_flax = kwargs.get("from_flax", False),
            # Bare load reads only ROOT weights; skip subdir weights. Ignored when a subfolder is set.
            weights_at_root = True,
            variant = kwargs.get("variant"),  # forward so the warm keeps the variant .bin
            gguf_file = kwargs.get(
                "gguf_file"
            ),  # forward so the warm fetches the GGUF (else ignored)
        )
        # Child did the forced download; clear the flag so the load reuses the warm cache.
        if _prefetched and kwargs.get("force_download", False):
            kwargs["force_download"] = False

        # Warm a SEPARATE tokenizer repo only (model_name is covered above). Not model_name here: this
        # runs before fast_inference_setup may remap the repo, so it would warm the wrong one.
        _tokenizer_repo = (
            tokenizer_name if (isinstance(tokenizer_name, str) and tokenizer_name) else model_name
        )
        _warm_tokenizer_repo = (
            isinstance(_tokenizer_repo, str)
            and bool(_tokenizer_repo)
            and _tokenizer_repo != model_name
        )
        if _warm_tokenizer_repo:
            maybe_prefetch_hf_snapshot(
                _tokenizer_repo,
                token = token,
                cache_dir = kwargs.get("cache_dir"),
                local_files_only = kwargs.get("local_files_only", False),
                tokenizer_only = True,
            )

        _skip_modules = SKIP_QUANTIZATION_MODULES.copy()
        # Nemotron-H uses 'mixer' (not 'mamba') for Mamba layers.
        # Mamba fused kernels pass out_proj.weight directly to F.linear,
        # which fails with quantized Params4bit. Skip out_proj from quantization.
        if any(mt == "nemotron_h" for mt in (model_types or [])):
            _skip_modules.append("out_proj")

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = bnb_compute_dtype,
                llm_int8_skip_modules = _skip_modules,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit = True,
                llm_int8_skip_modules = _skip_modules,
            )
        elif load_in_16bit:
            bnb_config = None
        elif not load_in_4bit and not load_in_8bit and not full_finetuning:
            print("Unsloth: QLoRA and full finetuning all not selected. Switching to 16bit LoRA.")

        if full_finetuning:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
            if dtype == torch.bfloat16:
                if float32_mixed_precision != True:
                    print(
                        f"Unsloth: Using bfloat16 full finetuning which cuts memory usage by 50%.\n"
                        f"To enable float32 training, use `float32_mixed_precision = True` during FastLanguageModel.from_pretrained"
                    )
                else:
                    print(
                        f"Unsloth: Using full float32 full finetuning. "
                        f"To enable bfloat16 training to reduce VRAM usage by 50% albeit with a slightly higher loss, do:\n"
                        "use `float32_mixed_precision = False` during FastLanguageModel.from_pretrained"
                    )
                    os.environ["UNSLOTH_BFLOAT16_MIXED_PRECISION"] = "1"
            else:
                print(
                    "Unsloth: Float16 full finetuning uses more memory since we upcast weights to float32."
                )
        else:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"

        # Fix AttributeError: 'BitsAndBytesConfig' object has no attribute 'get_loading_attributes'
        if bnb_config is not None and not hasattr(bnb_config, "get_loading_attributes"):
            bnb_config.get_loading_attributes = lambda *args, **kwargs: {}

        # Cannot be None, since HF now checks for the config
        if load_in_4bit or load_in_8bit:
            # Ignore load_in_4bit / load_in_8bit for MXFP4 - best to get config file
            if "gpt-oss-20b" in model_name.lower() or "gpt-oss-120b" in model_name.lower():
                pass
            else:
                if user_quantization_config is None:
                    kwargs["quantization_config"] = bnb_config
        else:
            if auto_config is None:
                auto_config = AutoConfig.from_pretrained(
                    model_name,
                    token = token,
                    trust_remote_code = trust_remote_code,
                    local_files_only = local_files_only,
                )
            if hasattr(auto_config, "quantization_config"):
                from transformers.quantizers.auto import (
                    AUTO_QUANTIZATION_CONFIG_MAPPING,
                )

                quantization_config = auto_config.quantization_config
                quant_method = quantization_config["quant_method"]
                # Sometimes bitsandbytes_4bit + bitsandbytes_8bit is provided
                if (
                    quant_method == "bitsandbytes"
                    and "bitsandbytes" not in AUTO_QUANTIZATION_CONFIG_MAPPING
                ):
                    if "bitsandbytes_4bit" not in AUTO_QUANTIZATION_CONFIG_MAPPING:
                        raise KeyError(
                            "Unsloth: AUTO_QUANTIZATION_CONFIG_MAPPING does not have `bitsandbytes_4bit`"
                        )
                    quantizer = AUTO_QUANTIZATION_CONFIG_MAPPING["bitsandbytes_4bit"]
                else:
                    quantizer = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
                quantizer_kwargs = {}
                if quant_method == "compressed-tensors":
                    # Ignore these
                    pass
                else:
                    # We cannot dequantize since gpt-oss-20b MXFP4 will now be gpt-oss-20b-BF16
                    if load_in_16bit and "dequantize" in inspect.signature(quantizer).parameters:
                        quantizer_kwargs["dequantize"] = True
                    try:
                        # Sometimes this fails so we wrap it in a try except
                        quantization_config = quantizer.from_dict(
                            quantization_config, **quantizer_kwargs
                        )
                    except:
                        pass
                    if user_quantization_config is None:
                        kwargs["quantization_config"] = quantization_config

        # Check if using forced float32 - we load it in bfloat16, then cast to float16!
        torch_dtype = dtype
        if do_forced_float32:
            torch_dtype = torch.bfloat16

        kwargs = add_dtype_kwargs(torch_dtype, kwargs)

        config_attn_impl = kwargs.get("attn_implementation", None)
        if config_attn_impl is None:
            config_attn_impl = "sdpa" if supports_sdpa else "eager"
        if auto_config is None:
            auto_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
                local_files_only = local_files_only,
            )
        _set_attn_impl(auto_config, config_attn_impl)
        model_config = auto_config

        verify_fp8_support_if_applicable(model_config)

        raise_handler = RaiseUninitialized()
        try:
            if offload_embedding and fast_inference:
                # vLLM manages its own weights; embedding offload does not apply.
                print(
                    "Unsloth: Not offloading embeddings; incompatible with fast_inference (vLLM)."
                )
                offload_embedding = False
            if not fast_inference:
                # Prevent load_in_fp8 from being forwarded into HF internal model loading
                load_in_fp8 = kwargs.pop("load_in_fp8", None)
                # Transformers 5.x @strict config classes reject unexpected kwargs.
                # Move config-level attributes onto the config object directly.
                _num_labels = kwargs.pop("num_labels", None)
                if _num_labels is not None:
                    set_task_config_attr(model_config, "num_labels", _num_labels)
                for _cfg_key in ("id2label", "label2id", "problem_type"):
                    _cfg_val = kwargs.pop(_cfg_key, None)
                    if _cfg_val is not None:
                        set_task_config_attr(model_config, _cfg_key, _cfg_val)
                _cfg_val = kwargs.pop("max_position_embeddings", None)
                if _cfg_val is not None:
                    setattr(model_config, "max_position_embeddings", _cfg_val)
                model = auto_model.from_pretrained(
                    model_name,
                    config = model_config,
                    device_map = device_map,
                    # torch_dtype           = torch_dtype, # Transformers removed torch_dtype
                    # quantization_config   = bnb_config,
                    token = token,
                    trust_remote_code = trust_remote_code,
                    # attn_implementation   = attn_implementation,
                    **kwargs,
                )
                # Attach dispatch hooks for bnb multi-device loads.
                _attach_bnb_multidevice_hooks(
                    model,
                    load_in_4bit = load_in_4bit,
                    load_in_8bit = load_in_8bit,
                    offload_embedding = offload_embedding,
                    fast_inference = fast_inference,
                )
                # Re-apply block-fp8 weight_scale_inv tensors transformers dropped on load (#6200).
                _restore_dropped_fp8_scales(
                    model,
                    model_name,
                    local_files_only = local_files_only,
                    token = token,
                    revision = kwargs.get("revision"),
                    subfolder = kwargs.get("subfolder"),
                    cache_dir = kwargs.get("cache_dir"),
                    variant = kwargs.get("variant"),
                )
                if hasattr(model, "generate"):
                    model.fast_generate = make_fast_generate_wrapper(model.generate)
                    model.fast_generate_batches = error_out_no_vllm
                if offload_embedding:
                    if bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")):
                        # WSL doesn't work with offloaded embeddings
                        pass
                    elif os.name == "nt":
                        # Windows doesn't work with offloaded embeddings
                        pass
                    else:
                        embed_tokens = model.get_input_embeddings()
                        out_embed = (
                            model.get_output_embeddings()
                            if hasattr(model, "get_output_embeddings")
                            else None
                        )
                        if _embeddings_are_tied(embed_tokens, out_embed):
                            raise NotImplementedError(
                                "offload_embedding = True is not supported for models with tied word "
                                "embeddings (embed_tokens shares its weight with lm_head). Offloading "
                                "would strand the output projection on CPU and saves no VRAM. Set "
                                "offload_embedding = False for this model."
                            )
                        nbytes = embed_tokens.weight.numel() * embed_tokens.weight.itemsize
                        ngb = round(nbytes / 1024 / 1024 / 1024, 2)
                        print(f"Unsloth: Offloading embeddings to RAM to save {ngb} GB.")
                        _embed_device = embed_tokens.weight.device  # decoder device, before offload
                        embed_tokens.to("cpu")

                        # Device-safe embedding offload.
                        _install_offload_embedding_hooks(embed_tokens, out_embed, _embed_device)
                        # Must free GPU memory otherwise will not free!
                        torch.cuda.empty_cache()
                        gc.collect()
            else:
                from unsloth_zoo.vllm_utils import (
                    load_vllm,
                    get_vllm_state_dict,
                    convert_vllm_to_huggingface,
                    generate_batches,
                    get_lora_supported_ranks,
                )

                if full_finetuning:
                    max_lora_rank = max(get_lora_supported_ranks())
                    raise NotImplementedError(
                        "Unsloth: `fast_inference=True` cannot be used together with `full_finetuning=True`.\n"
                        "Reason: fast_inference is optimized for inference-only workflows and "
                        "does not currently support full fine-tuning.\n"
                        "Workaround: disable fast_inference, or use parameter-efficient fine-tuning "
                        f"(e.g. LoRA with rank r={max_lora_rank})."
                    )

                model_config.model_name = model_name

                if fast_inference:
                    fast_inference, model_name = fast_inference_setup(model_name, model_config)

                fp8_mode = None
                if load_in_fp8 != False:
                    fp8_mode = _get_fp8_mode_and_check_settings(
                        load_in_fp8,
                        fast_inference,
                        full_finetuning,
                        load_in_4bit,
                        load_in_8bit,
                        load_in_16bit,
                    )

                allowed_args = inspect.getfullargspec(load_vllm).args
                load_vllm_kwargs = dict(
                    model_name = model_name,
                    config = model_config,
                    gpu_memory_utilization = gpu_memory_utilization,
                    max_seq_length = max_seq_length,
                    dtype = dtype,
                    float8_kv_cache = float8_kv_cache,
                    enable_lora = vllm_enable_lora,
                    max_lora_rank = max_lora_rank,
                    disable_log_stats = disable_log_stats,
                    use_bitsandbytes = load_in_4bit,
                    unsloth_vllm_standby = unsloth_vllm_standby,
                    is_vision_model = is_vlm_config,
                    fp8_mode = fp8_mode,
                )
                for allowed_arg in allowed_args:
                    if allowed_arg not in load_vllm_kwargs and allowed_arg in kwargs:
                        load_vllm_kwargs[allowed_arg] = kwargs[allowed_arg]

                # Load vLLM first
                llm = load_vllm(**load_vllm_kwargs)

                # Convert to HF format
                _, quant_state_dict = get_vllm_state_dict(
                    llm,
                    config = model_config,
                    is_vision_model = is_vlm_config,
                    load_in_fp8 = load_in_fp8,
                )
                model = convert_vllm_to_huggingface(
                    quant_state_dict,
                    model_config,
                    dtype,
                    bnb_config,
                    is_vision_model = is_vlm_config,
                )
                model.vllm_engine = llm
                llm.shared_weights = True
                model.fast_generate = model.vllm_engine.generate
                model.fast_generate_batches = functools.partial(generate_batches, model.vllm_engine)

        finally:
            raise_handler.remove()
            # Return old flag
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Check float32 norm weights
        if os.environ.get("UNSLOTH_HIGH_PRECISION_LAYERNORM", "0") == "1":
            for jj, (name, module) in enumerate(model.named_modules()):
                if (
                    name.endswith(("norm", "norm1", "norm2", "norm3", "norm4"))
                    or "layernorm" in name
                    or "layer_norm" in name
                ) and hasattr(module, "weight"):
                    module._pre_set_compute_dtype = torch.float32
        # Edit data-types
        if custom_datatype is not None:
            with torch.no_grad():
                for jj, (name, module) in enumerate(model.named_modules()):
                    exec(custom_datatype)
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name

        # On the vLLM path the tokenizer warm was deferred (fast_inference_setup may remap model_name).
        # Warm the now-final tokenizer repo so the load below hits the cache (a cached/local repo is a no-op).
        if _vllm_owns_weights and isinstance(tokenizer_name, str) and tokenizer_name:
            maybe_prefetch_hf_snapshot(
                tokenizer_name,
                token = token,
                revision = kwargs.get("revision"),
                cache_dir = kwargs.get("cache_dir"),
                local_files_only = kwargs.get("local_files_only", False),
                tokenizer_only = True,
            )

        # Fix _Unsloth_Patched_ prefix in local config files from old saves (issue #4085)
        if os.path.isdir(tokenizer_name):
            import json as _json
            for _cfg_name in (
                "processor_config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                _cfg_path = os.path.join(tokenizer_name, _cfg_name)
                if os.path.exists(_cfg_path):
                    try:
                        with open(_cfg_path, "r", encoding = "utf-8") as _f:
                            _cfg = _json.load(_f)
                        if _cfg.get("processor_class", "").startswith("_Unsloth_Patched_"):
                            _cfg["processor_class"] = _cfg["processor_class"][
                                len("_Unsloth_Patched_") :
                            ]
                            with open(_cfg_path, "w", encoding = "utf-8") as _f:
                                _json.dump(_cfg, _f, indent = 2, ensure_ascii = False)
                    except Exception:
                        pass

        # Functional load chain (AutoProcessor -> get_auto_processor -> manual VLM
        # fallback); offline is already forced upstream. Surfaces the error for the retry.
        def _acquire_processor(lfo):
            _err = None  # underlying load failure (used by the entry-point retry)
            if (whisper_language and whisper_task) or auto_model.__name__.endswith(
                "ForConditionalGeneration"
            ):
                try:
                    _tok = auto_processor.from_pretrained(
                        tokenizer_name,
                        padding_side = "left",
                        token = token,
                        language = whisper_language,
                        task = whisper_task,
                        trust_remote_code = trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )
                except Exception as _e:
                    _tok = None
                    _err = _e
            else:
                try:
                    _tok = auto_processor.from_pretrained(
                        tokenizer_name,
                        padding_side = "left",
                        token = token,
                        trust_remote_code = trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )
                except Exception as _e:
                    _err = _e
                    try:
                        _tok = get_auto_processor(
                            tokenizer_name,
                            padding_side = "left",
                            token = token,
                            trust_remote_code = trust_remote_code,
                            cache_dir = kwargs.get("cache_dir"),
                            local_files_only = lfo,
                        )
                    except Exception:
                        # Swallow so the manual fallback / entry-point retry can run.
                        _tok = None

            # Build the processor manually if it failed to load or silently degraded to
            # a text-only tokenizer (no image_processor) for a VLM (issue #4085).
            _processor_is_degraded = (
                is_vlm and _tok is not None and not hasattr(_tok, "image_processor")
            )
            if (_tok is None or _processor_is_degraded) and is_vlm:
                try:
                    _fallback, _fb_err = _construct_vlm_processor_fallback(
                        tokenizer_name,
                        model_type_arch,
                        token,
                        trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )
                except Exception as _fe:
                    _fallback, _fb_err = None, _fe
                if _fallback is not None:
                    _tok = _fallback
                elif _err is None or (_fb_err is not None and _is_offline_related_error(_fb_err)):
                    # Prefer a network fallback error over a permanent primary one so the
                    # offline retry still fires.
                    _err = _fb_err
            return _tok, _err

        def _is_degraded_vlm(_t):
            # VLM that loaded only a text-only tokenizer (no image_processor).
            return is_vlm and _t is not None and not hasattr(_t, "image_processor")

        tokenizer, _primary_err = _acquire_processor(local_files_only)
        # Online network failure/degrade: raise so @_offline_aware_load retries from cache.
        # Permanent / missing-file errors propagate; when already offline keep what we got.
        if (
            (tokenizer is None or _is_degraded_vlm(tokenizer))
            and not local_files_only
            and _is_offline_related_error(_primary_err)
        ):
            raise _primary_err
        # Missing torchvision silently degrades a VLM processor to text-only; surface the
        # real cause instead of a later collator error (#4202), incl. on a silent degrade.
        if is_vlm and (tokenizer is None or not hasattr(tokenizer, "image_processor")):
            if _missing_torchvision_error(_primary_err):
                raise ImportError(
                    f"Unsloth: Could not load the vision processor for `{tokenizer_name}` "
                    "because torchvision is not installed. transformers requires torchvision "
                    "for this model's vision (image/video) processors. Please install it, "
                    "e.g. `pip install torchvision`."
                )
            import sys
            print(
                f"Unsloth: Warning - VLM processor fallback returned None for model_type={model_type_arch}",
                file = sys.stderr,
            )
        # Backwards compat: if processor has no chat_template (e.g. old saves without
        # chat_template.jinja) but the inner tokenizer does, copy it to the processor.
        if (
            hasattr(tokenizer, "tokenizer")
            and getattr(tokenizer, "chat_template", None) is None
            and getattr(tokenizer.tokenizer, "chat_template", None) is not None
        ):
            tokenizer.chat_template = tokenizer.tokenizer.chat_template

        if hasattr(tokenizer, "tokenizer"):
            __tokenizer = tokenizer.tokenizer
            # Add padding side as well
            __tokenizer.padding_side = "left"
            # Check bos, eos, pad tokens
            if hasattr(__tokenizer, "bos_token"):
                tokenizer.bos_token = __tokenizer.bos_token
                tokenizer.bos_token_id = __tokenizer.bos_token_id
            if hasattr(__tokenizer, "eos_token"):
                tokenizer.eos_token = __tokenizer.eos_token
                tokenizer.eos_token_id = __tokenizer.eos_token_id
            if hasattr(__tokenizer, "pad_token"):
                tokenizer.pad_token = __tokenizer.pad_token
                tokenizer.pad_token_id = __tokenizer.pad_token_id
        # Fix other stuff like BnB compute data types
        model, tokenizer = patch_model_and_tokenizer(
            model,
            tokenizer,
            downcast_rope = False,
            fix_embeddings = False,
            do_forced_float32 = do_forced_float32,
            correct_dtype = correct_dtype,
        )

        try:
            model, tokenizer = patch_tokenizer(model, tokenizer)
        except Exception as _patch_err:
            # Some VLM processors (e.g. ERNIE VL) fail patching; fall back to AutoTokenizer.
            try:
                from transformers import AutoTokenizer as _AutoTokenizer

                _fallback_tok = _AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    padding_side = "left",
                    token = token,
                    trust_remote_code = trust_remote_code,
                    cache_dir = kwargs.get("cache_dir"),
                    local_files_only = local_files_only,
                )
                model, _fallback_tok = patch_tokenizer(model, _fallback_tok)
                # Re-attach as processor wrapper if original was a processor
                if hasattr(tokenizer, "image_processor"):
                    tokenizer.tokenizer = _fallback_tok
                else:
                    tokenizer = _fallback_tok
            except Exception as _fb_err:
                # Online network failure: propagate for the offline retry; else raise the patch error.
                if not local_files_only and _is_offline_related_error(_fb_err):
                    raise
                raise _patch_err
        model = post_patch_loss_function(model)

        # Log Unsloth version for future fastpaths for inference
        if hasattr(model, "config"):
            model.config.update({"unsloth_version": __version__})
        patch_saving_functions(model, vision = True)
        if tokenizer is None:
            # Last resort: AutoTokenizer, then PreTrainedTokenizerFast (raise on network failure to retry).
            def _last_resort_tokenizer(lfo):
                from transformers import AutoTokenizer as _AutoTokenizer
                try:
                    return _AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        padding_side = "left",
                        token = token,
                        trust_remote_code = trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )
                except Exception:
                    from transformers import PreTrainedTokenizerFast
                    return PreTrainedTokenizerFast.from_pretrained(
                        tokenizer_name,
                        padding_side = "left",
                        token = token,
                        trust_remote_code = trust_remote_code,
                        cache_dir = kwargs.get("cache_dir"),
                        local_files_only = lfo,
                    )

            _last_resort_err = None
            try:
                tokenizer = _last_resort_tokenizer(local_files_only)
            except Exception as _e:
                _last_resort_err = _e
                # Online network failure: let the entry point retry forced-offline.
                if not local_files_only and _is_offline_related_error(_e):
                    raise
            if tokenizer is None:
                del model
                raise RuntimeError(
                    "Unsloth: Could not load the tokenizer/processor. If you are "
                    "offline, make sure the tokenizer files exist in the checkpoint "
                    "folder or were previously downloaded to the Hugging Face cache, "
                    "or set HF_HUB_OFFLINE=1 to force local loading. "
                    "Otherwise please check that the model has a tokenizer."
                ) from _last_resort_err
        patch_saving_functions(tokenizer, vision = True)

        # Fix gradient accumulation. See issue #4982.
        from transformers.trainer import Trainer

        apply_accepts_loss_kwargs_fix(model)
        patch_gradient_accumulation_fix(Trainer)

        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left"  # Force inference
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "left"  # Force inference
        # Audio feature extractors must stay right padded: left (a text setting,
        # forwarded by from_pretrained) shifts Whisper mels and desyncs Gemma 4
        # audio token counts (crash on transformers < 5.10).
        feature_extractor = getattr(tokenizer, "feature_extractor", None)
        if (
            feature_extractor is not None
            and getattr(feature_extractor, "padding_side", None) == "left"
        ):
            feature_extractor.padding_side = "right"
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m._saved_temp_tokenizer = tokenizer
            m = m.model
        m.max_seq_length = max_seq_length
        # Save to modules as well
        for module in model.modules():
            module.max_seq_length = max_seq_length
        m._saved_temp_tokenizer = tokenizer
        # Prevent Transformers Trainer from auto-wrapping Unsloth LoRA models in DP.
        _mark_unsloth_disable_data_parallel(model, disable = not full_finetuning)

        # Patch generate
        if os.environ.get("UNSLOTH_DISABLE_FAST_GENERATION", "0") == "0" and hasattr(
            model, "generate"
        ):
            if model.generate.__name__ != "unsloth_base_fast_generate":
                model._old_generate = model.generate
                unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
                model.generate = types.MethodType(unsloth_base_fast_generate, model)
        model._unsloth_trust_remote_code = trust_remote_code
        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            trust_remote_code = trust_remote_code,
            model_type = model_type_arch,
            tokenizer = tokenizer,
            float32_mixed_precision = float32_mixed_precision,
        )
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = None,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        finetune_vision_layers = True,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        finetune_last_n_layers = None,
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,  # not used anymore
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        task_type = TaskType.CAUSAL_LM,
        temporary_location = "_unsloth_temporary_saved_buffers",
        qat_scheme = None,
        target_parameters = None,  # For MoE expert layers (nn.Parameter)
        ensure_weight_tying = False,  # [TODO] Add `ensure_weight_tying` for `modules_to_save` for vision models
        finetune_audio_layers = False,  # placed last to preserve existing positional argument order
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            # Full finetuning still compiles, so a stray pre-train forward can poison the
            # cache; install the detector here too (it is idempotent).
            _unsloth_install_pretrain_detector(model)
            return model
        transformers_set_seed(random_state)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM):
            raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")

        # Remember whether the CALLER explicitly opted into audio. "all-linear" turns
        # the flag on implicitly below, but an old unsloth_zoo that cannot do audio
        # must not make a plain all-linear (text/vision) run fail.
        _audio_explicitly_requested = bool(finetune_audio_layers)
        if target_modules == "all-linear":
            finetune_vision_layers = True
            finetune_language_layers = True
            finetune_attention_modules = True
            finetune_mlp_modules = True
            finetune_audio_layers = True
        # Older unsloth_zoo (before get_peft_regex gained finetune_audio_layers) does
        # not accept the kwarg. Pass it only when supported; if the caller EXPLICITLY
        # asked for audio but it is unsupported, fail loudly rather than silently
        # training a language-only adapter. (all-linear's implicit opt-in degrades
        # gracefully instead of raising.)
        if "finetune_audio_layers" in inspect.signature(get_peft_regex).parameters:
            _audio_kwargs = {"finetune_audio_layers": finetune_audio_layers}
        elif _audio_explicitly_requested:
            raise RuntimeError(
                "Unsloth: finetune_audio_layers=True requires a newer unsloth_zoo. "
                "Please upgrade with `pip install --upgrade --no-deps unsloth_zoo`."
            )
        else:
            _audio_kwargs = {}
        # Remember the caller's ORIGINAL explicit leaf list for MoE expert
        # detection. When an explicit list is routed through get_peft_regex for
        # family scoping below, the generated regex carries get_peft_regex's full
        # "mlp|feed_forward|ffn|dense" component block even when the caller named
        # only attention leaves (q/k/v/o_proj). Keying expert detection on that
        # regex would train the experts for an attention-only request. The
        # original list carries the true leaf intent, so use it for MoE detection;
        # only the auto (None / "all-linear") path relies on the regex, whose mlp
        # block is the sole remaining MLP-intent signal on fused-expert models.
        _moe_detect_target = target_modules if type(target_modules) in (list, tuple) else None
        if target_modules is None or target_modules == "all-linear":
            target_modules = get_peft_regex(
                model,
                finetune_vision_layers = finetune_vision_layers,
                finetune_language_layers = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules = finetune_mlp_modules,
                **_audio_kwargs,
            )
        else:
            assert type(target_modules) in (list, tuple, str)
            # Route an explicit list through get_peft_regex when the caller scoped a
            # layer family (one of the finetune_* below is off) OR opted into audio (so
            # the new audio/embedder branches are considered). finetune_audio_layers is
            # a POSITIVE term here: using `not finetune_audio_layers` would -- since it
            # defaults False -- force every explicit list through the filter.
            _scoping = (
                not finetune_vision_layers
                or not finetune_language_layers
                or not finetune_attention_modules
                or not finetune_mlp_modules
            )
            if type(target_modules) in (list, tuple) and (_scoping or finetune_audio_layers):
                if _scoping:
                    print(
                        "Unsloth: Explicit target_modules are constrained by the "
                        "finetune_(vision|language|attention|mlp) filters; adapters "
                        "attach only where both select."
                    )
                target_modules = get_peft_regex(
                    model,
                    finetune_vision_layers = finetune_vision_layers,
                    finetune_language_layers = finetune_language_layers,
                    finetune_attention_modules = finetune_attention_modules,
                    finetune_mlp_modules = finetune_mlp_modules,
                    target_modules = list(target_modules),
                    **_audio_kwargs,
                )

        if hasattr(model, "vllm_engine"):
            if (
                hasattr(model.vllm_engine, "llm_engine")
                and hasattr(model.vllm_engine.llm_engine, "vllm_config")
                and getattr(model.vllm_engine.llm_engine.vllm_config, "lora_config", None) is None
            ):
                # If vLLM is being used but lora is not enabled, throw an error
                # Ref https://github.com/vllm-project/vllm/blob/51ba839555a5d122eadd91e9c16463ac288f5fa1/vllm/v1/engine/processor.py#L148-L151
                raise RuntimeError("Unsloth: LoRA is not enabled for this model!")
            if finetune_vision_layers:
                # vLLM does not support LoRA on vision layers
                # https://github.com/vllm-project/vllm/blob/main/vllm/lora/models.py#L471-L477
                # TODO: Update this once vLLM V1 supports LoRA on vision layers (possibly not happening)
                raise RuntimeError(
                    "Unsloth: Finetuning vision layers is not supported for fast_inference. Only text layers are supported!"
                )
            if model.config.model_type in VLLM_NON_LORA_VLM:
                # mllama is still only in vllm v0 https://arc.net/l/quote/llwkfgmu
                # https://docs.vllm.ai/en/stable/models/supported_models.html#text-generation_1
                # vLLM V0 does not support LoRA on multi modal models.
                # TODO: Update this once vLLM V1 supports Llama 3.2 aka mllama
                raise RuntimeError(
                    "Unsloth: LoRA finetuning for Llama 3.2 aka mllama models is not supported with fast_inference!"
                )

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        max_seq_length = model.max_seq_length
        # If we pass loftq_config = None we will get an error
        loftq_config = validate_loftq_config(
            loftq_config, lora_dropout, bias, init_lora_weights, model
        )

        # Auto-detect MoE models and populate target_parameters for expert layers.
        # Prefer the caller's ORIGINAL explicit leaf list over the scoped regex so an
        # attention-only request does not train experts via get_peft_regex's mlp block,
        # but only when MLP and language families are both still in scope. If the caller
        # scoped MLP or language OFF (finetune_mlp_modules / finetune_language_layers
        # False), the scoped regex already dropped the experts, so honor it instead of
        # re-introducing the original list's gate/up/down leaves.
        if target_parameters is None:
            _moe_targets = _select_moe_detection_targets(
                _moe_detect_target,
                target_modules,
                finetune_mlp_modules = finetune_mlp_modules,
                finetune_language_layers = finetune_language_layers,
            )
            target_parameters = get_moe_target_parameters(model, _moe_targets)

        # Per-expert Linear expert layouts (e.g. gpt-oss bnb-4bit) target experts via
        # target_modules, not fused Parameters. Extend either form PEFT accepts: a leaf
        # list (explicit) or a regex string (auto / all-linear / scoped). No-op otherwise.
        _moe_module_detect = _select_moe_detection_targets(
            _moe_detect_target,
            target_modules,
            finetune_mlp_modules = finetune_mlp_modules,
            finetune_language_layers = finetune_language_layers,
        )
        _moe_module_targets = get_moe_target_modules(model, _moe_module_detect)
        if _moe_module_targets:
            if isinstance(target_modules, (list, tuple)):
                target_modules = list(target_modules) + [
                    target for target in _moe_module_targets if target not in target_modules
                ]
            elif isinstance(target_modules, str):
                _expert_leaves = sorted({t.rsplit(".", 1)[0] for t in _moe_module_targets})
                _expert_alt = (
                    r".*\.experts\.(?:"
                    + "|".join(re.escape(leaf) for leaf in _expert_leaves)
                    + r")\.\d+"
                )
                target_modules = f"(?:{target_modules})|(?:{_expert_alt})"
            print(
                f"Unsloth: Detected MoE model with per-expert Linear experts. "
                f"Enabling LoRA on {len(_moe_module_targets)} expert projection modules."
            )
            warn_if_zoo_cannot_merge_moe_experts()

        if finetune_last_n_layers is not None and layers_to_transform is None:
            _total_layers = _get_total_transformer_layers(model)
            if _total_layers is not None and _total_layers > 0:
                n = max(1, min(int(finetune_last_n_layers), _total_layers))
                layers_to_transform = list(range(_total_layers - n, _total_layers))

        # Get only allowed parameters for LoraConfig
        local_variables = {
            **locals(),
            **kwargs,
        }
        del local_variables["kwargs"]
        allowed_parameters = inspect.signature(LoraConfig).parameters.keys()
        lora_config = LoraConfig(
            **{k: v for k, v in local_variables.items() if k in allowed_parameters},
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        # Gemma4 ClippableLinear wraps nn.Linear -- PEFT can't inject LoRA on it directly.
        # Monkey-patch PEFT to target the inner .linear child instead.
        _clippable_linear_cls = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import (
                Gemma4ClippableLinear as _clippable_linear_cls,
            )
        except ImportError:
            pass
        if _clippable_linear_cls is not None:
            from peft.tuners.lora.model import LoraModel as _LoraModel

            _original_car = _LoraModel._create_and_replace

            def _patched_car(
                self,
                peft_config,
                adapter_name,
                target,
                target_name,
                parent,
                current_key = None,
                **kwargs,
            ):
                if isinstance(target, _clippable_linear_cls):
                    return _original_car(
                        self,
                        peft_config,
                        adapter_name,
                        target.linear,
                        "linear",
                        target,
                        current_key = current_key,
                        **kwargs,
                    )
                return _original_car(
                    self,
                    peft_config,
                    adapter_name,
                    target,
                    target_name,
                    parent,
                    current_key = current_key,
                    **kwargs,
                )

            _LoraModel._create_and_replace = _patched_car

        model = _get_peft_model(model, lora_config)

        # Restore original PEFT method
        if _clippable_linear_cls is not None:
            _LoraModel._create_and_replace = _original_car
        # Apply QAT + LoRA if specified
        if qat_scheme is not None:
            print("Unsloth: Applying QAT to mitigate quantization degradation")
            model = _prepare_model_for_qat(model, qat_scheme)
        # Fix LoraConfig.auto_mapping is None
        fix_lora_auto_mapping(model)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)
        trust_remote_code = getattr(model, "_unsloth_trust_remote_code", False)
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            trust_remote_code = trust_remote_code,
        )
        model.max_seq_length = max_seq_length
        # Save to modules as well
        for module in model.modules():
            module.max_seq_length = max_seq_length
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        patch_saving_functions(model, vision = True)
        patch_peft_fast_inference(model)

        # Add for_inference and for_training
        model.for_training = functools.partial(FastBaseModel.for_training, model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        # Detect a stray pre-train forward so train() can drop the torch.compile
        # graph cache it would otherwise poison (see prepare_for_training_mode).
        _unsloth_install_pretrain_detector(model)
        model = _exclude_rope_inv_freq_from_ddp(model)
        return model

    @staticmethod
    def post_patch_model(
        model,
        use_gradient_checkpointing = True,
        trust_remote_code = False,
        model_type = None,
        tokenizer = None,
        float32_mixed_precision = None,
    ):
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"

        if type(float32_mixed_precision) is bool:
            # Respect whatever it was set before
            pass
        else:
            float32_mixed_precision = True
            if _get_dtype(dtype_from_config(model.config)) == torch.bfloat16 and full_finetuning:
                # Use bfloat16 precision for full finetuning
                float32_mixed_precision = False

        # VLMs can hit DDP "marked ready twice" with re-entrant checkpointing.
        # See: https://github.com/unslothai/unsloth/issues/3713.
        use_reentrant = not is_distributed()
        if not use_reentrant:
            # Under DDP, avoid the offloaded/re-entrant checkpoint patch.
            unpatch_unsloth_gradient_checkpointing()
            unpatch_unsloth_smart_gradient_checkpointing()
            # Force native checkpoint to default to non-reentrant for downstream calls.
            _orig_checkpoint = torch_checkpoint.checkpoint

            def _nonre_checkpoint(function, *args, **kwargs):
                kwargs["use_reentrant"] = False
                return _orig_checkpoint(function, *args, **kwargs)

            torch_checkpoint.checkpoint = _nonre_checkpoint
            hf_modeling_utils.checkpoint = _nonre_checkpoint

        model = prepare_model_for_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = use_reentrant,
            full_finetuning = full_finetuning,
            train_layernorms = full_finetuning,
            train_embedding = full_finetuning,
            train_lm_head = full_finetuning,
            float32_mixed_precision = float32_mixed_precision,
            patch_modules_to_save = True,
        )
        # Persist the configured GC mode so the trainer restores it verbatim.
        # for_inference() clears the module flags (GRPO does this every generation
        # step), and a plain TrainingArguments defaults gradient_checkpointing=False,
        # which would otherwise silently disable this setting at train time (#4735).
        model._unsloth_gradient_checkpointing = use_gradient_checkpointing

        # Gemma3N audio conformer processes variable-length audio tensors
        # that cause stride mismatches in AOT autograd compiled backward
        # when non-reentrant checkpointing is used. The notebook or TRL
        # may override gradient_checkpointing_kwargs with use_reentrant=False
        # after this point, so we intercept gradient_checkpointing_enable
        # to always force use_reentrant=True for Gemma3N.
        _model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        if "gemma3n" in _model_type.lower() or "gemma4" in _model_type.lower():
            _original_gc_enable = model.gradient_checkpointing_enable

            def _gc_enable_reentrant(**kwargs):
                gc_kwargs = kwargs.get("gradient_checkpointing_kwargs", {}) or {}
                gc_kwargs["use_reentrant"] = True
                kwargs["gradient_checkpointing_kwargs"] = gc_kwargs
                return _original_gc_enable(**kwargs)

            model.gradient_checkpointing_enable = _gc_enable_reentrant

        from transformers.trainer import Trainer

        if (
            Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop"
            and trust_remote_code == False
        ):
            raise RuntimeError("Unsloth: Unsuccessfully patched inner_training_loop")
        patch_saving_functions(model, vision = True)

        # Patch tokenizer to pad to the left
        m = model
        while hasattr(m, "model"):
            if hasattr(m, "_saved_temp_tokenizer"):
                if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                    m._saved_temp_tokenizer.tokenizer.padding_side = "left"
            m = m.model
        if hasattr(m, "_saved_temp_tokenizer"):
            if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                m._saved_temp_tokenizer.tokenizer.padding_side = "left"
        # Prevent Transformers Trainer from auto-wrapping Unsloth LoRA models in DP.
        _mark_unsloth_disable_data_parallel(model, disable = not full_finetuning)

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.empty_cache()
            elif DEVICE_TYPE == "xpu":
                torch.xpu.empty_cache()
        # Add for_inference and for_training
        model.for_training = functools.partial(FastBaseModel.for_training, model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        # Set weight[padding_idx] = 0 for embeddings that are NOT tied with the
        # lm_head. When weights are tied, zeroing the padding row also zeros
        # the corresponding lm_head row, forcing logit = 0 for the pad token.
        # Only do this if tokenizer is defined since eos_token == pad_token sometimes!
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        lm_head = getattr(model, "lm_head", None)
        lm_head_weight = getattr(lm_head, "weight", None) if lm_head is not None else None
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) != pad_token_id:
            with torch.no_grad():
                for name, module in model.named_modules():
                    if type(module) is torch.nn.Embedding:
                        if (
                            getattr(module, "weight", None) is not None
                            and getattr(module, "padding_idx", None) is not None
                        ):
                            if (
                                module.padding_idx == pad_token_id
                                and module.padding_idx < module.weight.shape[0]
                            ):
                                # Skip if tied to lm_head
                                if (
                                    lm_head_weight is not None
                                    and module.weight.data_ptr() == lm_head_weight.data_ptr()
                                ):
                                    continue
                                module.weight[module.padding_idx] = 0
        return model

    @staticmethod
    def for_inference(model):
        if not hasattr(model, "parameters"):
            raise TypeError(
                "Unsloth: I think you're passing a tokenizer, not the model to for_inference!"
            )

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
            if hasattr(m, "training"):
                m.training = False
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "left"
            # Set a flag for generation!
            m._flag_for_generation = True

        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)
        model.eval()  # to turn off training on modules deeper in

        # Since transformers 4.53, must turn off explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = False
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = False
        # Restore use_cache values that prepare_model_for_training disabled
        # for gradient checkpointing (older unsloth_zoo has no restore helper)
        try:
            from unsloth_zoo.training_utils import restore_use_cache
            restore_use_cache(model)
        except ImportError:
            pass

        # Must disable returning hidden states in the case for GRPO
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        # Must enable returning logits
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        # Turn off skip guards and set stance to default
        if torch_compiler_set_stance is not None:
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError(
                "Unsloth: I think you're passing a tokenizer, not the model to for_training!"
            )

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"):
                m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"):
                m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"):
                try:
                    # Weirdly sometimes cannot succeed so do a try except
                    del m._flag_for_generation
                except:
                    pass

        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)
        model.train()  # to turn on training on modules deeper in

        # Since transformers 4.53, must turn on explicitly
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = use_gradient_checkpointing

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = True
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"):
                embeddings.training = True
        # Re-disable use_cache if prepare_model_for_training had disabled it
        # and for_inference restored it (record only exists after a disable)
        if (
            use_gradient_checkpointing
            and getattr(model, "_unsloth_use_cache_originals", None) is not None
        ):
            try:
                from unsloth_zoo.training_utils import disable_use_cache
                disable_use_cache(model)
            except ImportError:
                pass

        # Can re-enable not returning logits
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        # Turn off skip guards and set stance to default
        if torch_compiler_set_stance is not None:
            torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
        return model


def _looks_like_message_list(value):
    return isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict))


def _iter_message_lists(example, column):
    if _looks_like_message_list(example):
        yield example
        return
    if not isinstance(example, dict):
        return
    seen_keys = set()
    for key in (column, "messages", "conversations", "prompt", "completion"):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        value = example.get(key)
        if _looks_like_message_list(value):
            yield value


def _local_path_from_video_value(video_path):
    # data: URIs are inline payloads, not files, and contain no "://"
    if video_path.startswith("data:"):
        return None
    if "://" not in video_path:
        return video_path
    if not video_path.startswith("file://"):
        return None
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    parsed = urlparse(video_path)
    # RFC 8089: only an empty authority or "localhost" is the local machine
    if parsed.netloc and parsed.netloc != "localhost":
        return None
    path = url2pathname(parsed.path)
    return path or None


def check_dataset_for_missing_videos(
    dataset,
    column = "messages",
    raise_error = True,
    checked = None,
):
    """
    Validate that local video paths referenced in a dataset exist, catching
    missing files before training (torchvision otherwise returns an empty
    tensor and the model silently receives no video signal).

    Args:
        dataset:     Map-style Dataset, list of dicts, or iterable of examples
                     (not a streaming IterableDataset - iterating consumes it).
        column:      Chat-messages column, default "messages"; "conversations",
                     "prompt" and "completion" are also scanned.
        raise_error: True (default) raises FileNotFoundError listing missing
                     files; False warns and returns them.
        checked:     Optional set of known-good paths for cross-call dedup.

    Returns:
        List[str]: Missing file paths (empty when all exist).
    """
    try:
        from datasets import IterableDataset as _IterableDataset
        if isinstance(dataset, _IterableDataset):
            warnings.warn(
                "Unsloth: check_dataset_for_missing_videos received a streaming "
                "IterableDataset; iterating would exhaust it and training would "
                "see zero samples. Skipping validation - pass a map-style Dataset "
                "or rely on the UnslothVisionDataCollator's per-batch check.",
                stacklevel = 2,
            )
            return []
    except ImportError:
        pass

    missing = []
    # Report each missing path once; only confirmed-existing paths enter
    # `checked`, so retries after an error re-check previously missing files.
    seen_missing = set()
    if checked is None:
        checked = set()

    for example in dataset:
        for messages in _iter_message_lists(example, column):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", [])
                if not isinstance(content, (list, tuple)):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "video":
                        continue
                    video_path = item.get("video", "")
                    if not isinstance(video_path, str) or not video_path:
                        continue
                    path = _local_path_from_video_value(video_path)
                    if path is None or path in checked or path in seen_missing:
                        continue
                    if not os.path.isfile(path):
                        seen_missing.add(path)
                        missing.append(path)
                    else:
                        checked.add(path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        error_msg = (
            f"Unsloth: {len(missing)} video file(s) referenced in your dataset could not be found.\n"
            "Training would silently continue with empty video tensors - the model would receive\n"
            "no actual video signal while loss still appears to decrease.\n\n"
            f"Missing files:\n{missing_list}\n\n"
            "Fix: verify the video file paths in your dataset before calling the trainer."
        )
        if raise_error:
            raise FileNotFoundError(error_msg)
        warnings.warn(error_msg, stacklevel = 2)

    return missing


# Auto-enable grouped-GEMM MoE (transformers<5 ModuleList experts); see llama.py.
try:
    from unsloth_zoo.temporary_patches.moe_grouped_modulelist import wrap_loader_for_grouped_moe
    FastBaseModel.from_pretrained = staticmethod(
        wrap_loader_for_grouped_moe(FastBaseModel.from_pretrained)
    )
    FastBaseModel.get_peft_model = staticmethod(
        wrap_loader_for_grouped_moe(FastBaseModel.get_peft_model)
    )
except Exception:
    pass
