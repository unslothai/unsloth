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

__version__ = "2026.7.3"

__all__ = [
    "SUPPORTS_BFLOAT16",
    "is_bfloat16_supported",
    "is_vLLM_available",
    "prepare_model_for_kbit_training",
    "xformers",
    "xformers_attention",
    "xformers_version",
    "__version__",
    "importlib_version",
    "HAS_FLASH_ATTENTION",
    "HAS_FLASH_ATTENTION_SOFTCAPPING",
    "USE_MODELSCOPE",
    "platform_system",
    "resolve_hip_gpu_stats_name",
    "patch_tokenizer",
    "get_statistics",
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "offload_to_disk",
    "offload_input_embeddings",
    "offload_output_embeddings",
    "unsloth_offloaded_gradient_checkpoint",
    "torch_compile_options",
    "patch_linear_scaling",
    "patch_llama_rope_scaling",
    "create_boolean_mask",
    "torch_amp_custom_fwd",
    "torch_amp_custom_bwd",
    # "accelerate_old_send_to_device",
    # "accelerate_new_send_to_device",
    "patch_gradient_accumulation_fix",
    "apply_accepts_loss_kwargs_fix",
    "patch_compiling_bitsandbytes",
    "patch_regional_compilation",
    "patch_layernorm",
    "patch_torch_compile",
    "patch_model_and_tokenizer",
    "patch_unsloth_gradient_checkpointing",
    "unpatch_unsloth_gradient_checkpointing",
    "patch_gradient_checkpointing",
    "unpatch_gradient_checkpointing",
    "HAS_CUT_CROSS_ENTROPY",
    "EMPTY_LOGITS",
    "fused_linear_cross_entropy",
    "unsloth_fused_ce_loss",
    "patch_unsloth_smart_gradient_checkpointing",
    "unpatch_unsloth_smart_gradient_checkpointing",
    "apply_unsloth_gradient_checkpointing",
    "_unsloth_install_pretrain_detector",
    "_unsloth_reset_stray_compile_cache",
    "patch_compiled_autograd",
    "process_vision_info",
    "unsloth_compile_transformers",
    "resolve_model_class",
    "resolve_attention_implementation",
    "resolve_encoder_attention_implementation",
    "_set_attn_impl",
    "set_task_config_attr",
    "patch_fast_lora",
    "validate_loftq_config",
    "RaiseUninitialized",
    "fast_inference_setup",
    "patch_peft_fast_inference",
    "error_out_no_vllm",
    "dequantize_module_weight",
    "patch_hf_quantizer",
    "verify_fp8_support_if_applicable",
    "_get_inference_mode_context_manager",
    "hf_login",
    "maybe_prefetch_hf_snapshot",
    "is_moe_model",
    "get_moe_target_parameters",
    "get_moe_target_modules",
    "warn_if_zoo_cannot_merge_moe_experts",
    "_select_moe_detection_targets",
    "make_fast_generate_wrapper",
    "_mark_unsloth_disable_data_parallel",
    "_patch_transformers_trainer_data_parallel",
]

import torch
from typing import Union, Optional, List, Any, Callable, Tuple, Iterator
from platform import system as platform_system

platform_system = platform_system()
import numpy as np
import contextlib
import copy
import re
from dataclasses import dataclass, field
import functools
import textwrap
import logging
import warnings, subprocess, inspect, psutil, os, math
from unsloth_zoo.utils import Version, get_quant_type
from importlib.metadata import version as importlib_version
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)
from ..import_fixes import UNSLOTH_ENABLE_LOGGING
from unsloth_zoo.log import logger
from unsloth_zoo.tokenizer_utils import (
    patch_tokenizer as _patch_tokenizer,
)
from unsloth_zoo.rl_environments import (
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
    Benchmarker,
)
from unsloth_zoo.patching_utils import (
    patch_compiling_bitsandbytes,
    patch_layernorm,
    patch_torch_compile,
    patch_model_and_tokenizer,
    patch_compiled_autograd,
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
    patch_unsloth_smart_gradient_checkpointing,
    unpatch_unsloth_smart_gradient_checkpointing,
)
from unsloth_zoo.loss_utils import (
    HAS_CUT_CROSS_ENTROPY,
    fused_linear_cross_entropy,
    _unsloth_get_batch_samples,
    unsloth_fused_ce_loss,
)
from unsloth_zoo.vision_utils import (
    process_vision_info,
)
from unsloth_zoo.compiler import (
    get_transformers_model_type,
    unsloth_compile_transformers as _unsloth_compile_transformers,
)
from unsloth_zoo.training_utils import (
    prepare_model_for_training,
)


def _iter_wrapped_models(model):
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        next_model = getattr(current, "model", None)
        if next_model is None:
            next_model = getattr(current, "base_model", None)
        if next_model is None:
            next_model = getattr(current, "module", None)
        current = next_model


def _patch_transformers_trainer_data_parallel():
    try:
        from transformers.trainer import Trainer
    except (ImportError, ModuleNotFoundError):
        return False

    original_wrap_model = getattr(Trainer, "_wrap_model", None)
    if original_wrap_model is None:
        return False
    if getattr(original_wrap_model, "_unsloth_data_parallel_patched", False):
        return True
    try:
        supports_dataloader = "dataloader" in inspect.signature(original_wrap_model).parameters
    except (TypeError, ValueError):
        supports_dataloader = True

    def _call_original_wrap_model(self, model, wrap_args, wrap_kwargs):
        if supports_dataloader:
            return original_wrap_model(self, model, *wrap_args, **wrap_kwargs)

        if "dataloader" in wrap_kwargs:
            wrap_kwargs = {k: v for k, v in wrap_kwargs.items() if k != "dataloader"}
        return original_wrap_model(self, model, *wrap_args, **wrap_kwargs)

    @functools.wraps(original_wrap_model)
    def _unsloth_wrap_model(self, model, *wrap_args, **wrap_kwargs):
        args = getattr(self, "args", None)
        disable_data_parallel = getattr(model, "_unsloth_disable_data_parallel", False)
        is_real_8bit = getattr(model, "is_loaded_in_8bit", False)
        if (
            args is None
            or not disable_data_parallel
            or is_real_8bit
            or getattr(args, "n_gpu", 0) <= 1
        ):
            return _call_original_wrap_model(self, model, wrap_args, wrap_kwargs)

        had_n_gpu = hasattr(args, "_n_gpu")
        old_n_gpu = getattr(args, "_n_gpu", None)
        args._n_gpu = 1
        try:
            return _call_original_wrap_model(self, model, wrap_args, wrap_kwargs)
        finally:
            if had_n_gpu:
                args._n_gpu = old_n_gpu
            else:
                try:
                    delattr(args, "_n_gpu")
                except AttributeError:
                    pass

    _unsloth_wrap_model._unsloth_data_parallel_patched = True
    _unsloth_wrap_model._unsloth_original_wrap_model = original_wrap_model
    Trainer._wrap_model = _unsloth_wrap_model
    return True


def _mark_unsloth_disable_data_parallel(model, disable = True):
    if disable:
        _patch_transformers_trainer_data_parallel()
    for module in _iter_wrapped_models(model):
        setattr(module, "_unsloth_disable_data_parallel", bool(disable))
    return model


def resolve_hip_gpu_stats_name(gpu_stats):
    name = str(getattr(gpu_stats, "name", "") or "").strip()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    normalized_name = name.lower().strip(". ")
    if normalized_name and normalized_name not in ("amd radeon graphics",):
        return name + ". "

    try:
        torch_name = str(torch.cuda.get_device_name(0) or "").strip()
        torch_name = re.sub(r"\s*\([^)]*\)\s*$", "", torch_name).strip()
    except Exception:
        torch_name = ""
    normalized_torch_name = torch_name.lower().strip(". ")
    if normalized_torch_name and normalized_torch_name not in ("amd radeon graphics",):
        return torch_name + ". "

    arch_name = ""
    for key in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        value = getattr(gpu_stats, key, None)
        if value is not None and str(value).strip():
            arch_name = str(value).strip()
            break

    if arch_name:
        arch_name = arch_name.strip()
        match = re.search(r"(gfx[0-9a-z]+)", arch_name, flags = re.I)
        if match:
            return f"AMD {match.group(1).lower()} GPU. "
    return "AMD GPU. "


from unsloth_zoo.temporary_patches import (
    TEMPORARY_PATCHES,
)


def _unsloth_install_pretrain_detector(model):
    """Attach a one-shot forward pre-hook recording whether a forward ran before
    trainer.train(), so prepare_for_training_mode can drop a torch.compile graph cache poisoned
    by a stray manual forward/backward. Idempotent; no-op if the model cannot take hooks."""
    if model is None or not hasattr(model, "register_forward_pre_hook"):
        return model
    marker = getattr(model, "_unsloth_pretrain_marker", None)
    if isinstance(marker, dict):
        # A live hook is already recording: keep it (no duplicates) and DON'T clear seen -- a
        # grad-enabled probe may have already flagged the poisoned cache, and a re-entrant
        # get_peft_model/patch_peft_model call must not erase that before train() resets.
        if "hook" in marker:
            return model
        # Marker exists but its hook was torn down -> reinstall fresh, so reset seen.
        marker["seen"] = False
    else:
        marker = {"seen": False}
        try:
            model._unsloth_pretrain_marker = marker
        except Exception:
            return model

    def _mark(_module, _inp):
        # Only a grad-enabled forward poisons the AOTAutograd backward-graph cache; a no-grad
        # probe builds no backward graph, so treat it as clean (avoids a needless dynamo reset).
        if torch.is_grad_enabled():
            marker["seen"] = True

    try:
        marker["hook"] = model.register_forward_pre_hook(_mark)
    except Exception:
        pass
    return model


def _unsloth_reset_stray_compile_cache(self):
    # A manual forward/backward under torch.compile BEFORE trainer.train() (e.g. a grad-norm
    # probe) caches a forward + AOTAutograd backward graph in a one-off context; reusing it
    # poisons training with NaN/zero gradients. If such a forward was seen and compile is on,
    # drop the compiled-graph cache so training recompiles cleanly. No-op on the normal path.
    # Module-level (not just inside the RL trainer template) so the SFT auto-packing wrapper and
    # the plain-Trainer loop can import and run it too.
    import os

    model = getattr(self, "model", None)
    if model is None:
        return
    # The detector hook can sit on any wrapper in the chain, and the probe may have run on a
    # different one than self.model, so walk the chain: detect a "seen" marker anywhere and
    # collect every marker to tear down below.
    markers = []
    seen = False
    _curr = model
    _visited = set()
    while _curr is not None and id(_curr) not in _visited:
        _visited.add(id(_curr))
        _m = getattr(_curr, "_unsloth_pretrain_marker", None)
        if isinstance(_m, dict):
            markers.append(_m)
            if _m.get("seen"):
                seen = True
        # Follow the wrapper chain: Unsloth/HF (.model), PEFT (.base_model), DDP/FSDP (.module).
        _nxt = getattr(_curr, "model", None)
        if _nxt is None:
            _nxt = getattr(_curr, "base_model", None)
        if _nxt is None:
            _nxt = getattr(_curr, "module", None)
        _curr = _nxt
    if seen and os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") != "1":
        try:
            import torch._dynamo as _dynamo
            _dynamo.reset()
        except Exception:
            pass
        try:
            from unsloth_zoo.gradient_checkpointing import (
                reset_unsloth_gradient_checkpointing_buffers,
            )
            reset_unsloth_gradient_checkpointing_buffers()
        except Exception:
            pass
        try:
            model.zero_grad(set_to_none = True)
        except Exception:
            pass
        import warnings

        warnings.warn(
            "Unsloth: detected a manual forward/backward run before trainer.train(); "
            "reset the torch.compile graph cache it poisoned so training starts clean. "
            "To avoid this, run any pre-train probe under `with torch.no_grad():`."
        )
    # Tear down every one-shot detector hook in the chain so none adds per-step cost.
    for _m in markers:
        hook = _m.pop("hook", None)
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass
        _m["seen"] = False


def apply_unsloth_gradient_checkpointing(use_gradient_checkpointing, max_seq_length, dtype):
    """
    Apply gradient checkpointing with smart heuristics.

    For seq < 512, gc="unsloth" offloading overhead isn't worth it; standard gc is faster.

    Args:
        use_gradient_checkpointing: "unsloth", True, False, or None
        max_seq_length: The maximum sequence length
        dtype: The model dtype for patching

    Returns:
        The effective use_gradient_checkpointing value (may change from "unsloth" to True)
    """
    if use_gradient_checkpointing == "unsloth":
        # Offloading not worth it below ~512; standard gc is faster (crossover ~384-512).
        if max_seq_length < 512:
            unpatch_unsloth_smart_gradient_checkpointing()
            return True
        else:
            patch_unsloth_smart_gradient_checkpointing(dtype = dtype)
            return "unsloth"
    elif use_gradient_checkpointing in (True, False):
        # User explicitly set True or False - unpatch any previous "unsloth" patching
        unpatch_unsloth_smart_gradient_checkpointing()
        return use_gradient_checkpointing
    return use_gradient_checkpointing


# Models that don't work with flex_attention as the global Transformers
# attention implementation:
# GPT-OSS: training uses the custom flex sink patch, but inference intentionally
# falls back to eager because flex decoding gives incorrect outputs.
# Mllama: BlockMask Q_LEN!=KV_LEN ValueError on decode.
# NemotronH: hybrid Mamba-2 + Transformer, raises NotImplementedError.
# Gemma3N: timm vision wrappers don't support flex_attention.
# ModernBERT: create_block_mask with _compile=True hits CUDA illegal memory
# access on some GPU architectures (B200). Falls back to eager safely.
_FLEX_EXCLUDED_MODELS = ("gpt_oss", "mllama", "nemotron_h", "modernbert")
_FLEX_PREFERRED_MODELS = ("gemma3", "gemma3_text", "shieldgemma2")
_SDPA_EXCLUDED_MODELS = ("gpt_oss", "deepseek_v4")
# The loader (loader.py) forces supports_sdpa=False for these because their bundled
# SDPA modules are wrong. Kept here, not in loader.py, so _is_sdpa_excluded can honor
# them without a loader -> _utils import cycle (loader.py already imports from _utils
# and re-exports this name for callers like sentence_transformer.py). Entries are matched
# as substrings against a comma-joined model_types string ending in a comma, so "gemma3,"
# matches a distinct "gemma3" entry but not "gemma3n", and "gemma3_text" matches the
# EmbeddingGemma text model.
DISABLE_SDPA_MODEL_NAMES = [
    "gemma3,",  # Add comma bc gemma3 will match gemma3n
    "gemma3_text",  # Gemma3TextModel (EmbeddingGemma) - substring match, keep underscore
    "gpt_oss",
]
_FLASH_EXCLUDED_MODELS = ("gpt_oss", "deepseek_v4")
# deepseek_v4's custom attention is sdpa/flash-incompatible; force eager, and
# excluded above so an explicit sdpa/flash request cannot re-enable the crash.
_EAGER_ONLY_PREFIXES = ("gemma3n", "deepseek_v4")
_FLASH_ATTENTION_MAX_HEAD_DIM = 256
_FLASH_ATTENTION_DISABLED_WARNED = set()


def _is_flex_excluded(model_type):
    return model_type in _FLEX_EXCLUDED_MODELS


def _is_sdpa_disabled_by_name(model_type):
    # Mirror the loader's DISABLE_SDPA_MODEL_NAMES check: loader.py builds
    # model_types_all = ",".join(model_types) + "," and tests `name in model_types_all`.
    # Rebuild the same trailing-comma form for a single model_type so the match is
    # identical (e.g. "gemma3," matches "gemma3" but not "gemma3n", and "gemma3_text"
    # still matches "gemma3_text").
    model_types_all = model_type.lower() + ","
    return any(name.lower() in model_types_all for name in DISABLE_SDPA_MODEL_NAMES)


def _is_sdpa_excluded(model_type):
    # SDPA is known-broken for these models, so an explicit sdpa request must not
    # re-enable it. Two sources: _SDPA_EXCLUDED_MODELS (resolver-level, e.g. gpt_oss)
    # and DISABLE_SDPA_MODEL_NAMES (loader-level, e.g. gemma3 / gemma3_text, which the
    # loader also forces to supports_sdpa=False).
    lowered = model_type.lower()
    return lowered in _SDPA_EXCLUDED_MODELS or _is_sdpa_disabled_by_name(lowered)


def _is_flash_excluded(model_type):
    return model_type in _FLASH_EXCLUDED_MODELS


def _config_prefers_flex_attention(config):
    return any(
        _config_get(attention_config, "model_type", "").lower() in _FLEX_PREFERRED_MODELS
        for attention_config in _iter_attention_configs(config)
    )


def _is_eager_only(model_type):
    return any(model_type.startswith(p) for p in _EAGER_ONLY_PREFIXES)


def _supports_flex_attention(model_class, config, model_type):
    if os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0":
        return False
    if not getattr(model_class, "_supports_flex_attn", False):
        return False
    if _is_flex_excluded(model_type):
        return False
    for attention_config in _iter_attention_configs(config):
        attention_dropout = _config_get(attention_config, "attention_dropout", 0) or 0
        if attention_dropout != 0:
            return False
    try:
        from transformers.utils.import_utils import is_torch_flex_attn_available
        return is_torch_flex_attn_available()
    except Exception:
        return False


def _config_items(config):
    if isinstance(config, dict):
        return config.items()
    if hasattr(config, "__dict__"):
        return vars(config).items()
    return ()


def _config_get(
    config,
    field_name,
    default = None,
):
    if isinstance(config, dict):
        return config.get(field_name, default)
    return getattr(config, field_name, default)


def _config_set(config, field_name, value):
    if isinstance(config, dict):
        config[field_name] = value
    elif config is not None:
        setattr(config, field_name, value)


def set_task_config_attr(config, field_name, value):
    _config_set(config, field_name, value)
    text_config = None
    if isinstance(config, dict):
        text_config = config.get("text_config", None)
    elif config is not None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config()
            except Exception:
                text_config = None
        if text_config is None:
            text_config = getattr(config, "text_config", None)
    if (
        text_config is not None
        and text_config is not config
        and (isinstance(text_config, dict) or hasattr(text_config, "__dict__"))
    ):
        _config_set(text_config, field_name, value)


def _iter_attention_configs(config, seen = None):
    if config is None or (not isinstance(config, dict) and not hasattr(config, "__dict__")):
        return
    if seen is None:
        seen = set()
    config_id = id(config)
    if config_id in seen:
        return
    seen.add(config_id)
    yield config

    for field_name, child_config in _config_items(config):
        if not isinstance(field_name, str) or not field_name.endswith("_config"):
            continue
        if isinstance(child_config, dict) or hasattr(child_config, "__dict__"):
            yield from _iter_attention_configs(child_config, seen)


def _collect_attention_head_dims(config):
    explicit_head_dims = []

    for field_name in (
        "head_dim",
        "global_head_dim",
        "local_head_dim",
        "kv_head_dim",
    ):
        value = _config_get(config, field_name, None)
        if isinstance(value, int) and value > 0:
            explicit_head_dims.append(value)

    if len(explicit_head_dims) != 0:
        return explicit_head_dims

    head_dims = []

    hidden_size_names = ("hidden_size", "d_model", "embed_dim", "dim")
    num_heads_names = ("num_attention_heads", "num_heads", "n_heads")
    for hidden_size_name in hidden_size_names:
        hidden_size = _config_get(config, hidden_size_name, None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            continue
        for num_heads_name in num_heads_names:
            num_heads = _config_get(config, num_heads_name, None)
            if isinstance(num_heads, int) and num_heads > 0 and (hidden_size % num_heads) == 0:
                head_dims.append(hidden_size // num_heads)

    return head_dims


def _get_max_attention_head_dim(config):
    head_dims = []
    for attention_config in _iter_attention_configs(config):
        head_dims.extend(_collect_attention_head_dims(attention_config))
    return max(head_dims) if len(head_dims) != 0 else None


def _get_flash_attention_disable_reason(config):
    model_type = _config_get(config, "model_type", "").lower()
    if _is_flash_excluded(model_type):
        return f"{model_type} uses custom sink attention kernels"
    max_head_dim = _get_max_attention_head_dim(config)
    if max_head_dim is not None and max_head_dim > _FLASH_ATTENTION_MAX_HEAD_DIM:
        return (
            f"max attention head dim {max_head_dim} exceeds the Flash Attention 2 "
            f"limit of {_FLASH_ATTENTION_MAX_HEAD_DIM}"
        )
    return None


def _is_flash_attention_disabled(config):
    return _get_flash_attention_disable_reason(config) is not None


def _is_flash_attention_requested(attn_implementation):
    return isinstance(attn_implementation, str) and attn_implementation.startswith(
        "flash_attention"
    )


def _disable_flash_attention_if_needed(
    config,
    attn_implementation = None,
    supports_sdpa = False,
    supports_flex_attention = False,
    would_use_flash_attention = False,
    disable_reason = None,
):
    if disable_reason is None:
        disable_reason = _get_flash_attention_disable_reason(config)
    if disable_reason is None:
        return attn_implementation

    # Only an implementation passed by the caller counts as an explicit request.
    # Values read from the config are synthesized by the loaders (the language path
    # seeds the config with attn_implementation="sdpa") or come from Transformers
    # defaults, so they must not be treated as a deliberate user choice.
    explicit_request = attn_implementation

    requested_attn_implementation = attn_implementation
    if requested_attn_implementation is None:
        requested_attn_implementation = _config_get(config, "_attn_implementation", None)
    if requested_attn_implementation is None:
        requested_attn_implementation = _config_get(config, "attn_implementation", None)

    if requested_attn_implementation == "eager":
        return _set_attn_impl(config, "eager")

    model_type = _config_get(config, "model_type", "")

    # The disable reason is flash-specific: honor an explicit non-flash request from
    # the caller instead of downgrading it. SDPA is honored unless the model's SDPA is
    # known-broken - _SDPA_EXCLUDED_MODELS (e.g. gpt_oss) or DISABLE_SDPA_MODEL_NAMES
    # (e.g. gemma3 / gemma3_text); flex_attention
    # is honored only when it is actually usable, since supports_flex_attention already
    # rejects the excluded/broken/unavailable configs. This keeps an explicit request
    # from selecting a backend the repo marks as wrong.
    if explicit_request == "sdpa" and not _is_sdpa_excluded(model_type.lower()):
        return _set_attn_impl(config, "sdpa")
    if explicit_request == "flex_attention" and supports_flex_attention:
        return _set_attn_impl(config, "flex_attention")

    if supports_sdpa:
        fallback_attn_implementation = "sdpa"
    elif supports_flex_attention:
        fallback_attn_implementation = "flex_attention"
    else:
        fallback_attn_implementation = "eager"
    if _is_flash_attention_requested(requested_attn_implementation) or would_use_flash_attention:
        logged_attn_implementation = (
            requested_attn_implementation
            if _is_flash_attention_requested(requested_attn_implementation)
            else "flash_attention_2"
        )
        warning_key = (
            model_type,
            logged_attn_implementation,
            fallback_attn_implementation,
            disable_reason,
        )
        if warning_key not in _FLASH_ATTENTION_DISABLED_WARNED:
            _FLASH_ATTENTION_DISABLED_WARNED.add(warning_key)
            print(
                f"Unsloth: `{logged_attn_implementation}` is not supported "
                f"for `{model_type}` because {disable_reason} - "
                f"defaulting to `{fallback_attn_implementation}`."
            )

    return _set_attn_impl(config, fallback_attn_implementation)


def _set_attn_impl(config, impl):
    if config is not None:
        _config_set(config, "_attn_implementation", impl)
        if isinstance(config, dict) or hasattr(config, "attn_implementation"):
            _config_set(config, "attn_implementation", impl)
    return impl


def resolve_model_class(auto_model, config):
    mapping = getattr(auto_model, "_model_mapping", {})
    try:
        result = mapping[config.__class__]
    except Exception:
        result = None
        for key in list(getattr(mapping, "_model_mapping", {})):
            try:
                config_class = mapping._load_attr_from_module(key, mapping._config_mapping[key])
                if isinstance(config, config_class):
                    result = mapping._load_attr_from_module(key, mapping._model_mapping[key])
                    break
            except Exception:
                continue
        if result is None:
            for extra_cls, extra_model in getattr(mapping, "_extra_content", {}).items():
                try:
                    if isinstance(config, extra_cls):
                        result = extra_model
                        break
                except Exception:
                    continue
        if result is None:
            return None
    return result[0] if isinstance(result, (list, tuple)) else result


def _is_family_text_decoder(parent_model_type, text_model_type):
    # True only for the family's own text variant (gemma3 -> gemma3_text); a generic
    # reused decoder (llava -> llama) would load random weights, so keep the full model.
    return bool(parent_model_type) and str(text_model_type).startswith(parent_model_type)


def _get_text_only_config(model_config, model_name):
    # Text sub-config of a vision-language config so FastLanguageModel skips the vision tower (PR #5816).
    text_config = None
    if hasattr(model_config, "get_text_config"):
        text_config = model_config.get_text_config()
    if text_config is None:
        text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        raise ValueError(f"Cannot load {model_name} as text-only; use FastVisionModel")
    # Carry over quantization_config; copy first since get_text_config() shares the parent's object.
    qc = getattr(model_config, "quantization_config", None)
    if qc is not None and getattr(text_config, "quantization_config", None) is None:
        text_config = copy.copy(text_config)
        text_config.quantization_config = _remap_text_only_skip_modules(qc)
    return text_config


def _remap_text_only_skip_modules(qc):
    # Remap llm_int8_skip_modules off the VLM wrapper prefix (language_model.model.* ->
    # model.*) after text-only stripping, and drop vision/audio entries. See PR #5816.
    is_dict = isinstance(qc, dict)
    skip = (
        qc.get("llm_int8_skip_modules") if is_dict else getattr(qc, "llm_int8_skip_modules", None)
    )
    if not skip:
        return qc
    remapped = []
    for name in skip:
        for pref in (
            "language_model.model.",
            "model.language_model.",
            "language_model.",
        ):
            if name.startswith(pref):
                name = (
                    ("model." + name[len(pref) :])
                    if pref != "language_model."
                    else name[len(pref) :]
                )
                break
        if name.startswith(
            (
                "vision_tower",
                "multi_modal_projector",
                "audio_tower",
                "modality_projection",
            )
        ):
            continue
        remapped.append(name)
    remapped = list(dict.fromkeys(remapped))
    qc = dict(qc) if is_dict else copy.copy(qc)
    if is_dict:
        qc["llm_int8_skip_modules"] = remapped
    else:
        qc.llm_int8_skip_modules = remapped
    return qc


def _get_text_only_key_mapping(parent_config, text_config):
    # transformers >=5 stopped auto-stripping the VLM wrapper prefix (base_model_prefix
    # changed language_model -> model), so remap the text weights onto the decoder keys.
    # None on tf <5 (still strips; a mapping would break the load) or non-family. See PR #5816.
    if Version(transformers_version) < Version("5.0.0"):
        return None
    if not _is_family_text_decoder(
        getattr(parent_config, "model_type", ""),
        getattr(text_config, "model_type", ""),
    ):
        return None
    return {
        r"^language_model\.model\.": "model.",
        r"^model\.language_model\.": "model.",
        r"^language_model\.lm_head\.": "lm_head.",
    }


def _apply_text_only_key_mapping(kwargs, parent_config, text_config):
    # Add the text-only key_mapping to from_pretrained kwargs, under any user mapping.
    mapping = _get_text_only_key_mapping(parent_config, text_config)
    if not mapping:
        return
    user_mapping = kwargs.get("key_mapping", None)
    kwargs["key_mapping"] = {**mapping, **user_mapping} if user_mapping else mapping


def resolve_attention_implementation(
    model_class,
    config,
    requested_attn_implementation = None,
    supports_sdpa = None,
):
    model_type_name = _config_get(config, "model_type", "")
    model_type = model_type_name.lower()
    if supports_sdpa is None:
        supports_sdpa = model_class is not None and getattr(model_class, "_supports_sdpa", False)
    if _is_sdpa_excluded(model_type):
        supports_sdpa = False
    supports_flash_attention = (
        model_class is not None
        and (
            getattr(model_class, "_supports_flash_attn_2", False)
            or getattr(model_class, "_supports_flash_attn", False)
        )
        and not _is_flash_excluded(model_type)
    )
    supports_flex_attention = _supports_flex_attention(model_class, config, model_type)
    disable_reason = _get_flash_attention_disable_reason(config)
    flash_attention_disabled = disable_reason is not None

    if model_class is None:
        attn_impl = _set_attn_impl(config, "sdpa" if supports_sdpa else "eager")
    else:
        prefers_flex_attention = _config_prefers_flex_attention(config)
        if _is_eager_only(model_type):
            attn_impl = _set_attn_impl(config, "eager")
        elif prefers_flex_attention and supports_flex_attention:
            # Models in _FLEX_PREFERRED_MODELS (gemma3 family) prefer flex_attention
            # over flash. Caller can still override by passing
            # requested_attn_implementation="sdpa" (handled below).
            attn_impl = _set_attn_impl(config, "flex_attention")
        elif not flash_attention_disabled and HAS_FLASH_ATTENTION and supports_flash_attention:
            attn_impl = _set_attn_impl(config, "flash_attention_2")
        elif flash_attention_disabled:
            attn_impl = _disable_flash_attention_if_needed(
                config,
                supports_sdpa = supports_sdpa,
                supports_flex_attention = supports_flex_attention,
                would_use_flash_attention = (HAS_FLASH_ATTENTION and supports_flash_attention),
                disable_reason = disable_reason,
            )
        elif supports_sdpa:
            attn_impl = _set_attn_impl(config, "sdpa")
        elif supports_flex_attention:
            # Flex is only a fallback for models that don't support SDPA
            # (e.g. some custom configurations). Without this fallback such
            # models would land on eager.
            attn_impl = _set_attn_impl(config, "flex_attention")
        else:
            attn_impl = _set_attn_impl(config, "eager")

    if requested_attn_implementation is None:
        final_attn_impl = attn_impl
    elif flash_attention_disabled:
        final_attn_impl = _disable_flash_attention_if_needed(
            config,
            requested_attn_implementation,
            supports_sdpa = supports_sdpa,
            supports_flex_attention = supports_flex_attention,
            disable_reason = disable_reason,
        )
    else:
        final_attn_impl = requested_attn_implementation
        _set_attn_impl(config, final_attn_impl)

    # A caller who explicitly passes requested_attn_implementation="sdpa" keeps it even
    # on a conservatively unsupported model, mirroring _disable_flash_attention_if_needed
    # which honors an explicit sdpa request. The exception is a model whose SDPA is
    # known-broken - _SDPA_EXCLUDED_MODELS (e.g. gpt_oss) or DISABLE_SDPA_MODEL_NAMES
    # (e.g. gemma3 / gemma3_text, which the loader also forces to supports_sdpa=False):
    # an explicit request must not re-enable it, so it still downgrades to eager, just
    # like flex falls back for _FLEX_EXCLUDED_MODELS. A synthesized/default sdpa
    # (requested is None, so the value came from the model resolution above or the
    # config) also downgrades.
    honor_explicit_sdpa = requested_attn_implementation == "sdpa" and not _is_sdpa_excluded(
        model_type
    )
    if not supports_sdpa and final_attn_impl == "sdpa" and not honor_explicit_sdpa:
        print(
            f"Unsloth: {(model_type_name or 'model').title()} does not support SDPA - switching to fast eager."
        )
        final_attn_impl = _set_attn_impl(config, "eager")

    return final_attn_impl


def resolve_encoder_attention_implementation(
    auto_model,
    config,
    model_type = "",
    disable_sdpa_model_names = (),
):
    model_class = resolve_model_class(auto_model, config)
    supports_sdpa = model_class is not None and getattr(model_class, "_supports_sdpa", False)
    if any(name in model_type.lower() for name in disable_sdpa_model_names):
        return "eager"
    if supports_sdpa:
        return "sdpa"
    return None


def _run_temporary_patches(phase):
    import inspect
    for temporary_patch in TEMPORARY_PATCHES:
        try:
            sig = inspect.signature(temporary_patch)
            if "phase" in sig.parameters:
                temporary_patch(phase = phase)
            else:
                temporary_patch()
        except (ValueError, TypeError):
            temporary_patch()


_run_temporary_patches("init")

# =============================================
# Disable some warnings which can get annoying
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "torch")
warnings.filterwarnings(action = "ignore", category = FutureWarning, module = "torch")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = FutureWarning, module = "huggingface_hub")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "trl")
warnings.filterwarnings(action = "ignore", category = FutureWarning, module = "trl")
warnings.filterwarnings(action = "ignore", category = FutureWarning, module = "xformers")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "subprocess")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "transformers")
warnings.filterwarnings(action = "ignore", category = FutureWarning, module = "accelerate")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocessing")
warnings.filterwarnings(action = "ignore", category = RuntimeWarning, module = "multiprocess")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "triton")
warnings.filterwarnings(action = "ignore", category = UserWarning, module = "bitsandbytes")

# Stop "Special tokens have been added in the vocabulary, ..."
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL + 1)

TORCHAO_MSG = "Error: torchao not found, please install with `pip install torchao`"


# Artifacts a Transformers/PEFT load never reads (ONNX/TF/Flax/CoreML/GGUF/training state), skipped
# when prewarming so a mixed-format repo is not pulled in full.
_PREFETCH_IGNORE_PATTERNS = (
    "*.onnx",
    "onnx/*",
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "coreml/*",
    "*.mlpackage/*",
    "*.mlmodel",
    "*.gguf",
    # Training / checkpoint formats from_pretrained never reads.
    "*.pt",
    "*.pth",
    "*.ckpt",
    "optimizer.*",
    "scheduler.*",
    "rng_state*",
    "trainer_state.json",
    "events.out.tfevents*",
    "checkpoint-*/*",
)


# Repo-root tokenizer / config / processor files from_pretrained reads from root even when weights
# load from a subfolder. Exact names (no wildcard) so they match only root-level files.
_ROOT_AUX_PREFETCH_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    # More VOCAB_FILES_NAMES the slow tokenizer may fetch (DeBERTa-v2, Whisper, Mistral, XLM-R/mBART, Marian, FSMT/XLM, GPT-2).
    "spm.model",
    "normalizer.json",
    "tokenizer.model.v3",
    "sentencepiece.bpe.model",
    "source.spm",
    "target.spm",
    "bpe.codes",
    "vocab.bpe",
    # More VOCAB_FILES_NAMES (RemBERT, FSMT) a distinct-tokenizer-repo warm must cache too.
    "sentencepiece.model",
    "vocab-src.json",
    "vocab-tgt.json",
    "chat_template.jinja",
    "chat_template.json",
    # chat_template="<name>" fetches additional_chat_templates/<name>.jinja.
    "additional_chat_templates/*.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",  # Qwen2.5-VL-style video processors
    # trust_remote_code auto_map can name any module, so warm every *.py (tiny; none in a non-remote repo).
    "*.py",
    "*.tiktoken",  # tiktoken vocab (e.g. Qwen's qwen.tiktoken)
)


# Files a PEFT adapter load reads: config + weights (glob covers sharded adapters). Any merged
# full-model weights the repo also ships match none of these.
_ADAPTER_PREFETCH_PATTERNS = (
    "adapter_config.json",
    "adapter_model*",
)


# Weight files in a SUBDIRECTORY. A bare root load reads only root weights, so ignoring these drops
# alternate-precision/experimental dirs (fp16/, experimental/). "*/*" spans "/" (HF fnmatch), so nested
# weights match while root "model.safetensors" is kept. Only applied when weights_at_root (diffusion
# keeps weights in subfolders).
_SUBDIR_WEIGHT_IGNORE_PATTERNS = (
    "*/*.safetensors",
    "*/*.bin",
    "*/*.h5",
    "*/*.msgpack",
    "*/*.pt",
    "*/*.pth",
)


def _in_requested_load_scope(filename, subfolder):
    """True if *filename* is in the location being loaded (*subfolder*, else root). Scopes the ".bin is
    redundant when safetensors exist" test so a .bin-only subfolder keeps its .bin."""
    filename = filename.replace("\\", "/")
    if isinstance(subfolder, str) and subfolder.strip("/"):
        return filename.startswith(subfolder.strip("/") + "/")
    return "/" not in filename  # root load: no directory component


# .safetensors training-state files that are NOT model weights (e.g. optimizer.safetensors next to a
# real pytorch_model.bin); counting them as "model safetensors present" would drop the needed .bin.
_NON_MODEL_WEIGHT_STEMS = frozenset(
    {
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "training_args",
    }
)


def _is_model_weight_safetensors(filename):
    """True if *filename* is a model-weights safetensors, not a PEFT adapter/sidecar
    (adapter_model.safetensors) or trainer-state (optimizer.safetensors). Only a real one proves the
    .bin redundant; counting a sidecar would wrongly drop the needed .bin (fetched then without Xet fallback)."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.endswith((".safetensors", ".safetensors.index.json")):
        return False
    if name.startswith("adapter_"):
        return False
    # Stem before first dot: "optimizer.safetensors" -> "optimizer" (real shards kept); rng_state via prefix.
    stem = name.split(".", 1)[0].lower()
    if stem in _NON_MODEL_WEIGHT_STEMS or stem.startswith("rng_state"):
        return False
    return True


def _is_canonical_variant_model_weight_safetensors(filename, variant):
    """True for a canonical model-weights safetensors carrying the requested *variant*, in the forms
    transformers reads (single, either numbered-shard layout, or the index). Strict (base must be
    "model"): a sidecar like consolidated.<variant>.safetensors does not prove the variant .bin redundant."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    v = re.escape(variant)
    return bool(
        re.match(
            rf"^(?:model\.{v}\.safetensors"
            rf"|model\.{v}-\d{{5}}-of-\d{{5}}\.safetensors"
            rf"|model-\d{{5}}-of-\d{{5}}\.{v}\.safetensors"
            rf"|model\.safetensors\.index\.{v}\.json)$",
            base,
        )
    )


_CANONICAL_MODEL_WEIGHT_SAFETENSORS_RE = re.compile(
    r"^(?:model\.safetensors|model-\d{5}-of-\d{5}\.safetensors|model\.safetensors\.index\.json)$"
)


def _is_canonical_model_weight_safetensors(filename):
    """True for a canonical (non-variant) model-weights safetensors a default load reads (model.safetensors,
    a numbered shard, or the index). Strict: an unrecognized name keeps both formats, so a variant-only
    safetensors + pytorch_model.bin repo never has its .bin dropped for a no-variant load."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(_CANONICAL_MODEL_WEIGHT_SAFETENSORS_RE.match(name))


def _adapter_repo_has_safetensors(
    model_name,
    *,
    token = None,
    revision = None,
):
    """Best-effort: does the adapter repo ship a root safetensors adapter weight (making the .bin
    redundant)? Scoped to root adapter_model* files; any failure returns False."""
    try:
        from huggingface_hub import HfApi
        siblings = HfApi().model_info(model_name, revision = revision, token = token).siblings or []
        return any(
            "/" not in sibling.rfilename.replace("\\", "/")  # root only
            and sibling.rfilename.startswith("adapter_model")
            and sibling.rfilename.endswith(".safetensors")
            for sibling in siblings
        )
    except Exception:
        return False


def _prefetch_ignore_patterns(
    model_name,
    *,
    token = None,
    revision = None,
    subfolder = None,
    use_safetensors = None,
    from_tf = False,
    from_flax = False,
    variant = None,
    weights_at_root = False,
):
    """ignore_patterns for the prewarm snapshot: the static skip list, minus the checkpoint guard when
    loading from a checkpoint-* subfolder, minus the weight format the load will not read. use_safetensors
    is a format allowlist (True -> skip *.bin, False -> skip *.safetensors); auto (None) skips *.bin only
    when in-scope safetensors are shipped. from_tf/from_flax keep *.h5/*.msgpack.

    Suppressed for a whole multi-component snapshot (weights_at_root=False, no subfolder: ST/diffusers
    repos with per-subfolder weights, each in its own format), since "*" spans "/" so dropping "*.bin"
    would strip a module's only weight."""
    # Keep checkpoint-*/* under a checkpoint-* subfolder; keep *.h5 / *.msgpack under from_tf/flax.
    ignore_patterns = [
        pattern
        for pattern in _PREFETCH_IGNORE_PATTERNS
        if not (
            (
                pattern == "checkpoint-*/*"
                and isinstance(subfolder, str)
                and subfolder.startswith("checkpoint-")
            )
            or (from_tf and pattern == "*.h5")
            or (from_flax and pattern == "*.msgpack")
        )
    ]
    # Drop the format the load will not read (the other doubles the download); skipped for a whole
    # multi-component snapshot (see docstring).
    whole_multi_component = not weights_at_root and not (
        isinstance(subfolder, str) and subfolder.strip("/")
    )
    if whole_multi_component:
        pass
    elif from_tf or from_flax:
        # TF / Flax loads never read the PyTorch formats; drop safetensors and .bin.
        ignore_patterns.extend(
            (
                "*.safetensors",
                "*.safetensors.index.json",
                "*.bin",
                "*.bin.index.json",
            )
        )
    elif use_safetensors is True:
        # Explicit safetensors: load never reads .bin (no model_info call needed).
        ignore_patterns.extend(("*.bin", "*.bin.index.json"))
    elif use_safetensors is False:
        # Explicit .bin: load never reads safetensors.
        ignore_patterns.extend(("*.safetensors", "*.safetensors.index.json"))
    else:
        # Auto: skip .bin only once in-scope safetensors are confirmed (best-effort; any failure keeps both).
        try:
            from huggingface_hub import HfApi

            siblings = (
                HfApi()
                .model_info(
                    model_name,
                    revision = revision,
                    token = token,
                )
                .siblings
                or []
            )
            # Count only in-scope model-weights safetensors (not adapters/sidecars): variant-matching if
            # a variant is requested, else canonical, proving the .bin redundant.
            has_safetensors = any(
                _is_model_weight_safetensors(sibling.rfilename)
                and _in_requested_load_scope(sibling.rfilename, subfolder)
                and (
                    _is_canonical_variant_model_weight_safetensors(sibling.rfilename, variant)
                    if variant
                    else _is_canonical_model_weight_safetensors(sibling.rfilename)
                )
                for sibling in siblings
            )
            if has_safetensors:
                ignore_patterns.extend(("*.bin", "*.bin.index.json"))
        except Exception:
            pass
    return ignore_patterns


def maybe_prefetch_hf_snapshot(
    model_name,
    token = None,
    *,
    revision = None,
    cache_dir = None,
    local_files_only = False,
    fast_inference = False,
    subfolder = None,
    force_download = False,
    use_safetensors = None,
    from_tf = False,
    from_flax = False,
    tokenizer_only = False,
    adapter_only = False,
    weights_at_root = False,
    variant = None,
    gguf_file = None,
):
    """Warm the HF cache for a remote repo before the in-process load.

    Xet can hang on a blob with no progress or exception, and a blocked native Xet thread cannot be
    killed in-process. So pull the snapshot first in a killable subprocess that falls back Xet -> HTTP
    on a stall (unsloth_zoo.hf_xet_fallback), making from_pretrained a cache hit.

    Returns True iff warmed (caller can clear force_download), else False (skipped: local/offline/
    local_files_only/fast_inference/old unsloth_zoo, or failed). Only a both-transports-stalled
    DownloadStallError is raised; other failures are left for from_pretrained to surface.
    """
    try:
        from unsloth_zoo.hf_xet_fallback import (
            snapshot_download_with_xet_fallback,
            DownloadStallError,
        )
    except Exception:
        return False  # older unsloth_zoo without the helper: load normally

    if not isinstance(model_name, str) or not model_name:
        return False
    # Local path: nothing to download. Expand ~ first (os.path.exists does not).
    model_path = os.path.expanduser(model_name)
    if os.path.isdir(model_path) or os.path.exists(model_path):
        return False
    # Looks local but not yet on disk (e.g. an uncreated output dir): not a Hub repo id, so leave it
    # for from_pretrained rather than download it.
    if (
        os.path.isabs(model_path)
        or model_name.startswith(("~", "./", "../", ".\\", "..\\"))
        or "\\" in model_name
    ):
        return False
    if local_files_only:  # cache-only: never reach out
        return False
    if any(
        os.environ.get(flag, "0").lower() in ("1", "true", "yes", "on")
        for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        return False
    if fast_inference:  # vLLM has its own download path
        return False

    # tokenizer-only / adapter-only warms allow-list exact files below, so the weight-format ignore
    # list (and its auto-branch model_info call) is skipped.
    ignore_patterns = (
        None
        if tokenizer_only or adapter_only or gguf_file
        else _prefetch_ignore_patterns(
            model_name,
            token = token,
            revision = revision,
            subfolder = subfolder,
            use_safetensors = use_safetensors,
            from_tf = from_tf,
            from_flax = from_flax,
            variant = variant,
            weights_at_root = weights_at_root,
        )
    )
    # Narrow the warm to what the load reads (skip extra checkpoints/precisions); every branch still warms
    # root tokenizer/config/custom-code so those never fall in-process.
    allow_patterns = None
    if gguf_file:
        # gguf_file=NAME reads exactly that GGUF, but the static ignore list drops *.gguf; so warm just
        # that file (plus root aux), under <subfolder>/ if set.
        _gguf_path = (
            f"{subfolder.strip('/')}/{gguf_file}"
            if isinstance(subfolder, str) and subfolder.strip("/")
            else gguf_file
        )
        allow_patterns = [_gguf_path, *_ROOT_AUX_PREFETCH_PATTERNS]
    elif tokenizer_only:
        # A distinct tokenizer repo: warm only tokenizer / config / vocab files, never its weights.
        allow_patterns = list(_ROOT_AUX_PREFETCH_PATTERNS)
    elif adapter_only:
        # A PEFT adapter load reads only adapter_config.json + adapter_model.* (plus root aux), not any
        # merged weights the repo may also publish.
        allow_patterns = [*_ADAPTER_PREFETCH_PATTERNS, *_ROOT_AUX_PREFETCH_PATTERNS]
        # PeftModel reads one format (safetensors when present): explicit use_safetensors wins, else
        # prefer safetensors when shipped (best-effort; any failure keeps both).
        if use_safetensors is False:
            ignore_patterns = [
                "adapter_model*.safetensors",
                "adapter_model*.safetensors.index.json",
            ]
        elif use_safetensors is True or _adapter_repo_has_safetensors(
            model_name, token = token, revision = revision
        ):
            ignore_patterns = ["adapter_model*.bin", "adapter_model*.bin.index.json"]
    elif isinstance(subfolder, str) and subfolder.strip("/"):
        # subfolder=X: load resolves every weight under X/, so warm that subfolder (plus root aux).
        allow_patterns = [f"{subfolder.strip('/')}/*", *_ROOT_AUX_PREFETCH_PATTERNS]
    elif weights_at_root:
        # A bare load reads only root weights: drop subdir weights (fp16/, checkpoint dirs) while keeping
        # subdir configs. Diffusion leaves weights_at_root False.
        ignore_patterns = [*(ignore_patterns or []), *_SUBDIR_WEIGHT_IGNORE_PATTERNS]
    try:
        snapshot_download_with_xet_fallback(
            model_name,
            token = token,
            revision = revision,
            cache_dir = cache_dir,
            allow_patterns = allow_patterns,
            ignore_patterns = ignore_patterns,
            force_download = force_download,
            variant = variant,
        )
        return True
    except DownloadStallError:
        # Both transports stalled: surface a clear network error, not a silent in-process hang.
        raise
    except Exception as exception:
        logger.warning_once(
            f"Unsloth: Could not pre-download {model_name} "
            f"({type(exception).__name__}: {exception}); continuing with the normal load."
        )
        return False


# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def filter(self, x):
        return not (self.text in x.getMessage())


# Replace warning messages (analogous to HideLoggingMessage but for warnings.warn)
class ReplaceWarningMessage:
    """
    Intercepts warnings.warn calls and replaces matching messages with Unsloth branded ones.
    Uses a list of registered (match_text, replacement, category) rules checked in order.
    """

    _rules = []
    _original_showwarning = None
    _installed = False

    @classmethod
    def add_rule(
        cls,
        match_text,
        replacement,
        category = None,
    ):
        cls._rules.append((match_text, replacement, category))
        if not cls._installed:
            cls._install()

    @classmethod
    def _install(cls):
        cls._original_showwarning = warnings.showwarning
        cls._installed = True

        def _patched_showwarning(
            message,
            category,
            filename,
            lineno,
            file = None,
            line = None,
        ):
            msg_str = str(message)
            for match_text, replacement, match_category in cls._rules:
                if match_text in msg_str and (match_category is None or category is match_category):
                    print(replacement)
                    return
            cls._original_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _patched_showwarning


# Stop vLLM messages
if not UNSLOTH_ENABLE_LOGGING:
    try:
        from vllm.worker.worker import logger as vllm_worker_logger
        vllm_worker_logger.addFilter(HideLoggingMessage("Sleep mode freed"))
        del vllm_worker_logger
    except:
        pass
    try:
        from vllm.v1.worker.gpu_worker import logger as vllm_gpu_worker_logger
        vllm_gpu_worker_logger.addFilter(HideLoggingMessage("Sleep mode freed"))
        del vllm_gpu_worker_logger
    except:
        pass
    try:
        from vllm.executor.executor_base import logger as vllm_executor_logger

        vllm_executor_logger.addFilter(HideLoggingMessage("to fall asleep"))
        vllm_executor_logger.addFilter(HideLoggingMessage("to wake up"))
        vllm_executor_logger.addFilter(HideLoggingMessage("Executor is not sleeping"))
        del vllm_executor_logger
    except:
        pass
    try:
        from vllm.v1.executor.abstract import logger as vllm_v1_executor_logger

        vllm_v1_executor_logger.addFilter(HideLoggingMessage("to fall asleep"))
        vllm_v1_executor_logger.addFilter(HideLoggingMessage("to wake up"))
        vllm_v1_executor_logger.addFilter(HideLoggingMessage("Executor is not sleeping"))
        del vllm_v1_executor_logger
    except:
        pass
    try:
        from vllm.core.block.prefix_caching_block import (
            logger as vllm_prefix_caching_logger,
        )
        vllm_prefix_caching_logger.addFilter(HideLoggingMessage("reset prefix cache"))
        del vllm_prefix_caching_logger
    except:
        pass
    try:
        from vllm.v1.core.block_pool import logger as vllm_block_pool_logger
        vllm_block_pool_logger.addFilter(HideLoggingMessage("reset prefix cache"))
        del vllm_block_pool_logger
    except:
        pass
    try:
        from vllm.lora.models import logger as vllm_lora_model_logger
        vllm_lora_model_logger.addFilter(
            HideLoggingMessage("Regarding multimodal models, vLLM currently only supports adding")
        )
        del vllm_lora_model_logger
    except:
        pass
    try:
        from vllm.attention.utils.fa_utils import (
            logger as vllm_attention_utils_fa_utils_logger,
        )
        vllm_attention_utils_fa_utils_logger.addFilter(HideLoggingMessage("Cannot use FA version"))
        del vllm_attention_utils_fa_utils_logger
    except:
        pass

# The speedups for torchdynamo mostly come with GPU Ampere or higher and which is not detected here.
from transformers.training_args import logger as transformers_training_args_logger

transformers_training_args_logger.addFilter(HideLoggingMessage("The speedups"))
# torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED.
transformers_training_args_logger.addFilter(HideLoggingMessage("torch.distributed"))
# average_tokens_across_devices is set to True but it is invalid when world size is1
transformers_training_args_logger.addFilter(HideLoggingMessage("average_tokens_across_devices"))
del transformers_training_args_logger

# No label_names provided for model class
from transformers.trainer import logger as transformers_trainer_logger

transformers_trainer_logger.addFilter(HideLoggingMessage("No label_names"))

# The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config.
transformers_trainer_logger.addFilter(HideLoggingMessage("The tokenizer has new"))
del transformers_trainer_logger

# Strip the "[transformers] " prefix transformers>=5 adds; keep the messages.
try:
    try:
        import transformers.utils.logging as _tf_log
        _tf_log._configure_library_root_logger()
    except Exception:
        pass
    _tf_root = logging.getLogger("transformers")
    _tf_fmt = logging.Formatter("%(message)s")
    for _h in list(_tf_root.handlers):
        _h.setFormatter(_tf_fmt)
    del _tf_root, _tf_fmt
except Exception:
    pass

# Using the default loss: `ForCausalLMLoss`.
try:
    from transformers.modeling_utils import logger as transformers_modeling_utils_logger
    transformers_modeling_utils_logger.addFilter(HideLoggingMessage("ForCausalLMLoss"))
    del transformers_modeling_utils_logger
except:
    pass

# The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
try:
    from accelerate.utils.modeling import logger as accelerate_utils_modeling_logger
    accelerate_utils_modeling_logger.addFilter(HideLoggingMessage("The model weights are not tied"))
    del accelerate_utils_modeling_logger
except:
    pass

# Setting `pad_token_id` to `eos_token_id`
try:
    from transformers.generation.utils import (
        logger as transformers_generation_utils_logger,
    )

    transformers_generation_utils_logger.addFilter(
        HideLoggingMessage("Setting `pad_token_id` to `eos_token_id`")
    )
    # "You have set `compile_config`
    transformers_generation_utils_logger.addFilter(HideLoggingMessage("compile_config"))
    del transformers_generation_utils_logger
except:
    pass

# The following generation flags are not valid and may be ignored:
try:
    from transformers.generation.configuration_utils import (
        logger as configuration_logger,
    )
    configuration_logger.addFilter(HideLoggingMessage("following generation flags"))
    del configuration_logger
except:
    pass

# Gemma3 It is strongly recommended to train Gemma3 models with the `eager`
try:
    from transformers.models.gemma3.modeling_gemma3 import logger as gemma3_logger
    gemma3_logger.addFilter(HideLoggingMessage("strongly recommended"))
    del gemma3_logger
except:
    pass

# Gemma4 It is strongly recommended to train Gemma4 models with the `eager`
try:
    from transformers.models.gemma4.modeling_gemma4 import logger as gemma4_logger
    gemma4_logger.addFilter(HideLoggingMessage("strongly recommended"))
    del gemma4_logger
except:
    pass

# Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed.
try:
    from huggingface_hub.file_download import logger as hub_logger
    hub_logger.addFilter(HideLoggingMessage("hf_xet"))
    del hub_logger
except:
    pass

# MXFP4 quantization requires triton >= 3.4.0
try:
    from transformers.quantizers.quantizer_mxfp4 import logger as mxfp4_logger
    mxfp4_logger.addFilter(HideLoggingMessage("requires triton"))
    del mxfp4_logger
except:
    pass

# You passed `quantization_config` or equivalent parameters
try:
    warnings.filterwarnings(
        action = "ignore",
        message = r".*quantization_config.*",
        category = UserWarning,
        append = True,
    )
except:
    pass

# UserWarning: Logical operators 'and' and 'or' are deprecated for non-scalar tensors; please use '&' or '|' instead
# Will be fixed in torch 2.8.1 https://github.com/pytorch/pytorch/issues/158463
try:
    warnings.filterwarnings(
        action = "ignore",
        message = r".*Logical operators 'and' and 'or'.*",
        category = UserWarning,
        append = True,
    )
except:
    pass

# Using a slow image processor as `use_fast`
try:
    from transformers.processing_utils import logger as processing_utils_logger
    processing_utils_logger.addFilter(HideLoggingMessage("`use_fast`"))
    del processing_utils_logger
except:
    pass

# Using a slow image processor as `use_fast`
try:
    from transformers.models.auto.image_processing_auto import (
        logger as processing_utils_logger,
    )
    processing_utils_logger.addFilter(HideLoggingMessage("`use_fast`"))
    del processing_utils_logger
except:
    pass

# `use_cache=True` is incompatible with gradient checkpointing
try:
    from transformers.trainer import logger as trainer_logger
    trainer_logger.addFilter(HideLoggingMessage("`use_cache=True`"))
    del trainer_logger
except:
    pass

# `use_cache=True` is incompatible with gradient checkpointing
try:
    from transformers.utils.generic import logger as trainer_logger
    trainer_logger.addFilter(HideLoggingMessage("`use_cache=True`"))
    del trainer_logger
except:
    pass

# We detected that you are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')
try:
    from transformers.modeling_utils import logger as modeling_utils_logger
    modeling_utils_logger.addFilter(HideLoggingMessage("anti-pattern"))
    del modeling_utils_logger
except:
    pass

# Errors out on
# Some weights of Gemma3nForConditionalGeneration were not initialized from the model checkpoint
from transformers.modeling_utils import logger as transformers_logger


# NVIDIA DGX Spark (GB10) / N1X "RTX Spark" unified-memory support.
# Names vary ("NVIDIA GB10", "JMJWOA-Generic-GPU" on N1X); the aarch64 + CUDA
# gate keeps every Spark workaround a no-op elsewhere.
_DGX_SPARK_DEVICE_TOKENS = ("GB10", "JMJWOA", "N1X", "DGX SPARK", "GB110")


def _name_has_spark_token(names_upper):
    # Whole-token match so "GB10" doesn't match discrete "GB100"/"GB10X".
    import re
    return any(
        re.search(r"(?<![A-Z0-9])" + re.escape(tok) + r"(?![A-Z0-9])", names_upper)
        for tok in _DGX_SPARK_DEVICE_TOKENS
    )


@functools.lru_cache(maxsize = None)
def is_dgx_spark():
    """True only on DGX Spark / N1X (gate: aarch64 + CUDA + device-name token).
    UNSLOTH_FORCE_DGX_SPARK=1/0 forces on/off."""
    _force = os.environ.get("UNSLOTH_FORCE_DGX_SPARK")
    if _force == "1":
        return True
    if _force == "0":
        return False
    try:
        import platform

        if platform.machine().lower() not in ("aarch64", "arm64"):
            return False
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return False
        names = " ".join(
            str(torch.cuda.get_device_name(i)).upper() for i in range(torch.cuda.device_count())
        )
        return _name_has_spark_token(names)
    except Exception:
        return False


@functools.lru_cache(maxsize = None)
def _is_dgx_spark_no_cuda_init():
    """Spark detection that never inits CUDA: reads names via `nvidia-smi`, not
    torch, so PYTORCH_CUDA_ALLOC_CONF can still be set afterwards. Same
    UNSLOTH_FORCE_DGX_SPARK override; False on any error."""
    _force = os.environ.get("UNSLOTH_FORCE_DGX_SPARK")
    if _force == "1":
        return True
    if _force == "0":
        return False
    try:
        import platform

        if platform.machine().lower() not in ("aarch64", "arm64"):
            return False
        import subprocess
        import shutil

        # The WoA shim execs the venv binary directly (no login shell), where
        # /usr/lib/wsl/lib can be off PATH -- resolve WSL's nvidia-smi explicitly.
        _smi = "nvidia-smi"
        if shutil.which(_smi) is None and os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
            _smi = "/usr/lib/wsl/lib/nvidia-smi"
        out = subprocess.run(
            [_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            timeout = 5,
        )
        names = (out.stdout or "").upper()
        return _name_has_spark_token(names)
    except Exception:
        return False


def patch_dgx_spark_caching_allocator_warmup():
    """No-op `caching_allocator_warmup` on Spark UMA.

    `cudaMemGetInfo()` undercounts free UMA memory, so HF's warmup
    `torch.empty(...)` raises `AcceleratorError: invalid argument`, aborting
    bitsandbytes 4/8-bit loads. The warmup is only a speed hint, so skip it.
    Gated by `is_dgx_spark()`; idempotent via `_unsloth_spark_noop` marker.
    """
    if not is_dgx_spark():
        return
    try:
        from transformers import modeling_utils as _mu
    except Exception:
        return
    if not hasattr(_mu, "caching_allocator_warmup"):
        return
    if getattr(_mu.caching_allocator_warmup, "_unsloth_spark_noop", False):
        return

    def _noop(*args, **kwargs):
        return None

    _noop._unsloth_spark_noop = True
    _mu.caching_allocator_warmup = _noop


def patch_dgx_spark_memory_config():
    """Enable allocator `expandable_segments` on Spark UMA to cut fragmentation
    OOMs (no-op off-Spark).

    Appends to PYTORCH_CUDA_ALLOC_CONF only when absent; opt out with
    UNSLOTH_NO_EXPANDABLE_SEGMENTS=1. Must run before the first CUDA allocation,
    hence the CUDA-free `_is_dgx_spark_no_cuda_init()` gate -- `is_dgx_spark()`
    would init the allocator before the env var could take effect.
    """
    if not _is_dgx_spark_no_cuda_init():
        return
    if os.environ.get("UNSLOTH_NO_EXPANDABLE_SEGMENTS") == "1":
        return
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in conf:
        return  # respect user's setting
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        conf + "," if conf else ""
    ) + "expandable_segments:True"


def patch_dgx_spark_runtime_defaults():
    """Spark UMA runtime defaults (no-op off-Spark; env-overridable).

    - UNSLOTH_DISABLE_DOUBLE_BUFFER=1 (setdefault): zoo's grad-checkpointing
      double-buffer gates on a mem_get_info check that UNDERCOUNTS on UMA, and
      its staging buffer is pure waste on a shared pool.
    - UNSLOTH_SPARK_MEM_FRACTION=<0..1> (opt-in, default NO cap): caps the
      allocator so over-allocation raises OutOfMemoryError early instead of
      wedging the box (untracked UMA allocations may never trip a catchable OOM).
    """
    if not is_dgx_spark():
        return
    os.environ.setdefault("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    _frac = os.environ.get("UNSLOTH_SPARK_MEM_FRACTION")
    if _frac:
        try:
            # Out-of-range = no cap (0 OOMs everything; torch rejects > 1).
            _frac_val = float(_frac)
            if 0.0 < _frac_val <= 1.0:
                torch.cuda.set_per_process_memory_fraction(_frac_val)
        except Exception:
            pass


def patch_dgx_spark_dataloader_defaults():
    """Default `dataloader_pin_memory` to False on Spark UMA.

    On one shared pool, pinning only reserves non-pageable RAM and adds a staging
    copy (mirrors transformers' use_cpu precedent). Wrapping the base
    `TrainingArguments.__post_init__` covers SFT + every TRL trainer in one
    idempotent patch. Opt out: UNSLOTH_SPARK_KEEP_PIN_MEMORY=1. No-op off-Spark.
    """
    if not is_dgx_spark():
        return
    if os.environ.get("UNSLOTH_SPARK_KEEP_PIN_MEMORY") == "1":
        return
    try:
        from transformers import training_args as _ta
        Base = _ta.TrainingArguments
    except Exception:
        return
    if getattr(Base.__post_init__, "_unsloth_spark_uma", False):
        return
    _orig_post_init = Base.__post_init__

    # *args/**kwargs: tolerate future InitVar params in __post_init__.
    def __post_init__(self, *args, **kwargs):
        _orig_post_init(self, *args, **kwargs)
        if getattr(self, "dataloader_pin_memory", None) is True:
            self.dataloader_pin_memory = False

    __post_init__._unsloth_spark_uma = True
    Base.__post_init__ = __post_init__


patch_dgx_spark_memory_config()
patch_dgx_spark_caching_allocator_warmup()
patch_dgx_spark_runtime_defaults()
patch_dgx_spark_dataloader_defaults()


def _all_missing_keys_are_position_ids(record_str):
    """True only when EVERY key in the 'newly initialized: [...]' list is a position_ids
    buffer.

    transformers reports all missing keys in a single record, so a substring test would
    wrongly suppress the warning when a real missing weight is listed alongside a benign
    position_ids buffer. position_ids is a deterministic arange buffer that transformers
    itself lists in _keys_to_ignore_on_load_missing (some VLMs, e.g. DeepSeek-OCR, ship it
    non-persistently), so a record listing ONLY position_ids keys is safe to ignore;
    anything else must still raise.
    """
    import ast
    import re

    match = re.search(r"newly initialized:\s*(\[[^\]]*\])", record_str)
    if not match:
        return False
    try:
        keys = ast.literal_eval(match.group(1))
    except Exception:
        return False
    return bool(keys) and all("position_ids" in str(key) for key in keys)


class _RaiseUninitialized(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        record_str = str(record)
        record_lower = record_str.lower()
        if (
            ("some weights of" in record_lower)
            and ("score.weight" not in record_lower)
            and ("classifier.weight" not in record_lower)
            and ("cls.predictions" not in record_lower)
            and ("predictions.decoder" not in record_lower)
            and not _all_missing_keys_are_position_ids(record_str)
            and (os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1") == "1")
        ):
            raise Exception(
                f"Unsloth: Critical error since some weights are not initialized.\n"
                f"Please try updating Unsloth, transformers and timm via:\n"
                f"`pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo transformers timm`\n"
                f"{str(record)}"
            )


class RaiseUninitialized:
    def __init__(self):
        self.error_handler = _RaiseUninitialized()
        transformers_logger.addHandler(self.error_handler)

    def remove(self):
        transformers_logger.removeHandler(self.error_handler)


try:
    from transformers.trainer import logger as transformers_trainer_logger
    transformers_trainer_logger.addFilter(
        HideLoggingMessage("The model is already on multiple devices.")
    )
except:
    pass

# Hide HF Hub unauthenticated request warnings
try:
    from huggingface_hub.utils._http import logger as hf_http_logger
    hf_http_logger.addFilter(HideLoggingMessage("You are sending unauthenticated requests"))
    del hf_http_logger
except:
    pass

# Replace PEFT target_parameters warning with Unsloth branded message for MoE models
ReplaceWarningMessage.add_rule(
    match_text = "target_parameters",
    replacement = (
        "Unsloth: PEFT set target_parameters but found no matching parameters.\n"
        "This is expected for MoE models - Unsloth handles MoE expert LoRA targeting separately."
    ),
    category = RuntimeWarning,
)

# Patch get_model_param_count to record correct 4bit / 8bit
from transformers.trainer_pt_utils import is_deepspeed_zero3_enabled


def extract_quant_model_param_count(model):
    """
    Calculate quant model param count based on difference in param class. Returns int for param count.
    """
    count: int = 0
    for name, p in model.named_parameters():
        if p.__class__.__name__ == "Params4bit":
            count += 2 * p.numel()
        else:
            count += p.numel()
    return count


def get_model_param_count(model, trainable_only = False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled():

        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()
    else:

        def numel(p):
            return p.numel()

    s = sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)
    if (
        (not trainable_only)
        and hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
    ):
        approx = extract_quant_model_param_count(model)
        if approx is not None:
            s = approx
    return s


import transformers.trainer_pt_utils

transformers.trainer_pt_utils.get_model_param_count = get_model_param_count
import transformers.trainer

transformers.trainer.get_model_param_count = get_model_param_count
# =============================================

# =============================================
# Edits all Config files to enable RoPE Scaling for all models


# Transformers had to update for Mistral Nemo 12b since Attention is (5120, 4096) now.
def patch_mistral_nemo_config(config):
    if "head_dim (" not in config:
        add_head_dim = (
            "If it is not specified, will default to `8`.\n"
            "        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):\n"
            "            The attention head dimension."
        )
        config = config.replace("If it is not specified, will default to `8`.", add_head_dim)

        add_head_dim = "num_key_value_heads=8,\n        head_dim=None,"
        config = config.replace("num_key_value_heads=8,", add_head_dim)

        add_head_dim = "self.sliding_window = sliding_window\n        self.head_dim = head_dim or hidden_size // num_attention_heads\n"
        config = config.replace("self.sliding_window = sliding_window", add_head_dim)
    return config


try:
    # Some Config files use layer_type_validation
    # for eg Gemma-2, so we must import it to stop errors.
    from transformers.configuration_utils import layer_type_validation
except:
    pass

try:
    # Transformers 5.0+ uses RotaryEmbeddingConfigMixin as a base class for configs
    from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
except:
    pass
from transformers import __version__ as transformers_version

try:
    from transformers import PreTrainedConfig
except:
    from transformers import PretrainedConfig

model_architectures = [
    "llama",
    "mistral",
    "gemma",
    "gemma2",
    "qwen2",
    "granite",
    "qwen3",
    "qwen3_moe",
    "falcon_h1",
]

# Transformers 5.x uses class-level annotations with @strict, @auto_docstring,
# and interval() in config classes. exec(inspect.getsource(...)) fails because
# those symbols are not in scope. Skip the exec-based config patching for 5.x
# since those configs already use rope_parameters (the v5 replacement for
# rope_scaling).
_skip_config_exec_patch = Version(transformers_version) >= Version("5.0.0")

for model_name in model_architectures:
    if _skip_config_exec_patch:
        break
    config_filepath = f"transformers.models.{model_name}.configuration_{model_name}"
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    config_filename = f"{model_name.title().replace('_','')}Config"  # qwen3 arch folder is qwen3_moe but config is Qwen3Config. Need to remove underscore(_) for now
    try:
        exec(f"from {config_filepath} import {config_filename}", globals())
    except:
        continue

    try:
        config = inspect.getsource(eval(config_filename))
    except:
        continue
    if "RopeParameters" in config:
        try:
            exec(f"from {config_filepath} import RopeParameters", globals())
        except:
            continue

    if "rope_scaling" in config:
        continue
    config = re.sub(
        r"(\*\*kwargs)[\s]{0,}\,[\s]{0,}\)[\s]{0,}\:",
        r"rope_scaling=None,"
        r"\n        **kwargs):\n"
        r"\n        self.rope_scaling = rope_scaling\n",
        config,
    )

    # Just for Mistral Nemo
    if model_name == "mistral":
        if Version(transformers_version) <= Version("4.42.4"):
            config = patch_mistral_nemo_config(config)

    try:
        exec(config, globals())
        exec(f"import {config_filepath}", globals())
        exec(f"{config_filepath}.{config_filename} = {config_filename}", globals())
    except Exception:
        continue
# =============================================

# =============================================
# torch.cuda.amp.custom_fwd is deprecated >= 2.4
torch_version = torch.__version__
if DEVICE_TYPE in ("cuda", "hip"):
    if Version(torch_version) < Version("2.4.0"):
        torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
        torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
    else:
        torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
        torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
elif DEVICE_TYPE == "xpu":
    if Version(torch_version) < Version("2.6.0"):
        raise RuntimeError("torch.xpu currently only supports torch.version >= 2.6.0")
    else:
        torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "xpu")
        torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "xpu")
# =============================================

# =============================================
# Fix KeyError: 'Cache only has 0 layers, attempted to access layer with index 0'
# import transformers.cache_utils
# if hasattr(transformers.cache_utils, "DynamicCache") and \
#     transformers.cache_utils.DynamicCache.__getitem__.__name__ != "__cache_utils_getitem__":

#     source = inspect.getsource(transformers.cache_utils.DynamicCache.__getitem__)
#     start = source.find("def")
#     spaces = start*" "
#     source = source.split("\n")
#     source = "\n".join(x[start:] for x in source)
#     where = source.find("raise KeyError")
#     source = source[:where] + \
#         f"if len(self) == 0:\n{spaces}{spaces}"\
#         "    raise RuntimeError('Unsloth: You must call `FastLanguageModel.for_inference(model)` before doing inference for Unsloth models.')\n" + \
#         f"{spaces}{spaces}else:\n{spaces}{spaces}{spaces}" + source[where:]
#     source = source.replace("__getitem__", "__cache_utils_getitem__", 1)
#     exec(source)
#     transformers.cache_utils.DynamicCache.__getitem__ = __cache_utils_getitem__
# pass
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

        def _is_openai_available():
            return False

        transformers.utils.is_openai_available = _is_openai_available

# =============================================
# Get Flash Attention v2 if Ampere (RTX 30xx, A100)
from transformers import AutoTokenizer
from transformers.utils.import_utils import _is_package_available


def _package_available(pkg_name: str) -> bool:
    # transformers >= 5.x makes `_is_package_available` always return a
    # `(exists, version)` tuple, which is truthy even when the package is
    # absent; older versions returned a plain bool. Normalise to a bool so
    # callers don't take "package present" branches for missing packages.
    result = _is_package_available(pkg_name)
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


SUPPORTS_BFLOAT16 = False
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False

if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    torch.cuda.get_device_capability = functools.cache(torch.cuda.get_device_capability)

    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
        if _package_available("flash_attn"):
            # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
            try:
                try:
                    # See https://github.com/unslothai/unsloth/issues/1437
                    from flash_attn.flash_attn_interface import flash_attn_gpu
                except:
                    from flash_attn.flash_attn_interface import flash_attn_cuda
                HAS_FLASH_ATTENTION = True

                # Also check for softcapping
                from flash_attn import __version__ as flash_attn_version

                HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
                if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                    print(
                        "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"
                        "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"
                        "To update flash-attn, do the below:\n"
                        '\npip install --no-deps --no-build-isolation --upgrade "flash-attn>=2.6.3"'
                    )
            except:
                print(
                    "Unsloth: Your Flash Attention 2 installation seems to be broken. "
                    "Using Xformers instead. No performance changes will be seen."
                )

                # Stop Flash Attention from importing!
                import transformers.utils.import_utils

                transformers.utils.import_utils.is_flash_attn_2_available = (
                    lambda *args, **kwargs: False
                )
                import transformers.utils

                transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

                HAS_FLASH_ATTENTION = False
        else:
            HAS_FLASH_ATTENTION = False
    else:
        # Tri Dao's benchmark shows xformers is faster for now.
        HAS_FLASH_ATTENTION = False
elif DEVICE_TYPE == "hip":
    SUPPORTS_BFLOAT16 = True
    if _package_available("flash_attn"):
        # Check for CUDA linking errors "undefined symbol: _ZNK3c106SymIntltEl"
        try:
            try:
                # See https://github.com/unslothai/unsloth/issues/1437
                from flash_attn.flash_attn_interface import flash_attn_gpu
            except:
                from flash_attn.flash_attn_interface import flash_attn_cuda
            HAS_FLASH_ATTENTION = True

            # Also check for softcapping
            from flash_attn import __version__ as flash_attn_version

            HAS_FLASH_ATTENTION_SOFTCAPPING = Version(flash_attn_version) >= Version("2.6.3")
            if not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"
                    "To update flash-attn, do the below:\n"
                    '\npip install --no-deps --no-build-isolation --upgrade "flash-attn>=2.6.3"'
                )
        except:
            print(
                "Unsloth: Your Flash Attention 2 installation seems to be broken. "
                "Using Xformers instead. No performance changes will be seen."
            )

            # Stop Flash Attention from importing!
            import transformers.utils.import_utils

            transformers.utils.import_utils.is_flash_attn_2_available = (
                lambda *args, **kwargs: False
            )
            import transformers.utils

            transformers.utils.is_flash_attn_2_available = lambda *args, **kwargs: False

            HAS_FLASH_ATTENTION = False
elif DEVICE_TYPE == "xpu":
    SUPPORTS_BFLOAT16 = True

# =============================================
# Get Xformers
# Silence xformers CUDA mismatch warnings before import
try:
    _xformers_logger = logging.getLogger("xformers")
    _xformers_logger.setLevel(logging.ERROR)
    del _xformers_logger
except:
    pass
try:
    from xformers import __version__ as xformers_version

    # Xformers <= 0.0.32.post2 has a broken FA3 dispatch on Blackwell/RTX 50x GPUs.
    # The FA3 check used `capability >= (9, 0)` which matches SM 10.0/11.0/12.0,
    # causing sm_90a kernels to be attempted on non-Hopper GPUs (CUDA error in
    # flash_fwd_launch_template.h:188). Fixed in 0.0.33 with `<= (9, 0)`.
    # See https://github.com/facebookresearch/xformers/issues/1329
    if DEVICE_TYPE == "cuda":
        major_version, minor_version = torch.cuda.get_device_capability()
        if (f"{major_version}.{minor_version}" in ("10.0", "11.0", "12.0")) and (
            Version(xformers_version) <= Version("0.0.32.post2")
        ):
            raise NotImplementedError(
                f"Unsloth: Xformers {xformers_version} has a broken FA3 dispatch on "
                f"SM {major_version}.{minor_version} GPUs. Please upgrade to >= 0.0.33 or build from source via\n"
                "```\n"
                "pip install ninja\n"
                "pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers\n"
                "```\n"
            )

    # Temporarily disable 0.0.27 and higher - inference issues
    if False:  # Version(xformers_version) >= Version("0.0.27"):
        raise ImportError(
            "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "
            "then press Disconnect Runtime and then Restart it.\n"
            "\n"
            "%%capture\n"
            "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
            '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
            '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'
            "\n"
            f"Otherwise in local machines, your xformers version of {xformers_version} is too new.\n"
            'Please downgrade xformers via `pip install --force-reinstall "xformers<=0.0.27"'
        )

    if Version(torch_version) < Version("2.2.0") and Version(xformers_version) >= Version("0.0.24"):
        raise ImportError(
            f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"
            f"Please install xformers < 0.0.24 for torch = {torch_version}."
        )
    elif Version(torch_version) < Version("2.3.0") and Version(xformers_version) >= Version(
        "0.0.26"
    ):
        raise ImportError(
            f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"
            f"Please install xformers < 0.0.26 for torch = {torch_version}."
        )
    elif Version(torch_version) < Version("2.4.0") and Version(xformers_version) > Version(
        "0.0.27"
    ):
        raise ImportError(
            f"Unsloth: You have torch = {torch_version} but xformers = {xformers_version}.\n"
            f"Please install xformers <= 0.0.27 for torch = {torch_version}."
        )

    from xformers._cpp_lib import _register_extensions

    try:
        _register_extensions()  # Check if C++ modules are loaded correctly
    except Exception as error:
        raise ImportError(
            "Unsloth: Xformers was not installed correctly.\n"
            "Please install xformers separately first.\n"
            "Then confirm if it's correctly installed by running:\n"
            "python -m xformers.info\n\n"
            "Longer error message:\n" + str(error)
        )
    import xformers.ops.fmha as xformers

    xformers_attention = xformers.memory_efficient_attention
except ModuleNotFoundError:
    xformers = None
    xformers_attention = None
    xformers_version = None
except Exception as e:
    if UNSLOTH_ENABLE_LOGGING:
        print("========\nSwitching to PyTorch attention since your Xformers is broken.\n========\n")
        print(str(e))
    xformers = None
    xformers_attention = None
    xformers_version = None

# Check TRL version
from trl import __version__ as trl_version

# Unsloth now supports all TRL versions!
if False:  # Version(trl_version) >= Version("0.9.0"):
    raise ImportError(
        "Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below "
        "then press Disconnect Runtime and then Restart it.\n"
        "\n"
        "%%capture\n"
        "# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n"
        '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        '!pip install --no-deps "xformers<=0.0.27" trl peft accelerate bitsandbytes\n'
        "\n"
        f"Otherwise in local machines, your TRL version of {trl_version} is too new.\n"
        "Please downgrade TRL via `pip install --force-reinstall trl"
    )

# =============================================
# Fix new Xformers versions TypeError: Multiple dispatch failed for 'torch._ops.aten.to.dtype_layout'
# accelerate_old_send_to_device = None
# accelerate_new_send_to_device = None
# if xformers_version is not None and Version(xformers_version) >= Version("0.0.27"):
#     import accelerate.utils.operations
#     if hasattr(accelerate.utils.operations, "send_to_device") and \
#         accelerate.utils.operations.send_to_device.__name__ != "_fixed_send_to_device":
#         accelerate_old_send_to_device = accelerate.utils.operations.send_to_device
#         from accelerate.utils.operations import *
#         send_to_device = inspect.getsource(accelerate.utils.operations.send_to_device)
#         send_to_device = re.sub(
#             r"([ ]{4,})return tensor\.to\(device\)",
#             r"\1try: return tensor.to(device)\n\1except: return tensor",
#             send_to_device,
#         ).replace("def send_to_device", "def _fixed_send_to_device")
#         exec(send_to_device)
#         # accelerate.utils.operations.send_to_device = _fixed_send_to_device
#         accelerate_new_send_to_device = _fixed_send_to_device
#     pass
# pass

# Transformers 4.46 breaks dynamic caching. This is a hack
import transformers.generation.configuration_utils

if hasattr(transformers.generation.configuration_utils, "ALL_CACHE_IMPLEMENTATIONS"):
    if type(transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS) is list:
        if "dynamic" not in transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS:
            transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS.append("dynamic")
# =============================================

# =============================================
# Torch compile settings
UNSLOTH_COMPILE_DEBUG = os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1"
UNSLOTH_COMPILE_MAXIMUM = os.environ.get("UNSLOTH_COMPILE_MAXIMUM", "0") == "1"
UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "1") == "1"
# Just remove max_autotune_gemm warning
from torch._inductor.runtime.hints import DeviceProperties


@functools.lru_cache(None)
def is_big_gpu(index) -> bool:
    if DEVICE_TYPE == "xpu":
        prop = DeviceProperties.create(torch.device("xpu", index) if type(index) is int else index)
        min_sms = 16
    else:
        prop = DeviceProperties.create(torch.device("cuda", index) if type(index) is int else index)
        min_sms = 80

    avail_sms = prop.multi_processor_count
    if avail_sms < min_sms:
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
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": UNSLOTH_COMPILE_DEBUG,
    "triton.cudagraphs": False,
}
# Spark's 48 SMs are under inductor's 68-SM is_big_gpu bar; max_autotune just wastes search time.
if is_dgx_spark():
    torch_compile_options["max_autotune"] = False

import accelerate


def torch_compile_kwargs(*args, **kwargs):
    print("Unsloth: Enabled auto compiling")
    return {
        "dynamic": True,
        "fullgraph": False,
        "options": torch_compile_options,
    }


accelerate.utils.dataclasses.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
accelerate.utils.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
accelerate.accelerator.TorchDynamoPlugin.to_kwargs = torch_compile_kwargs
del accelerate


def patch_regional_compilation():
    # Regional torch 2.5 Recompilation - weirdly very slow??
    if torch.nn.ModuleList.__name__ == "UnslothModuleList":
        return
    # Only works for torch 2.5
    if Version(torch.__version__) < Version("2.5.0"):
        return

    old_module_list = torch.nn.ModuleList
    os.environ["UNSLOTH_PATCHED"] = "1"

    def UnslothModuleList(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) is list:
            args = [
                old_module_list(
                    [
                        torch.compile(
                            x,
                            dynamic = True,
                            options = torch_compile_options,
                            fullgraph = False,
                        )
                        for x in args[0]
                    ]
                )
            ]
        return old_module_list(*args, **kwargs)

    UnslothModuleList.__doc__ = old_module_list.__doc__

    torch.nn.ModuleList = UnslothModuleList
    return


# =============================================


def prepare_model_for_kbit_training(
    model: Any,
    use_gradient_checkpointing: Optional = True,
    use_reentrant: Optional[bool] = True,
) -> Any:
    return prepare_model_for_training(
        model = model,
        use_gradient_checkpointing = use_gradient_checkpointing,
        use_reentrant = use_reentrant,
        full_finetuning = False,
        train_layernorms = False,
        train_embedding = False,
        train_lm_head = False,
        float32_mixed_precision = True,
    )


# =============================================
# Weirdly LoraLayer.update_layer downcasts PEFT layers to float16??
# For mixed precision, we need it to be in float32 not float16.
from peft import __version__ as peft_version
from peft.utils.integrations import dequantize_module_weight

if Version(peft_version) < Version("0.12.0"):
    from peft.tuners.lora.layer import LoraLayer
    try:
        source = inspect.getsource(LoraLayer.update_layer)
        text = "if weight is not None:\n"
        start = source.find(text) + len(text)
        end = source.find("self.to(weight.device)", start)
        spaces = re.findall(r"^([ ]{1,})break", source, flags = re.MULTILINE)[0]
        source = source.replace(source[start:end], spaces)
        spaces = len(re.match(r"[\s]{1,}", source).group(0))
        lines = source.split("\n")
        source = "\n".join(x[spaces:] for x in lines)
        source = re.sub(r"([^\.])nn\.", r"\1torch.nn.", source)
        source = source.replace("def update_layer", "def LoraLayer_update_layer")
        exec(source, globals())

        # Fix up incorrect downcasting of LoRA weights
        from peft.tuners.lora.layer import LoraLayer

        LoraLayer.update_layer = LoraLayer_update_layer
        from peft.tuners.lora import LoraLayer

        LoraLayer.update_layer = LoraLayer_update_layer
    except:
        logger.warning_once(
            "Unsloth unsuccessfully patched LoraLayer.update_layer. Please file a bug report.\n"
            "Luckily, your training run will still work in the meantime!"
        )

# =============================================
import importlib

global USE_MODELSCOPE
USE_MODELSCOPE = os.environ.get("UNSLOTH_USE_MODELSCOPE", "0") == "1"
if USE_MODELSCOPE:
    if importlib.util.find_spec("modelscope") is None:
        raise ImportError(
            f"You are using the modelscope hub, please install modelscope by `pip install modelscope -U`"
        )

import socket


@functools.lru_cache(1)
def has_internet(
    host = "8.8.8.8",
    port = 53,
    timeout = 3,
):
    if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
        return False

    OFFLINE_TRUE = {"1", "true", "yes", "on"}

    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in OFFLINE_TRUE:
        return False
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            return True
        finally:
            sock.close()
    except socket.error as ex:
        return False


import psutil


def _get_statistics(statistics = None, force_download = True):
    # We log some basic stats about which environment is being used.
    # We simply download a README.md file from HF - all data is made public.
    # This is simply so we can check if some envs are broken or not.
    # You can disable this by commenting the below out
    n_cpus = psutil.cpu_count(logical = False)
    keynames = "\n" + "\n".join(os.environ.keys())
    # Check modelscope for down detection
    global USE_MODELSCOPE
    USE_MODELSCOPE = os.environ.get("UNSLOTH_USE_MODELSCOPE", "0") == "1"

    if statistics is None:
        # Prefer filesystem markers (harder to misidentify) before env-key matching
        try:
            from pathlib import Path
            if Path("/kaggle/working").exists():
                statistics = "kaggle"
            elif Path("/content").exists() and Path("/opt/colab").exists():
                statistics = "colab" if n_cpus == 1 else "colabpro"
            elif Path("/runpod-volume").exists():
                statistics = "runpod"
        except Exception:
            pass

        # Fallback to env-key detection
        if statistics is None:
            if "\nKAGGLE_" in keynames:
                statistics = "kaggle"
            elif "\nCOLAB_" in keynames and n_cpus == 1:
                statistics = "colab"
            elif "\nCOLAB_" in keynames:
                statistics = "colabpro"
            elif "\nRUNPOD_" in keynames:
                statistics = "runpod"
            elif "\nAWS_" in keynames:
                statistics = "aws"
            elif "\nAZURE_" in keynames:
                statistics = "azure"
            # elif "\nK_" in keynames or "\nFUNCTION_" in keynames: statistics = "gcp"
            elif "\nINVOCATION_ID" in keynames:
                statistics = "lambda"
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

                    for vendor_file in vendor_files:
                        path = Path(vendor_file)
                        if path.is_file():
                            file_content = path.read_text().lower()
                            if "amazon" in file_content:
                                return "aws"
                            elif "microsoft corporation" in file_content:
                                return "azure"
                            elif "google" in file_content:
                                return "gcp"
                    return "other"

                try:
                    statistics = try_vllm_check()
                except Exception:
                    statistics = "other"

    if statistics is not None:
        import tempfile
        from huggingface_hub import snapshot_download
        from unsloth_zoo.rl_environments import execute_with_time_limit

        if has_internet():

            def stats_check():
                with tempfile.TemporaryDirectory(ignore_cleanup_errors = True) as f:
                    snapshot_download(
                        f"unslothai/{statistics}",
                        force_download = True,
                        cache_dir = f,
                        local_dir = f,
                    )

            time_limited_stats_check = execute_with_time_limit(120)(stats_check)
            try:
                time_limited_stats_check()
            except TimeoutError:
                raise TimeoutError(
                    "Unsloth: HuggingFace seems to be down after trying for 120 seconds :(\n"
                    "Check https://status.huggingface.co/ for more details.\n"
                    "As a temporary measure, use modelscope with the same model name ie:\n"
                    "```\n"
                    "pip install modelscope\n"
                    "import os; os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'\n"
                    "from unsloth import FastLanguageModel\n"
                    "model = FastLanguageModel.from_pretrained('unsloth/gpt-oss-20b')\n"
                    "```"
                )
            except Exception:
                logger.debug("Unsloth: stats_check failed with an exception.")
                # Don't retry without a time limit — would freeze offline


def get_statistics(local_files_only = False):
    # We log some basic stats about which environment is being used.
    # This is also to check if HuggingFace is down or not!
    # We simply download a README.md file from HF - all data is made public.
    # This is simply so we can check if some envs are broken or not.
    # You can disable this by setting UNSLOTH_DISABLE_STATISTICS
    import os

    if (
        "UNSLOTH_DISABLE_STATISTICS" in os.environ
        or os.environ.get("UNSLOTH_USE_MODELSCOPE", "0") == "1"
    ):
        return
    if local_files_only:
        return
    # Also skip when HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set.
    _offline_vals = {"1", "true", "yes", "on"}
    if (
        os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _offline_vals
        or os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _offline_vals
    ):
        return
    from huggingface_hub.utils import (
        disable_progress_bars,
        enable_progress_bars,
        are_progress_bars_disabled,
    )

    disabled = False
    if not are_progress_bars_disabled():
        disable_progress_bars()
        disabled = True
    _get_statistics(None)
    _get_statistics("repeat", force_download = False)
    total_memory = (
        torch.xpu.get_device_properties(0).total_memory
        if DEVICE_TYPE == "xpu"
        else torch.cuda.get_device_properties(0).total_memory
    )
    vram = total_memory / 1024 / 1024 / 1024
    if vram <= 8:
        vram = 8
    elif vram <= 16:
        vram = 16
    elif vram <= 20:
        vram = 20
    elif vram <= 24:
        vram = 24
    elif vram <= 40:
        vram = 40
    elif vram <= 48:
        vram = 48
    elif vram <= 80:
        vram = 80
    else:
        vram = 96
    _get_statistics(f"vram-{vram}")
    _get_statistics(f"{DEVICE_COUNT if DEVICE_COUNT <= 8 else 9}")
    if disabled:
        enable_progress_bars()


# =============================================
# Fixes Bitsandbytes to remove missing warnings
from transformers.utils.quantization_config import (
    BitsAndBytesConfig,
    QuantizationMethod,
)

BitsAndBytesConfig__init__ = inspect.getsource(BitsAndBytesConfig.__init__)
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
exec(BitsAndBytesConfig__init__, globals())

if DEVICE_COUNT == 1 and int(os.environ.get("WORLD_SIZE", "1")) <= 1:
    from accelerate.utils.dataclasses import DistributedType

    def _prepare_backend(self, *args, **kwargs):
        return None, DistributedType.NO

    import accelerate.state

    accelerate.state.PartialState._prepare_backend = _prepare_backend
    accelerate.accelerator.Accelerator.distributed_type = lambda *args, **kwargs: DistributedType.NO


# to move multiple tensors to the same device
def move_to_device(target_device, *tensors):
    """Move tensors to target_device (returns same objects if already there)."""
    if isinstance(target_device, int):
        target_device = torch.device(target_device)
    elif isinstance(target_device, str):
        # if string we expect it to be a device name like "cuda:0"
        target_device = torch.device(target_device)
    elif isinstance(target_device, torch.device):
        pass
    else:
        raise ValueError(f"Invalid target device: {target_device}")
    moved_tensors = []
    for tensor in tensors:
        if tensor.device != target_device:
            moved_tensors.append(tensor.to(target_device))
        else:
            moved_tensors.append(tensor)
    return tuple(moved_tensors) if len(moved_tensors) > 1 else moved_tensors[0]


import transformers.utils.quantization_config

transformers.utils.quantization_config.BitsAndBytesConfig.__init__ = _BitsAndBytesConfig__init__
# =============================================

# Offloading to disk for modules (lm_head, embed_tokens)
import pickle


def offload_to_disk(
    W,
    model,
    name,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
):
    file_location = os.path.join(temporary_location, model.config._name_or_path)
    if not os.path.exists(file_location):
        os.makedirs(file_location)

    filename = os.path.join(file_location, f"{name}.pt")
    W = W.weight if hasattr(W, "weight") else W
    torch.save(
        W,
        filename,
        pickle_module = pickle,
        pickle_protocol = pickle.HIGHEST_PROTOCOL,
    )
    # We must use weights_only = False due to pickling
    offloaded_W = torch.load(filename, map_location = "cpu", mmap = True, weights_only = False)
    offloaded_W._offloaded_file_location = filename
    return offloaded_W


def offload_input_embeddings(model, temporary_location: str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(
        model.get_input_embeddings(), model, "input_embeddings", temporary_location
    )
    new_input_embeddings = torch.nn.Embedding.from_pretrained(offloaded_W)
    new_input_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_input_embeddings(new_input_embeddings)
    return


def offload_output_embeddings(model, temporary_location: str = "_unsloth_temporary_saved_buffers"):
    offloaded_W = offload_to_disk(
        model.get_output_embeddings(), model, "output_embeddings", temporary_location
    )

    new_output_embeddings = torch.nn.Linear(1, 1, bias = None)
    del new_output_embeddings.weight
    new_output_embeddings.weight = offloaded_W
    new_output_embeddings.in_features = offloaded_W.shape[1]
    new_output_embeddings.out_features = offloaded_W.shape[0]

    new_output_embeddings._offloaded_file_location = offloaded_W._offloaded_file_location
    model.set_output_embeddings(new_output_embeddings)
    return


# Fixes a weird Torch 2.3 bug which says T4s have bfloat16
def is_bfloat16_supported():
    return SUPPORTS_BFLOAT16


def is_vLLM_available():
    return _package_available("vllm")


# Patches models to add RoPE Scaling
def patch_linear_scaling(
    model_name = "gemma2",
    rope_module = None,
    scaled_rope_module = None,
    attention_module = None,
):
    assert rope_module is not None and scaled_rope_module is not None
    assert attention_module is not None

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = (
        f"import torch.nn as nn\n"
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"
        f"from {model_filepath} import logger, "
        f"{model_name.title()}Attention, {model_name.title()}Config"
    )

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
        rope_function = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
    )
    rotary_emb = re.findall(
        r"self\.rotary\_emb \= .+?\)",
        function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0:
        return None, exec_code + "\n\n" + function

    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function


# Patches for Llama-3 LlamaExtendedRotaryEmbedding
def patch_llama_rope_scaling(
    model_name = "llama",
    rope_module = None,
    scaled_rope_module = None,
    extended_rope_module = None,
    attention_module = None,
    longrope_module = None,
):
    assert (
        rope_module is not None
        and scaled_rope_module is not None
        and extended_rope_module is not None
    )
    assert attention_module is not None

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = (
        f"import torch.nn as nn\n"
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"
        f"from {model_filepath} import logger, "
        f"{model_name.title()}Attention, {model_name.title()}Config"
    )

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
                config=self.config,
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
        rope_function = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
        extended_rope_function = extended_rope_module.__name__,
        longrope_rope_function = (
            longrope_module if longrope_module is not None else rope_module
        ).__name__,
    )
    rotary_emb = re.findall(
        r"self\.rotary\_emb \= .+?\)",
        function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0:
        return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function


def create_boolean_mask(n = 4096, sliding_window = 2048):
    # Creates a boolean mask for attention
    mask = torch.ones(n, n, dtype = torch.bool)
    if sliding_window == 0:
        return torch.triu(mask, diagonal = 1, out = mask)
    torch.triu(mask, diagonal = 0, out = mask)
    torch.triu(mask.T, diagonal = -sliding_window, out = mask.T)
    mask = mask.T
    torch.logical_not(mask, out = mask)
    return mask


def test_mask_creation():
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    for n in range(2, 23):
        for s in range(1, 23):
            correct_mask = (
                AttentionMaskConverter(
                    is_causal = True,
                    sliding_window = s,
                )
                .to_causal_4d(
                    1,
                    n,
                    n,
                    dtype = torch.float16,
                )
                .squeeze(0)
                .squeeze(0)
            )
            correct_mask = correct_mask == correct_mask.min()
            our_mask = create_boolean_mask(n = n, sliding_window = s)
            assert torch.all(correct_mask == our_mask)
        correct_mask = (
            AttentionMaskConverter(
                is_causal = True,
                sliding_window = None,
            )
            .to_causal_4d(
                1,
                n,
                n,
                dtype = torch.float16,
            )
            .squeeze(0)
            .squeeze(0)
        )
        correct_mask = correct_mask == correct_mask.min()
        our_mask = create_boolean_mask(n = n, sliding_window = 0)
        assert torch.all(correct_mask == our_mask)


def _unsloth_pre_compute_loss(self, model, inputs, *args, **kwargs):
    num_items_in_batch = None

    if "num_items_in_batch" in kwargs:
        num_items_in_batch = kwargs["num_items_in_batch"]
        if num_items_in_batch is None:
            # Remove it since the model does not support it!
            kwargs.pop("num_items_in_batch")
        elif "num_items_in_batch" not in inputs:
            inputs["num_items_in_batch"] = num_items_in_batch

    # Get gradient accumulation steps if possible
    if (
        num_items_in_batch is None
        and getattr(getattr(self, "args", self), "gradient_accumulation_steps", 1) != 1
    ):
        inner_model = model
        if hasattr(inner_model, "base_model"):
            inner_model = inner_model.base_model
        if hasattr(inner_model, "model"):
            inner_model = inner_model.model
        name = inner_model.__class__.__name__

        logger.warning_once(
            f"Unsloth: Not an error, but {name} does not accept `num_items_in_batch`.\n"
            "Using gradient accumulation will be very slightly less accurate.\n"
            "Read more on gradient accumulation issues here: https://unsloth.ai/blog/gradient"
        )
    # Gemma3 multimodal models in transformers 5.x require token_type_ids during training.
    # For text-only SFT, token_type_ids should be all zeros (no image tokens).
    if "token_type_ids" not in inputs and "input_ids" in inputs:
        _inner = model
        for _attr in ("base_model", "model", "model"):
            _inner = getattr(_inner, _attr, _inner)
        if getattr(getattr(_inner, "config", None), "model_type", "") in ("gemma3",):
            import sys as _sys

            _mod = _sys.modules.get(type(_inner).__module__)
            _has_ccm = _mod is not None and hasattr(_mod, "create_causal_mask_mapping")
            if _has_ccm and _inner.training:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    # Gemma4 uses mm_token_type_ids (not token_type_ids) for VLM masking
    if "mm_token_type_ids" not in inputs and "input_ids" in inputs:
        _inner = model
        for _attr in ("base_model", "model", "model"):
            _inner = getattr(_inner, _attr, _inner)
        if getattr(getattr(_inner, "config", None), "model_type", "") in ("gemma4",):
            import sys as _sys

            _mod = _sys.modules.get(type(_inner).__module__)
            _has_ccm = _mod is not None and hasattr(_mod, "create_causal_mask_mapping")
            if _has_ccm and _inner.training:
                inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    outputs = self._old_compute_loss(model, inputs, *args, **kwargs)
    return outputs


def patch_gradient_accumulation_fix(Trainer):
    # Fixes gradient accumulation
    # Fixes Output 0 of UnslothFusedLossBackward is a view and is being modified inplace.
    import inspect

    if hasattr(Trainer, "get_batch_samples"):
        if Trainer.get_batch_samples.__name__ == "_unsloth_get_batch_samples":
            return
        if (
            not inspect.getsource(Trainer.get_batch_samples)
            .strip()
            .endswith("return batch_samples, num_items_in_batch")
        ):
            raise NotImplementedError("Unsloth: Please make a Github issue immediately!!")
        else:
            if Trainer.get_batch_samples.__name__ != "_unsloth_get_batch_samples":
                Trainer.get_batch_samples = _unsloth_get_batch_samples

            # Also fix passing in num_items_in_batch
            if not hasattr(Trainer, "_old_compute_loss"):
                # Fix transformers 4.57.0 causing `Output 0 of UnslothFusedLossBackward is a view and is being modified inplace.`
                function = inspect.getsource(Trainer.compute_loss)
                if "loss *=" in function or "loss*=" in function:
                    where = function.find("def")
                    function = function.split("\n")
                    function = "\n".join(x[where:] for x in function)

                    # Import all variables that need importing
                    import transformers.trainer

                    items_in_trainer = dir(transformers.trainer)
                    good_items = []
                    for item in items_in_trainer:
                        if item in function:
                            good_items.append(item)
                    exec(
                        "from transformers.trainer import ("
                        + ", ".join(x for x in good_items)
                        + ")",
                        globals(),
                    )

                    # Replace loss*= with loss = loss *
                    function = re.sub(
                        r"loss[\s]{0,}\*\=",
                        "loss = loss *",
                        function,
                    )
                    exec(function, globals())
                    Trainer.compute_loss = compute_loss
                Trainer._old_compute_loss = Trainer.compute_loss
                Trainer.compute_loss = _unsloth_pre_compute_loss
    else:
        logger.warning_once(
            "Unsloth: We fixed a gradient accumulation bug, "
            "but it seems like you don't have the latest transformers version!\n"
            "Please update transformers, TRL and unsloth via:\n"
            "`pip install --upgrade --no-cache-dir --no-deps unsloth transformers git+https://github.com/huggingface/trl.git`"
        )

    # Also fix up loss scaling ie negate loss *= self.args.gradient_accumulation_steps
    if not (
        Trainer.training_step.__name__ == "_unsloth_training_step"
        or "num_items_in_batch" not in inspect.signature(Trainer.training_step).parameters
    ):
        function = inspect.getsource(Trainer.training_step)
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        # Import all variables that need importing
        import transformers.trainer

        items_in_trainer = dir(transformers.trainer)
        good_items = []
        for item in items_in_trainer:
            if item in function:
                good_items.append(item)
        exec(
            "from transformers.trainer import (" + ", ".join(x for x in good_items) + ")",
            globals(),
        )

        # Accelerate does / self.args.gradient_accumulation_steps internally, so if we already
        # summed it up and did the division before hand, we have to negate it.
        function = function.replace(
            "loss *= self.args.gradient_accumulation_steps",
            "if num_items_in_batch is not None: loss *= self.args.gradient_accumulation_steps",
        )
        function = function.replace("def training_step", "def _unsloth_training_step", 1)

        # Fix 4.47.0 issue where num_items_in_batch was removed
        # See https://github.com/huggingface/transformers/pull/35121
        function = function.replace(
            "if self.model_accepts_loss_kwargs:",
            "if False:",
        )

        # Fix when num_items_in_batch is nothing
        # https://github.com/huggingface/transformers/pull/35207
        function = re.sub(
            r"else:\n"
            r"([\s]{4,})self\.accelerator\.backward\(loss, \*\*kwargs\)\n"
            r"(.+?)if num_items_in_batch is None\:\n"
            r"(.+?)return loss\.detach\(\) \/ self\.args\.gradient_accumulation_steps",
            "else:\n"
            "\2if num_items_in_batch is None:\n"
            "\3loss = loss / self.args.gradient_accumulation_steps\n"
            "\1self.accelerator.backward(loss, **kwargs)",
            function,
        )

        exec(function, globals())
        Trainer.training_step = _unsloth_training_step

    # Wrap Trainer.__init__: (1) pre-init, shadow accepts_loss_kwargs on whatever
    # model was passed in (covers PEFT wrapping done after FastModel.from_pretrained);
    # (2) post-init, clamp accelerator GA to 1 for the transformers 5.0-5.5
    # GradientAccumulationPlugin regression. No-op on 4.x and 5.6+. See #4982.
    if not getattr(Trainer, "_unsloth_init_wrapped_for_accelerate_gas", False):
        _original_trainer_init = Trainer.__init__

        def _unsloth_trainer_init(self, *args, **kwargs):
            model = kwargs.get("model")
            if model is None and len(args) > 0:
                model = args[0]
            if model is not None:
                try:
                    apply_accepts_loss_kwargs_fix(model)
                except Exception:
                    pass
            _original_trainer_init(self, *args, **kwargs)
            try:
                accelerator = getattr(self, "accelerator", None)
                if (
                    accelerator is not None
                    and getattr(accelerator, "gradient_accumulation_steps", 1) > 1
                ):
                    accelerator.gradient_accumulation_steps = 1
                    gs = getattr(accelerator, "gradient_state", None)
                    if gs is not None and hasattr(gs, "plugin_kwargs"):
                        try:
                            gs.plugin_kwargs["num_steps"] = 1
                        except Exception:
                            pass
            except Exception:
                pass

        _unsloth_trainer_init.__wrapped__ = _original_trainer_init
        Trainer.__init__ = _unsloth_trainer_init
        Trainer._unsloth_init_wrapped_for_accelerate_gas = True


def _unsloth_compile_cache_leaves():
    # Accepts `UNSLOTH_COMPILE_LOCATION` overrides (the env var unsloth_zoo honors).
    leaves = {"unsloth_compiled_cache", "unsloth_cache", "unsloth_compiled"}
    loc = os.environ.get("UNSLOTH_COMPILE_LOCATION", "") or ""
    loc = loc.rstrip("/\\")
    if loc:
        leaves.add(os.path.basename(loc) or loc)
    return leaves


def _forward_is_unsloth_compiled(model):
    # True iff forward was installed from the Unsloth compile cache directory.
    # __module__ stays as the transformers module, so check co_filename.
    leaves = _unsloth_compile_cache_leaves()

    def check(m):
        if m is None:
            return False
        fwd = getattr(type(m), "forward", None)
        if fwd is None:
            return False
        code = getattr(fwd, "__code__", None)
        fn = getattr(code, "co_filename", "") if code is not None else ""
        fn = fn.replace("\\", "/")
        parts = set(fn.split("/"))
        return any(leaf in parts for leaf in leaves)

    if check(model):
        return True
    seen = set()
    m = model
    for _ in range(4):
        if m is None or id(m) in seen:
            break
        seen.add(id(m))
        nxt = getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            nxt = getattr(m, "model", None)
        if nxt is None or nxt is m:
            break
        if check(nxt):
            return True
        m = nxt
    return False


def _find_concrete_accepts_loss_kwargs(model):
    # Walk wrapper chain for first class that declares accepts_loss_kwargs in its
    # own __mro__ dict. Avoids PEFT __getattr__ forwarding and our own shadow.
    seen = set()
    m = model
    for _ in range(6):
        if m is None or id(m) in seen:
            break
        seen.add(id(m))
        for klass in type(m).__mro__:
            if "accepts_loss_kwargs" in klass.__dict__:
                return klass.__dict__[
                    "accepts_loss_kwargs"
                ], f"{klass.__name__}.accepts_loss_kwargs"
        nxt = getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            nxt = getattr(m, "model", None)
        if nxt is None or nxt is m:
            break
        m = nxt
    return None, "no explicit accepts_loss_kwargs on any wrapper level"


def _shadow_accepts_loss_kwargs(model, value):
    # Set the attribute at every wrapper level so HF's hasattr check resolves
    # regardless of where accelerator / peft unwrap lands.
    seen = set()
    m = model
    for _ in range(8):
        if m is None or id(m) in seen:
            break
        seen.add(id(m))
        try:
            setattr(m, "accepts_loss_kwargs", value)
        except Exception:
            pass
        nxt = getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            nxt = getattr(m, "model", None)
        if nxt is None or nxt is m:
            break
        m = nxt


def apply_accepts_loss_kwargs_fix(model):
    # Shadow the correct accepts_loss_kwargs on the model so HF Trainer picks it
    # up via hasattr(unwrapped_model, ...). Replaces the old Trainer.__init__
    # source rewrite. Priority: compiled forward -> True; else first class attr
    # in wrapper chain; else leave HF default. Issue #4982.
    if _forward_is_unsloth_compiled(model):
        _shadow_accepts_loss_kwargs(model, True)
        return "True (Unsloth compiled forward)"

    value, reason = _find_concrete_accepts_loss_kwargs(model)
    if value is None:
        return f"default (signature inspection, {reason})"
    _shadow_accepts_loss_kwargs(model, value)
    return f"{value} ({reason})"


def patch_tokenizer(model, tokenizer):
    model, tokenizer = _patch_tokenizer(model, tokenizer)
    if model is not None:
        model.config.update({"unsloth_version": __version__})
    return model, tokenizer


def patch_fast_lora():
    import peft.tuners.lora.bnb
    from ..kernels.fast_lora import fast_lora_forward
    peft.tuners.lora.bnb.Linear4bit.forward = fast_lora_forward


def unsloth_compile_transformers(
    dtype,
    model_name,
    model_types,
    token = None,
    revision = None,
    trust_remote_code = False,
    sdpa_dynamic_mask = True,
    sdpa_bool_masks = True,
    sdpa_gqa_replace = True,
    sdpa_dynamic_compile = True,
    compile_attention = True,
    disable_causal_masks = True,
    compile_torch_modules = True,
    compile_custom_modules = True,
    compile_function_calls = True,
    fuse_lm_head = True,
    gradient_checkpointing = True,
    manual_replacements = True,
    fast_lora_forwards = True,
    fast_residual_stream = True,
    accurate_accumulation = True,
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    cudagraphs = False,
    debug = False,
    fullgraph = True,
    import_from_cache = False,
    disable = False,
    return_logits = False,
    unsloth_force_compile = False,
):
    if Version(torch_version) < Version("2.4.0"):
        print(
            "="
            * 30
            + "Unsloth: Unfortunately Unsloth vision and other newer optimized models need Torch 2.4 or later.\n"
            f"You have Torch version {torch_version}. Please upgrade your Torch version by visiting https://pytorch.org/\n"
            "For now your models will not get optimized, but will still work for now!"
        )
        return
    if trust_remote_code and unsloth_force_compile == False:
        print(
            "Unsloth: We can't trace models if `trust_remote_code = True`, "
            "so turning off some optimizations!"
        )
        return model_types, False
    model_types = list(dict().fromkeys(model_types).keys())
    if disable:
        return model_types, False

    supports_sdpa = [True]

    # Run patches BEFORE compiler so class replacements (e.g. GptOssTopKRouter,
    # GptOssExperts) are in place before the compiler caches references to them.
    _run_temporary_patches("pre_compile")

    for model_type in model_types:
        _unsloth_compile_transformers(
            model_type,
            sdpa_dynamic_mask = sdpa_dynamic_mask,
            sdpa_bool_masks = sdpa_bool_masks,
            sdpa_gqa_replace = sdpa_gqa_replace,
            sdpa_dynamic_compile = sdpa_dynamic_compile,
            compile_attention = compile_attention,
            disable_causal_masks = disable_causal_masks,
            compile_torch_modules = compile_torch_modules,
            compile_custom_modules = compile_custom_modules,
            compile_function_calls = compile_function_calls,
            fuse_lm_head = fuse_lm_head,
            gradient_checkpointing = gradient_checkpointing,
            manual_replacements = manual_replacements,
            fast_lora_forwards = fast_lora_forwards,
            fast_residual_stream = fast_residual_stream,
            accurate_accumulation = accurate_accumulation,
            epilogue_fusion = epilogue_fusion,
            max_autotune = max_autotune,
            shape_padding = shape_padding,
            cudagraphs = cudagraphs,
            debug = debug,
            fullgraph = fullgraph,
            import_from_cache = import_from_cache,
            disable = disable,
            return_logits = return_logits,
            supports_sdpa = supports_sdpa,
        )
    # Redo patches which override compiler
    _run_temporary_patches("post_compile")
    return model_types, supports_sdpa[0]


# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = (
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'
    "```\nimport os\n"
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"
    "trainer.train()\n```\n"
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"
)


def raise_logits_error(*args, **kwargs):
    raise NotImplementedError(LOGITS_ERROR_STRING)


def return_none(*args, **kwargs):
    return None


class EmptyLogits:
    def __init__(self):
        return

    def raise_getattr_error(self, attr):
        return return_none if attr == "to" else raise_logits_error

    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error

    def __repr__(self):
        return LOGITS_ERROR_STRING

    def __str__(self):
        return LOGITS_ERROR_STRING

    def __reduce__(self):
        # Stateless pickling so gather_object works on the sentinel
        return (type(self), ())

    def __eq__(self, other):
        # Gathered copies must compare equal in accelerate debug mode
        return type(other).__name__ == "EmptyLogits"

    __hash__ = object.__hash__


EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try:
            exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except:
            continue
# The loop above stomps pickle hooks with stubs returning None, which breaks
# gather_object on EMPTY_LOGITS in distributed runs. Restore default pickling.
for function in ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"):
    try:
        delattr(EMPTY_LOGITS, function)
    except Exception:
        pass


def validate_loftq_config(loftq_config, lora_dropout, bias, init_lora_weights, model):
    from peft import LoraConfig

    if loftq_config is None:
        loftq_config = {}

    signature = str(inspect.signature(LoraConfig))
    SUPPORTS_LOFTQ = "loftq_config" in signature

    if lora_dropout != 0:
        logger.warning_once(
            f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
            f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
        )

    if bias != "none":
        logger.warning_once(
            f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
            f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
        )

    if not (
        type(init_lora_weights) is bool
        or init_lora_weights == "gaussian"
        or init_lora_weights == "loftq"
        or init_lora_weights == "corda"
    ):
        raise ValueError(
            'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq", "corda"].'
        )

    if init_lora_weights == "loftq":
        if not SUPPORTS_LOFTQ:
            import peft
            raise RuntimeError(
                f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"
                "Please install PEFT 0.7.2 or higher.\n"
                "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
            )

        if loftq_config == {}:
            from peft import LoftQConfig
            logger.warning_once(
                "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"
                "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
            )
            loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)

        if hasattr(model.config, "quantization_config"):
            raise ValueError(
                "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"
                "Reload your model without any quantization by setting `load_in_4bit = False`."
            )

    return loftq_config


def fast_inference_setup(model_name, model_config):
    fast_inference = True
    if not is_vLLM_available():
        logger.warning_once("Unsloth: vLLM is not installed! Will use Unsloth inference!")
        fast_inference = False
    from unsloth_zoo.vllm_utils import (
        patch_vllm,
        vllm_dynamic_quant_supported,
    )

    patch_vllm()
    if model_name.endswith("unsloth-bnb-4bit"):
        if not vllm_dynamic_quant_supported(model_name, model_config):
            # Instead use -bnb-4bit variant
            logger.warning_once(
                f"Unsloth: Switching from Unsloth dynamic quant to normal quant since\n"
                f"we do not yet support fast inference for {model_name}"
            )
            model_name = model_name[: -len("unsloth-bnb-4bit")] + "bnb-4bit"
    return fast_inference, model_name


def patch_peft_fast_inference(model):
    vllm_engine = getattr(model.model, "vllm_engine", None)
    if vllm_engine is not None:
        model.vllm_engine = model.model.vllm_engine
        model.fast_generate = model.model.fast_generate
        model.fast_generate_batches = model.model.fast_generate_batches

        # Also saving and loading LoRA
        from unsloth_zoo.vllm_utils import save_lora, load_lora

        model.save_lora = functools.partial(save_lora, model)
        model.load_lora = functools.partial(load_lora, model)


def error_out_no_vllm(*args, **kwargs):
    raise NotImplementedError(
        "Unsloth: vLLM is not yet supported for fast inference for this model! Please use `.generate` instead"
    )


try:
    from torchao.core.config import AOBaseConfig
    try:
        from torchao.quantization import Int4WeightOnlyConfig
    except:
        print("Unsloth: TorchAO changed `torchao.quantization.Int4WeightOnlyConfig`")
        Int4WeightOnlyConfig = None
except:
    AOBaseConfig = None
    Int4WeightOnlyConfig = None


@dataclass
class TorchAOConfig:
    qat_scheme: Optional[str] = "int4"

    # Each (config, filter_fn) pair defines a quantization rule
    base_config_and_filter_fns: List[
        Tuple["AOBaseConfig", Optional[Callable[[torch.nn.Module, str], bool]]]
    ] = field(
        default_factory = lambda: [
            (
                Int4WeightOnlyConfig(group_size = 128),
                lambda m, _: isinstance(m, torch.nn.Linear) and getattr(m, "in_features", 0) >= 128,
            ),
        ]
    )

    # Optional transformation to apply before quantization setup
    prequantization_transform: Optional[Callable[[torch.nn.Module], None]] = None


def _untie_input_output_embeddings(model: torch.nn.Module) -> None:
    """
    Utility to untie input/output embeddings in a HuggingFace model.
    This is useful if we want to quantize the input/output embeddings differently.
    Model is modified in-place.
    """

    # 1) Persist setting in config
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    # 2) Find input and output embeddings
    in_emb = model.get_input_embeddings()
    out_proj = model.get_output_embeddings() or getattr(model, "lm_head", None)
    if out_proj is None:
        raise AttributeError("Couldn't locate output projection (lm_head).")

    # (Optional) sanity: shapes should match [vocab, hidden]
    assert (
        out_proj.weight.shape == in_emb.weight.shape
    ), f"Shape mismatch: out_proj {out_proj.weight.shape} vs in_emb {in_emb.weight.shape}"

    # 3) Only clone if they are actually tied (shared storage)
    if out_proj.weight.data_ptr() == in_emb.weight.data_ptr():
        with torch.no_grad():
            W = in_emb.weight.detach().clone()
        out_proj.weight = torch.nn.Parameter(W)  # new storage, keeps dtype/device

    # 4) Prevent future automatic re-tying
    def _no_tie(self):
        return

    model.tie_weights = _no_tie.__get__(model, model.__class__)

    # 5) Verify no shared storage
    assert out_proj.weight.data_ptr() != in_emb.weight.data_ptr(), "Embeddings still tied!"


def _filter_fn_to_fqns(
    model: torch.nn.Module, filter_fn: Callable[[torch.nn.Module, str], bool]
) -> Iterator[str]:
    """
    Given a model and a filter function (m, fqn) -> bool,
    yield fully qualified names (FQNs) of modules that match.
    """
    for fqn, module in model.named_modules():
        if filter_fn(module, fqn):
            yield fqn


def _convert_torchao_model(model):
    from transformers import TorchAoConfig
    from torchao.quantization import quantize_, ModuleFqnToConfig
    from torchao.quantization.qat import QATConfig
    from torchao.utils import TorchAOBaseTensor

    module_to_fqn_dict = {}
    for base_config, filter_fn in model._torchao_config.base_config_and_filter_fns:
        quantize_(model, QATConfig(base_config, step = "convert"), filter_fn = filter_fn)

        # Default filter function used for quantize_
        if filter_fn is None:
            if "_default" in module_to_fqn_dict:
                raise ValueError("Cannot use multiple default quantization configs")
            module_to_fqn_dict["_default"] = base_config
        else:
            for fqn in _filter_fn_to_fqns(model, filter_fn):
                if fqn in module_to_fqn_dict:
                    raise ValueError(f"Found multiple quantization configs for {fqn}")
                module_to_fqn_dict[fqn] = base_config

    in_emb = model.get_input_embeddings()
    out_proj = model.get_output_embeddings() or getattr(model, "lm_head", None)
    kwargs = {}
    if isinstance(in_emb.weight, TorchAOBaseTensor) or (
        out_proj is not None and isinstance(out_proj.weight, TorchAOBaseTensor)
    ):
        kwargs["include_input_output_embeddings"] = True
        kwargs["modules_to_not_convert"] = []

    quant_config = ModuleFqnToConfig(module_to_fqn_dict)
    quantization_config = TorchAoConfig(quant_type = quant_config, **kwargs)
    model.config.quantization_config = quantization_config


def _prepare_model_for_qat(
    model: torch.nn.Module, qat_scheme: Union[str, TorchAOConfig]
) -> torch.nn.Module:
    """
    Transform a model for Quantization-Aware Training (QAT) during fine-tuning.

    On a high level, this means fake quantizing the base (frozen) model during training.
    Fake quantization refers to simulating quantization numerics in high precision (e.g. bf16).
    This helps mitigate quantization degradations when the model is quantized after training.

    QAT can be optionally combined with LoRA fine-tuning to for additional throughput improvement.
    For more details: https://dev-discuss.pytorch.org/t/speeding-up-qat-by-1-89x-with-lora/2700
    """
    try:
        from torchao.quantization import PerRow, quantize_
        from torchao.quantization.granularity import PerGroup, PerAxis
        from torchao.quantization.qat import QATConfig
    except ImportError:
        raise ImportError(TORCHAO_MSG)

    # Gemma3 models have issues with int8 embedding quantization due to their
    # large vocabulary size (262144). Auto-switch to int4 weight-only instead.
    if qat_scheme == "int8-int4":
        model_types = get_transformers_model_type(model.config)
        is_gemma3 = any("gemma3" in mt or "gemma_3" in mt for mt in model_types)
        if is_gemma3:
            print(
                "Unsloth: Gemma3 has a large vocabulary causing int8 embedding issues. "
                "Switching to int4 weight-only QAT for training stability."
            )
            qat_scheme = "int4"

    if not isinstance(qat_scheme, TorchAOConfig):
        torchao_config: Optional[TorchAOConfig] = None
        if qat_scheme == "fp8-int4":
            try:
                from torchao.quantization import Float8DynamicActivationInt4WeightConfig
            except ImportError:
                raise ImportError(TORCHAO_MSG)
            group_size = 128
            base_config = Float8DynamicActivationInt4WeightConfig()
            filter_fn = lambda m, _: isinstance(m, torch.nn.Linear) and m.in_features >= group_size
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme,
                base_config_and_filter_fns = [(base_config, filter_fn)],
            )
        elif qat_scheme == "fp8-fp8":
            try:
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                )
            except ImportError:
                raise ImportError(TORCHAO_MSG)
            base_config = Float8DynamicActivationFloat8WeightConfig(granularity = PerRow())
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme, base_config_and_filter_fns = [(base_config, None)]
            )
        elif qat_scheme == "int8-int4":
            try:
                from torchao.quantization import (
                    Int8DynamicActivationIntxWeightConfig,
                    IntxWeightOnlyConfig,
                )
            except ImportError:
                raise ImportError(TORCHAO_MSG)
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme,
                base_config_and_filter_fns = [
                    (
                        IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)),
                        lambda m, fqn: isinstance(m, torch.nn.Embedding),
                    ),
                    (
                        Int8DynamicActivationIntxWeightConfig(
                            weight_dtype = torch.int4, weight_granularity = PerGroup(32)
                        ),
                        None,
                    ),
                ],
                prequantization_transform = _untie_input_output_embeddings,
            )
        elif qat_scheme == "int4":
            try:
                from torchao.quantization import Int4WeightOnlyConfig
            except ImportError:
                raise ImportError(TORCHAO_MSG)
            group_size = 128
            base_config = Int4WeightOnlyConfig(group_size = group_size)
            filter_fn = lambda m, _: isinstance(m, torch.nn.Linear) and m.in_features >= group_size
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme,
                base_config_and_filter_fns = [(base_config, filter_fn)],
            )
        elif qat_scheme == "int8":
            try:
                from torchao.quantization import IntxWeightOnlyConfig
                from torchao.quantization.granularity import PerAxis
            except ImportError:
                raise ImportError(TORCHAO_MSG)

            base_config = IntxWeightOnlyConfig(
                weight_dtype = torch.int8,
                granularity = PerAxis(0),
            )
            filter_fn = lambda m, _: isinstance(m, torch.nn.Linear)
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme,
                base_config_and_filter_fns = [(base_config, filter_fn)],
            )
        elif qat_scheme == "cactus":
            try:
                from torchao.quantization import IntxWeightOnlyConfig
            except ImportError:
                raise ImportError(TORCHAO_MSG)

            # IntxWeightOnlyConfig already defaults to
            # `mapping_type = MappingType.SYMMETRIC`, so we intentionally do not
            # import `MappingType` here. Matches the upstream Cactus runtime
            # int8 / per-group-32 / symmetric weight-only configuration.
            group_size = 32
            base_config = IntxWeightOnlyConfig(
                weight_dtype = torch.int8,
                granularity = PerGroup(group_size),
            )
            filter_fn = (
                lambda m, _: isinstance(m, torch.nn.Linear)
                and m.in_features >= group_size
                and m.in_features % group_size == 0
            )
            # Warn if any Linear layer is skipped by the cactus filter because
            # its in_features is not divisible by `group_size`. torchao's
            # PerGroup(32) quantizer rejects non-divisible widths at
            # `quantize_()` time, so the filter excludes those layers to keep
            # the QAT prepare step from crashing. Surface that silently-skipped
            # coverage gap to the user so they know some Linears will stay in
            # full precision during training.
            skipped_cactus_layers = [
                name
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
                and module.in_features >= group_size
                and module.in_features % group_size != 0
            ]
            if skipped_cactus_layers:
                preview = ", ".join(skipped_cactus_layers[:8])
                if len(skipped_cactus_layers) > 8:
                    preview += f", ... ({len(skipped_cactus_layers) - 8} more)"
                warnings.warn(
                    f"Unsloth: qat_scheme='cactus' uses PerGroup({group_size}) "
                    "which requires in_features to be divisible by "
                    f"{group_size}. The following Linear layers will be kept "
                    f"in full precision during QAT: {preview}",
                    stacklevel = 2,
                )
            torchao_config = TorchAOConfig(
                qat_scheme = qat_scheme,
                base_config_and_filter_fns = [(base_config, filter_fn)],
            )
        else:
            raise ValueError(f"Unexpected QAT scheme {qat_scheme}")
        assert torchao_config is not None, f"TorchAOConfig was not set for {qat_scheme}"
    else:
        torchao_config = qat_scheme

    # Save Torchao metadata everywhere
    inner_model = model
    while hasattr(inner_model, "model"):
        inner_model._torchao_config = torchao_config
        inner_model = inner_model.model
    inner_model._torchao_config = torchao_config

    if torchao_config.prequantization_transform is not None:
        torchao_config.prequantization_transform(model)
    for base_config, filter_fn in torchao_config.base_config_and_filter_fns:
        quantize_(model, QATConfig(base_config, step = "prepare"), filter_fn = filter_fn)

    return model


def patch_hf_quantizer():
    # To tell hf trainer that the quantized model is trainable
    def make_trainable(self):
        return True

    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer,
        )
        FineGrainedFP8HfQuantizer.is_trainable = property(make_trainable)
        FineGrainedFP8HfQuantizer.is_qat_trainable = property(make_trainable)
    except Exception as e:
        logger.warning(f"Failed to patch FineGrainedFP8HfQuantizer. Error {e}")

    try:
        from transformers.quantizers.quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer
        FbgemmFp8HfQuantizer.is_trainable = property(make_trainable)
        FbgemmFp8HfQuantizer.is_qat_trainable = property(make_trainable)
    except Exception as e:
        logger.warning(f"Failed to patch FbgemmFp8HfQuantizer. Error {e}")

    try:
        from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer
        TorchAoHfQuantizer.is_trainable = property(make_trainable)
        TorchAoHfQuantizer.is_qat_trainable = property(make_trainable)
    except Exception as e:
        logger.warning(f"Failed to patch TorchAoHfQuantizer. Error {e}")


patch_hf_quantizer()


def verify_fp8_support_if_applicable(model_config):
    quant_method = get_quant_type(model_config)
    if quant_method in ["fbgemm_fp8", "fp8"] and DEVICE_TYPE != "cuda":
        raise ValueError(
            f"Unsloth: FP8 quantization is only supported on CUDA GPUs. You are using {DEVICE_TYPE}."
        )

    # [TODO] Need to add FP8 support for Intel XPUs
    if DEVICE_TYPE == "cuda":
        major_version, minor_version = torch.cuda.get_device_capability()
        if quant_method == "fbgemm_fp8" and major_version < 9:
            # While L4 does support FP8 as data type, it doesn't have fbgemm (package) support yet. So we restrict it.
            raise ValueError(
                f"Unsloth: FBGEMM FP8 quantization is only supported on H100 and higher GPUs. L4 is not supported. You are using {torch.cuda.get_device_name()}. Refer to https://developer.nvidia.com/cuda-gpus for more details."
            )
        if quant_method == "fp8" and major_version * 10 + minor_version < 89:
            # In case of block quantized, we allow L4 because we fall back to torchao kernels.
            raise ValueError(
                f"Unsloth: FP8 quantization is only supported on L4 and higher GPUs with compute capability 8.9 or higher. You are using {torch.cuda.get_device_name()}. Refer to https://developer.nvidia.com/cuda-gpus for more details."
            )


def _get_inference_mode_context_manager(model: torch.nn.Module):
    """
    If the state dict was quantized using torchao, we will run into
    the following error when calling ops like aten.t() in inference mode.
    This is a bug in PyTorch that affects all tensor subclasses.

        Cannot set version_counter for inference tensor

    For now, we work around this issue by using `torch.no_grad()` in this case.
    See https://github.com/pytorch/pytorch/issues/164872 for more details.
    Otherwise, just return `torch.inference_mode()`.
    """
    torchao_config = getattr(model, "torchao_config", None)
    if torchao_config is not None and torchao_config.qat_scheme is None:
        return torch.no_grad()
    else:
        return torch.inference_mode()


def hf_login(token: Optional[str] = None) -> Optional[str]:
    if token is None:
        try:
            from huggingface_hub import get_token
            token = get_token()
            if token is None:
                return None
        except:
            return None
    try:
        from huggingface_hub import login
        login(token = token)
        return token
    except Exception as e:
        logger.info(f"Failed to login to huggingface using token with error: {e}")
    return token


# =============================================
# MoE (Mixture of Experts) Detection and LoRA Utilities


def is_moe_model(model) -> bool:
    """
    Detect if a model is a Mixture of Experts (MoE) model.

    Args:
        model: The model to check (can be HF model or config)

    Returns:
        True if the model is an MoE model, False otherwise
    """
    config = getattr(model, "config", model)

    # Different MoE models use different config attribute names:
    # - Qwen3-MoE: num_experts
    # - GLM4-MoE: n_routed_experts, num_local_experts
    # - Mixtral: num_local_experts
    num_experts = None
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        num_experts = getattr(config, attr, None)
        if num_experts is not None:
            break

    # Check text_config for VL models
    if num_experts is None and hasattr(config, "text_config"):
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            num_experts = getattr(config.text_config, attr, None)
            if num_experts is not None:
                break

    return num_experts is not None and num_experts > 0


def _resolve_moe_parameter_name(model, default_name: str, alternate_name: str) -> str:
    """
    Resolve the actual parameter path for MoE expert weights.

    Most current Unsloth MoE models expose expert weights under
    ``mlp.experts.*``. Gemma4 stores them directly under ``experts.*``.
    Prefer the path that exists on the loaded module when possible.
    """
    if hasattr(model, "named_parameters"):
        try:
            for name, _ in model.named_parameters():
                if name == default_name or name.endswith("." + default_name):
                    return default_name
                if name == alternate_name or name.endswith("." + alternate_name):
                    return alternate_name
        except Exception:
            pass

    config = getattr(model, "config", model)
    model_types = {getattr(config, "model_type", None)}
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        model_types.add(getattr(text_config, "model_type", None))

    if any(
        isinstance(model_type, str) and model_type.startswith("gemma4")
        for model_type in model_types
    ):
        return alternate_name

    return default_name


_MOE_BROAD_MLP_TARGETS = frozenset(("gate_proj", "up_proj", "down_proj", "gate_up_proj"))


def _moe_target_set_from_string(target_modules: str) -> set[str]:
    if target_modules in _MOE_BROAD_MLP_TARGETS:
        return {target_modules}

    is_regex = re.search(r"[*+?()[\]{}|\\^$]", target_modules) is not None
    # Key detection on the mlp/ffn/experts path segment (absent from an
    # attention-only regex), never on q/k/v/o leaves alone.
    targets_mlp_path = any(
        tag in target_modules for tag in ("mlp", "ffn", "feed_forward", "experts")
    )
    if not is_regex or not targets_mlp_path:
        return set()
    # Explicit expert leaves scope the target set to exactly those leaves.
    named = {name for name in _MOE_BROAD_MLP_TARGETS if name in target_modules}
    if named:
        return named
    # A generic projection under an mlp path (e.g. ".*mlp.*proj"): any proj
    # occurrence that is not an attention leaf name.
    if re.search(r"(?<![qkvo]_)(?<!out_)(?<!in_)proj", target_modules):
        return set(_MOE_BROAD_MLP_TARGETS)
    # The auto regex on fused-expert models lists only attention Linears as
    # leaves; its mlp tag block is the remaining MLP-intent signal. A regex
    # like "(mlp|self_attn).(q_proj|o_proj)" has neither and stays attention-only.
    if "mlp|feed_forward|ffn|dense" in target_modules:
        return set(_MOE_BROAD_MLP_TARGETS)

    return set()


def get_moe_target_parameters(model, target_modules = None) -> Optional[List[str]]:
    """
    Get the target_parameters for MoE expert layers if applicable.

    For MoE models, returns the parameter paths for expert weights
    (gate_up_proj, down_proj) that should be targeted by PEFT's
    target_parameters for LoRA on nn.Parameter. The exact parameter path
    depends on the model layout, for example ``mlp.experts.*`` or
    ``experts.*``.

    Only includes MoE parameters that match what's in target_modules:
    - If "down_proj" is in target_modules -> includes "mlp.experts.down_proj"
    - If "gate_proj" or "up_proj" is in target_modules -> includes "mlp.experts.gate_up_proj"

    Args:
        model: The model to get target parameters for
        target_modules: List/tuple of target module names to match against

    Returns:
        List of parameter paths for MoE experts, or None if not an MoE model
    """
    if not is_moe_model(model):
        return None

    config = getattr(model, "config", model)
    # Get num_experts from various possible config attributes
    num_experts = None
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        num_experts = getattr(config, attr, None)
        if num_experts is not None:
            break
    if num_experts is None and hasattr(config, "text_config"):
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            num_experts = getattr(config.text_config, attr, None)
            if num_experts is not None:
                break
    if num_experts is None:
        num_experts = 0

    if target_modules is None:
        return None
    elif isinstance(target_modules, str):
        target_set = _moe_target_set_from_string(target_modules)
    else:
        target_set = {
            target
            for target in target_modules or ()
            if (isinstance(target, str) and "." not in target and target in _MOE_BROAD_MLP_TARGETS)
        }

    moe_params = []

    gate_up_name = _resolve_moe_parameter_name(
        model,
        default_name = "mlp.experts.gate_up_proj",
        alternate_name = "experts.gate_up_proj",
    )
    down_name = _resolve_moe_parameter_name(
        model,
        default_name = "mlp.experts.down_proj",
        alternate_name = "experts.down_proj",
    )

    # gate_up_proj combines gate_proj and up_proj; also match the fused name directly.
    # Only target a fused expert Parameter that exists: per-expert Linear layouts
    # (e.g. gpt-oss bnb-4bit) have no fused Parameter and are handled by
    # get_moe_target_modules, so skip them rather than pass PEFT a dead path.
    if "gate_proj" in target_set or "up_proj" in target_set or "gate_up_proj" in target_set:
        if _moe_parameter_exists(model, gate_up_name):
            moe_params.append(gate_up_name)

    if "down_proj" in target_set:
        if _moe_parameter_exists(model, down_name):
            moe_params.append(down_name)

    if moe_params:
        print(
            f"Unsloth: Detected MoE model with {num_experts = } and {target_modules = }. Enabling LoRA on MoE parameters: {moe_params}"
        )
        return moe_params

    return None


def _moe_parameter_exists(model, name: str) -> bool:
    """True if ``name`` is an exact suffix of some parameter path on the model."""
    if not hasattr(model, "named_parameters"):
        return False
    try:
        for parameter_name, _ in model.named_parameters():
            if parameter_name == name or parameter_name.endswith("." + name):
                return True
    except Exception:
        return False
    return False


def get_moe_target_modules(model, target_modules = None) -> List[str]:
    """Per-expert ``target_modules`` suffixes for MoE models whose experts are stored
    as per-expert ``nn.Linear`` ModuleLists rather than fused nn.Parameters.

    gpt-oss bnb-4bit is the canonical case (mlp.experts.gate_up_projs.<i> /
    down_projs.<i> as Linear4bit): no fused Parameter, and the plain
    gate/up/down_proj leaves do not match, so LoRA skips them. Returning the
    per-expert suffixes makes PEFT attach via ordinary suffix matching (the
    module-LoRA counterpart of get_moe_target_parameters). Returns [] for non-MoE,
    fused-parameter MoEs, an absent per-expert layout, or a request that omits the
    MLP experts (so an attention-only run does not train experts).
    """
    if not is_moe_model(model):
        return []
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        target_set = _moe_target_set_from_string(target_modules)
    else:
        target_set = {
            target
            for target in target_modules or ()
            if (isinstance(target, str) and "." not in target and target in _MOE_BROAD_MLP_TARGETS)
        }
    if not (target_set & _MOE_BROAD_MLP_TARGETS):
        return []

    if not hasattr(model, "named_modules"):
        return []

    # Scope the returned suffixes to the requested projection leaves, matching
    # get_moe_target_parameters: gate_proj/up_proj/gate_up_proj map to the fused
    # gate_up ModuleList (e.g. gate_up_projs); down_proj maps to the down ModuleList
    # (e.g. down_projs). A down-only (or gate/up-only) request must not pull in the
    # other projection.
    want_gate_up = bool(target_set & {"gate_proj", "up_proj", "gate_up_proj"})
    want_down = "down_proj" in target_set

    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        parent, _, leaf = name.rpartition(".")
        # ModuleList directly under an ``experts`` container, holding only Linear
        # leaves (bnb Linear4bit / Linear8bitLt subclass nn.Linear). After PEFT has
        # wrapped the experts the child is a LoRA layer whose ``base_layer`` is the
        # Linear, so accept that too (keeps this idempotent across a re-wrapped model).
        if not parent.endswith("experts"):
            continue
        if not all(
            isinstance(child, torch.nn.Linear)
            or isinstance(getattr(child, "base_layer", None), torch.nn.Linear)
            for child in module
        ):
            continue
        # Honor the requested subset: classify the ModuleList by projection role.
        leaf_lower = leaf.lower()
        is_down = "down" in leaf_lower
        is_gate_up = (not is_down) and ("gate" in leaf_lower or "up" in leaf_lower)
        if is_down and not want_down:
            continue
        if is_gate_up and not want_gate_up:
            continue
        # One entry per expert index; ``leaf.<i>`` matches expert i in every layer.
        for expert_index in range(len(module)):
            targets.add(f"{leaf}.{expert_index}")

    return sorted(targets)


def warn_if_zoo_cannot_merge_moe_experts():
    """Warn once when the installed unsloth_zoo cannot fold per-expert Linear MoE LoRA
    into a merged_16bit checkpoint. Older zoo releases keep the fused gate_up_proj /
    down_proj tensors and drop the per-expert gate_up_projs.<i> / down_projs.<i> deltas,
    so save_pretrained_merged("merged_16bit") would silently lose the expert training
    (the LoRA adapter itself still saves and reloads correctly)."""
    try:
        from unsloth_zoo import saving_utils as _saving_utils

        # _fold_perexpert_lora_into_fused is the helper that folds these experts.
        if hasattr(_saving_utils, "_fold_perexpert_lora_into_fused"):
            return
    except Exception:
        return  # cannot introspect zoo -> stay quiet rather than false-alarm
    logger.warning_once(
        "Unsloth: the installed unsloth_zoo will not fold these per-expert experts into "
        "a merged_16bit checkpoint, so save_pretrained_merged('merged_16bit') would drop "
        "the expert LoRA. Upgrade unsloth_zoo to merge them; saving the LoRA adapter is "
        "unaffected."
    )


def _select_moe_detection_targets(
    original_target_modules,
    scoped_target_modules,
    finetune_mlp_modules = True,
    finetune_language_layers = True,
):
    """Pick what get_moe_target_parameters keys expert detection on.

    Prefer the caller's ORIGINAL explicit leaf list over the scoped regex so an
    attention-only request is not pushed into the experts by get_peft_regex's
    ``mlp|feed_forward|ffn|dense`` component block (which the string fallback
    cannot tell apart from a fused-expert auto regex).

    But only when the MLP and language families are BOTH still in scope. If the
    caller scoped MLP or language OFF (``finetune_mlp_modules=False`` or
    ``finetune_language_layers=False``) the scoped regex already drops the MoE
    experts, and reusing the original list -- which may still name gate/up/down
    leaves -- would wrongly re-introduce them. In that case honor the scoped
    result so the frozen-MLP / vision-only request is respected.
    """
    if original_target_modules is not None and finetune_mlp_modules and finetune_language_layers:
        return original_target_modules
    return scoped_target_modules


def make_fast_generate_wrapper(original_generate):
    """
    Creates a wrapper around model.generate that checks for incorrect
    vLLM-style usage when fast_inference=False.
    """

    @functools.wraps(original_generate)
    def _fast_generate_wrapper(*args, **kwargs):
        def _has_sampling_params(a):
            # SamplingParams passed directly or inside a positional list/tuple
            return type(a).__name__ == "SamplingParams" or (
                isinstance(a, (list, tuple))
                and any(type(i).__name__ == "SamplingParams" for i in a)
            )

        def _is_vllm_prompt(a):
            # str prompt, a vLLM prompt dict (prompt / prompt_token_ids / prompt_embeds /
            # multi_modal_data), or a list/tuple of those
            head = a[0] if isinstance(a, (list, tuple)) and len(a) > 0 else a
            return isinstance(head, str) or (
                isinstance(head, dict)
                and any(
                    k in head
                    for k in ("prompt", "prompt_token_ids", "prompt_embeds", "multi_modal_data")
                )
            )

        # vLLM-only; also catch SamplingParams passed positionally (fast_generate(prompt, params))
        if "sampling_params" in kwargs or any(_has_sampling_params(a) for a in args):
            raise ValueError(
                "Unsloth: `sampling_params` is only supported when `fast_inference=True` (vLLM). "
                "Since `fast_inference=False`, use HuggingFace generate arguments instead:\n"
                "  model.fast_generate(**tokens.to('cuda'), max_new_tokens=64, temperature=1.0, top_p=0.95)"
            )

        if "lora_request" in kwargs:
            raise ValueError(
                "Unsloth: `lora_request` is only supported when `fast_inference=True` (vLLM). "
                "Since `fast_inference=False`, LoRA weights are already merged into the model."
            )

        # A vLLM-style prompt (string, {"prompt":..., "multi_modal_data":...} dict, or a list/tuple
        # of either) only works under vLLM; tokenize first when fast_inference=False. A positional
        # arg may be HF token ids, so check it conservatively with _is_vllm_prompt. The `prompts` /
        # `prompt_token_ids` / `prompt_embeds` keywords are vLLM-only names that HuggingFace generate
        # does not accept, so any of them being present is a vLLM-style call (even a bare token list,
        # or an explicit None from a defaulted kwargs dict), hence membership rather than a value check.
        vllm_prompt_kwarg = any(
            k in kwargs for k in ("prompts", "prompt_token_ids", "prompt_embeds")
        )
        if (len(args) > 0 and _is_vllm_prompt(args[0])) or vllm_prompt_kwarg:
            raise ValueError(
                "Unsloth: Passing vLLM-style prompts to `fast_generate` is only supported when "
                "`fast_inference=True` (vLLM). Since `fast_inference=False`, tokenize first:\n\n"
                "  inputs = tokenizer.apply_chat_template(\n"
                '      [{"role": "user", "content": "Your prompt here"}],\n'
                "      tokenize=True, add_generation_prompt=True,\n"
                '      return_tensors="pt", return_dict=True,\n'
                "  )\n"
                "  output = model.fast_generate(**inputs.to('cuda'), max_new_tokens=64, temperature=1.0)"
            )

        # Call original generate
        return original_generate(*args, **kwargs)

    return _fast_generate_wrapper


# Fix llm_int8_skip_modules not being respected for VLMs with dynamic quantization.
# Dynamic quant checkpoints (eg gemma-3-4b-it-unsloth-bnb-4bit) encode skip paths as
# "language_model.model.layers.*", but the live module tree surfaces them as
# "model.language_model.layers.*". This prefix mismatch causes should_convert_module
# to miss the skip list, so modules meant to stay in 16-bit get wrapped in Linear4bit
# without a quant_state, producing "Skipping ... no quant_state found" warnings.
# We patch should_convert_module to expand both the module name and the skip patterns
# into all equivalent alias forms before delegating to the original matcher.
# Ref: https://github.com/unslothai/unsloth/issues/4208
import transformers.quantizers.quantizers_utils as _quantizers_utils

if (
    hasattr(_quantizers_utils, "should_convert_module")
    and getattr(_quantizers_utils.should_convert_module, "__name__", "")
    != "patched_should_convert_module"
):
    _original_should_convert_module = _quantizers_utils.should_convert_module

    def _get_full_name_aliases(full_name):
        aliases = {full_name}
        if not isinstance(full_name, str):
            return aliases

        if full_name.startswith("model.language_model."):
            aliases.add(full_name[len("model.") :])
        if "language_model.model." in full_name:
            aliases.add(full_name.replace("language_model.model.", "language_model."))
        if full_name.startswith("model.language_model.model."):
            aliases.add(
                full_name[len("model.") :].replace("language_model.model.", "language_model.")
            )
        return aliases

    def _get_pattern_aliases(pattern):
        aliases = {pattern}
        if not isinstance(pattern, str):
            return aliases

        if "language_model.model." in pattern:
            aliases.add(pattern.replace("language_model.model.", "language_model."))
        return aliases

    def _expand_patterns(patterns):
        expanded = set()
        for pattern in patterns:
            expanded.update(_get_pattern_aliases(pattern))
        return expanded

    def patched_should_convert_module(full_name, patterns = None):
        if patterns is None:
            return _original_should_convert_module(full_name, patterns)

        expanded_patterns = _expand_patterns(patterns)
        return all(
            _original_should_convert_module(candidate, expanded_patterns)
            for candidate in _get_full_name_aliases(full_name)
        )

    patched_should_convert_module._original_should_convert_module = _original_should_convert_module
    _quantizers_utils.should_convert_module = patched_should_convert_module

    try:
        import transformers.integrations.bitsandbytes
        transformers.integrations.bitsandbytes.should_convert_module = patched_should_convert_module
    except Exception:
        pass
