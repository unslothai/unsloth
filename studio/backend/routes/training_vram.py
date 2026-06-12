# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
VRAM coordination between chat/inference and training.

When a training run starts it competes with any resident chat model for GPU
memory. These helpers decide -- based on live free VRAM -- whether the chat
model can stay loaded (so the user can train and chat at the same time) or must
be unloaded first, and perform the unload across all inference backends
(HF/transformers + MLX orchestrator, and the llama.cpp GGUF server).

Kept in the route layer (not core/) because the GGUF singleton accessor lives in
routes/inference.py; backend accessors are imported lazily inside each function
to avoid import-time cycles and the heavy LlamaCppBackend() construction.
"""

from typing import Any, Dict, List, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

# Conservative headroom for keeping the chat model loaded during training. The
# probe only sees the chat model's *current* footprint, so we reserve extra for
# imperfect estimates and KV-cache growth during long conversations:
#   keep iff usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
# KEEP_FLOOR_GB folds the estimator's ~2 GB load buffer plus ~2 GB chat reserve.
SAFETY_MARGIN = 1.15
KEEP_FLOOR_GB = 4.0

# Sharding has inter-GPU overhead, so each extra GPU contributes less than its
# raw free memory. Mirrors auto_select_gpu_ids' empirical factor.
_MULTI_GPU_OVERHEAD = 0.85


def summarize_resident_chat() -> Dict[str, Any]:
    """Report which chat/inference models currently hold GPU memory.

    A model counts as resident even while it is still loading (it already holds
    VRAM at that point). Never raises.
    """
    hf_name: Optional[str] = None
    gguf_name: Optional[str] = None

    try:
        from core.inference import get_inference_backend
        inf = get_inference_backend()
        # active_model_name is set only on a *successful* load; a model mid-load
        # sits in loading_models (set in the parent before the load) while
        # already holding VRAM, so treat both as resident. A bare-alive
        # subprocess with no model loaded (e.g. after an unload) holds only the
        # CUDA context and is intentionally NOT treated as resident.
        if inf.active_model_name or inf.loading_models:
            hf_name = inf.active_model_name or next(iter(inf.loading_models), None)
    except Exception as e:
        logger.warning("Could not inspect inference backend: %s", e)

    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        # is_active (process exists) rather than is_loaded (process exists AND
        # healthy) -- a server mid-start is already allocating VRAM.
        if llama.is_active:
            gguf_name = llama.model_identifier or "gguf"
    except Exception as e:
        logger.warning("Could not inspect GGUF backend: %s", e)

    return {
        "hf": hf_name,
        "gguf": gguf_name,
        "any": bool(hf_name or gguf_name),
    }


def can_keep_chat_during_training(
    *,
    model_name: str,
    hf_token: Optional[str],
    training_type: str,
    load_in_4bit: bool,
    batch_size: int,
    max_seq_length: int,
    lora_rank: int,
    target_modules: Optional[list],
    gradient_checkpointing: str,
    optimizer: str,
    gpu_ids: Optional[list],
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether a resident chat model can coexist with a training run
    given current free VRAM.

    Reuses the same estimator/selector training itself will use, so the decision
    matches the placement computed later in TrainingBackend.start_training.
    Default-deny: anything we cannot confidently size returns ``False`` (unload),
    which is the historical always-unload behavior.
    """
    try:
        from utils.hardware import (
            DeviceType,
            auto_select_gpu_ids,
            estimate_required_model_memory_gb,
            get_device,
            get_visible_gpu_utilization,
            resolve_requested_gpu_ids,
        )

        if get_device() != DeviceType.CUDA:
            return False, {"mode": "non_cuda", "reason": "non_cuda"}

        # Full finetuning always runs in 16-bit (mirrors training.py:242-243), so
        # the estimate must ignore the 4-bit request or it under-counts VRAM.
        effective_4bit = False if training_type == "Full Finetuning" else load_in_4bit
        hf_token_arg = hf_token or None

        est_kwargs = dict(
            hf_token = hf_token_arg,
            training_type = training_type,
            load_in_4bit = effective_4bit,
            batch_size = batch_size,
            max_seq_length = max_seq_length,
            lora_rank = lora_rank,
            target_modules = target_modules,
            gradient_checkpointing = gradient_checkpointing,
            optimizer = optimizer,
        )

        if gpu_ids:
            # Explicit GPUs: the selector does no VRAM math (it just shards over
            # the requested set), so size it here against those GPUs' free VRAM.
            required_gb, _ = estimate_required_model_memory_gb(model_name, **est_kwargs)
            if required_gb is None:
                return False, {"mode": "explicit", "reason": "estimate_unavailable"}

            resolved = resolve_requested_gpu_ids(gpu_ids)
            devices = get_visible_gpu_utilization().get("devices", [])
            free_by_index: Dict[int, float] = {}
            for device in devices:
                total_gb = device.get("vram_total_gb")
                used_gb = device.get("vram_used_gb")
                if total_gb is None or used_gb is None:
                    continue
                free_by_index[device["index"]] = max(total_gb - used_gb, 0.0)

            # A requested GPU missing from the device list contributes 0.
            frees = sorted((free_by_index.get(i, 0.0) for i in resolved), reverse = True)
            usable_gb = frees[0] + sum(f * _MULTI_GPU_OVERHEAD for f in frees[1:]) if frees else 0.0
            keep = usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
            return keep, {
                "mode": "explicit",
                "required_gb": required_gb,
                "usable_gb": round(usable_gb, 3),
            }

        # Auto: this is the identical call start_training makes later. Reuse its
        # metadata (already encodes per-GPU floors + multi-GPU overhead).
        _selected, meta = auto_select_gpu_ids(model_name, **est_kwargs)
        mode = meta.get("selection_mode")
        required_gb = meta.get("required_gb")
        usable_gb = meta.get("usable_gb")
        keep = (
            mode == "auto"
            and required_gb is not None
            and usable_gb is not None
            and usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
        )
        return keep, {
            "mode": mode,
            "required_gb": required_gb,
            "usable_gb": usable_gb,
        }
    except Exception as e:
        # Never let a sizing failure keep a chat model loaded into a training OOM.
        logger.warning("Chat-coexistence probe failed; will unload: %s", e)
        return False, {"reason": "probe_error", "error": str(e)}


def free_chat_models_for_training(reason: str) -> List[str]:
    """Unload every resident chat/inference model to free GPU memory for training.

    Covers the HF/transformers + MLX orchestrator subprocess and the llama.cpp
    GGUF server. Each backend is isolated so one failing does not block the
    other. Returns labels of what was freed.
    """
    freed: List[str] = []

    try:
        from core.inference import get_inference_backend
        inf = get_inference_backend()
        if inf.active_model_name or inf.loading_models:
            name = inf.active_model_name or next(iter(inf.loading_models), None)
            logger.info(
                "Unloading inference model '%s' to free GPU memory for training (%s)",
                name,
                reason,
            )
            inf._shutdown_subprocess()
            inf.active_model_name = None
            inf.models.clear()
            inf.loading_models.clear()
            freed.append(f"hf:{name}")
    except Exception as e:
        logger.warning("Could not unload inference model: %s", e)

    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        if llama.is_active:
            name = llama.model_identifier or "gguf"
            logger.info(
                "Unloading GGUF chat model '%s' to free GPU memory for training (%s)",
                name,
                reason,
            )
            llama.unload_model()
            freed.append(f"gguf:{name}")
    except Exception as e:
        logger.warning("Could not unload GGUF chat model: %s", e)

    return freed
