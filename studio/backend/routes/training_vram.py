# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory coordination between inference and training.

Uses live free VRAM to keep resident chat and STT models when they fit. STT is
evicted before chat when training needs memory.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

# keep iff usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB. Conservative:
# the probe sees only the chat model's current footprint, so reserve headroom for
# estimate error + KV-cache growth (KEEP_FLOOR_GB ~= 2 GB load buffer + 2 GB chat).
SAFETY_MARGIN = 1.15
KEEP_FLOOR_GB = 4.0

# Each extra GPU contributes less than its raw free memory (sharding overhead).
_MULTI_GPU_OVERHEAD = 0.85


def _free_vram_by_index(devices: List[Dict[str, Any]]) -> Dict[int, float]:
    """Map GPU index -> free VRAM (GB) from a get_visible_gpu_utilization() device list."""
    free_by_index: Dict[int, float] = {}
    for device in devices:
        total_gb = device.get("vram_total_gb")
        used_gb = device.get("vram_used_gb")
        if total_gb is None or used_gb is None:
            continue
        free_by_index[device["index"]] = max(total_gb - used_gb, 0.0)
    return free_by_index


def summarize_resident_chat() -> Dict[str, Any]:
    """Report which chat models hold GPU memory (resident even while loading). Never raises."""
    hf_name: Optional[str] = None
    gguf_name: Optional[str] = None
    loading: bool = False

    try:
        from core.inference import get_inference_backend
        inf = get_inference_backend()
        # active_model_name is set only on success; a mid-load model sits in
        # loading_models while already holding VRAM -> both count as resident.
        if inf.active_model_name or inf.loading_models:
            hf_name = inf.active_model_name or next(iter(inf.loading_models), None)
            # Any in-flight load (incl. a replacement while the old model is still
            # active) can't be sized -> flag it so the caller frees instead of keeps.
            if inf.loading_models:
                loading = True
    except Exception as e:
        logger.warning("Could not inspect inference backend: %s", e)

    try:
        from routes.inference import get_llama_cpp_backend
        llama = get_llama_cpp_backend()
        # is_active (not is_loaded): a mid-start server already allocates VRAM.
        # A confirmed CPU-only server (_gpu_offload_active is False) holds no VRAM.
        if llama.is_active and getattr(llama, "_gpu_offload_active", None) is not False:
            gguf_name = llama.model_identifier or "gguf"
            if not getattr(llama, "is_loaded", False):  # still loading -> size unknown
                loading = True
    except Exception as e:
        logger.warning("Could not inspect GGUF backend: %s", e)

    return {
        "hf": hf_name,
        "gguf": gguf_name,
        "loading": loading,
        "any": bool(hf_name or gguf_name),
    }


def summarize_resident_stt() -> Dict[str, Any]:
    """Report the resident dictation model (either engine). Never raises."""
    try:
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        from core.inference.stt_sidecar import get_stt_sidecar

        sidecar = get_stt_sidecar()
        model = sidecar.loaded_model
        device = sidecar.device
        loading = sidecar.is_loading()
        # The whisper.cpp engine holds GPU memory through its own subprocess, and
        # both engines can be live at once (an engine switch or direct
        # /audio/stt/load calls). Always fold the GGUF sidecar in -- a resident
        # Transformers model (e.g. after its CPU fallback) must not mask a GGUF
        # server still binding its accelerator backend, or admission would let
        # training launch into that startup and OOM.
        ggml = get_ggml_stt_sidecar()
        if not model:
            model = ggml.loaded_model
            device = device or ggml.device
        loading = loading or ggml.is_loading()
        return {
            "model": model,
            "device": device,
            "loading": loading,
            "any": bool(model or loading),
        }
    except Exception as e:
        logger.warning("Could not inspect STT sidecar: %s", e)
        return {"model": None, "device": None, "loading": False, "any": False}


def can_keep_chat_during_training(
    *,
    model_name: str,
    hf_token: Optional[str],
    training_type: str,
    load_in_4bit: bool,
    batch_size: int,
    max_seq_length: int,
    lora_rank: int,
    target_modules: Optional[List[str]],
    gradient_checkpointing: str,
    optimizer: str,
    gpu_ids: Optional[List[int]],
) -> Tuple[bool, Dict[str, Any]]:
    """Decide if a resident chat model can coexist with training given free VRAM.

    Reuses training's own estimator/selector so the decision matches later
    placement. Default-deny: anything we can't size returns False (unload).
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

        # Full finetuning runs in 16-bit, so ignore the 4-bit request or we under-count.
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
            # Explicit GPUs: the selector does no VRAM math, so size it here.
            try:
                resolved = resolve_requested_gpu_ids(gpu_ids)
            except ValueError:
                # Invalid ids -> start_training will 400 first, so don't unload.
                return True, {"mode": "explicit", "reason": "invalid_gpu_ids"}

            required_gb, est_meta = estimate_required_model_memory_gb(model_name, **est_kwargs)
            if required_gb is None:
                return False, {"mode": "explicit", "reason": "estimate_unavailable"}

            free_by_index = _free_vram_by_index(get_visible_gpu_utilization().get("devices", []))

            # A requested GPU missing from the device list contributes 0.
            free_vals = [free_by_index.get(i, 0.0) for i in resolved]
            ranked = sorted(free_vals, reverse = True)
            usable_gb = (
                ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:]) if ranked else 0.0
            )
            aggregate_fits = usable_gb >= required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB

            # Activations don't shard: enforce a per-GPU floor so an uneven split
            # (e.g. free [45, 10]) can't be kept into an OOM the aggregate misses.
            per_gpu_fits = True
            min_free_gb = min(free_vals) if free_vals else 0.0
            if len(resolved) > 1:
                min_per_gpu_gb = est_meta.get("vram_breakdown", {}).get(
                    f"min_per_gpu_{len(resolved)}"
                )
                if min_per_gpu_gb is not None:
                    per_gpu_fits = min_free_gb >= min_per_gpu_gb

            keep = aggregate_fits and per_gpu_fits
            return keep, {
                "mode": "explicit",
                "required_gb": required_gb,
                "usable_gb": round(usable_gb, 3),
                "min_free_gb": round(min_free_gb, 3),
            }

        # Auto: same call start_training makes later; reuse its sizing metadata.
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


def can_load_chat_during_training(
    *,
    model_name: str,
    hf_token: Optional[str],
    load_in_4bit: bool,
    max_seq_length: int,
    requested_gpu_ids: Optional[List[int]],
    is_gguf: bool = False,
    required_override_gb: Optional[float] = None,
    single_device_gpu: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide if a NEW chat model can load without OOMing active training (inverse
    of can_keep_chat_during_training: training is already resident, so size the
    chat model against the free VRAM that remains). Sizes/places it the same way
    the loader will: HF auto reuses auto_select_gpu_ids; HF explicit requires an
    even-share per-GPU floor for device_map="balanced"; GGUF sizes from
    required_override_gb over the visible pool. ``single_device_gpu`` is the
    exact physical device token selected by a single-device runner.
    `load_in_4bit` must be effective (LoRA can flip 4-bit -> 16-bit). Non-CUDA
    allows the load; default-deny on any CUDA case it can't size, so a load never
    OOMs training."""
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
            return True, {"mode": "non_cuda", "reason": "non_cuda"}

        est_kwargs = dict(
            hf_token = hf_token or None,
            training_type = None,  # inference sizing of the chat model itself
            load_in_4bit = load_in_4bit,
            max_seq_length = max_seq_length or 2048,
        )

        # HF auto: reuse the loader's selector; fits iff its pick clears the margin.
        if not requested_gpu_ids and not is_gguf:
            _selected, meta = auto_select_gpu_ids(model_name, **est_kwargs)
            mode = meta.get("selection_mode")
            required_gb = meta.get("required_gb")
            usable_gb = meta.get("usable_gb")
            needed_gb = (
                round(required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB, 3)
                if required_gb is not None
                else None
            )
            fits = (
                mode == "auto"
                and required_gb is not None
                and usable_gb is not None
                and usable_gb >= needed_gb
            )
            return fits, {
                "mode": mode,
                "required_gb": required_gb,
                "usable_gb": usable_gb,
                "needed_gb": needed_gb,
            }

        # Explicit GPUs, or GGUF: size directly and check live free VRAM.
        if single_device_gpu is not None:
            mode = "single_device"
        elif is_gguf:
            mode = "gguf"
        else:
            mode = "explicit"
        required_gb = required_override_gb
        if required_gb is None:
            required_gb, _meta = estimate_required_model_memory_gb(model_name, **est_kwargs)
        if required_gb is None:
            return False, {"mode": mode, "reason": "estimate_unavailable"}

        free_by_index = _free_vram_by_index(get_visible_gpu_utilization().get("devices", []))
        if single_device_gpu is not None:
            token = str(single_device_gpu).strip()
            if not token:
                # Empty token = a CPU-only single-device runner (e.g. a CPU
                # diffusion GGUF): it uses no GPU VRAM, so it never threatens
                # active training and can always load.
                return True, {"mode": "single_device", "reason": "cpu_only"}
            try:
                selected_gpu = int(token)
                if selected_gpu < 0:
                    raise ValueError
            except (TypeError, ValueError):
                # A non-numeric device token (e.g. a CUDA UUID / MIG handle)
                # can't be mapped to a free-VRAM index, but the runner still
                # drives ONE device. Size against the worst-case visible device
                # (min free), never the aggregate pool, so a single-device load
                # is never OK'd on capacity it can't use and OOMs training.
                free_vals = [min(free_by_index.values())] if free_by_index else []
            else:
                free_vals = [free_by_index.get(selected_gpu, 0.0)]
        elif requested_gpu_ids:
            # Invalid ids -> load_model 400s first, so don't block; missing id = 0.
            try:
                resolved = resolve_requested_gpu_ids(requested_gpu_ids)
            except ValueError:
                return True, {"mode": mode, "reason": "invalid_gpu_ids"}
            free_vals = [free_by_index.get(i, 0.0) for i in resolved]
        else:
            # GGUF: llama.cpp picks the GPU(s); any visible GPU is a candidate.
            free_vals = list(free_by_index.values())

        if not free_vals:
            return False, {"mode": mode, "reason": "no_visible_gpus"}

        ranked = sorted(free_vals, reverse = True)
        usable_gb = ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:])
        needed_gb = required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
        aggregate_fits = usable_gb >= needed_gb

        # device_map="balanced" shards across GPUs: an even-share floor stops one
        # near-full GPU hiding behind aggregate capacity. GGUF self-places, no floor.
        min_free_gb = min(free_vals)
        per_gpu_fits = True
        if mode == "explicit" and len(free_vals) > 1:
            per_gpu_fits = min_free_gb >= needed_gb / len(free_vals)

        return aggregate_fits and per_gpu_fits, {
            "mode": mode,
            "required_gb": round(required_gb, 3),
            "usable_gb": round(usable_gb, 3),
            "needed_gb": round(needed_gb, 3),
            "min_free_gb": round(min_free_gb, 3),
        }
    except Exception as e:
        # Never let a sizing failure load a chat model into a training OOM.
        logger.warning("Chat-load coexistence probe failed; will refuse: %s", e)
        return False, {"reason": "probe_error", "error": str(e)}


def free_chat_models_for_training(reason: str) -> List[str]:
    """Unload every resident chat model (HF/MLX orchestrator + GGUF server) to free
    VRAM for training. Each backend isolated. Returns labels of what was freed."""
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
        # CPU-only GGUF holds no VRAM, so killing it can't help (see summarize).
        if llama.is_active and getattr(llama, "_gpu_offload_active", None) is not False:
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


def free_stt_model_for_training(reason: str) -> List[str]:
    """Unload the dictation model(s) before training. Never raises.

    The Transformers and GGUF sidecars are freed under independent exception
    boundaries so a failure unloading one backend never skips freeing the other
    (both can hold accelerator memory at once after an engine switch).
    """
    freed: List[str] = []
    try:
        from core.inference.stt_sidecar import get_stt_sidecar
        sidecar = get_stt_sidecar()
        if sidecar.is_loading() and sidecar.cancel_pending_load():
            logger.info("Cancelling STT model load for training (%s)", reason)
            # The loader may still be inside from_pretrained()/.to(device) and
            # holding VRAM; wait for it to observe the cancel and release before
            # training claims the memory.
            sidecar.wait_for_load_to_settle()
            # A load that finished before observing the cancel leaves a resident
            # model; unload it so training actually gets the memory back.
            if sidecar.loaded_model:
                sidecar.unload()
            freed.append("stt:loading")
        else:
            model = sidecar.loaded_model
            if model:
                logger.info("Unloading STT model '%s' for training (%s)", model, reason)
                sidecar.unload()
                freed.append(f"stt:{model}")
    except Exception as e:
        logger.warning("Could not unload Transformers STT model: %s", e)

    # Check the GGUF sidecar even after a cancelled/failed Transformers unload;
    # both engines can hold memory at once (engine switch or direct load calls).
    try:
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        ggml = get_ggml_stt_sidecar()
        if ggml.is_loading() and ggml.cancel_pending_load():
            logger.info("Cancelling GGUF STT model load for training (%s)", reason)
            # whisper-server may still be binding its accelerator backend; wait
            # for the cancelled startup to be killed and reaped before training
            # claims the memory (its loaded_model stays unset until it is ready).
            ggml.wait_for_load_to_settle()
            if ggml.loaded_model:
                ggml.unload()
            freed.append("stt:gguf-loading")
        else:
            ggml_model = ggml.loaded_model
            if ggml_model:
                logger.info("Unloading GGUF STT model '%s' for training (%s)", ggml_model, reason)
                ggml.unload()
                freed.append(f"stt:{ggml_model}")
    except Exception as e:
        logger.warning("Could not unload GGUF STT model: %s", e)

    return freed


def coordinate_models_for_training(
    can_keep: Callable[[], Tuple[bool, Dict[str, Any]]],
) -> List[str]:
    """Keep resident models when they fit, evicting STT before chat."""
    resident_chat = summarize_resident_chat()
    resident_stt = summarize_resident_stt()
    if not resident_chat["any"] and not resident_stt["any"]:
        return []

    if resident_chat.get("loading"):
        freed = free_stt_model_for_training(reason = "chat model still loading")
        freed += free_chat_models_for_training(reason = "chat model still loading")
        return freed

    freed: List[str] = []
    if resident_stt.get("loading"):
        released_stt = free_stt_model_for_training(reason = "STT model still loading")
        freed += released_stt
        resident_stt = (
            {"model": None, "device": None, "loading": False, "any": False}
            if released_stt
            else summarize_resident_stt()
        )
        if not resident_chat["any"] and not resident_stt["any"]:
            return freed

    keep, info = can_keep()
    if keep:
        logger.info(
            "Keeping resident models loaded during training (free ~%s GB, needs ~%s GB): %s",
            info.get("usable_gb"),
            info.get("required_gb"),
            {"chat": resident_chat, "stt": resident_stt},
        )
        return freed

    if resident_stt["any"]:
        freed += free_stt_model_for_training(reason = "insufficient training memory")
        if not resident_chat["any"]:
            return freed
        keep, _info = can_keep()
        if keep:
            logger.info("Keeping chat model loaded after freeing STT: %s", resident_chat)
            return freed

    freed += free_chat_models_for_training(
        reason = "insufficient VRAM to run training alongside chat",
    )
    return freed
