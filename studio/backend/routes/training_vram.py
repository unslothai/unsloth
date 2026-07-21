# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VRAM coordination between chat/inference and training.

Decides, from live free VRAM, whether a resident chat model can stay loaded
during training or must be unloaded, and unloads it across all backends
(HF/MLX orchestrator + llama.cpp GGUF server). In the route layer because the
GGUF accessor lives in routes/inference.py; backends are imported lazily.
"""

from typing import Any, Dict, List, Optional, Tuple

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
    gpu_ids_are_vulkan_ordinals: bool = False,
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
        elif requested_gpu_ids and gpu_ids_are_vulkan_ordinals:
            # Vulkan ordinals enumerate independently of the CUDA index space free_by_index
            # uses (the loader skips resolve_requested_gpu_ids for them), so the target
            # physical card is unknown. Require a fit on the least-free visible GPU so no
            # ordinal->physical mapping can approve a load that then OOMs training (#7188).
            free_vals = list(free_by_index.values())
            if not free_vals:
                return False, {"mode": "gguf_vulkan", "reason": "no_visible_gpus"}
            needed_gb = required_gb * SAFETY_MARGIN + KEEP_FLOOR_GB
            # A multi-GPU pin shards the model (~needed/N per device), so require each
            # visible GPU to hold one shard, not the whole model. With the mapping unknown,
            # min_free is the safe bound: if the least-free card holds a shard, any mapping
            # does. Also apply the aggregate multi-GPU overhead so a sharded load does not
            # start with too little protected headroom and OOM the training run (#7188).
            per_gpu_needed_gb = needed_gb / len(requested_gpu_ids)
            min_free_gb = min(free_vals)
            # The pin lands on some N-of-visible subset (mapping unknown), so budget the
            # aggregate over the least-free N cards, not all visible: the N smallest minimize
            # usable_gb = max + overhead*rest, so if they fit, any N-subset does. Using all
            # visible would over-count headroom the pinned cards may not have (e.g. four 50 GB
            # cards make a two-card 100 GB pin look like it fits when no pair holds it).
            n_pins = min(len(requested_gpu_ids), len(free_vals))
            ranked = sorted(sorted(free_vals)[:n_pins], reverse = True)
            usable_gb = ranked[0] + sum(f * _MULTI_GPU_OVERHEAD for f in ranked[1:])
            aggregate_fits = usable_gb >= needed_gb
            per_gpu_fits = min_free_gb >= per_gpu_needed_gb
            return per_gpu_fits and aggregate_fits, {
                "mode": "gguf_vulkan",
                "required_gb": round(required_gb, 3),
                "needed_gb": round(needed_gb, 3),
                "usable_gb": round(usable_gb, 3),
                "per_gpu_needed_gb": round(per_gpu_needed_gb, 3),
                "min_free_gb": round(min_free_gb, 3),
            }
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
