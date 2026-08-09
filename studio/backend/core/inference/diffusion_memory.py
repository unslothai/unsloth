# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured-budget memory policy for the local diffusion backend.

From the resolved device target, a free-memory snapshot, and a coarse model footprint
estimate, this picks a CPU-offload policy and VAE slice/tile settings, then applies them to a
built diffusers pipeline. A model that won't fit resident is kept running by streaming weights
through the GPU one module at a time, which is lossless (offload / VAE slicing change placement
and decode chunking, not numerics).

The choice is coarse (sizes the model, not every activation), so ``auto`` is best-effort and
the explicit ``fast`` / ``balanced`` / ``low_vram`` modes are a hard override. torch / psutil
imported lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


# ── memory modes (operator intent) ───────────────────────────────────────────
MEMORY_MODE_AUTO = "auto"
MEMORY_MODE_FAST = "fast"
MEMORY_MODE_BALANCED = "balanced"
MEMORY_MODE_LOW_VRAM = "low_vram"
MEMORY_MODES = (
    MEMORY_MODE_AUTO,
    MEMORY_MODE_FAST,
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
)

# none   -- all weights resident (fastest; fits only with room).
# model  -- enable_model_cpu_offload(): one top-level module on the GPU at a time.
# group  -- apply_group_offloading() on the transformer: stream a few blocks at a time with a prefetch stream.
# sequential -- enable_sequential_cpu_offload(): submodule-level (broken for GGUF on diffusers 0.38, kept as an escape hatch).
OFFLOAD_NONE = "none"
OFFLOAD_MODEL = "model"
OFFLOAD_GROUP = "group"
OFFLOAD_SEQUENTIAL = "sequential"

# Transformer blocks resident per group under group offloading: fewer = lower VRAM, more host-to-device traffic.
DEFAULT_GROUP_BLOCKS = 1

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
# Flat allowance for fixed pipeline costs (scheduler, embeddings, CUDA context, fragmentation).
DEFAULT_BASE_OVERHEAD_MIB = 2048


def normalize_memory_mode(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested mode (accepting dashes); None passes through. Raises ValueError
    for an unsupported mode so the route rejects it as a 4xx before any GPU work."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return None
    if normalized not in MEMORY_MODES:
        valid = ", ".join(MEMORY_MODES)
        raise ValueError(f"Unsupported diffusion memory_mode '{value}'. Use one of: {valid}.")
    return normalized


@dataclass(frozen = True)
class DeviceMemory:
    """Point-in-time view of the active device's memory, in MiB.

    ``memory_kind`` distinguishes discrete VRAM (CPU offload helps) from unified / system memory
    (offload moves bytes within the same pool, so it does not)."""

    backend: str
    device: str
    memory_kind: str  # "discrete_vram" | "unified_memory" | "system_memory" | "unknown"
    free_mib: Optional[int] = None
    total_mib: Optional[int] = None

    @property
    def is_unified(self) -> bool:
        return self.memory_kind in ("unified_memory", "system_memory")

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "memory_kind": self.memory_kind,
            "free_mib": self.free_mib,
            "total_mib": self.total_mib,
        }


@dataclass(frozen = True)
class MemoryPlan:
    """The chosen runtime profile for one load."""

    requested_mode: str
    offload_policy: str
    vae_tiling: bool
    vae_slicing: bool
    device_memory: DeviceMemory
    estimates: dict[str, Optional[int]]
    reasons: tuple[str, ...] = ()
    # Under group offload, stream the TEXT ENCODERS alongside the transformer instead of keeping
    # them resident. Defaulted so every existing construction is unchanged; set only where that
    # is what makes group offload fit at all (see plan_diffusion_memory).
    stream_text_encoders: bool = False

    @property
    def engages_offload(self) -> bool:
        return self.offload_policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL)

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "offload_policy": self.offload_policy,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "device_memory": self.device_memory.as_public_dict(),
            "estimates": dict(self.estimates),
            "reasons": list(self.reasons),
            "stream_text_encoders": self.stream_text_encoders,
        }


# ── hardware snapshot ─────────────────────────────────────────────────────────


def snapshot_device_memory(target: Any) -> DeviceMemory:
    """Free / total memory for ``target``'s device. Never raises: a probe failure yields None
    counts, which the planner treats as "budget unknown" (stay resident)."""
    device = getattr(target, "device", "cpu")
    backend = getattr(target, "backend", device)

    if device == "cuda":
        free, total, kind = _cuda_memory(backend)
        return DeviceMemory(backend, device, kind, free, total)
    if device == "xpu":
        free, total = _xpu_memory()
        return DeviceMemory(backend, device, "discrete_vram", free, total)
    if device == "mps":
        # Apple Silicon shares one CPU/GPU pool: system memory is the budget, offload pointless.
        total, free = _system_memory_mib()
        return DeviceMemory(backend, device, "unified_memory", free, total)

    total, free = _system_memory_mib()
    return DeviceMemory(backend, device, "system_memory", free, total)


def settled_snapshot_device_memory(
    target: Any,
    attempts: int = 3,
    delay_s: float = 1.0,
) -> DeviceMemory:
    """``snapshot_device_memory`` hardened against TRANSIENT free-VRAM undercounts on cuda.

    ``torch.cuda.mem_get_info`` is device-wide and instantaneous: a neighbouring process (or a
    just-spawned subprocess context) briefly holding tens of GB at the wrong moment makes an
    empty card look full, and the planner then silently declines the resident/quant fast path
    (measured on B200: a cold FLUX.2-dev int8 load saw free < 74 GB on an idle 183 GB card and
    fell back to offloaded GGUF; the identical retry saw >= 124 GB and went resident). Settle
    the allocator (synchronize + empty_cache, best-effort) and take the MAX free over a few
    spaced reads: a transient can only SHRINK free, so the max rejects transient undercounts
    while a persistent tenant still caps every read. Non-cuda targets keep the single read."""
    if getattr(target, "device", "cpu") != "cuda":
        return snapshot_device_memory(target)
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — settle is best-effort; the snapshot below still runs
        pass
    best = snapshot_device_memory(target)
    for _ in range(max(0, attempts - 1)):
        if best.free_mib is not None and best.total_mib is not None:
            # Free already within the reserve of total: nothing transient to wait out.
            if best.free_mib >= best.total_mib - max(2048, int(best.total_mib * 0.10)):
                break
        try:
            import time
            time.sleep(delay_s)
        except Exception:  # noqa: BLE001
            break
        nxt = snapshot_device_memory(target)
        if nxt.free_mib is not None and (best.free_mib is None or nxt.free_mib > best.free_mib):
            best = nxt
    return best


def _cuda_memory(backend: str) -> tuple[Optional[int], Optional[int], str]:
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        kind = "discrete_vram"
        try:
            # Query the CURRENT device (mem_get_info reports it); hardcoding 0 would inspect the wrong GPU and misclassify it.
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if bool(getattr(props, "integrated", False) or getattr(props, "is_integrated", False)):
                kind = "unified_memory"  # e.g. Jetson / integrated SoC
        except Exception:
            pass
        return int(free // (1024 * 1024)), int(total // (1024 * 1024)), kind
    except Exception:
        return None, None, "discrete_vram"


def _xpu_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import torch
        mem_get_info = getattr(getattr(torch, "xpu", None), "mem_get_info", None)
        if callable(mem_get_info):
            free, total = mem_get_info()
            return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:
        pass
    return None, None


def _system_memory_mib() -> tuple[Optional[int], Optional[int]]:
    """(total, available) host RAM in MiB, via psutil then POSIX sysconf."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        avail = os.sysconf("SC_AVPHYS_PAGES") * page
        return int(total // (1024 * 1024)), int(avail // (1024 * 1024))
    except Exception:
        return None, None


# ── size estimates ────────────────────────────────────────────────────────────


def file_size_mib(path: Any) -> Optional[int]:
    """On-disk size of ``path`` in MiB, or None if it can't be stat'd."""
    try:
        from pathlib import Path
        return max(1, int(Path(path).expanduser().stat().st_size // (1024 * 1024)))
    except Exception:
        return None


def estimate_gguf_resident_mib(storage_mib: Optional[int]) -> Optional[int]:
    """Approximate the RESIDENT device size of a GGUF transformer under ``GGUFQuantizationConfig``.

    Weights stay PACKED as quantised bytes; ``GGUFLinear.forward`` dequantises each transiently
    for its matmul and frees it, so the persistent footprint is ~= on-disk size, not unpacked
    bf16. Measured on Z-Image-Turbo: Q2_K 3.64 -> 3.68 GiB, Q8_0 7.22 -> 7.25 GiB resident. The
    transient dequant is covered by the separate runtime headroom. (The prior per-quant expansion
    assumed a full unpack that never happens, over-estimating Q2 ~7.6x and forcing needless offload.)"""
    if storage_mib is None:
        return None
    return int(storage_mib * 1.05)  # small margin for allocator + bf16 norms/biases


def estimate_safetensors_dense_mib(
    storage_mib: Optional[int], *, fp8_upcast: bool = False
) -> Optional[int]:
    """Resident size of a safetensors checkpoint, in MiB.

    Usually loads near on-disk size (None passes through). Exception: ``fp8_upcast`` -- an fp8
    single-file transformer loads with no quantization_config, so diffusers upcasts to bf16 (~2x)."""
    if storage_mib is None:
        return None
    if fp8_upcast:
        return storage_mib * 2
    return storage_mib


def estimate_image_runtime_mib(
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
) -> int:
    """Per-call activation / latent headroom for an image gen, scaled by pixel area and batch.
    Distilled / turbo models (few steps, no CFG) need less."""
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    pixel_scale = (w * h * batch) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    fam = (family or "").lower()
    multiplier = 1.0
    if "edit" in fam:
        multiplier *= 1.35
    if "turbo" in fam or "distilled" in fam or "schnell" in fam:
        multiplier *= 0.85
    return max(1024, int(8192 * max(0.25, pixel_scale) * multiplier))


def estimate_video_runtime_mib(
    *, width: Optional[int], height: Optional[int], num_frames: Optional[int]
) -> int:
    """Per-call activation / latent / decode headroom for a video generation.

    The pixel-area image estimator undershoots video: the VAE DECODE is the peak -- the clip
    materialises as num_frames full-res fp32 frames plus decoder intermediates. Scale by the
    decoded-clip footprint (frames x H x W x 3 x 4 bytes) with a 3x factor for intermediates +
    the export copy, on top of a fixed denoise-side base.
    """
    w = max(64, int(width or 768))
    h = max(64, int(height or 512))
    frames = max(1, int(num_frames or 121))
    decoded_mib = (frames * w * h * 3 * 4) / float(1024 * 1024)
    return max(3072, int(4096 + 3.0 * decoded_mib))


def _reserve_mib(memory_kind: str, base: int) -> int:
    if memory_kind == "unified_memory":
        return max(2048, int(base * 0.20))  # OS + CPU share this pool
    if memory_kind == "system_memory":
        return max(1024, int(base * 0.10))
    return max(2048, int(base * 0.10))


def _safe_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus a headroom reserve (room for fragmentation + other tenants). None when
    free memory is unknown."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    return max(0, int(memory.free_mib) - _reserve_mib(memory.memory_kind, base))


def plan_fits_total_capacity(plan: Any) -> bool:
    """Whether ``plan``'s resident requirement fits TOTAL device capacity under the standard
    reserve + the 0.85 resident margin -- i.e. an offload decision can only stem from the
    instantaneous FREE reading (something else held VRAM at snapshot time), never from the
    device being too small. Used to retry a declined resident/quant plan once with a fresh
    settled snapshot instead of trusting a single transient undercount. False on any missing
    input (unknown sizes keep today's behaviour)."""
    try:
        required = plan.estimates.get("resident_required_mib")
        memory = plan.device_memory
        total = memory.total_mib
        kind = memory.memory_kind
    except Exception:  # noqa: BLE001 — malformed plan: no retry
        return False
    if required is None or total is None:
        return False
    return int(required) <= int((int(total) - _reserve_mib(kind, int(total))) * 0.85)


def _sum_required(*values: Optional[int]) -> Optional[int]:
    total = 0
    for value in values:
        if value is None:
            return None
        total += int(value)
    return total


# ── the planner ───────────────────────────────────────────────────────────────


def plan_diffusion_memory(
    *,
    target: Any,
    device_memory: DeviceMemory,
    model_dense_mib: Optional[int],
    runtime_headroom_mib: int,
    companion_dense_mib: Optional[int] = None,
    text_encoder_dense_mib: Optional[int] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    requested_mode: Optional[str] = None,
    explicit_offload: bool = False,
) -> MemoryPlan:
    """Pick an offload policy + VAE memory savers for the current load.

    ``model_dense_mib`` is the resident size of all weights; ``companion_dense_mib`` is just the
    companions, which stay resident under group offload while the transformer streams block by
    block. ``text_encoder_dense_mib`` is the TEXT-ENCODER share of that companion total, which
    unlocks a second group tier (below); None means "no split available" and reproduces the
    pre-split decision exactly. ``explicit_offload`` is the back-compat ``cpu_offload=True``
    request (forces model offload).

    Policies by speed/VRAM tradeoff:
      none  - everything resident: fastest, highest VRAM.
      group - stream the transformer, companions resident: near-resident speed, moderate cut.
      group + streamed text encoders - as above, but the encoders stream too: they run ONCE,
              before step 0, so this costs one extra host-to-device pass per call rather than a
              per-step one, and it frees their bytes for every denoising step.
      model - offload every component: lowest VRAM, slow.
    """
    mode = normalize_memory_mode(requested_mode) or MEMORY_MODE_AUTO
    can_offload = bool(getattr(target, "supports_model_cpu_offload", False))
    budget = _safe_device_budget_mib(device_memory)
    required = _sum_required(model_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # The resident floor under group offload: companions stay, the transformer streams.
    group_floor = _sum_required(companion_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # A SECOND floor, for the same tier with the text encoders streamed as well. The encoders are
    # the largest companion on most families (Z-Image: 8.0 of 8.2 GB) and they are used exactly
    # once, before step 0, so holding them resident for the whole denoise reserves their bytes for
    # nothing. Streaming them leaves the VAE as the only resident companion. Computed only when
    # BOTH terms are known: an unknown split must reproduce the previous decision, never guess a
    # smaller floor. Clamped at 0 because the two terms can come from different sources.
    group_floor_streamed_te = (
        _sum_required(
            max(0, int(companion_dense_mib) - int(text_encoder_dense_mib)),
            runtime_headroom_mib,
            base_overhead_mib,
        )
        if companion_dense_mib is not None and text_encoder_dense_mib is not None
        else None
    )
    reasons: list[str] = []
    stream_text_encoders = False
    estimates: dict[str, Optional[int]] = {
        "safe_device_budget_mib": budget,
        "model_dense_mib": model_dense_mib,
        "companion_dense_mib": companion_dense_mib,
        "text_encoder_dense_mib": text_encoder_dense_mib,
        "runtime_headroom_mib": runtime_headroom_mib,
        "base_overhead_mib": base_overhead_mib,
        "resident_required_mib": required,
        "group_floor_mib": group_floor,
        "group_floor_streamed_te_mib": group_floor_streamed_te,
    }

    def _group_fits() -> bool:
        # Group offload only helps if the resident companions fit; a too-big text encoder needs whole-module offload.
        return group_floor is not None and budget is not None and group_floor <= budget

    def _group_fits_streamed_te() -> bool:
        # The same tier once the text encoders stream too; only reachable with a known split.
        return (
            group_floor_streamed_te is not None
            and budget is not None
            and group_floor_streamed_te <= budget
        )

    # The best tier available when the weights do not fit resident, in speed order: plain group
    # (companions resident) beats group with streamed encoders (one extra host-to-device pass per
    # CALL) beats whole-module offload (every component paged per STEP -- the 48-minute case).
    def _offload_tier() -> tuple[str, bool]:
        if _group_fits():
            return OFFLOAD_GROUP, False
        if _group_fits_streamed_te():
            return OFFLOAD_GROUP, True
        return OFFLOAD_MODEL, False

    _STREAMED_TE_REASON = (
        "companions exceed budget, but they fit with the text encoders streamed too "
        "(they run once, before step 0); streaming them beats paging every component per step"
    )

    if not can_offload or device_memory.is_unified:
        # MPS / CPU cannot stream to a separate device; on unified memory offload just shuffles bytes within the same pool.
        policy = OFFLOAD_NONE
        if device_memory.is_unified:
            reasons.append("unified/system memory: CPU offload frees no device memory")
        else:
            reasons.append(f"{device_memory.backend}: CPU offload unavailable; staying resident")
    elif mode == MEMORY_MODE_FAST:
        policy = OFFLOAD_NONE
        if budget is not None and required is not None and required > budget:
            # Doesn't fit resident: streamed transformer is the fastest offload.
            policy, stream_text_encoders = _offload_tier()
            reasons.append("fast requested but weights do not fit resident; offloading")
            if stream_text_encoders:
                reasons.append(_STREAMED_TE_REASON)
        else:
            reasons.append("fast requested; weights resident on device")
    elif mode == MEMORY_MODE_BALANCED:
        policy = OFFLOAD_GROUP
        reasons.append("balanced requested; streamed block-level transformer offload")
    elif mode == MEMORY_MODE_LOW_VRAM:
        policy = OFFLOAD_MODEL
        reasons.append("low_vram requested; whole-module offload of every component")
    elif budget is None or required is None:
        policy = OFFLOAD_NONE
        reasons.append("device budget or model size unknown; staying resident")
    elif required <= int(budget * 0.85):
        policy = OFFLOAD_NONE
        reasons.append("weights fit resident with headroom")
    elif _group_fits():
        policy = OFFLOAD_GROUP
        reasons.append("tight fit; stream the transformer, companions resident")
    elif _group_fits_streamed_te():
        policy = OFFLOAD_GROUP
        stream_text_encoders = True
        reasons.append(_STREAMED_TE_REASON)
    else:
        policy = OFFLOAD_MODEL
        reasons.append("companions exceed budget; whole-module offload of every component")

    # The legacy cpu_offload flag applies only when no memory_mode was supplied, so an explicit `fast` stays resident.
    if (
        explicit_offload
        and normalize_memory_mode(requested_mode) is None
        and policy == OFFLOAD_NONE
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload overrides resident placement")

    # VAE savers cap the high-res decode spike. Slicing (one image at a time) is EXACT, so enable it on any offload tier. Tiling
    # is only bit-identical for a single tile (<=1MP), so restrict it to the lowest tiers. Group offload keeps the VAE resident.
    any_offload = policy != OFFLOAD_NONE or device_memory.backend in ("mps", "cpu")
    tile = policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL) or device_memory.backend in ("mps", "cpu")
    return MemoryPlan(
        requested_mode = mode,
        offload_policy = policy,
        vae_tiling = tile,
        vae_slicing = any_offload,
        device_memory = device_memory,
        estimates = estimates,
        reasons = tuple(reasons),
        # Only ever meaningful under group offload; every other tier already places the encoders.
        stream_text_encoders = stream_text_encoders and policy == OFFLOAD_GROUP,
    )


# ── apply to a built pipeline ─────────────────────────────────────────────────


def apply_memory_plan(
    pipe: Any,
    plan: MemoryPlan,
    *,
    device: str,
    logger: Any = None,
) -> tuple[str, bool]:
    """Apply ``plan`` to a built diffusers pipeline: enable the VAE savers then place / offload
    the weights. Exactly one placement call runs (fully resident or wired for offload, never both).

    Returns the ``(offload_policy, vae_tiling)`` ACTUALLY engaged, which can differ from the plan:
    tiling is a no-op where there's no tiling control, and group / sequential offload fall back to
    whole-module offload if unsupported (e.g. sequential is broken for GGUF on diffusers 0.38)."""
    tiling_engaged = False
    if plan.vae_tiling:
        tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)
    if plan.vae_slicing:
        _enable_vae_saver(pipe, "enable_vae_slicing", "enable_slicing", logger)

    def _fallback_to_model_offload() -> None:
        # The GROUP plan set vae_tiling=False (the VAE stays resident). Dropping to whole-module offload is the low-VRAM case where the decode spike can OOM, so turn tiling on now.
        nonlocal tiling_engaged
        pipe.enable_model_cpu_offload(device = device)
        if not tiling_engaged:
            tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)

    policy = plan.offload_policy
    if policy == OFFLOAD_MODEL:
        pipe.enable_model_cpu_offload(device = device)
    elif policy == OFFLOAD_GROUP:
        # getattr, not attribute access: manually built / duck-typed plans predate this field.
        if not _apply_group_offload(
            pipe,
            device,
            logger,
            stream_text_encoders = bool(getattr(plan, "stream_text_encoders", False)),
        ):
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    elif policy == OFFLOAD_SEQUENTIAL:
        try:
            pipe.enable_sequential_cpu_offload(device = device)
        except Exception as exc:  # noqa: BLE001 — keep the model loadable
            if logger is not None:
                logger.warning(
                    "diffusion.memory: sequential offload failed (%s); "
                    "falling back to whole-module offload",
                    exc,
                )
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    else:
        pipe.to(device)
    return policy, tiling_engaged


def _enable_vae_saver(pipe: Any, pipe_method: str, vae_method: str, logger: Any) -> bool:
    """Turn on a VAE memory saver, trying the pipeline shortcut first then the VAE submodule
    (some pipelines, e.g. Z-Image, only expose it on ``pipe.vae``). Returns whether it engaged."""
    for owner, method in ((pipe, pipe_method), (getattr(pipe, "vae", None), vae_method)):
        fn = getattr(owner, method, None)
        if not callable(fn):
            continue
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 — a VAE saver is an optimisation, never fatal
            if logger is not None:
                logger.warning("diffusion.memory: %s() failed: %s", method, exc)
    return False


def _apply_group_offload(
    pipe: Any,
    device: str,
    logger: Any,
    *,
    stream_text_encoders: bool = False,
) -> bool:
    """Stream the transformer a few blocks at a time via diffusers group offloading, keeping the
    smaller components resident. Returns False (caller falls back to whole-module) on any failure.

    ``stream_text_encoders`` extends the streamed set to every ``text_encoder*`` module. Off by
    default: keeping them resident is faster when there is room. The planner turns it on only
    where it is the difference between group offload and whole-module offload."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    installed = 0  # streamed modules that already carry group-offload hooks
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        # A dual-DiT pipeline (Ideogram 4) carries a second denoiser as large as the first, so stream every DiT and keep only smaller companions resident.
        streamed: dict[str, Any] = {"transformer": transformer}
        for extra in ("transformer_2", "unconditional_transformer"):
            module = getattr(pipe, extra, None)
            if isinstance(module, torch.nn.Module):
                streamed[extra] = module
        if stream_text_encoders:
            # A text encoder runs ONCE, before step 0, so residency buys it nothing while it costs
            # its bytes for every step of the denoise. Adding it to `streamed` does two things: the
            # resident loop below skips it (it is no longer placed with comp.to(onload)), and group
            # hooks page it in for that single encode. Component names, not attributes, so a family
            # with text_encoder / text_encoder_2 / text_encoder_3 is covered without a per-family list.
            for name, comp in getattr(pipe, "components", {}).items():
                if name.startswith("text_encoder") and isinstance(comp, torch.nn.Module):
                    streamed[name] = comp

        onload = torch.device(device)
        use_stream = onload.type == "cuda"  # overlap H2D copies with compute
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        # On the CUDA stream path, overlap each block's H2D copy with compute. Lossless, and gated on the signature so older diffusers still works.
        if use_stream:
            _params = inspect.signature(apply_group_offloading).parameters
            if "non_blocking" in _params:
                gkwargs["non_blocking"] = True
            if "record_stream" in _params:
                gkwargs["record_stream"] = True
        # Place the smaller components resident BEFORE attaching the transformer group-offload hooks: a companion .to() OOM then returns False with no hooks installed, and diffusers rejects enable_model_cpu_offload once group hooks exist.
        for name, comp in getattr(pipe, "components", {}).items():
            if name in streamed:
                continue
            if isinstance(comp, torch.nn.Module):
                comp.to(onload)
        for module in streamed.values():
            apply_group_offloading(module, **gkwargs)
            installed += 1
        return True
    except Exception as exc:  # noqa: BLE001 — fall back to whole-module offload
        if installed:
            # An earlier streamed module already has hooks but a later one failed: the pipe is in a PARTIAL group-offload state enable_model_cpu_offload rejects, so propagate the real failure instead of a misleading hook error.
            if logger is not None:
                logger.warning(
                    "diffusion.memory: group offload failed after installing hooks on %d "
                    "module(s) (%s); cannot fall back to whole-module offload",
                    installed,
                    exc,
                )
            raise
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload failed (%s); falling back to "
                "whole-module offload",
                exc,
            )
        return False


# ── generate-time activation guard ────────────────────────────────────────────
# The load-time plan cannot know the output resolution: a model is loaded once and then generates
# at whatever size the sliders say, so ``_plan_memory`` budgets the 1024x1024 default. That is the
# right call for PLACEMENT, but it means a request for a much larger frame is never checked against
# anything. This is the second half: a per-generation re-check, with the real dimensions.

# Opt-in escape hatch, mirroring the load-time one: the activation estimate is coarse, so an
# operator who believes it is wrong keeps a way through.
OVERSIZED_GENERATE_ENV = "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_GENERATE"


def _oversized_generate_override() -> bool:
    return os.environ.get(OVERSIZED_GENERATE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def image_activation_shortfall_message(
    *,
    device_memory: DeviceMemory,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
) -> Optional[str]:
    """A user-facing refusal when this generation's ACTIVATIONS cannot fit the free device
    budget, else None.

    Why this is a refusal and not another tuning knob: weights can be offloaded, activations
    cannot. Every offload tier moves WEIGHTS between host and device; the latents, attention
    buffers and VAE intermediates a forward pass allocates have to be on the device while it
    runs. So an activation estimate above the free budget is an overrun at every tier, including
    the lowest one, rather than a hint that a different placement would help. There is nothing
    left to degrade to, which is exactly the situation where refusing beats trying.

    Refusing also has to happen HERE rather than being left to torch. On Linux the overrun raises
    ``torch.OutOfMemoryError`` and the job dies cleanly. On Windows WDDM (including ROCm) it does
    not raise at all: the driver satisfies the overflow from system RAM as "non-local" GPU memory,
    so the process quietly grows past the card into tens of GB of host RAM and pagefile, and the
    desktop stops responding with no error anywhere. A ValueError here is a 400 with the reason,
    which is the clean refusal that failure mode never produces on its own.

    What it does NOT do is second-guess the load. The flat headroom this estimate is built on is a
    deliberately generous PLANNING figure whose job is to pick an offload tier, and the tier it
    picks already runs the 1024x1024 default on cards whose whole budget is under 8 GB (measured:
    a 8 GB card's safe budget is 5898 MiB against a 6963 MiB default-resolution estimate, and
    those generations complete). Treating that figure as a hard limit at or below the default
    would refuse work that succeeds today. So the refusal needs BOTH conditions: over the free
    budget, and over what the load already budgeted. That confines it to the resolution-driven
    overrun it is for -- the load reserved a 1 MP frame and the request is several times that --
    and leaves every generation at or below the default resolution exactly as it is.

    Fail-open on anything unknown (no free reading, no budget) and on any device class where the
    estimate or the offload story means something different, so a broken probe can never block a
    generation that would have worked."""
    if _oversized_generate_override():
        return None
    try:
        # Unified / system memory: offload moves bytes within one pool, "free" is a moving target
        # shared with the OS, and the load-time unified refusal already owns that device class.
        if getattr(device_memory, "is_unified", False):
            return None
        # CUDA / ROCm only. ROCm's torch reports device "cuda", so this covers both. XPU / MPS /
        # CPU keep today's behaviour exactly: their allocators and offload semantics differ and
        # this estimate was measured against a discrete VRAM pool.
        if getattr(device_memory, "device", None) != "cuda":
            return None
        free = getattr(device_memory, "free_mib", None)
        if free is None:
            return None  # no reading -> no verdict
        budget = _safe_device_budget_mib(device_memory)
        if budget is None:
            return None
        needed = estimate_image_runtime_mib(
            width = width,
            height = height,
            batch_size = batch_size,
            family = family,
        )
        # What the LOAD budgeted: the same estimator at the default resolution, i.e. the exact
        # call _plan_memory makes. Same function and same family hint, so the comparison is
        # between two points on one curve rather than between two different guesses.
        planned = estimate_image_runtime_mib(
            width = None,
            height = None,
            batch_size = 1,
            family = family,
        )
    except Exception:  # noqa: BLE001 — a broken probe must never block a generation
        return None
    if needed <= int(budget) or needed <= planned:
        return None
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    batch_note = f" at a batch of {batch}" if batch > 1 else ""
    # Two decimals, not one: the refusal is often decided by tens of MiB (the 1088x1920 report
    # needed 13,872 MiB against a 13,822 MiB budget), and one decimal prints both as "13.5 GB".
    return (
        f"Generating at {w}x{h}{batch_note} needs about {needed / 1024:.2f} GB of working memory, "
        f"but only about {int(budget) / 1024:.2f} GB is usable on this device (of the "
        f"{int(free) / 1024:.2f} GB currently free, after reserving room for fragmentation and "
        "other processes). Working memory holds this pass's latents and attention buffers, which "
        "cannot be offloaded to the CPU the way weights can, so no memory mode recovers this. "
        "Generate at a smaller resolution or a smaller batch size, free device memory by closing "
        f"other applications, or set {OVERSIZED_GENERATE_ENV}=1 to attempt it anyway."
    )


def raise_on_image_activation_shortfall(
    *,
    device_memory: DeviceMemory,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
    logger: Any = None,
) -> None:
    """Refuse a generation whose activations cannot fit the free device budget. No-op whenever
    ``image_activation_shortfall_message`` declines to produce a verdict.

    ``ValueError`` on purpose: ``/images/generate`` maps ValueError to HTTP 400 with the message
    as the reason, so this surfaces as an actionable refusal in the UI. RuntimeError there is
    reserved for the two client-state sentinels (not loaded / cancelled) and otherwise becomes an
    opaque 500."""
    message = image_activation_shortfall_message(
        device_memory = device_memory,
        width = width,
        height = height,
        batch_size = batch_size,
        family = family,
    )
    if message is None:
        return
    if logger is not None:
        logger.error("diffusion.memory: refusing oversized generation: %s", message)
    raise ValueError(message)
