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

# ── offload policies (what the loader does) ──────────────────────────
# none   -> all weights resident (fastest; fits only with room).
# model  -> enable_model_cpu_offload(): one top-level module on the GPU at a time.
# group  -> apply_group_offloading() on the transformer: stream a few blocks at a time with a
#           prefetch stream (lowest practical VRAM for the dominant module).
# sequential -> enable_sequential_cpu_offload(): submodule-level (broken for GGUF on diffusers
#           0.38, kept as an explicit escape hatch).
OFFLOAD_NONE = "none"
OFFLOAD_MODEL = "model"
OFFLOAD_GROUP = "group"
OFFLOAD_SEQUENTIAL = "sequential"

# Transformer blocks resident per group under group offloading: fewer = lower VRAM, more
# host<->device traffic. One is the lowest-VRAM setting.
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
    target: Any, attempts: int = 3, delay_s: float = 1.0
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
            # Query the CURRENT device (mem_get_info reports it); hardcoding 0 would inspect the
            # wrong GPU and misclassify discrete vs unified when the active device isn't 0.
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
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    requested_mode: Optional[str] = None,
    explicit_offload: bool = False,
) -> MemoryPlan:
    """Pick an offload policy + VAE memory savers for the current load.

    ``model_dense_mib`` is the resident size of all weights; ``companion_dense_mib`` is just the
    companions, which stay resident under group offload while the transformer streams block by
    block. ``explicit_offload`` is the back-compat ``cpu_offload=True`` request (forces model
    offload).

    Policies by speed/VRAM tradeoff:
      none  - everything resident: fastest, highest VRAM.
      group - stream the transformer, companions resident: near-resident speed, moderate cut.
      model - offload every component: lowest VRAM, slow.
    """
    mode = normalize_memory_mode(requested_mode) or MEMORY_MODE_AUTO
    can_offload = bool(getattr(target, "supports_model_cpu_offload", False))
    budget = _safe_device_budget_mib(device_memory)
    required = _sum_required(model_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # The resident floor under group offload: companions stay, the transformer streams.
    group_floor = _sum_required(companion_dense_mib, runtime_headroom_mib, base_overhead_mib)
    reasons: list[str] = []
    estimates: dict[str, Optional[int]] = {
        "safe_device_budget_mib": budget,
        "model_dense_mib": model_dense_mib,
        "companion_dense_mib": companion_dense_mib,
        "runtime_headroom_mib": runtime_headroom_mib,
        "base_overhead_mib": base_overhead_mib,
        "resident_required_mib": required,
        "group_floor_mib": group_floor,
    }

    def _group_fits() -> bool:
        # Group offload only helps if the resident companions fit; a too-big text encoder needs
        # whole-module offload.
        return group_floor is not None and budget is not None and group_floor <= budget

    if not can_offload or device_memory.is_unified:
        # MPS / CPU can't stream to a separate device; on unified memory offload just shuffles
        # bytes within the same pool.
        policy = OFFLOAD_NONE
        if device_memory.is_unified:
            reasons.append("unified/system memory: CPU offload frees no device memory")
        else:
            reasons.append(f"{device_memory.backend}: CPU offload unavailable; staying resident")
    elif mode == MEMORY_MODE_FAST:
        policy = OFFLOAD_NONE
        if budget is not None and required is not None and required > budget:
            # Doesn't fit resident: streamed transformer is the fastest offload.
            policy = OFFLOAD_GROUP if _group_fits() else OFFLOAD_MODEL
            reasons.append("fast requested but weights do not fit resident; offloading")
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
    else:
        policy = OFFLOAD_MODEL
        reasons.append("companions exceed budget; whole-module offload of every component")

    # The legacy cpu_offload flag applies only when no memory_mode was supplied (memory_mode
    # overrides it), so an explicit `fast` request stays resident even with the old flag on.
    if (
        explicit_offload
        and normalize_memory_mode(requested_mode) is None
        and policy == OFFLOAD_NONE
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload overrides resident placement")

    # VAE savers cap the high-res decode spike. Slicing (one image at a time) is EXACT, so enable
    # it on any offload tier / non-discrete backend. Tiling (spatial chunks) is only bit-identical
    # for a single tile (<=1MP), so restrict it to the lowest tiers (model / sequential) or no
    # spare device pool (MPS / CPU). Group offload keeps the VAE resident -> exact full-image
    # decode. On a roomy discrete GPU both stay off.
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
        # The GROUP plan set vae_tiling=False (VAE stays resident). Dropping to whole-module
        # offload is the low-VRAM case where the decode spike can OOM, so turn tiling on now.
        nonlocal tiling_engaged
        pipe.enable_model_cpu_offload(device = device)
        if not tiling_engaged:
            tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)

    policy = plan.offload_policy
    if policy == OFFLOAD_MODEL:
        pipe.enable_model_cpu_offload(device = device)
    elif policy == OFFLOAD_GROUP:
        if not _apply_group_offload(pipe, device, logger):
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


def _apply_group_offload(pipe: Any, device: str, logger: Any) -> bool:
    """Stream the transformer a few blocks at a time via diffusers group offloading, keeping the
    smaller components resident. Returns False (caller falls back to whole-module) on any failure."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    installed = 0  # streamed modules that already carry group-offload hooks
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        # A dual-DiT pipeline (e.g. Ideogram 4) carries a second denoiser as large as the first;
        # leaving it resident defeats this tier. Stream every DiT, keep only smaller companions.
        streamed: dict[str, Any] = {"transformer": transformer}
        for extra in ("transformer_2", "unconditional_transformer"):
            module = getattr(pipe, extra, None)
            if isinstance(module, torch.nn.Module):
                streamed[extra] = module

        onload = torch.device(device)
        use_stream = onload.type == "cuda"  # overlap H2D copies with compute
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        # On the CUDA stream path, overlap each block's H2D copy with compute: non_blocking
        # issues it async, record_stream defers the free until the copy's stream is done. Lossless
        # (only transfer scheduling changes). Gated on the signature so older diffusers still works.
        if use_stream:
            _params = inspect.signature(apply_group_offloading).parameters
            if "non_blocking" in _params:
                gkwargs["non_blocking"] = True
            if "record_stream" in _params:
                gkwargs["record_stream"] = True
        # Place the smaller components resident BEFORE attaching the transformer's group-offload
        # hooks: if a companion .to() OOMs we return False with NO hooks installed, so the caller's
        # whole-module fallback works (diffusers REJECTS enable_model_cpu_offload on a pipe that
        # already has group-offload hooks). The streamed transformer places itself via the hooks next.
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
            # An earlier streamed module already has hooks but a later one failed: the pipe is in a
            # PARTIAL group-offload state that enable_model_cpu_offload rejects, so the caller's
            # fallback would crash. Propagate the real failure (e.g. the OOM) instead of a
            # misleading hook error; the "no hooks installed" cases below fall back cleanly.
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
